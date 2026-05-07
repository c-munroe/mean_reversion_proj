"""
Reusable backtesting engine for trading strategies.

This module implements a straightforward event-driven backtester
supporting both long and short mean reversion strategies. It accepts a
strategy object, price data, configuration objects for backtesting and
risk, and an optional position sizer. The engine produces a history of
portfolio values and a record of trades, and can compute basic
performance metrics via `quant_trading.metrics`.

The design intentionally avoids lookahead bias: signals are computed
using only historical bars, and the current bar's price is used for
execution. Transaction costs, slippage, and stop losses are incorporated
to approximate real-world trading conditions.

Short-selling is modeled using simple marked-to-market accounting:
short entries increase cash by the sale proceeds, short positions are
stored as negative share quantities, and portfolio value is computed as
cash plus position times current price. This does not model borrow fees,
margin calls, hard-to-borrow restrictions, or short-sale locate rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from .config import BacktestConfig, RiskConfig
from .metrics import compute_returns, max_drawdown, sharpe_ratio, volatility, win_rate
from .risk import FixedFractionalSizer, compute_stop_price
from .strategy import MeanReversionStrategy, Signal


@dataclass
class Trade:
    """Record of a single completed trade.

    Attributes
    ----------
    entry_idx:
        Index into the price series at which the trade was opened.
    exit_idx:
        Index into the price series at which the trade was closed.
    entry_price:
        Execution price on entry, including transaction cost and slippage.
    exit_price:
        Execution price on exit, including transaction cost and slippage.
    size:
        Number of shares traded. Positive means long, negative means short.
    side:
        "LONG" for long trades and "SHORT" for short trades.
    pnl:
        Profit or loss in dollars.
    return_pct:
        Percentage return on the trade relative to absolute entry notional.
    exit_reason:
        Reason the position was closed, such as "signal", "stop_loss",
        "max_drawdown", or "end_of_test".
    """

    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    size: float
    side: str
    pnl: float
    return_pct: float
    exit_reason: str


@dataclass
class BacktestResult:
    """Container for backtest outputs.

    Attributes
    ----------
    portfolio_values:
        Time series of portfolio values at each bar.
    trades:
        List of completed trades.
    metrics:
        Dictionary of computed performance statistics including cumulative
        return, Sharpe ratio, max drawdown, volatility, win rate, and number
        of trades.
    """

    portfolio_values: List[float]
    trades: List[Trade]
    metrics: dict


class BacktestEngine:
    """Engine for running a mean reversion strategy on historical prices."""

    def __init__(
        self,
        strategy: MeanReversionStrategy,
        prices: pd.Series,
        config: BacktestConfig,
        risk_config: RiskConfig,
        sizer: Optional[FixedFractionalSizer] = None,
        allow_fractional: bool = True,
    ) -> None:
        if not isinstance(prices, pd.Series):
            raise TypeError("prices must be a pandas Series")

        if prices.empty:
            raise ValueError("price series is empty")

        if prices.isna().any():
            raise ValueError(
                "price series contains NaNs; fill or drop missing values before backtesting"
            )

        if not prices.index.is_monotonic_increasing:
            raise ValueError("price series must be sorted in ascending order")

        try:
            numeric_prices = prices.astype(float)
        except Exception as exc:
            raise ValueError("price series must be numeric") from exc

        if (numeric_prices <= 0).any():
            raise ValueError("price series must contain only positive prices")

        if config.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")

        if risk_config.transaction_cost < 0:
            raise ValueError("transaction_cost must be non-negative")

        if risk_config.slippage < 0:
            raise ValueError("slippage must be non-negative")

        if risk_config.stop_loss_pct < 0 or risk_config.stop_loss_pct >= 1:
            raise ValueError("stop_loss_pct must be in the range [0, 1)")

        if risk_config.max_drawdown < 0 or risk_config.max_drawdown > 1:
            raise ValueError("max_drawdown must be between 0 and 1")

        self.strategy = strategy
        self.prices = numeric_prices
        self.config = config
        self.risk_config = risk_config
        self.sizer = sizer
        self.allow_fractional = allow_fractional

    def _buy_price(self, raw_price: float) -> float:
        """Execution price for a buy order, including cost and slippage."""
        return raw_price * (
            1.0 + self.risk_config.transaction_cost + self.risk_config.slippage
        )

    def _sell_price(self, raw_price: float) -> float:
        """Execution price for a sell order, including cost and slippage."""
        return raw_price * (
            1.0 - self.risk_config.transaction_cost - self.risk_config.slippage
        )

    @staticmethod
    def _portfolio_value(cash: float, position: float, mark_price: float) -> float:
        """Compute marked-to-market portfolio value."""
        return cash + position * mark_price

    @staticmethod
    def _trade_pnl(entry_price: float, exit_price: float, size: float) -> float:
        """Compute P&L for either a long or short trade.

        Positive size means long. Negative size means short.
        """
        return (exit_price - entry_price) * size

    @staticmethod
    def _trade_return(entry_price: float, size: float, pnl: float) -> float:
        """Compute trade return relative to absolute entry notional."""
        entry_notional = abs(entry_price * size)

        if entry_notional == 0:
            return 0.0

        return pnl / entry_notional

    def _round_quantity(self, qty: float) -> float:
        """Apply fractional-share settings to a desired quantity."""
        if qty <= 0:
            return 0.0

        if self.allow_fractional:
            return float(qty)

        return float(np.floor(qty))

    def _desired_quantity(
        self,
        cash: float,
        raw_entry_price: float,
        entry_price: float,
        planned_stop_exit_price: float,
    ) -> float:
        """Compute desired trade quantity using the sizer or simple cash allocation.

        If a sizer is provided, quantity is based on account risk and stop distance.
        Otherwise, the engine uses approximately one-times cash notional.

        Quantity is capped so the backtest does not intentionally open more than
        one-times current equity notional.
        """
        if cash <= 0:
            return 0.0

        if self.sizer is not None:
            desired_qty = self.sizer.size_position(
                account_value=cash,
                entry_price=entry_price,
                stop_price=planned_stop_exit_price,
            )
        else:
            desired_qty = cash / raw_entry_price

        max_qty = cash / raw_entry_price
        desired_qty = min(desired_qty, max_qty)

        return self._round_quantity(desired_qty)

    def run(self) -> BacktestResult:
        """Execute the backtest and return a result object."""
        lookback = self.strategy.lookback
        prices = self.prices
        n = len(prices)

        cash = float(self.config.initial_cash)
        position = 0.0

        entry_price: Optional[float] = None
        entry_idx: Optional[int] = None
        entry_stop_price: Optional[float] = None
        entry_side: Optional[str] = None

        portfolio_values: List[float] = [cash]
        trades: List[Trade] = []

        peak_value = cash

        def close_position(exit_idx: int, raw_price: float, exit_reason: str) -> None:
            """Close the current position and record the completed trade."""
            nonlocal cash
            nonlocal position
            nonlocal entry_price
            nonlocal entry_idx
            nonlocal entry_stop_price
            nonlocal entry_side
            nonlocal peak_value

            if position == 0:
                return

            size = position
            side = entry_side if entry_side is not None else ("LONG" if size > 0 else "SHORT")

            if size > 0:
                # Closing a long means selling shares.
                exit_price = self._sell_price(raw_price)
                cash += size * exit_price
            else:
                # Closing a short means buying shares back.
                exit_price = self._buy_price(raw_price)
                cash -= abs(size) * exit_price

            trade_entry_price = entry_price if entry_price is not None else raw_price
            trade_entry_idx = entry_idx if entry_idx is not None else exit_idx

            pnl = self._trade_pnl(
                entry_price=trade_entry_price,
                exit_price=exit_price,
                size=size,
            )
            return_pct = self._trade_return(
                entry_price=trade_entry_price,
                size=size,
                pnl=pnl,
            )

            trades.append(
                Trade(
                    entry_idx=trade_entry_idx,
                    exit_idx=exit_idx,
                    entry_price=trade_entry_price,
                    exit_price=exit_price,
                    size=size,
                    side=side,
                    pnl=pnl,
                    return_pct=return_pct,
                    exit_reason=exit_reason,
                )
            )

            position = 0.0
            entry_price = None
            entry_idx = None
            entry_stop_price = None
            entry_side = None
            peak_value = cash

        def open_long(entry_idx_value: int, raw_price: float) -> None:
            """Open a long position."""
            nonlocal cash
            nonlocal position
            nonlocal entry_price
            nonlocal entry_idx
            nonlocal entry_stop_price
            nonlocal entry_side

            entry_exec_price = self._buy_price(raw_price)

            stop_trigger_price = compute_stop_price(
                entry_exec_price,
                self.risk_config.stop_loss_pct,
                side="long",
            )
            planned_stop_exit_price = self._sell_price(stop_trigger_price)

            desired_qty = self._desired_quantity(
                cash=cash,
                raw_entry_price=raw_price,
                entry_price=entry_exec_price,
                planned_stop_exit_price=planned_stop_exit_price,
            )

            if desired_qty <= 0:
                return

            max_affordable_qty = cash / entry_exec_price
            desired_qty = min(desired_qty, max_affordable_qty)
            desired_qty = self._round_quantity(desired_qty)

            cost = desired_qty * entry_exec_price

            if desired_qty <= 0 or cost > cash:
                return

            cash -= cost
            position = desired_qty
            entry_price = entry_exec_price
            entry_idx = entry_idx_value
            entry_stop_price = stop_trigger_price
            entry_side = "LONG"

        def open_short(entry_idx_value: int, raw_price: float) -> None:
            """Open a short position."""
            nonlocal cash
            nonlocal position
            nonlocal entry_price
            nonlocal entry_idx
            nonlocal entry_stop_price
            nonlocal entry_side

            entry_exec_price = self._sell_price(raw_price)

            stop_trigger_price = compute_stop_price(
                entry_exec_price,
                self.risk_config.stop_loss_pct,
                side="short",
            )
            planned_stop_exit_price = self._buy_price(stop_trigger_price)

            desired_qty = self._desired_quantity(
                cash=cash,
                raw_entry_price=raw_price,
                entry_price=entry_exec_price,
                planned_stop_exit_price=planned_stop_exit_price,
            )

            if desired_qty <= 0:
                return

            # Entering short means selling borrowed shares.
            proceeds = desired_qty * entry_exec_price

            cash += proceeds
            position = -desired_qty
            entry_price = entry_exec_price
            entry_idx = entry_idx_value
            entry_stop_price = stop_trigger_price
            entry_side = "SHORT"

        if n <= lookback + 1:
            metrics = {
                "cumulative_return": 0.0,
                "sharpe_ratio": float("nan"),
                "max_drawdown": 0.0,
                "volatility": 0.0,
                "win_rate": 0.0,
                "num_trades": 0,
            }
            return BacktestResult(
                portfolio_values=portfolio_values,
                trades=trades,
                metrics=metrics,
            )

        for i in range(lookback, n - 1):
            history_slice = prices.iloc[i - lookback : i].to_list()
            current_price = float(prices.iloc[i])

            current_portfolio = self._portfolio_value(
                cash=cash,
                position=position,
                mark_price=current_price,
            )

            peak_value = max(peak_value, current_portfolio)

            drawdown = (
                (current_portfolio - peak_value) / peak_value
                if peak_value > 0
                else 0.0
            )

            # Liquidate if the account violates the max drawdown rule.
            if position != 0 and drawdown < -self.risk_config.max_drawdown:
                close_position(
                    exit_idx=i,
                    raw_price=current_price,
                    exit_reason="max_drawdown",
                )
                portfolio_values.append(cash)
                continue

            # Long stop-loss: price falls to or below the stop.
            if position > 0 and entry_stop_price is not None:
                if current_price <= entry_stop_price:
                    close_position(
                        exit_idx=i,
                        raw_price=current_price,
                        exit_reason="stop_loss",
                    )
                    portfolio_values.append(cash)
                    continue

            # Short stop-loss: price rises to or above the stop.
            if position < 0 and entry_stop_price is not None:
                if current_price >= entry_stop_price:
                    close_position(
                        exit_idx=i,
                        raw_price=current_price,
                        exit_reason="stop_loss",
                    )
                    portfolio_values.append(cash)
                    continue

            signal: Signal = self.strategy.generate_signal(
                history=history_slice,
                current_price=current_price,
                current_position=position,
            )

            # Flat: BUY enters long, SELL enters short.
            if position == 0:
                if signal.action == "BUY":
                    open_long(entry_idx_value=i, raw_price=current_price)
                elif signal.action == "SELL":
                    open_short(entry_idx_value=i, raw_price=current_price)

            # Long: SELL exits the long.
            elif position > 0:
                if signal.action == "SELL":
                    close_position(
                        exit_idx=i,
                        raw_price=current_price,
                        exit_reason="signal",
                    )

            # Short: BUY covers the short.
            elif position < 0:
                if signal.action == "BUY":
                    close_position(
                        exit_idx=i,
                        raw_price=current_price,
                        exit_reason="signal",
                    )

            portfolio_value = self._portfolio_value(
                cash=cash,
                position=position,
                mark_price=current_price,
            )
            portfolio_values.append(portfolio_value)

        # Liquidate any remaining long or short position at the final price.
        final_price = float(prices.iloc[-1])

        if position != 0:
            close_position(
                exit_idx=n - 1,
                raw_price=final_price,
                exit_reason="end_of_test",
            )

        final_portfolio_value = self._portfolio_value(
            cash=cash,
            position=position,
            mark_price=final_price,
        )
        portfolio_values.append(final_portfolio_value)

        returns = compute_returns(portfolio_values)
        trade_returns = [trade.return_pct for trade in trades]

        if portfolio_values and all(value > 0 for value in portfolio_values):
            mdd = max_drawdown(portfolio_values)
        else:
            # If equity becomes non-positive, treat drawdown as a total loss.
            mdd = -1.0

        metrics = {
            "cumulative_return": (
                (portfolio_values[-1] / self.config.initial_cash) - 1.0
                if portfolio_values
                else 0.0
            ),
            "sharpe_ratio": sharpe_ratio(
                returns,
                risk_free_rate=self.config.risk_free_rate,
            ),
            "max_drawdown": mdd,
            "volatility": volatility(returns),
            "win_rate": win_rate(trade_returns),
            "num_trades": len(trades),
        }

        return BacktestResult(
            portfolio_values=portfolio_values,
            trades=trades,
            metrics=metrics,
        )