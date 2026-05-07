"""
Configuration dataclasses for the quant_trading package.

These dataclasses centralize all configurable parameters related to the
strategy, backtesting environment, risk management, and live trading. By
encapsulating parameters in well-typed objects, we make it easy to
document defaults and pass around settings in a single object rather than
relying on global variables or ad hoc dictionaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class StrategyConfig:
    """Parameters specific to the mean reversion strategy.

    Attributes
    ----------
    symbol:
        The ticker symbol to trade. Defaults to "SPY".

    lookback:
        Number of bars to use when computing the rolling mean and
        standard deviation for the z-score. Must be >= 2.

    long_entry_z:
        Z-score threshold below which a long position is entered.
        Example: -2.0 means buy when price is far below its rolling mean.

    long_exit_z:
        Z-score threshold at or above which an existing long position is
        exited. This is usually closer to zero than long_entry_z.

    short_entry_z:
        Z-score threshold above which a short position is entered.
        Example: 2.0 means short when price is far above its rolling mean.

    short_exit_z:
        Z-score threshold at or below which an existing short position is
        exited. This is usually closer to zero than short_entry_z.

    use_fractional_shares:
        If True, permits fractional share sizes when sizing long positions.
        Short positions should generally use whole-share quantities.
    """

    symbol: str = "SPY"
    lookback: int = 40

    long_entry_z: float = -2.0
    long_exit_z: float = -0.75

    short_entry_z: float = 2.25
    short_exit_z: float = 0.0

    use_fractional_shares: bool = True


@dataclass
class RiskConfig:
    """Parameters for risk management and transaction cost assumptions.

    Attributes
    ----------
    risk_per_trade:
        Fraction of total capital to risk on any single trade.
        For example, 0.01 corresponds to 1%.

    max_drawdown:
        Maximum allowable drawdown on the total portfolio during the
        backtest. If the running drawdown exceeds this value, the
        backtest can liquidate open positions and stop. Expressed as a
        decimal fraction, such as 0.2 for 20%.

    transaction_cost:
        Proportional transaction cost applied on both buys and sells,
        expressed as a decimal. For example, 0.0005 equals 5 basis points.

    slippage:
        Slippage as a fraction of the trade price. This parameter inflates
        buy prices and deflates sell prices to approximate execution costs.

    stop_loss_pct:
        Hard stop-loss as a fraction of the entry price. For long positions,
        the stop is below entry. For short positions, the stop is above entry.
        Example: 0.03 means a 3% stop.
    """

    risk_per_trade: float = 0.01
    max_drawdown: float = 0.2
    transaction_cost: float = 0.0005
    slippage: float = 0.0001
    stop_loss_pct: float = 0.03


@dataclass
class BacktestConfig:
    """Parameters governing the backtesting environment.

    Attributes
    ----------
    initial_cash:
        Starting cash balance for the backtest.

    start_date:
        ISO format date string for the beginning of the backtest.

    end_date:
        ISO format date string for the end of the backtest, non-inclusive.

    risk_free_rate:
        Annualized risk-free rate used when computing Sharpe ratios.
        When computing daily returns, this rate is divided by 252 trading days.
    """

    initial_cash: float = 10_000.0
    start_date: str = "2022-01-01"
    end_date: str = "2024-01-01"
    risk_free_rate: float = 0.0


@dataclass
class LiveConfig:
    """Configuration parameters for live trading via the Alpaca API.

    Attributes
    ----------
    api_key:
        Alpaca API key. This should be provided via environment variables
        or another secure method.

    api_secret:
        Alpaca secret key. This should be provided via environment variables
        or another secure method.

    base_url:
        Base URL for the Alpaca API. Use the paper trading URL for testing
        and the live URL for production.

    symbol:
        The ticker symbol to trade. Defaults to match the strategy.

    timeframe:
        Timeframe for historical bar data and the live polling loop. An
        `alpaca.data.timeframe.TimeFrame` object may be passed at runtime.

    dollar_position:
        Dollar amount allocated to each new trade. Long entries may use
        notional orders, while short entries should be converted into
        whole-share quantities before submitting the order.

    sleep_seconds:
        Number of seconds to sleep between polling the market in the live loop.
        This should be greater than or equal to the bar duration.
    """

    api_key: Optional[str] = field(default=None)
    api_secret: Optional[str] = field(default=None)
    base_url: str = "https://paper-api.alpaca.markets"
    symbol: str = "SPY"
    timeframe: Any = None
    dollar_position: float = 300.0
    sleep_seconds: int = 60