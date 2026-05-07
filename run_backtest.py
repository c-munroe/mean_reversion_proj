"""
Example script for running a backtest of the mean reversion strategy.

This script downloads historical SPY data from Yahoo Finance, configures
the long/short mean reversion strategy, runs the backtest, prints a
summary of the results, and saves a portfolio value chart.

It also runs a small parameter comparison so you can see whether the
short side is helping or hurting.
"""

import itertools

import matplotlib.pyplot as plt
import pandas as pd

from quant_trading import (
    BacktestConfig,
    BacktestEngine,
    FixedFractionalSizer,
    MeanReversionStrategy,
    RiskConfig,
    StrategyConfig,
)
from quant_trading.data import fetch_yfinance_data


def get_price_data(strategy_cfg: StrategyConfig, backtest_cfg: BacktestConfig) -> pd.Series:
    """Fetch adjusted close prices from Yahoo Finance, with synthetic fallback."""
    try:
        data = fetch_yfinance_data(
            strategy_cfg.symbol,
            start=backtest_cfg.start_date,
            end=backtest_cfg.end_date,
            interval="1d",
        )

        prices = data["Adj Close"].dropna().squeeze()

        if prices.empty:
            raise ValueError("Empty price series returned from data provider")

        return prices

    except Exception as e:
        print(
            f"Warning: could not fetch real data ({type(e).__name__}: {e}). "
            "Using synthetic data."
        )

        import numpy as np

        num_days = 252 * 2
        np.random.seed(42)
        returns = np.random.normal(0.0001, 0.01, num_days)
        prices_array = 100 * np.cumprod(1 + returns)
        index = pd.date_range(
            start=backtest_cfg.start_date,
            periods=num_days,
            freq="B",
        )

        return pd.Series(prices_array, index=index)


def run_single_backtest(
    prices: pd.Series,
    strategy_cfg: StrategyConfig,
    risk_cfg: RiskConfig,
    backtest_cfg: BacktestConfig,
    use_sizer: bool = True,
    allow_fractional: bool = False,
):
    """Run one backtest and return the result."""
    strategy = MeanReversionStrategy(
        lookback=strategy_cfg.lookback,
        long_entry_z=strategy_cfg.long_entry_z,
        long_exit_z=strategy_cfg.long_exit_z,
        short_entry_z=strategy_cfg.short_entry_z,
        short_exit_z=strategy_cfg.short_exit_z,
    )

    sizer = None

    if use_sizer:
        sizer = FixedFractionalSizer(risk_per_trade=risk_cfg.risk_per_trade)

    engine = BacktestEngine(
        strategy=strategy,
        prices=prices,
        config=backtest_cfg,
        risk_config=risk_cfg,
        sizer=sizer,
        allow_fractional=allow_fractional,
    )

    return engine.run()


def print_summary(result, backtest_cfg: BacktestConfig) -> None:
    """Print overall metrics and long/short breakdown."""
    final_value = result.portfolio_values[-1]
    pnl = final_value - backtest_cfg.initial_cash
    pnl_pct = pnl / backtest_cfg.initial_cash

    long_trades = [trade for trade in result.trades if trade.side == "LONG"]
    short_trades = [trade for trade in result.trades if trade.side == "SHORT"]

    long_pnl = sum(trade.pnl for trade in long_trades)
    short_pnl = sum(trade.pnl for trade in short_trades)

    print("Backtest Results:")
    print("-----------------")

    for key, value in result.metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print()
    print(f"Initial portfolio value: ${backtest_cfg.initial_cash:.2f}")
    print(f"Final portfolio value: ${final_value:.2f}")
    print(f"PnL: ${pnl:.2f}")
    print(f"PnL %: {pnl_pct:.2%}")
    print(f"Number of trades: {len(result.trades)}")
    print()
    print(f"Long trades: {len(long_trades)}")
    print(f"Short trades: {len(short_trades)}")
    print(f"Long PnL: ${long_pnl:.2f}")
    print(f"Short PnL: ${short_pnl:.2f}")

    #print("\nTrades:")
    #for trade in result.trades:
    #    print(trade)


def plot_result(result) -> None:
    """Plot and save portfolio value."""
    plt.figure(figsize=(10, 5))
    plt.plot(result.portfolio_values)
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Portfolio Value ($)")
    plt.tight_layout()
    plt.savefig("results.png", dpi=200)
    plt.close()


def parameter_comparison(
    prices: pd.Series,
    risk_cfg: RiskConfig,
    backtest_cfg: BacktestConfig,
) -> None:
    """Test a broad set of long/short parameter combinations."""
    results = []

    lookbacks = [20, 30, 40, 50]
    long_entries = [-1.5, -2.0, -2.5]
    long_exits = [-1.0, -0.75, -0.5]
    short_entries = [2.0, 2.25, 2.5, 2.75]
    short_exits = [0.0, 0.25, 0.5, 0.75]

    total_tests = (
        len(lookbacks)
        * len(long_entries)
        * len(long_exits)
        * len(short_entries)
        * len(short_exits)
    )

    test_num = 0
    print(f"\nRunning {total_tests} parameter tests...")

    for lookback, long_entry, long_exit, short_entry, short_exit in itertools.product(
        lookbacks,
        long_entries,
        long_exits,
        short_entries,
        short_exits,
    ):
        test_num += 1

        if test_num % 100 == 0:
            print(f"Completed {test_num}/{total_tests} tests...")

        strategy_cfg = StrategyConfig(
            symbol="SPY",
            lookback=lookback,
            long_entry_z=long_entry,
            long_exit_z=long_exit,
            short_entry_z=short_entry,
            short_exit_z=short_exit,
        )

        result = run_single_backtest(
            prices=prices,
            strategy_cfg=strategy_cfg,
            risk_cfg=risk_cfg,
            backtest_cfg=backtest_cfg,
            use_sizer=True,
            allow_fractional=False,
        )

        final_value = result.portfolio_values[-1]
        pnl = final_value - backtest_cfg.initial_cash

        long_pnl = sum(trade.pnl for trade in result.trades if trade.side == "LONG")
        short_pnl = sum(trade.pnl for trade in result.trades if trade.side == "SHORT")
        short_count = sum(1 for trade in result.trades if trade.side == "SHORT")

        results.append(
            {
                "lookback": lookback,
                "long_entry": long_entry,
                "long_exit": long_exit,
                "short_entry": short_entry,
                "short_exit": short_exit,
                "final_value": final_value,
                "pnl": pnl,
                "sharpe": result.metrics["sharpe_ratio"],
                "max_drawdown": result.metrics["max_drawdown"],
                "num_trades": result.metrics["num_trades"],
                "long_pnl": long_pnl,
                "short_pnl": short_pnl,
                "short_trades": short_count,
            }
        )

    comparison = pd.DataFrame(results)
    comparison = comparison.sort_values(by="pnl", ascending=False)

    print("\nParameter Comparison:")
    print("---------------------")
    print(
        comparison[
            [
                "lookback",
                "long_entry",
                "long_exit",
                "short_entry",
                "short_exit",
                "final_value",
                "pnl",
                "sharpe",
                "max_drawdown",
                "num_trades",
                "long_pnl",
                "short_pnl",
                "short_trades",
            ]
        ].head(10).to_string(index=False)
    )


def main() -> None:
    strategy_cfg = StrategyConfig(
        symbol="SPY",
        lookback=40,
        long_entry_z=-2.0,
        long_exit_z=-0.75,
        short_entry_z=2.25,
        short_exit_z=0.0,
    )

    risk_cfg = RiskConfig(
        risk_per_trade=0.01,
        max_drawdown=0.2,
        transaction_cost=0.0005,
        slippage=0.0001,
        stop_loss_pct=0.03,
    )

    backtest_cfg = BacktestConfig(
        initial_cash=10_000.0,
        start_date="2022-01-01",
        end_date="2026-01-01",
        risk_free_rate=0.0,
    )

    prices = get_price_data(strategy_cfg, backtest_cfg)

    result = run_single_backtest(
        prices=prices,
        strategy_cfg=strategy_cfg,
        risk_cfg=risk_cfg,
        backtest_cfg=backtest_cfg,
        use_sizer=True,
        allow_fractional=False,
    )

    print_summary(result, backtest_cfg)
    #parameter_comparison(prices, risk_cfg, backtest_cfg)
    plot_result(result)


if __name__ == "__main__":
    main()