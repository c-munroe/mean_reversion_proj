"""
quant_trading package.

This package contains a modular implementation of a long/short
mean-reversion trading strategy together with a reusable backtesting
engine, risk management helpers, metrics utilities, and an execution
layer for live trading via Alpaca's API. The goal of this package is to
provide clean and maintainable building blocks for quantitative trading
projects.

All modules are designed with type hints, clear docstrings, and minimal
external state to encourage testability and readability. See the
README.md at the root of this repository for details on how to use
these components in backtests and live trading.
"""

from .backtest import BacktestEngine, BacktestResult, Trade
from .config import BacktestConfig, LiveConfig, RiskConfig, StrategyConfig
from .metrics import (
    compute_returns,
    max_drawdown,
    sharpe_ratio,
    volatility,
    win_rate,
)
from .risk import FixedFractionalSizer, compute_stop_price
from .strategy import MeanReversionStrategy, Signal

__all__ = [
    "StrategyConfig",
    "BacktestConfig",
    "RiskConfig",
    "LiveConfig",
    "MeanReversionStrategy",
    "Signal",
    "BacktestEngine",
    "BacktestResult",
    "Trade",
    "FixedFractionalSizer",
    "compute_stop_price",
    "compute_returns",
    "sharpe_ratio",
    "max_drawdown",
    "volatility",
    "win_rate",
]