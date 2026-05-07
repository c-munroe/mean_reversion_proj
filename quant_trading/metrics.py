"""
Performance metrics for trading strategies.

This module contains functions to compute common performance statistics
including cumulative returns, Sharpe ratio, maximum drawdown, volatility,
and win rate. These metrics allow objective comparison of trading
strategies and help identify risk/return trade-offs.

Definitions
-----------
* **Sharpe Ratio**: A measure of risk-adjusted return defined as
  mean excess return divided by the standard deviation of returns.
  A higher Sharpe ratio indicates more return per unit of risk.
* **Maximum Drawdown (MDD)**: The largest percentage decline from a
  portfolio's peak value to its subsequent trough over a period of time.
  It is calculated as (trough - peak) / peak.
* **Volatility**: The annualized standard deviation of returns.
* **Win Rate**: The fraction of trades that are profitable.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def compute_returns(values: Sequence[float]) -> np.ndarray:
    """Compute simple period-to-period returns from portfolio values.

    Returns are computed as `(V_t - V_{t-1}) / V_{t-1}`. The result
    has length one less than the input sequence.
    """
    arr = np.asarray(values, dtype=float)

    if arr.size < 2:
        return np.array([], dtype=float)

    previous_values = arr[:-1]

    if np.any(previous_values == 0):
        raise ValueError("Portfolio values must be nonzero to compute returns.")

    return np.diff(arr) / previous_values


def sharpe_ratio(
    returns: Sequence[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Compute the annualized Sharpe ratio of a return series.

    Parameters
    ----------
    returns: Sequence[float]
        Sequence of periodic returns, such as daily or minute returns.
    risk_free_rate: float
        Annual risk-free rate expressed as a decimal fraction.
    periods_per_year: int
        Number of return periods per year, such as 252 for daily returns.

    Returns
    -------
    float
        The annualized Sharpe ratio. If there are too few returns or the
        standard deviation is zero, the function returns NaN.
    """
    r = np.asarray(returns, dtype=float)

    if r.size < 2:
        return float("nan")

    excess_returns = r - risk_free_rate / periods_per_year
    sigma = excess_returns.std(ddof=1)

    if sigma == 0:
        return float("nan")

    return float(np.sqrt(periods_per_year) * excess_returns.mean() / sigma)


def max_drawdown(values: Sequence[float]) -> float:
    """Compute the maximum drawdown of a portfolio value series.

    The maximum drawdown is the minimum of `(V_t - peak_t) / peak_t`
    across the time series. A more negative value indicates a larger
    drawdown.
    """
    arr = np.asarray(values, dtype=float)

    if arr.size == 0:
        return 0.0

    if np.any(arr <= 0):
        raise ValueError("Portfolio values must be positive to compute drawdown.")

    cumulative_max = np.maximum.accumulate(arr)
    drawdowns = (arr - cumulative_max) / cumulative_max

    return float(drawdowns.min())


def volatility(returns: Sequence[float], periods_per_year: int = 252) -> float:
    """Compute annualized volatility of a return series.

    Parameters
    ----------
    returns: Sequence[float]
        Sequence of periodic returns.
    periods_per_year: int
        Number of return periods per year.

    Returns
    -------
    float
        Annualized standard deviation of returns.
    """
    r = np.asarray(returns, dtype=float)

    if r.size < 2:
        return 0.0

    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def win_rate(trade_returns: Sequence[float]) -> float:
    """Compute the fraction of trades that were profitable.

    A trade is considered profitable if its P&L is strictly greater
    than zero.

    Parameters
    ----------
    trade_returns: Sequence[float]
        Sequence of realized trade returns or dollar profits. The
        definition must be consistent across trades.

    Returns
    -------
    float
        Ratio of profitable trades to total trades. If there are no
        trades, returns 0.
    """
    tr = np.asarray(trade_returns, dtype=float)

    if tr.size == 0:
        return 0.0

    winners = (tr > 0).sum()

    return float(winners / tr.size)