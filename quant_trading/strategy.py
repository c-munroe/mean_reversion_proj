"""
Trading strategy implementations.

This module defines data structures and classes for implementing trading
strategies.  Strategies are designed to be stateful so that they can
maintain internal history (e.g. rolling windows) while processing
streaming market data.  Each strategy exposes a single method,
`generate_signal`, which consumes a sequence of prices and returns a
`Signal` indicating what action to take at the current bar.

Currently included is a simple mean-reversion strategy based on the
z-score of the most recent price relative to a moving average. The
strategy can enter long positions when price is far below its mean,
enter short positions when price is far above its mean, and exit when
price reverts closer to the mean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class Signal:
    """A trading signal returned by a strategy.

    Attributes
    ----------
    action:
        One of "BUY", "SELL", or "HOLD". The meaning depends on the current
        position. When flat, "BUY" enters a long position and "SELL" enters
        a short position. When long, "SELL" exits the long position. When
        short, "BUY" covers the short position.
    confidence:
        A float measuring how strong the signal is. In mean reversion,
        a larger absolute z-score corresponds to a stronger signal. The
        backtest engine may use this value to scale position sizes.
    z_score:
        The raw z-score computed by the strategy.  Exposed for
        diagnostic purposes.
    """

    action: str
    confidence: float
    z_score: float


class MeanReversionStrategy:
    """Simple mean reversion strategy using z-scores.

    The strategy computes a rolling mean and standard deviation over a
    lookback window. When the latest price is sufficiently below the
    mean (z < long_entry_z), the strategy signals a long entry. When the
    price has reverted closer to the mean (z >= long_exit_z), the strategy
    signals an exit. Likewise, when the latest price is sufficiently above
    the mean (z > short_entry_z), the strategy signals a short entry. When
    the price has moved closer to the mean (z <= short_exit_z), the strategy
    signals an exit. Additional fields such as the computed z-score and
    confidence are returned for downstream consumers.

    Parameters
    ----------
    lookback: int
        Length of the rolling window in bars. Must be >= 2.

    long_entry_z: float
        Z-score threshold below which the strategy enters a long position.
        More negative values mean the price is further below its rolling mean.

    long_exit_z: float
        Z-score threshold at or above which the strategy exits a long position.
        This is usually closer to zero than long_entry_z, indicating the price
        has moved back toward its rolling mean.

    short_entry_z: float
        Z-score threshold above which the strategy enters a short position.
        More positive values mean the price is further above its rolling mean.

    short_exit_z: float
        Z-score threshold at or below which the strategy exits a short position.
        This is usually closer to zero than short_entry_z, indicating the price
        has moved back toward its rolling mean.
    """

    def __init__(self, 
            lookback: int, 
            long_entry_z: float, 
            long_exit_z: float,
            short_entry_z: float, 
            short_exit_z: float,
        ) -> None:
        if lookback < 2:
            raise ValueError("lookback must be at least 2")

        if long_entry_z >= long_exit_z:
            raise ValueError("long_entry_z should be less than long_exit_z")

        if short_entry_z <= short_exit_z:
            raise ValueError("short_entry_z should be greater than short_exit_z")

        self.lookback = lookback
        self.long_entry_z = long_entry_z
        self.long_exit_z = long_exit_z
        self.short_entry_z = short_entry_z
        self.short_exit_z = short_exit_z

    def generate_signal(
            self, history: Sequence[float], 
            current_price: float,
            current_position: float = 0.0,
        ) -> Signal:
        """Compute the trading signal based on historical prices and the latest price.

        Parameters
        ----------
        history: Sequence[float]
            Sequence of past prices used to compute the rolling mean and
            standard deviation. The length of `history` must equal
            `self.lookback`. It must not include the current price to avoid
            lookahead bias.

        current_price: float
            The price at which a trade would be executed if the signal triggers.
            This value is not included when computing the mean and standard
            deviation of the history.

        current_position: float
            Current position size. Positive means long, negative means short,
            and zero means flat.

        Returns
        -------
        Signal
            A signal indicating whether to buy, sell, or hold, along with
            a confidence measure and the computed z-score.
        """
        if len(history) != self.lookback:
            raise ValueError(f"history length {len(history)} != lookback {self.lookback}")

        window = np.asarray(history, dtype=float)

        mean = window.mean()
        std = window.std(ddof=0)
        if std == 0:
            return Signal(action="HOLD", confidence=0.0, z_score=0.0)

        z = (float(current_price) - mean) / std
        confidence = abs(z)

        # If flat, look for a new long or short entry.
        if current_position == 0:
            if z < self.long_entry_z:
                return Signal(action="BUY", confidence=confidence, z_score=z)

            if z > self.short_entry_z:
                return Signal(action="SELL", confidence=confidence, z_score=z)

            return Signal(action="HOLD", confidence=confidence, z_score=z)

        # If long, only look for long exit.
        if current_position > 0:
            if z >= self.long_exit_z:
                return Signal(action="SELL", confidence=confidence, z_score=z)

            return Signal(action="HOLD", confidence=confidence, z_score=z)

        # If short, only look for short exit / buy to cover.
        if current_position < 0:
            if z <= self.short_exit_z:
                return Signal(action="BUY", confidence=confidence, z_score=z)

            return Signal(action="HOLD", confidence=confidence, z_score=z)