# Mean Reversion Trading Strategy (Z-Score Based)

## Overview

This project implements a long/short mean reversion trading strategy using Z-scores on SPY price data. The strategy identifies short-term deviations from a rolling average and trades on the expectation that prices will partially revert toward the mean.

The repository includes:

- A modular backtesting engine
- Long and short trading logic
- Risk management and performance metrics
- Live paper trading integration using Alpaca
- Clean, extensible architecture for further strategy development

---

## Strategy Logic

The strategy computes the Z-score of the current price relative to a rolling lookback window:

```text
z = (price - rolling mean) / rolling standard deviation
```

Trading rules:

- **Long entry:** `z < -2.0`  
  Price is significantly below its rolling mean, so the strategy buys.

- **Long exit:** `z >= -0.5`  
  Price has partially reverted toward the mean, so the strategy sells the long position.

- **Short entry:** `z > 2.0`  
  Price is significantly above its rolling mean, so the strategy sells short.

- **Short exit / cover:** `z <= 0.5`  
  Price has reverted to the mean, so the strategy buys to cover the short position.

---

## Parameters

| Parameter | Value |
|---|---:|
| Symbol | SPY |
| Lookback Window | 40 periods |
| Long Entry Threshold | `z < -2.0` |
| Long Exit Threshold | `z >= -0.75` |
| Short Entry Threshold | `z > 2.25` |
| Short Exit Threshold | `z <= 0.0` |
| Live Dollar Position | `$1000` per new live trade|

> Note: Parameters were selected to balance trade frequency, drawdown, and risk-adjusted returns. Further optimization and walk-forward testing would be needed before using this strategy with real capital.

---

## Backtest Results

Using historical SPY data, the strategy produced the following sample results:

```text
cumulative_return: 0.0706
sharpe_ratio: 1.0106
max_drawdown: -0.0181
volatility: 0.0362
win_rate: 0.9231
num_trades: 13
total_portfolio_value: $10706.23
```

### Key Takeaways

- Positive return of roughly 7% over the test period
- Sharpe ratio near 1.0, suggesting solid risk-adjusted performance
- Low maximum drawdown of roughly 1.8%
- High win rate, consistent with mean reversion-style behavior
- Strategy supports both long and short positions

> These results are historical and do not guarantee future performance. They also do not account for every real-world trading constraint, such as borrow fees, hard-to-borrow restrictions, margin requirements, or live execution delays.

---

## Portfolio Performance

![Portfolio Value](results.png)

---

## Project Structure

```text
run_backtest.py      Runs the full backtest
strategy.py         Mean reversion signal logic
backtest.py         Long/short backtesting engine
risk.py             Position sizing and stop-loss helpers
metrics.py          Performance evaluation metrics
data.py             Data fetching from Yahoo Finance / Alpaca
execution.py        Alpaca live trading execution layer
live_trading.py     Real-time paper trading script
config.py           Centralized configuration dataclasses
```

---

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Backtest

```bash
python run_backtest.py
```

### 3. Run Live Trading in Alpaca Paper Mode

Set your Alpaca paper trading API keys:

```bash
export APCA_API_KEY_ID=your_key
export APCA_API_SECRET_KEY=your_secret
```

Then run:

```bash
python live_trading.py
```

---

## Features

- Long and short mean reversion signals
- Modular, production-style project architecture
- No lookahead bias in backtesting
- Transaction cost and slippage modeling
- Stop-loss and drawdown risk controls
- Performance metrics including Sharpe ratio, drawdown, volatility, and win rate
- Alpaca paper trading integration
- Configurable strategy, risk, backtest, and live trading settings

---

## Limitations

- Backtest does not model borrow fees or hard-to-borrow restrictions
- Backtest does not model margin calls or short-sale locate requirements
- Live shorting depends on Alpaca account permissions and asset borrow availability
- Current implementation uses a simple Z-score mean reversion signal
- Strategy has only been tested on SPY in this setup
- Live stop-loss prices are currently computed and logged, but true live stop orders may require additional order logic

---

## Future Improvements

- Parameter optimization using grid search or Bayesian optimization
- Walk-forward testing across multiple market regimes
- Multi-asset testing across ETFs and liquid equities
- Pairs trading or spread-based mean reversion extension
- More realistic short-selling assumptions, including borrow fees and margin requirements
- Bracket orders or true stop-loss orders in live trading
- Docker or cloud deployment for persistent paper trading
- Logging dashboard for trades, signals, and portfolio value

---

## Author

Christopher Munroe