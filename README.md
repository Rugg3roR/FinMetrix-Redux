# FinMetrix

An algorithmic forex trading scanner built on MetaTrader 5, Marimo, and XGBoost.

Scans 28 currency pairs on M30 candles using a predictive EMA alignment strategy,
ML-gated signal scoring, ADR-based position sizing, and progressive exit management.

## What this repository contains

`finmetrix.py` — the live scanner and dashboard (Marimo reactive notebook).

The strategy modules, research pipeline, and configuration parameters are private.

## Requirements

- Python 3.11+
- MetaTrader 5 (connected broker account)
- [Marimo](https://marimo.io) — `pip install marimo`
- `polars`, `numpy`, `xgboost`, `altair`
- Private strategy modules (not included)

## Running

```bash
marimo run finmetrix.py
```

## Architecture

```
finmetrix.py          ← this file (public)
modules/              ← strategy modules (private)
  config.py           ← all constants and signal parameters
  features.py         ← feature engineering, triple barrier labels
  data_engine.py      ← MT5 data fetching, signal logging
  model_handler.py    ← XGBoost training and inference
  journal_engine.py   ← trade journal, Kelly sizing
research/             ← backtesting and optimisation pipeline (private)
  backtest_engine.py  ← shared computation core
  strategy_analyst.py ← manual backtesting tool
  optuna_search.py    ← Bayesian parameter optimisation
  gridsearch.py       ← targeted dense search
data/                 ← all data files (gitignored)
```

## Signal logic

The scanner fires on five simultaneous conditions:

1. **Trend gate** — ema144 (72-hour EMA on M30) sloping in signal direction
2. **Predictive alignment** — forecast ≥3/4 EMA pairs bullish at n+1 (BUY) or ≤1/4 (SELL)
3. **Alignment delta** — alignment stable or improving
4. **RSI MA** — smoothed RSI above 52 (BUY) or below 48 (SELL)
5. **Volume** — short-term volume above its long-term average

Signals are filtered by ML probability (XGBoost), signal age, cooldown, and
correlation with open positions.

## Position sizing

SL and TP are derived from the pair's 14-day Average Daily Range (ADR):
- SL = 25% of ADR
- TP1 = 25% of ADR (close 50% of position)
- TP2 = 50% of ADR (close 25% of position)
- Trailing stop = 20% of ADR behind extreme (final 25%)

Position size is calculated via fractional Kelly from live trade history.

## Licence

MIT
