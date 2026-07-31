# Methodology

## Research objective

The research asks whether apparent short-horizon directional signals in large
U.S. equities remain credible after controlling for lookahead, temporal
dependence, strategy selection, transaction costs, universe drift, and repeated
hypothesis testing.

The system is designed to reject weak evidence. It is not designed to maximize
a displayed backtest metric, and no implementation path connects to a broker.

## Prediction target and labels

The default binary target in `features/targets.py` is:

```text
target[t] = +1 if close[t + horizon] / close[t] - 1 > 0
            -1 otherwise
```

The final unavailable horizon row is `NaN`, not assigned to either class. Most
configs set `horizon_bars: 1`.

The code also supports:

- a ternary target with either an ATR-relative or fixed-basis-point flat band;
- a triple-barrier target defined by an entry lag, upper and lower ATR
  barriers, and a maximum holding period.

For triple-barrier labels, stop-loss contact is evaluated before take-profit
contact when both could occur within the same OHLC bar. This is a conservative
but still bar-resolution-dependent convention.

## Instruments and universe

The early configurations use a static set of ten contemporary U.S. mega-cap
equities. Later configurations use `configs/universe_pit.yaml`, a year-specific
top-ten membership table whose union contains 19 tickers. In
`evaluation_mask` mode, the engine can process the union while retaining only
predictions from dates when each ticker belongs to the point-in-time universe.

The point-in-time file is a manually researched membership table, not a
licensed CRSP-like security master. Its notes disclose the ranking convention
and boundary judgments. Universe membership is therefore a documented
approximation, not ground truth.

## Timeframes and historical coverage

`data/loader.py` supports six configured intervals: `1m`, `5m`, `15m`, `1h`,
`1d`, and `1wk`. The archived machine-readable result directories contain five
of them: `5m`, `15m`, `1h`, `1d`, and `1wk`. No archived
`performance_summary.csv` in this workspace provides a verified `1m` result,
so the public documentation does not claim a six- or seven-timeframe empirical
comparison.

Historical coverage differs by provider and interval:

- yfinance daily data are requested over a configurable multi-year window;
- yfinance hourly data are limited to a shorter recent window;
- yfinance minute data have short retention windows;
- the long-history 5-minute configuration uses Alpaca IEX data beginning in
  2020; and
- weekly bars are resampled from daily OHLCV bars using the configured weekly
  rule.

Every claim about a sample period must therefore identify its configuration and
data provider.

## Data loading and cleaning

`data/loader.py` normalizes provider output to lowercase `open`, `high`, `low`,
`close`, and `volume` columns. It:

- sorts timestamps and removes duplicate timestamps, keeping the last value;
- drops rows with missing OHLC prices;
- fills missing volume with zero;
- does not forward-fill price bars or create missing calendar bars;
- caches normalized data as Parquet; and
- merges incremental refreshes by timestamp.

When `auto_adjust: true`, provider-adjusted OHLC data are requested. The exact
corporate-action method remains provider-dependent.

## Feature groups

The main feature pipeline in `features/pipeline.py` includes:

- lagged log returns;
- simple and exponential moving averages;
- price-to-average ratios;
- MACD, RSI, stochastic oscillator, and rate of change;
- ATR, rolling volatility, and Bollinger features;
- OBV, volume z-scores, and dollar volume;
- day-of-week and month fields for daily/weekly data;
- optional market-relative features;
- intraday VWAP, opening-range, time-of-day, and signed-volume features; and
- optional FRED, CBOE SKEW, and FINRA short-interest transforms.

Rolling and exponentially weighted transforms at timestamp *t* use observations
through *t* only. Labels and execution outcomes are added later.

## Model and strategy families

The implemented hypotheses are:

- four baselines: always-up, train-majority, seeded random, and a
  buy-and-hold-like long direction;
- moving-average crossover and RSI/Bollinger rules;
- logistic regression, random forest, and gradient boosting;
- opening-range breakout, VWAP mean reversion, and intraday momentum;
- cross-sectional momentum ranking; and
- a logistic second-stage meta-labeling filter over a primary model.

Model preprocessing, feature selection, calibration, and nested confidence
threshold selection are fitted on training folds. Gradient boosting can use
LightGBM, XGBoost, or scikit-learn depending on installed optional dependencies;
cross-backend results are not assumed to be bit-identical.

## Experiment unit

A row in `performance_summary.csv` normally represents one:

```text
configuration × ticker × timeframe × strategy × sample_type
```

This is the operational experiment unit for FDR adjustment within a generated
summary. It is not a preregistered enumeration of every human research decision
made over the project's lifetime. Repeated configurations and robustness runs
share data and hypotheses and must not be treated as independent evidence.

## Execution timing and returns

For the default one-bar binary target:

1. features and the signal are evaluated at timestamp *t*;
2. `entry_open[t]` is the open at *t + entry_lag*;
3. with the usual one-bar lag, `exit_close[t]` is the close of that same next
   bar; and
4. `execution_return` is next-bar close divided by next-bar open minus one.

The backtest multiplies execution return by the direction signal and subtracts
the configured round-trip cost once for active rows. Default historical configs
usually use 5 bps. Intraday configs can define spread plus slippage and
time-of-day cost multipliers.

Triple-barrier execution ends at the first modeled barrier or at expiry. Because
OHLC bars do not reveal within-bar event order, barrier outcomes remain an
approximation.

## Point-in-time and publication-lag treatment

External features cannot appear until `available_from`. The loaders retain the
original observation date, add a configured business-day release lag, transform
observations in their own time order, then merge onto bars with a backward-only
as-of join. A staleness tolerance prevents old values from being silently
carried indefinitely.

Verified configuration examples include:

- 1 business day for DGS10, T10Y2Y, VIXCLS, and CBOE SKEW;
- 3 business days for DCOILWTICO;
- 6 business days for DTWEXBGS; and
- 10 business days for FINRA short interest.

These are conservative configuration assumptions, not a vendor-grade database
of exact publication timestamps.

FINRA features are stored and joined per ticker. Alias validity windows handle
symbol changes and prevent a recycled symbol from contaminating another
issuer's history.

## Missing values

Warm-up windows, missing external observations, and stale releases remain
`NaN`. ML strategies can impute features inside their training pipeline;
rule-based strategies operate on the available columns. Frames without valid
target, entry, exit, and execution-return values are excluded before splitting.

## Reproducibility assumptions

- Random seeds come from the configuration and are applied to Python, NumPy,
  and supported model backends.
- Config-relative paths resolve relative to the config file.
- Synthetic data use a fixed seed and fictional symbol.
- Raw vendor data, exact provider revisions, and multi-gigabyte historical
  outputs are not distributed.
- Optional ML backends and platform-specific floating-point behavior can affect
  exact numerical reproduction.
- A full reproduction requires the original data licenses, provider access,
  environment details, and the exact configuration.

## Implementation map

| Concern | Primary implementation |
|---|---|
| Configuration | `core/config.py` |
| PIT universe | `core/universe.py` |
| Price providers and caching | `data/loader.py` |
| External data and release lag | `data/macro_loader.py`, `data/cboe_loader.py`, `data/finra_loader.py` |
| Features | `features/pipeline.py`, `features/intraday.py` |
| Labels and execution alignment | `features/targets.py` |
| Splits | `backtest/splitter.py` |
| Fold execution and costs | `backtest/engine.py` |
| Strategies | `strategies/` |
| FDR and DSR | `metrics/multiple_testing.py`, `reports/performance.py` |
| Reports | `reports/performance.py` |
