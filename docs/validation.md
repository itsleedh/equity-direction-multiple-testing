# Validation Design

## Why an ordinary random split is unsuitable

Financial bars are ordered, autocorrelated, and often linked through overlapping
labels and common market regimes. Randomly shuffling them can:

- train on observations later than the test observation;
- mix nearly identical adjacent windows across train and test;
- leak overlapping label horizons across the boundary; and
- make regime-specific patterns look stable.

The repository therefore uses temporal splits and treats the order of
observations as part of the research design.

## Expanding walk-forward validation

`WalkForwardSplitter` builds a chronological training slice followed by a held
out test slice. In expanding mode the training start remains fixed and the
training end moves forward as test windows advance.

For this implementation, the ordinary walk-forward gap before a test window is
the configured `purge_lookback_bars + embargo_bars`. The names should not be
read as proof of a textbook implementation: the code's walk-forward mode places
both components before the test window. The tests assert the actual positional
gap.

No test observation is used to fit the model that predicts it. Each model
instance is cloned and fitted within a fold. Nested confidence thresholds use
training probabilities and are applied once to the test fold.

## Purge

The purge width protects the boundary from feature and label overlap. For
triple-barrier targets, the splitter automatically widens the configured purge
to at least:

```text
entry_lag_bars + max_holding_bars - 1
```

This prevents a training label's modeled holding interval from reaching the
first test observation.

## Embargo

In ordinary walk-forward mode, the configured embargo contributes to the
pre-test exclusion gap described above. In CPCV mode, each selected test group
is blocked directly, the purge width is applied before that group, and the
embargo width is applied after it.

This implementation-specific distinction matters when comparing results with
papers or libraries that use a different embargo convention.

## Combinatorial Purged Cross-Validation

The CPCV mode divides ordered observations into groups, selects combinations of
test groups, and removes surrounding training observations. It can reveal
path-dependence and unstable performance across alternative temporal
partitions.

CPCV test observations appear in multiple folds. The repository does not treat
those duplicated fold rows as independent observations. `reports/performance.py`
marks overlap-sensitive inference and compounded returns invalid for raw CPCV
fold rows and provides bar-level collapse summaries for inspection.

## Overlapping labels and effective sample size

Triple-barrier events can remain open for several bars, so adjacent labels share
future price intervals. Raw row counts overstate independent information.

The code calculates interval concurrency and average label uniqueness. Weighted
win-rate inference uses effective sample size, and sequential reporting admits a
new trade only after the previous modeled position has exited. This avoids
compounding mutually overlapping positions as though all could be held
independently.

## Out-of-sample prediction generation

The backtest engine:

1. removes rows without a valid target and execution outcome;
2. obtains chronological folds;
3. fits a fresh strategy clone on that fold's training slice;
4. performs any nested threshold selection on training data;
5. predicts the held-out test slice; and
6. labels persisted records as `train` or `test`.

Only `sample_type == "test"` rows enter the FDR headline counts. Long intraday
runs can disable persistence of the growing training predictions without
changing test predictions.

## Cross-symbol contamination controls

Price-derived features are constructed separately for each ticker. Market-level
external series may be broadcast to all tickers, while FINRA short-interest
frames are stored in an explicit per-symbol mapping. The as-of merge selects
only the matching ticker's frame. Unit tests inject distinct values and assert
that one symbol's feature never appears in another symbol's frame.

Cross-sectional ranking is intentionally the exception: it compares symbols at
the same timestamp for a cross-sectional hypothesis. It is implemented in a
separate pipeline and should not be confused with accidental feature sharing.

## Statistical controls

For valid test rows, the reporting layer calculates:

- Wilson confidence intervals;
- two-sided binomial p-values against 0.5;
- two-sided binomial p-values against the target-majority rate;
- Benjamini–Hochberg adjusted p-values for both nulls; and
- Deflated Sharpe Ratio diagnostics using the number of strategies in each
  ticker/timeframe/sample group.

`beats_majority_after_fdr` requires both majority-null rejection and a tested
win rate above the majority rate. This prevents a significantly worse strategy
from being counted as an edge.

BH-FDR controls the expected false-discovery proportion under its assumptions;
it does not remove dependence, undisclosed researcher degrees of freedom, or
economic irrelevance. Robustness runs and transparent negative reporting remain
necessary.

## Major invariants protected by tests

The `unittest` suite covers:

- removing future tail data does not change earlier features;
- binary and triple-barrier labels align with their documented horizons;
- entry price occurs after the signal timestamp;
- train and test positions do not overlap;
- purge and embargo boundaries match configuration;
- triple-barrier purge expands with the holding horizon;
- models and meta-models fit training data only;
- nested thresholds are selected on training data only;
- active-row transaction costs are subtracted with the correct sign and units;
- random baselines reproduce under a fixed seed;
- symbol-specific external values do not cross tickers;
- release lag blocks unpublished observations;
- BH-FDR matches known reference cases;
- CPCV and overlapping-label invalid metrics are masked;
- point-in-time membership respects year boundaries; and
- the complete non-ML pipeline runs on deterministic synthetic OHLCV data.

Run:

```bash
python -m unittest discover -s tests
```
