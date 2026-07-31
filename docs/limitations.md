# Limitations

## Historical backtests

All substantive results are historical simulations. They can reveal
inconsistency and fragility, but they cannot establish future profitability.
Repeated inspection of the same history creates researcher degrees of freedom
even when each individual run is evaluated out of sample.

## Limited universe

The study concentrates on large U.S. equities. The static universe is narrow,
and the point-in-time extension is a manually assembled annual top-ten table
with a 19-symbol union. Results need not generalize to smaller firms, other
countries, delisted securities, asset classes, or a broader investable
universe.

## Market-regime dependence

The primary period contains strong equity uptrends as well as specific
volatility and rate regimes. Long-biased classifiers can inherit market drift.
The majority-class null addresses one symptom but does not make the sample
regime-neutral.

## Data-vendor dependence

Price histories, adjustments, timestamps, and coverage depend on yfinance,
Alpaca IEX, Polygon-compatible interfaces, FRED, FINRA, and CBOE data as
configured. Historical revisions and vendor-specific corporate-action handling
can change results. Alpaca IEX coverage is not equivalent to the consolidated
SIP.

## Data licensing and reproducibility

The original raw market data are not redistributed. Users must obtain lawful
access and satisfy each provider's terms. The code license does not grant rights
to third-party data. This limits exact third-party reproduction and prevents the
repository from shipping a turnkey historical dataset.

## Transaction-cost approximation

Fixed basis-point costs and time-of-day multipliers are approximations.
Spreads, fees, rebates, volatility, order type, fill probability, and liquidity
vary by symbol and time. The long-intraday cost stress is informative, but not a
complete transaction-cost analysis.

## Latency and market impact

The engine does not simulate network latency, data latency, order routing,
queue position, partial fills, adverse selection, or market impact. It assumes
the configured next-bar price is executable. This can be optimistic, especially
for short-horizon strategies.

## Bar resolution

OHLC bars do not reveal the path within a bar. Triple-barrier logic must choose
an event-order convention when both barriers are touched. Intraday rules cannot
represent quote-level microstructure.

## Hyperparameter search

Some thresholds and features are selected inside training folds, which limits
direct test leakage. The full historical research process still includes
choices about models, configurations, transformations, robustness checks, and
which observations to investigate. The repository has no complete preregistered
search ledger.

## Multiple-testing assumptions

BH-FDR controls an expected proportion under assumptions about the tested
hypotheses and their dependence. Strategy rows share data and can be strongly
correlated. FDR does not correct for hypotheses that were tried but not recorded,
nor does statistical significance imply economic significance.

The Deflated Sharpe Ratio is stored as a diagnostic, but no project-wide pass
threshold is encoded.

## Overlapping observations

Effective sample size and sequential trade simulation reduce overconfidence
from overlapping labels. They remain estimates based on the modeled holding
intervals and do not create true independence.

## Point-in-time universe quality

The annual membership table improves on backfilling today's winners but is not
a licensed point-in-time database. Annual reconstitution, manual boundary
decisions, ticker aliases, and the omission of a broader delisting universe
limit fidelity.

## Publication timing

External-series `available_from` dates use configured business-day lags. They
are deliberately conservative, but they are not verified against a timestamped
release database for every observation. Holidays, revisions, and exact
publication times can differ.

## Live validation

There is no paper-trading or live-trading validation, no broker integration,
and no operational risk control. The repository should not be described as a
production trading engine.

## Generalization

A negative result is conditional on the data, periods, universes, labels,
features, models, costs, and controls tested. It neither proves that all
short-horizon signals fail nor licenses extrapolation to other settings.

## Result-inventory limitation

The historical “0/442” v1–v4 statement is repeated in local reports but cannot
be reconstructed from one machine-readable manifest. Public materials identify
it as a historical reported count. The independently rebuilt flagship audit
uses five named source summaries and does not aggregate them into a fictional
independent experiment count.
