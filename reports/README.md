# Report Artifacts

## Public files

`reports/public/` contains small, reviewable result audits:

- `verified_flagship_runs.csv` — five named local summary files, their SHA-256
  checksums, test-row counts, aggregated OOS predictions, historical 0.5-null
  FDR passes, recomputed majority-null FDR rejections, and final
  above-majority passes.
- `intraday_cost_survival.csv` — aggregate 5-minute strategy–symbol survival at
  each cost level.

Generate both with:

```bash
python scripts/build_public_summary.py
```

The generator reads these local source files:

```text
reports/output_pit_top10/performance_summary.csv
reports/output_ext_macro_on/performance_summary.csv
reports/output_ext_skew_on/performance_summary.csv
reports/output_pit_union_static/performance_summary.csv
reports/output_intraday_long/performance_summary.csv
reports/output_intraday_long/breakeven_survival.csv
```

The source paths are included in the public CSV. The source summaries and the
larger prediction artifacts are ignored from the public repository because
they derive from non-redistributed inputs and the output tree totals several
gigabytes.

## Local generated reports

`main.py` writes configuration-specific directories containing some or all of:

- `data_summary.csv`
- `predictions.csv`
- `performance_summary.csv`
- `significance_summary.csv`
- `overfitting_gap.csv`
- `breakeven_survival.csv`
- `skipped_strategies.csv`
- threshold and label summaries
- CPCV fold/bar summaries
- sequential triple-barrier summaries
- intraday daily-P&L and cost-sensitivity tables
- optional PNG visualizations

These are generated research artifacts, not source code. They remain local
under ignored `reports/output*` paths.

## Result provenance

The public aggregate script does not:

- read raw market data;
- rerun a model;
- merge outputs across runs as if they were independent;
- invent missing values; or
- change the stored historical 0.5-null verdict.

It recomputes the majority-null BH-FDR decision from the p-values already
stored in each named summary using `metrics.multiple_testing.benjamini_hochberg`.

The historical v1–v4 “0/442” statement is not represented as a reconstructed
CSV result because no single experiment manifest recreates that denominator.
See `docs/results.md`.
