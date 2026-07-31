"""Compare survivorship-biased vs point-in-time universe runs.

Reads predictions.csv from multiple run output directories and computes an
equal-weight daily portfolio of each run's evaluated tickers over a common
window, so the universe-selection effect is isolated from cost and strategy
effects. Uses B1_always_up test rows because execution_return there is the
bar's tradable long return independent of any strategy signal.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def load_portfolio_returns(run_dir: str | Path) -> pd.Series:
    predictions = pd.read_csv(Path(run_dir) / "predictions.csv", parse_dates=["date"])
    rows = predictions[(predictions["sample_type"] == "test") & (predictions["strategy"] == "B1_always_up")]
    if rows.empty:
        raise ValueError(f"No B1_always_up test rows in {run_dir}")
    return rows.groupby("date")["execution_return"].mean().sort_index()


def portfolio_stats(returns: pd.Series) -> dict[str, float]:
    growth = float((1.0 + returns).prod())
    years = len(returns) / TRADING_DAYS_PER_YEAR
    annualized = growth ** (1.0 / years) - 1.0 if years > 0 else float("nan")
    return {
        "days": int(len(returns)),
        "start": returns.index.min().date().isoformat(),
        "end": returns.index.max().date().isoformat(),
        "cumulative_return": growth - 1.0,
        "annualized_return": annualized,
        "annualized_volatility": float(returns.std(ddof=1)) * TRADING_DAYS_PER_YEAR**0.5,
    }


def compare_runs(run_dirs: dict[str, str | Path], *, start: str | None = None) -> pd.DataFrame:
    portfolios = {name: load_portfolio_returns(path) for name, path in run_dirs.items()}
    window_start = max(series.index.min() for series in portfolios.values())
    if start is not None:
        window_start = max(window_start, pd.Timestamp(start))
    window_end = min(series.index.max() for series in portfolios.values())
    rows = []
    for name, series in portfolios.items():
        trimmed = series[(series.index >= window_start) & (series.index <= window_end)]
        rows.append({"run": name, **portfolio_stats(trimmed)})
    return pd.DataFrame(rows)


def summarize_fdr(run_dirs: dict[str, str | Path]) -> pd.DataFrame:
    rows = []
    for name, path in run_dirs.items():
        summary = pd.read_csv(Path(path) / "performance_summary.csv")
        test = summary[summary["sample_type"] == "test"]
        rows.append(
            {
                "run": name,
                "test_rows": int(len(test)),
                "significant_after_fdr": int(test["significant_after_fdr"].sum()),
                "min_pvalue_fdr_bh": float(test["pvalue_fdr_bh"].min()),
                "median_ml_win_rate": float(test[test["strategy"].str.startswith("M")]["win_rate"].median()),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="PIT vs survivorship-biased run comparison")
    parser.add_argument("--pit", required=True, help="Output dir of the PIT-masked run")
    parser.add_argument("--current", required=True, help="Output dir of the current-universe run")
    parser.add_argument("--union", help="Optional output dir of the union no-mask ablation run")
    parser.add_argument("--start", help="Optional common-window start date (YYYY-MM-DD)")
    parser.add_argument("--output", help="Optional CSV path for the portfolio comparison table")
    args = parser.parse_args()

    run_dirs: dict[str, str | Path] = {"pit_top10": args.pit, "current10_static": args.current}
    if args.union:
        run_dirs["union_static"] = args.union

    comparison = compare_runs(run_dirs, start=args.start)
    fdr = summarize_fdr(run_dirs)
    print("Equal-weight long portfolio over common test window:")
    print(comparison.to_string(index=False))
    print("\nFDR / strategy summary:")
    print(fdr.to_string(index=False))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        comparison.merge(fdr, on="run").to_csv(output_path, index=False)
        print(f"\nWrote comparison: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
