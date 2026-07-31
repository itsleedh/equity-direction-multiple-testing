#!/usr/bin/env python3
"""Build small public result audits from named local summary artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from metrics.multiple_testing import benjamini_hochberg  # noqa: E402

RUNS = {
    "pit_top10_daily": "reports/output_pit_top10/performance_summary.csv",
    "fred_macro_daily": "reports/output_ext_macro_on/performance_summary.csv",
    "cboe_skew_daily": "reports/output_ext_skew_on/performance_summary.csv",
    "pit_union_static_daily": "reports/output_pit_union_static/performance_summary.csv",
    "intraday_long_5m": "reports/output_intraday_long/performance_summary.csv",
}
INTRADAY_COST_SOURCE = "reports/output_intraday_long/breakeven_survival.csv"


def number(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_run(run_id: str, relative_source: str) -> dict[str, object]:
    source = REPOSITORY_ROOT / relative_source
    with source.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    test_rows = [row for row in rows if row.get("sample_type") == "test"]
    if not test_rows:
        raise ValueError(f"{relative_source} has no test rows.")

    pvalues = [number(row.get("binom_pvalue_majority")) for row in test_rows]
    _, reject_majority = benjamini_hochberg(pvalues, alpha=0.05)
    beats = 0
    for row, rejected in zip(test_rows, reject_majority, strict=True):
        weighted = number(row.get("win_rate_weighted"))
        tested = (
            weighted
            if row.get("target_mode") == "triple_barrier" and math.isfinite(weighted)
            else number(row.get("win_rate"))
        )
        majority = number(row.get("target_majority_rate"))
        beats += int(bool(rejected and math.isfinite(tested) and math.isfinite(majority) and tested > majority))

    return {
        "run_id": run_id,
        "source_file": relative_source,
        "source_sha256": sha256(source),
        "test_rows": len(test_rows),
        "oos_predictions": sum(int(number(row.get("predictions"))) for row in test_rows),
        "fdr_pass_vs_0_5": sum(str(row.get("significant_after_fdr", "")).lower() in {"true", "1"} for row in test_rows),
        "fdr_rejections_vs_majority": int(reject_majority.sum()),
        "beats_majority_after_fdr": beats,
    }


def build_cost_summary() -> list[dict[str, object]]:
    source = REPOSITORY_ROOT / INTRADAY_COST_SOURCE
    with source.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_cost: dict[float, dict[str, int]] = {}
    for row in rows:
        if row.get("sample_type") != "test":
            continue
        cost = number(row.get("cost_bps"))
        aggregate = by_cost.setdefault(cost, {"surviving": 0, "total": 0})
        aggregate["surviving"] += int(number(row.get("surviving_strategies")))
        aggregate["total"] += int(number(row.get("total_strategies")))
    return [
        {
            "cost_bps": cost,
            "surviving_strategy_symbol_combinations": values["surviving"],
            "total_strategy_symbol_combinations": values["total"],
            "source_file": INTRADAY_COST_SOURCE,
            "source_sha256": sha256(source),
        }
        for cost, values in sorted(by_cost.items())
    ]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty public result file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY_ROOT / "reports" / "public")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()

    audits = [audit_run(run_id, source) for run_id, source in RUNS.items()]
    write_csv(output_dir / "verified_flagship_runs.csv", audits)
    write_csv(output_dir / "intraday_cost_survival.csv", build_cost_summary())
    print(f"Wrote {len(audits)} flagship audits and the intraday cost aggregate to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
