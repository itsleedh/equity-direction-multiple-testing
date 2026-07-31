#!/usr/bin/env python3
"""Run the public end-to-end smoke experiment on fictional OHLCV data."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from core.config import load_config, resolve_path  # noqa: E402
from main import run_research_pipeline  # noqa: E402
from reports.data_summary import build_data_summary, write_data_summary  # noqa: E402
from reports.performance import summarize_predictions, write_performance_reports  # noqa: E402

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def load_synthetic_sample(path: str | Path) -> pd.DataFrame:
    """Load and validate a source-controlled fictional sample."""
    sample_path = Path(path)
    frame = pd.read_csv(sample_path, parse_dates=["date"])
    missing = sorted(set(["date", "synthetic", *OHLCV_COLUMNS]) - set(frame.columns))
    if missing:
        raise ValueError(f"Synthetic sample is missing required columns: {missing}")
    marker = frame["synthetic"].astype(str).str.lower()
    if not marker.isin({"true", "1"}).all():
        raise ValueError("Refusing to run: every sample row must be explicitly marked synthetic.")
    frame = frame.set_index("date").loc[:, OHLCV_COLUMNS]
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("Synthetic sample timestamps must be unique and increasing.")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Synthetic OHLC prices must be positive.")
    if (frame["high"] < frame[["open", "close"]].max(axis=1)).any():
        raise ValueError("Synthetic high must be at least max(open, close).")
    if (frame["low"] > frame[["open", "close"]].min(axis=1)).any():
        raise ValueError("Synthetic low must be at most min(open, close).")
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/example_config.yaml")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    synthetic = dict(config.get("synthetic", {}))
    sample_path = resolve_path(config_path, synthetic.get("sample_file", "../data/sample/synthetic_ohlcv.csv"))
    symbol = str(synthetic.get("symbol", "SYNTH"))
    timeframe = str(synthetic.get("timeframe", "1d"))

    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)

    bars = load_synthetic_sample(sample_path)
    data = {(symbol, timeframe): bars}
    predictions, skipped = run_research_pipeline(
        data=data,
        config=config,
        include_ml=bool(config.get("ml", {}).get("enabled", False)),
        intraday_enabled=False,
    )
    if predictions.empty:
        reasons = "; ".join(sorted({item.get("reason", "unknown") for item in skipped}))
        raise RuntimeError(f"Synthetic smoke run produced no predictions: {reasons}")

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else resolve_path(config_path, config.get("reports", {}).get("output_dir", "../artifacts/smoke"))
    )
    data_summary = build_data_summary(data)
    write_data_summary(data_summary, output_dir)
    performance_summary = summarize_predictions(predictions)
    paths = write_performance_reports(
        predictions=predictions,
        summary=performance_summary,
        skipped=pd.DataFrame(skipped),
        output_dir=output_dir,
    )

    test_rows = performance_summary[performance_summary["sample_type"] == "test"]
    print(f"Synthetic smoke run complete: {len(test_rows)} OOS summary rows.")
    print(f"Sample: {sample_path}")
    print(f"Output: {output_dir}")
    print(f"Headline passes: {int(test_rows['beats_majority_after_fdr'].sum())}")
    print(f"Reports written: {len(paths) + 1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
