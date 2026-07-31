#!/usr/bin/env python3
"""Generate a small deterministic OHLCV sample for offline interface tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPOSITORY_ROOT / "data" / "sample" / "synthetic_ohlcv.csv"
DEFAULT_SEED = 20260731
DEFAULT_ROWS = 360


def generate_synthetic_ohlcv(*, rows: int = DEFAULT_ROWS, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Return fictional daily OHLCV bars with a fixed seed and explicit marker."""
    if rows < 240:
        raise ValueError("Synthetic sample requires at least 240 rows for feature warm-up and walk-forward testing.")

    rng = np.random.default_rng(seed)
    index = pd.date_range("2018-01-02", periods=rows, freq="B", name="date")
    innovations = rng.normal(loc=0.0, scale=0.008, size=rows)
    close = 100.0 * np.exp(np.cumsum(innovations))
    open_ = close * (1.0 + rng.normal(0.0, 0.0015, size=rows))
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.001, 0.007, size=rows))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.001, 0.007, size=rows))
    volume = rng.integers(100_000, 900_000, size=rows)

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "synthetic": True,
        },
        index=index,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = generate_synthetic_ohlcv(rows=args.rows, seed=args.seed)
    frame.to_csv(output, index=True, float_format="%.10f")
    print(f"Wrote {len(frame)} deterministic synthetic rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
