from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_ohlcv(rows: int = 320, *, freq: str = "B", seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-01", periods=rows, freq=freq)
    drift = 0.0004
    returns = rng.normal(drift, 0.012, size=rows)
    close = 100.0 * np.exp(np.cumsum(returns))
    open_ = close * (1.0 + rng.normal(0, 0.002, size=rows))
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.001, 0.01, size=rows))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.001, 0.01, size=rows))
    volume = rng.integers(1_000_000, 5_000_000, size=rows)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )
