from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_data_summary(data: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (ticker, timeframe), frame in sorted(data.items()):
        rows.append(
            {
                "ticker": ticker,
                "timeframe": timeframe,
                "rows": int(len(frame)),
                "start": "" if frame.empty else str(frame.index.min()),
                "end": "" if frame.empty else str(frame.index.max()),
                "missing_ohlc_rows_after_cleaning": int(
                    frame[["open", "high", "low", "close"]].isna().any(axis=1).sum()
                )
                if not frame.empty
                else 0,
            }
        )
    return pd.DataFrame(rows)


def write_data_summary(summary: pd.DataFrame, output_dir: str | Path) -> Path:
    path = Path(output_dir) / "data_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False)
    return path
