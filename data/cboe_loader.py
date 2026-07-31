from __future__ import annotations

import argparse
import csv
import io
import logging
import urllib.request
from pathlib import Path

import pandas as pd

from data.macro_loader import MacroDataError, MacroDataLoader, normalize_macro_frame

LOGGER = logging.getLogger(__name__)

CBOE_INDEX_CSV_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{index_id}_History.csv"


class CboeIndexLoader(MacroDataLoader):
    """Load daily CBOE index histories (e.g. SKEW) with explicit release timing.

    Reuses the macro loader's cache/lag semantics: the parquet cache stores raw
    observations only, and available_from = observation date + release lag is
    applied at load time. Only the provider fetch differs.
    """

    @classmethod
    def from_external_config(
        cls, external: dict, config_path: str | Path, *, offline: bool | None = None
    ) -> "CboeIndexLoader":
        from core.config import resolve_path

        cboe = dict(external.get("cboe", {}))
        series = cboe.get("series", [])
        if not isinstance(series, list) or not series:
            raise MacroDataError("features.external.cboe.series must be a non-empty list when present.")
        for spec in series:
            if not isinstance(spec, dict) or not spec.get("id"):
                raise MacroDataError("Each features.external.cboe.series entry must be a mapping with an 'id'.")
        return cls(
            cache_dir=resolve_path(config_path, cboe.get("cache_dir", "data/cache_cboe")),
            series=series,
            offline=bool(cboe.get("offline", False) if offline is None else offline),
            start=str(cboe.get("start", "2000-01-01")),
            rate_limit_seconds=float(cboe.get("rate_limit_seconds", 0.0)),
        )

    def _fetch(self, series_id: str) -> pd.DataFrame:
        url = CBOE_INDEX_CSV_URL.format(index_id=series_id)
        LOGGER.info("Downloading CBOE index %s from %s", series_id, url)
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = response.read().decode("utf-8")
        except Exception as exc:
            raise MacroDataError(f"Failed to download CBOE index {series_id}: {exc}") from exc
        return parse_cboe_csv(payload, series_id)


def parse_cboe_csv(payload: str, series_id: str) -> pd.DataFrame:
    """Parse a CBOE *_History.csv payload (DATE,<INDEX> with MM/DD/YYYY dates).

    Blank or non-numeric observations are dropped rather than filled, mirroring
    the FRED parser: gaps must surface as staleness downstream.
    """
    rows = list(csv.reader(io.StringIO(payload)))
    if not rows or len(rows[0]) < 2:
        raise MacroDataError(f"CBOE payload for {series_id} is not a two-column CSV.")
    header = [column.strip().lower() for column in rows[0]]
    if header[0] != "date":
        raise MacroDataError(f"CBOE payload for {series_id} has unexpected header {rows[0]!r}.")
    dates: list[pd.Timestamp] = []
    values: list[float] = []
    for row in rows[1:]:
        if len(row) < 2:
            continue
        raw_value = row[1].strip()
        if raw_value in {"", "."}:
            continue
        try:
            value = float(raw_value)
        except ValueError:
            continue
        dates.append(pd.Timestamp(row[0].strip()))
        values.append(value)
    if not dates:
        raise MacroDataError(f"CBOE payload for {series_id} contains no valid observations.")
    frame = pd.DataFrame({"value": values}, index=pd.DatetimeIndex(dates, name="observation_date"))
    return normalize_macro_frame(frame)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch and cache CBOE index series (e.g. SKEW)")
    parser.add_argument("--config", required=True, help="Config YAML with a features.external.cboe block")
    parser.add_argument("--offline", action="store_true", help="Only verify the existing cache; no downloads")
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    from core.config import load_config
    from data.macro_loader import external_config, external_features_enabled

    config = load_config(args.config)
    if not external_features_enabled(config):
        print("features.external.enabled is false; nothing to fetch.")
        return 0
    loader = CboeIndexLoader.from_external_config(external_config(config), args.config, offline=args.offline)
    series_data = loader.load_all(refresh=not args.offline)
    for series_id, frame in series_data.items():
        print(
            f"{series_id:>12}: {len(frame)} observations, "
            f"{frame.index.min().date()} .. {frame.index.max().date()}, "
            f"cache {loader.cache_path(series_id)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
