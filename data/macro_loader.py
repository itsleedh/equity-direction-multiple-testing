from __future__ import annotations

import argparse
import csv
import io
import logging
import time
import urllib.request
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


class MacroDataError(RuntimeError):
    """Raised when an external macro series cannot be served."""


class MacroDataLoader:
    """Load non-price daily time series (FRED) with explicit release timing.

    The parquet cache stores raw observations only (a 'value' column indexed by
    observation date). Release timing is applied at load time: each series gets
    an 'available_from' column = observation date + release_lag_business_days,
    so a config change in the lag never requires a refetch. Merging into price
    bars must key on 'available_from', never on the observation date.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        series: list[dict],
        offline: bool = False,
        start: str = "2000-01-01",
        rate_limit_seconds: float = 0.0,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.series = [dict(spec) for spec in series]
        self.offline = offline
        self.start = pd.Timestamp(start)
        self.rate_limit_seconds = rate_limit_seconds
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(cls, config: dict, config_path: str | Path, *, offline: bool | None = None) -> "MacroDataLoader":
        from core.config import resolve_path

        external = external_config(config)
        series = external.get("series", [])
        if not isinstance(series, list) or not series:
            raise MacroDataError(
                "features.external.series must be a non-empty list when external features are enabled."
            )
        for spec in series:
            if not isinstance(spec, dict) or not spec.get("id"):
                raise MacroDataError("Each features.external.series entry must be a mapping with an 'id'.")
        return cls(
            cache_dir=resolve_path(config_path, external.get("cache_dir", "data/cache_macro")),
            series=series,
            offline=bool(external.get("offline", False) if offline is None else offline),
            start=str(external.get("start", "2000-01-01")),
            rate_limit_seconds=float(external.get("rate_limit_seconds", 0.0)),
        )

    def cache_path(self, series_id: str) -> Path:
        safe_id = str(series_id).upper().replace("/", "-")
        return self.cache_dir / f"{safe_id}.parquet"

    def load(self, spec: dict, *, refresh: bool = True) -> pd.DataFrame:
        """Return one series as columns [value, available_from] indexed by observation date."""
        series_id = str(spec["id"]).upper().strip()
        cache_path = self.cache_path(series_id)
        cached = self._read_cache(cache_path)

        if not refresh or self.offline:
            if cached is None:
                raise MacroDataError(f"No cached macro data for {series_id} at {cache_path}")
            LOGGER.info("Using cached macro series %s from %s", series_id, cache_path)
            observations = cached
        else:
            observations = self._fetch(series_id)
            self._write_cache(cache_path, observations)

        observations = observations[observations.index >= self.start]
        if observations.empty:
            raise MacroDataError(f"Macro series {series_id} has no observations on or after {self.start.date()}")
        lag_days = int(spec.get("release_lag_business_days", 1))
        if lag_days < 1:
            raise MacroDataError(
                f"Macro series {series_id} requires release_lag_business_days >= 1; "
                "same-bar availability would risk lookahead."
            )
        output = observations.copy()
        output["available_from"] = observations.index + pd.offsets.BusinessDay(lag_days)
        return output

    def load_all(self, *, refresh: bool = True) -> dict[str, pd.DataFrame]:
        result: dict[str, pd.DataFrame] = {}
        for position, spec in enumerate(self.series):
            if position > 0 and refresh and not self.offline and self.rate_limit_seconds > 0:
                time.sleep(self.rate_limit_seconds)
            result[str(spec["id"]).upper().strip()] = self.load(spec, refresh=refresh)
        return result

    def _fetch(self, series_id: str) -> pd.DataFrame:
        """Provider hook: subclasses (e.g. the CBOE index loader) override this."""
        return self._fetch_fred(series_id)

    def _fetch_fred(self, series_id: str) -> pd.DataFrame:
        url = FRED_CSV_URL.format(series_id=series_id)
        LOGGER.info("Downloading macro series %s from %s", series_id, url)
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = response.read().decode("utf-8")
        except Exception as exc:
            raise MacroDataError(f"Failed to download FRED series {series_id}: {exc}") from exc
        return parse_fred_csv(payload, series_id)

    def _read_cache(self, path: Path) -> pd.DataFrame | None:
        if not path.exists():
            return None
        frame = pd.read_parquet(path)
        return normalize_macro_frame(frame)

    def _write_cache(self, path: Path, frame: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        normalize_macro_frame(frame).to_parquet(path)


def parse_fred_csv(payload: str, series_id: str) -> pd.DataFrame:
    """Parse a fredgraph.csv payload into a [value] frame indexed by observation date.

    Missing observations ('.') are dropped rather than filled: within-series gaps
    must surface as staleness downstream, mirroring the no-forward-fill price rule.
    """
    rows = list(csv.reader(io.StringIO(payload)))
    if not rows or len(rows[0]) < 2:
        raise MacroDataError(f"FRED payload for {series_id} is not a two-column CSV.")
    header = [column.strip().lower() for column in rows[0]]
    if header[0] not in {"observation_date", "date"}:
        raise MacroDataError(f"FRED payload for {series_id} has unexpected header {rows[0]!r}.")
    dates: list[pd.Timestamp] = []
    values: list[float] = []
    for row in rows[1:]:
        if len(row) < 2:
            continue
        raw_value = row[1].strip()
        if raw_value in {"", "."}:
            continue
        dates.append(pd.Timestamp(row[0].strip()))
        values.append(float(raw_value))
    if not dates:
        raise MacroDataError(f"FRED payload for {series_id} contains no valid observations.")
    frame = pd.DataFrame({"value": values}, index=pd.DatetimeIndex(dates, name="observation_date"))
    return normalize_macro_frame(frame)


def normalize_macro_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.index = pd.to_datetime(normalized.index)
    normalized.index.name = "observation_date"
    normalized = normalized.sort_index()
    normalized = normalized[~normalized.index.duplicated(keep="last")]
    normalized = normalized.loc[:, ["value"]].astype(float)
    return normalized.dropna(subset=["value"])


def external_config(config: dict) -> dict:
    return dict(config.get("features", {}).get("external", {}))


def external_features_enabled(config: dict) -> bool:
    return bool(external_config(config).get("enabled", False))


def load_external_feature_set(config: dict, config_path: str | Path, *, offline: bool = False):
    """Return an ExternalFeatureSet when features.external.enabled, else None.

    Composes up to three sources: FRED macro series (`series`), CBOE index
    series (`cboe.series`) — both market-level, broadcast to every ticker —
    and FINRA short interest (`finra_short_interest`), which is per-ticker.
    """
    from features.pipeline import (
        ExternalFeatureSet,
        build_external_feature_set,
        build_symbol_external_feature_set,
    )

    if not external_features_enabled(config):
        return None
    external = external_config(config)
    has_fred = bool(external.get("series"))
    has_cboe = bool(dict(external.get("cboe", {})).get("series"))
    has_finra = bool(external.get("finra_short_interest"))
    if not (has_fred or has_cboe or has_finra):
        raise MacroDataError(
            "features.external.enabled requires at least one of: series (FRED), cboe.series, or finra_short_interest."
        )

    series_data: dict[str, "pd.DataFrame"] = {}
    market_specs: list[dict] = []
    if has_fred:
        loader = MacroDataLoader.from_config(config, config_path, offline=offline)
        series_data.update(loader.load_all(refresh=not offline))
        market_specs.extend(loader.series)
    if has_cboe:
        from data.cboe_loader import CboeIndexLoader

        cboe_loader = CboeIndexLoader.from_external_config(external, config_path, offline=offline)
        series_data.update(cboe_loader.load_all(refresh=not offline))
        market_specs.extend(cboe_loader.series)
    base = build_external_feature_set(series_data, market_specs) if market_specs else ExternalFeatureSet()

    if not has_finra:
        return base
    from data.finra_loader import ShortInterestLoader

    finra = dict(external.get("finra_short_interest", {}))
    finra_loader = ShortInterestLoader.from_external_config(external, config_path, offline=offline)
    symbol_data = finra_loader.load_all(
        refresh=not offline,
        release_lag_business_days=int(finra.get("release_lag_business_days", 10)),
    )
    symbol_frames, symbol_staleness = build_symbol_external_feature_set(symbol_data, finra)
    return ExternalFeatureSet(
        frames=base.frames,
        staleness=base.staleness,
        symbol_frames=symbol_frames,
        symbol_staleness=symbol_staleness,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch and cache external macro series (FRED)")
    parser.add_argument("--config", required=True, help="Config YAML with a features.external block")
    parser.add_argument("--offline", action="store_true", help="Only verify the existing cache; no downloads")
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    from core.config import load_config

    config = load_config(args.config)
    if not external_features_enabled(config):
        print("features.external.enabled is false; nothing to fetch.")
        return 0
    loader = MacroDataLoader.from_config(config, args.config, offline=args.offline)
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
