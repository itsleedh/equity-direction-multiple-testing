from __future__ import annotations

import argparse
import json
import logging
import time
import urllib.request
from pathlib import Path

import pandas as pd

from data.macro_loader import MacroDataError

LOGGER = logging.getLogger(__name__)

FINRA_DATA_URL = "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"
FINRA_PARTITIONS_URL = "https://api.finra.org/partitions/group/otcMarket/name/consolidatedShortInterest"

# Raw fields kept in the cache. The feature value served downstream is
# days_to_cover (short interest normalized by average daily volume), which is
# split-neutral; raw share counts are kept for audit only.
CACHE_COLUMNS = ["short_interest", "avg_daily_volume", "days_to_cover"]


class ShortInterestLoader:
    """Load FINRA consolidated short interest per symbol with explicit release timing.

    The dataset is partitioned by settlement date (two reports per month). The
    parquet cache stores raw observations per repository ticker; the publication
    lag is applied at load time as available_from = settlement date +
    release_lag_business_days, so a config lag change never requires a refetch.
    FINRA publishes 7-8 weekdays after settlement (9 across holidays), so the
    lag must stay a strict upper bound of the real publication delay.

    `symbols` maps repository tickers to the FINRA symbol codes that identify
    the issue over time. Entries are either a bare code ("AAPL") or a mapping
    with a validity window ({"code": "FB", "until": "2022-06-08"}). Windows are
    required when a code was recycled by another issuer — e.g. META traded as
    FB until 2022-06-08, and FINRA later reassigned FB to a ProShares ETF, so
    an unbounded FB alias would poison META's series. Berkshire class B is
    BRKB (no separator) on FINRA.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        symbols: dict[str, list],
        offline: bool = False,
        start: str = "2017-12-01",
        rate_limit_seconds: float = 0.5,
        request_timeout_seconds: float = 60.0,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.symbols = {
            str(ticker).upper().strip(): normalize_alias_entries(ticker, entries) for ticker, entries in symbols.items()
        }
        self.offline = offline
        self.start = pd.Timestamp(start)
        self.rate_limit_seconds = rate_limit_seconds
        self.request_timeout_seconds = request_timeout_seconds
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_external_config(
        cls, external: dict, config_path: str | Path, *, offline: bool | None = None
    ) -> "ShortInterestLoader":
        from core.config import resolve_path

        finra = dict(external.get("finra_short_interest", {}))
        symbols = finra.get("symbols", {})
        if not isinstance(symbols, dict) or not symbols:
            raise MacroDataError(
                "features.external.finra_short_interest.symbols must be a non-empty mapping when present."
            )
        for ticker, entries in symbols.items():
            if not isinstance(entries, list) or not entries:
                raise MacroDataError(
                    f"finra_short_interest.symbols['{ticker}'] must be a non-empty list of FINRA symbol codes."
                )
        return cls(
            cache_dir=resolve_path(config_path, finra.get("cache_dir", "data/cache_finra")),
            symbols=symbols,
            offline=bool(finra.get("offline", False) if offline is None else offline),
            start=str(finra.get("start", "2017-12-01")),
            rate_limit_seconds=float(finra.get("rate_limit_seconds", 0.5)),
        )

    def cache_path(self, ticker: str) -> Path:
        safe_ticker = str(ticker).upper().replace("/", "-")
        return self.cache_dir / f"{safe_ticker}.parquet"

    def load(self, ticker: str, *, release_lag_business_days: int) -> pd.DataFrame:
        """Return one ticker as columns [value, available_from] indexed by settlement date."""
        ticker = str(ticker).upper().strip()
        cache_path = self.cache_path(ticker)
        if not cache_path.exists():
            raise MacroDataError(f"No cached FINRA short interest for {ticker} at {cache_path}")
        cached = normalize_short_interest_frame(pd.read_parquet(cache_path))
        cached = cached[cached.index >= self.start]
        if cached.empty:
            raise MacroDataError(
                f"FINRA short interest for {ticker} has no observations on or after {self.start.date()}"
            )
        lag_days = int(release_lag_business_days)
        if lag_days < 1:
            raise MacroDataError(
                f"FINRA short interest for {ticker} requires release_lag_business_days >= 1; "
                "same-bar availability would risk lookahead."
            )
        output = pd.DataFrame({"value": cached["days_to_cover"].astype(float)}, index=cached.index)
        output = output.dropna(subset=["value"])
        if output.empty:
            raise MacroDataError(f"FINRA short interest for {ticker} has no usable days_to_cover observations.")
        output["available_from"] = output.index + pd.offsets.BusinessDay(lag_days)
        return output

    def load_all(self, *, refresh: bool = True, release_lag_business_days: int) -> dict[str, pd.DataFrame]:
        if refresh and not self.offline:
            self.refresh_cache()
        result: dict[str, pd.DataFrame] = {}
        for ticker in self.symbols:
            result[ticker] = self.load(ticker, release_lag_business_days=release_lag_business_days)
        return result

    def refresh_cache(self) -> None:
        """Bulk-download every partition once and rewrite the per-ticker caches."""
        partitions = [date for date in self.fetch_partitions() if date >= self.start]
        if not partitions:
            raise MacroDataError(f"FINRA reports no short-interest partitions on or after {self.start.date()}")
        all_codes = sorted({entry["code"] for entries in self.symbols.values() for entry in entries})
        rows: dict[str, list[dict]] = {ticker: [] for ticker in self.symbols}
        for position, settlement in enumerate(sorted(partitions)):
            if position > 0 and self.rate_limit_seconds > 0:
                time.sleep(self.rate_limit_seconds)
            for record in self.fetch_partition(settlement, all_codes):
                code = str(record.get("symbolCode", "")).upper().strip()
                settlement_date = pd.Timestamp(str(record.get("settlementDate")))
                for ticker in match_tickers(self.symbols, code, settlement_date):
                    rows[ticker].append(
                        {
                            "settlement_date": settlement_date,
                            "short_interest": _to_float(record.get("currentShortPositionQuantity")),
                            "avg_daily_volume": _to_float(record.get("averageDailyVolumeQuantity")),
                            "days_to_cover": _to_float(record.get("daysToCoverQuantity")),
                        }
                    )
        for ticker, records in rows.items():
            if not records:
                raise MacroDataError(f"FINRA returned no short-interest rows for {ticker} ({self.symbols[ticker]}).")
            frame = pd.DataFrame(records).set_index("settlement_date")
            frame = normalize_short_interest_frame(frame)
            path = self.cache_path(ticker)
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path)
            LOGGER.info(
                "Cached %s: %d settlements %s..%s",
                ticker,
                len(frame),
                frame.index.min().date(),
                frame.index.max().date(),
            )

    def fetch_partitions(self) -> list[pd.Timestamp]:
        payload = self._request(FINRA_PARTITIONS_URL, body=None)
        partitions = []
        for entry in payload.get("availablePartitions", []):
            for value in entry.get("partitions", []):
                partitions.append(pd.Timestamp(str(value)))
        if not partitions:
            raise MacroDataError("FINRA partitions endpoint returned no settlement dates.")
        return sorted(set(partitions))

    def fetch_partition(self, settlement: pd.Timestamp, codes: list[str]) -> list[dict]:
        body = {
            "limit": max(100, 5 * len(codes)),
            "compareFilters": [
                {"compareType": "EQUAL", "fieldName": "settlementDate", "fieldValue": settlement.strftime("%Y-%m-%d")}
            ],
            "domainFilters": [{"fieldName": "symbolCode", "values": codes}],
        }
        payload = self._request(FINRA_DATA_URL, body=body)
        if not isinstance(payload, list):
            raise MacroDataError(f"FINRA data endpoint returned a non-list payload for {settlement.date()}.")
        return payload

    def _request(self, url: str, *, body: dict | None):
        data = None
        headers = {"Accept": "application/json"}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.request_timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise MacroDataError(f"FINRA request failed for {url}: {exc}") from exc


def normalize_alias_entries(ticker: str, entries: list) -> list[dict]:
    """Normalize alias config entries to [{code, from, until}] with Timestamps or None."""
    normalized: list[dict] = []
    for entry in entries:
        if isinstance(entry, str):
            entry = {"code": entry}
        if not isinstance(entry, dict) or not entry.get("code"):
            raise MacroDataError(
                f"finra_short_interest.symbols['{ticker}'] entries must be codes or mappings with a 'code'."
            )
        normalized.append(
            {
                "code": str(entry["code"]).upper().strip(),
                "from": pd.Timestamp(str(entry["from"])) if entry.get("from") else None,
                "until": pd.Timestamp(str(entry["until"])) if entry.get("until") else None,
            }
        )
    return normalized


def match_tickers(symbols: dict[str, list[dict]], code: str, settlement: pd.Timestamp) -> list[str]:
    """Return the tickers whose alias windows accept this (code, settlement) pair."""
    matched: list[str] = []
    for ticker, entries in symbols.items():
        for entry in entries:
            if entry["code"] != code:
                continue
            if entry["from"] is not None and settlement < entry["from"]:
                continue
            if entry["until"] is not None and settlement > entry["until"]:
                continue
            matched.append(ticker)
            break
    return matched


def normalize_short_interest_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.index = pd.to_datetime(normalized.index)
    normalized.index.name = "settlement_date"
    normalized = normalized.sort_index()
    normalized = normalized[~normalized.index.duplicated(keep="last")]
    for column in CACHE_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = float("nan")
    return normalized.loc[:, CACHE_COLUMNS].astype(float)


def _to_float(value) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch and cache FINRA consolidated short interest")
    parser.add_argument(
        "--config", required=True, help="Config YAML with a features.external.finra_short_interest block"
    )
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
    external = external_config(config)
    loader = ShortInterestLoader.from_external_config(external, args.config, offline=args.offline)
    lag = int(dict(external.get("finra_short_interest", {})).get("release_lag_business_days", 10))
    series_data = loader.load_all(refresh=not args.offline, release_lag_business_days=lag)
    for ticker, frame in sorted(series_data.items()):
        print(
            f"{ticker:>6}: {len(frame)} settlements, "
            f"{frame.index.min().date()} .. {frame.index.max().date()}, "
            f"cache {loader.cache_path(ticker)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
