from __future__ import annotations

import logging
import json
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd

LOGGER = logging.getLogger(__name__)

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
YFINANCE_COLUMN_MAP = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}


@dataclass(frozen=True)
class TimeframeSpec:
    name: str
    yfinance_interval: str | None
    source_timeframe: str | None = None
    note: str = ""


TIMEFRAME_SPECS: dict[str, TimeframeSpec] = {
    "1h": TimeframeSpec(
        name="1h",
        yfinance_interval="1h",
        note="yfinance hourly data is limited to roughly the most recent 730 days.",
    ),
    "1d": TimeframeSpec(name="1d", yfinance_interval="1d"),
    "1wk": TimeframeSpec(
        name="1wk",
        yfinance_interval=None,
        source_timeframe="1d",
        note="weekly bars are resampled from adjusted daily bars.",
    ),
    "1m": TimeframeSpec(
        name="1m",
        yfinance_interval="1m",
        note="yfinance 1m data is limited to a very short recent window; sample-size warnings are required.",
    ),
    "5m": TimeframeSpec(
        name="5m",
        yfinance_interval="5m",
        note="yfinance 5m data is limited to roughly the most recent 60 days.",
    ),
    "15m": TimeframeSpec(
        name="15m",
        yfinance_interval="15m",
        note="yfinance 15m data is limited to roughly the most recent 60 days.",
    ),
}


class DataProviderError(RuntimeError):
    """Raised when a market data provider cannot serve a request."""


class MarketDataLoader:
    """Load adjusted OHLCV bars with parquet caching and incremental refresh."""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        provider: str = "yfinance",
        auto_adjust: bool = True,
        daily_years: int = 10,
        hourly_period: str = "730d",
        weekly_resample_rule: str = "W-FRI",
        intraday_periods: dict[str, str] | None = None,
        provider_rate_limit_seconds: float = 0.0,
        offline: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.provider = provider
        self.auto_adjust = auto_adjust
        self.daily_years = daily_years
        self.hourly_period = hourly_period
        self.weekly_resample_rule = weekly_resample_rule
        self.intraday_periods = intraday_periods or {"1m": "7d", "5m": "60d", "15m": "60d"}
        self.provider_rate_limit_seconds = provider_rate_limit_seconds
        self.offline = offline
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(
        cls, config: dict, *, cache_dir: str | Path | None = None, offline: bool | None = None
    ) -> "MarketDataLoader":
        data_config = config.get("data", {})
        return cls(
            cache_dir=cache_dir or data_config.get("cache_dir", "data/cache"),
            provider=data_config.get("provider", "yfinance"),
            auto_adjust=bool(data_config.get("auto_adjust", True)),
            daily_years=int(data_config.get("daily_years", 10)),
            hourly_period=str(data_config.get("hourly_period", "730d")),
            weekly_resample_rule=str(data_config.get("weekly_resample_rule", "W-FRI")),
            intraday_periods=dict(data_config.get("intraday_periods", {"1m": "7d", "5m": "60d", "15m": "60d"})),
            provider_rate_limit_seconds=float(data_config.get("provider_rate_limit_seconds", 0.0)),
            offline=bool(data_config.get("offline", False) if offline is None else offline),
        )

    def load_universe(
        self,
        tickers: Iterable[str],
        timeframes: Iterable[str],
        *,
        refresh: bool = True,
    ) -> dict[tuple[str, str], pd.DataFrame]:
        """Load all requested ticker/timeframe pairs."""
        result: dict[tuple[str, str], pd.DataFrame] = {}
        for ticker in tickers:
            for timeframe in timeframes:
                result[(ticker, timeframe)] = self.load(ticker, timeframe, refresh=refresh)
        return result

    def load(self, ticker: str, timeframe: str, *, refresh: bool = True) -> pd.DataFrame:
        ticker = ticker.upper().strip()
        if timeframe not in TIMEFRAME_SPECS:
            supported = ", ".join(sorted(TIMEFRAME_SPECS))
            raise ValueError(f"Unsupported timeframe '{timeframe}'. Supported: {supported}")

        if timeframe == "1wk":
            return self._load_weekly(ticker, refresh=refresh)

        cache_path = self.cache_path(ticker, timeframe)
        cached = self._read_cache(cache_path)

        if not refresh or self.offline:
            if cached is None:
                raise DataProviderError(f"No cached data for {ticker} {timeframe} at {cache_path}")
            LOGGER.info("Using cached %s %s bars from %s", ticker, timeframe, cache_path)
            return cached

        fetched = self._fetch_from_provider(ticker, timeframe, cached)
        combined = self._merge_bars(cached, fetched)
        self._write_cache(cache_path, combined)
        return combined

    def cache_path(self, ticker: str, timeframe: str) -> Path:
        safe_ticker = ticker.upper().replace("/", "-")
        return self.cache_dir / f"{safe_ticker}_{timeframe}.parquet"

    def _load_weekly(self, ticker: str, *, refresh: bool) -> pd.DataFrame:
        daily = self.load(ticker, "1d", refresh=refresh)
        weekly = resample_ohlcv(daily, self.weekly_resample_rule)
        cache_path = self.cache_path(ticker, "1wk")
        self._write_cache(cache_path, weekly)
        return weekly

    def _fetch_from_provider(self, ticker: str, timeframe: str, cached: pd.DataFrame | None) -> pd.DataFrame:
        if self.provider == "yfinance":
            return self._fetch_yfinance(ticker, timeframe, cached)
        if self.provider == "alpaca":
            return self._fetch_alpaca(ticker, timeframe, cached)
        if self.provider == "polygon":
            return self._fetch_polygon(ticker, timeframe, cached)
        raise DataProviderError(f"Provider '{self.provider}' is not implemented.")

    def _fetch_yfinance(self, ticker: str, timeframe: str, cached: pd.DataFrame | None) -> pd.DataFrame:
        try:
            import yfinance as yf  # type: ignore
        except ModuleNotFoundError as exc:
            if cached is not None and not cached.empty:
                LOGGER.warning("yfinance is not installed; falling back to cached %s %s bars.", ticker, timeframe)
                return cached
            raise DataProviderError("yfinance is required for live data downloads. Install requirements.txt.") from exc

        if timeframe == "1h":
            download_kwargs = {"period": self.hourly_period, "interval": "1h"}
        elif timeframe == "1d":
            start = self._daily_fetch_start(cached)
            download_kwargs = {"start": start.strftime("%Y-%m-%d"), "interval": "1d"}
        elif timeframe in {"1m", "5m", "15m"}:
            download_kwargs = {"period": self.intraday_periods.get(timeframe, "60d"), "interval": timeframe}
        else:
            raise ValueError(f"Unsupported direct yfinance timeframe: {timeframe}")

        LOGGER.info("Downloading %s %s from yfinance with %s", ticker, timeframe, download_kwargs)
        raw = yf.download(
            ticker,
            auto_adjust=self.auto_adjust,
            progress=False,
            threads=False,
            **download_kwargs,
        )
        bars = normalize_yfinance_frame(raw)
        if bars.empty:
            raise DataProviderError(f"Provider returned no rows for {ticker} {timeframe}")
        return bars

    def _fetch_alpaca(self, ticker: str, timeframe: str, cached: pd.DataFrame | None) -> pd.DataFrame:
        key = os.environ.get("ALPACA_API_KEY")
        secret = os.environ.get("ALPACA_SECRET_KEY")
        if not key or not secret:
            if cached is not None and not cached.empty:
                LOGGER.warning("Alpaca credentials are missing; falling back to cached %s %s bars.", ticker, timeframe)
                return cached
            raise DataProviderError("ALPACA_API_KEY and ALPACA_SECRET_KEY are required for Alpaca data.")

        alpaca_timeframe = {"1m": "1Min", "5m": "5Min", "15m": "15Min", "1h": "1Hour", "1d": "1Day"}.get(timeframe)
        if alpaca_timeframe is None:
            raise DataProviderError(f"Alpaca provider does not support timeframe {timeframe}")
        # yfinance-style class-share tickers use "-" (BRK-B); Alpaca expects "." (BRK.B).
        # Translate only the outbound request symbol; cache/frame identity stays the yfinance ticker.
        alpaca_symbol = ticker.replace("-", ".")
        start = self._daily_fetch_start(cached).isoformat()
        query = {
            "timeframe": alpaca_timeframe,
            "start": start,
            "adjustment": "all" if self.auto_adjust else "raw",
            "feed": "iex",
            "limit": 10000,
        }
        rows = []
        while True:
            params = urllib.parse.urlencode(query)
            url = f"https://data.alpaca.markets/v2/stocks/{alpaca_symbol}/bars?{params}"
            request = urllib.request.Request(url, headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
            rows.extend(payload.get("bars", []))
            token = payload.get("next_page_token")
            if not token:
                break
            query["page_token"] = token
            if self.provider_rate_limit_seconds > 0:
                time.sleep(self.provider_rate_limit_seconds)
        frame = pd.DataFrame(rows)
        if frame.empty:
            raise DataProviderError(f"Alpaca returned no rows for {ticker} {timeframe}")
        frame = frame.rename(columns={"t": "date", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        frame = frame.set_index(pd.to_datetime(frame["date"]))
        return normalize_cached_frame(frame[OHLCV_COLUMNS])

    def _fetch_polygon(self, ticker: str, timeframe: str, cached: pd.DataFrame | None) -> pd.DataFrame:
        api_key = os.environ.get("POLYGON_API_KEY")
        if not api_key:
            if cached is not None and not cached.empty:
                LOGGER.warning("Polygon API key is missing; falling back to cached %s %s bars.", ticker, timeframe)
                return cached
            raise DataProviderError("POLYGON_API_KEY is required for Polygon data.")

        multiplier, timespan = {
            "1m": (1, "minute"),
            "5m": (5, "minute"),
            "15m": (15, "minute"),
            "1h": (1, "hour"),
            "1d": (1, "day"),
        }.get(timeframe, (None, None))
        if multiplier is None or timespan is None:
            raise DataProviderError(f"Polygon provider does not support timeframe {timeframe}")
        start = self._daily_fetch_start(cached).strftime("%Y-%m-%d")
        end = datetime.now(tz=UTC).strftime("%Y-%m-%d")
        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start}/{end}"
            f"?adjusted={'true' if self.auto_adjust else 'false'}&sort=asc&limit=50000&apiKey={api_key}"
        )
        rows = []
        while url:
            exception_name = None
            try:
                with urllib.request.urlopen(url, timeout=30) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            except Exception as exc:
                exception_name = type(exc).__name__
            if exception_name is not None:
                # Raise outside the handler so a key-bearing HTTPError is not retained as context.
                raise DataProviderError(f"Polygon request failed for {ticker} {timeframe}: {exception_name}") from None
            rows.extend(payload.get("results", []))
            next_url = payload.get("next_url")
            if next_url and "apiKey=" not in next_url:
                separator = "&" if "?" in next_url else "?"
                next_url = f"{next_url}{separator}apiKey={api_key}"
            url = next_url
            if url and self.provider_rate_limit_seconds > 0:
                time.sleep(self.provider_rate_limit_seconds)
        frame = pd.DataFrame(rows)
        if frame.empty:
            raise DataProviderError(f"Polygon returned no rows for {ticker} {timeframe}")
        frame = frame.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "t": "date"})
        frame = frame.set_index(pd.to_datetime(frame["date"], unit="ms", utc=True))
        return normalize_cached_frame(frame[OHLCV_COLUMNS])

    def _daily_fetch_start(self, cached: pd.DataFrame | None) -> datetime:
        if cached is not None and not cached.empty:
            last = pd.Timestamp(cached.index.max()).to_pydatetime()
            if last.tzinfo is None:
                last = last.replace(tzinfo=UTC)
            return last - timedelta(days=7)
        return datetime.now(tz=UTC) - timedelta(days=365 * self.daily_years + 10)

    def _read_cache(self, path: Path) -> pd.DataFrame | None:
        if not path.exists():
            return None
        frame = pd.read_parquet(path)
        return normalize_cached_frame(frame)

    def _write_cache(self, path: Path, frame: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        normalize_cached_frame(frame).to_parquet(path)

    def _merge_bars(self, cached: pd.DataFrame | None, fetched: pd.DataFrame) -> pd.DataFrame:
        if cached is None or cached.empty:
            return normalize_cached_frame(fetched)
        combined = pd.concat([cached, fetched], axis=0)
        combined = combined[~combined.index.duplicated(keep="last")]
        return normalize_cached_frame(combined)


def normalize_yfinance_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance output to adjusted lower-case OHLCV bars."""
    if frame is None or frame.empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    normalized = frame.copy()
    if isinstance(normalized.columns, pd.MultiIndex):
        if len(normalized.columns.levels) > 1:
            normalized.columns = normalized.columns.get_level_values(0)
        else:
            normalized.columns = normalized.columns.to_flat_index()

    normalized = normalized.rename(columns=YFINANCE_COLUMN_MAP)
    available = [column for column in OHLCV_COLUMNS if column in normalized.columns]
    normalized = normalized.loc[:, available]
    return normalize_cached_frame(normalized)


def normalize_cached_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply explicit missing-bar rules without manufacturing price data."""
    normalized = frame.copy()
    normalized.index = pd.to_datetime(normalized.index)
    normalized = normalized.sort_index()
    normalized = normalized[~normalized.index.duplicated(keep="last")]

    for column in OHLCV_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = pd.NA
    normalized = normalized.loc[:, OHLCV_COLUMNS]
    normalized["volume"] = normalized["volume"].fillna(0)
    normalized = normalized.dropna(subset=["open", "high", "low", "close"])
    normalized = normalized[normalized["close"] > 0]
    return normalized.astype(float)


def resample_ohlcv(frame: pd.DataFrame, rule: str = "W-FRI") -> pd.DataFrame:
    """Resample OHLCV bars using first/high/low/last/sum aggregation."""
    normalized = normalize_cached_frame(frame)
    if normalized.empty:
        return normalized
    resampled = normalized.resample(rule).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    return normalize_cached_frame(resampled)
