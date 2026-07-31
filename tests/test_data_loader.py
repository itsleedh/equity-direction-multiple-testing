from __future__ import annotations

import json
import tempfile
import traceback
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

import pandas as pd

from data.loader import DataProviderError, MarketDataLoader, normalize_cached_frame, resample_ohlcv


class DataLoaderTests(unittest.TestCase):
    def test_normalize_drops_missing_prices_without_forward_fill(self) -> None:
        frame = pd.DataFrame(
            {
                "open": [10.0, None, 12.0],
                "high": [11.0, 12.0, 13.0],
                "low": [9.5, 10.0, 11.5],
                "close": [10.5, 11.0, 12.5],
                "volume": [1000, None, 1200],
            },
            index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        )

        normalized = normalize_cached_frame(frame)

        self.assertEqual(list(normalized.index), list(pd.to_datetime(["2024-01-02", "2024-01-04"])))
        self.assertEqual(float(normalized.loc[pd.Timestamp("2024-01-04"), "volume"]), 1200.0)

    def test_weekly_resample_uses_ohlcv_aggregation(self) -> None:
        frame = pd.DataFrame(
            {
                "open": [10.0, 11.0, 12.0, 13.0, 14.0],
                "high": [11.0, 12.0, 14.0, 13.5, 15.0],
                "low": [9.0, 10.0, 11.0, 12.5, 13.0],
                "close": [10.5, 11.5, 13.5, 13.2, 14.8],
                "volume": [100, 110, 120, 130, 140],
            },
            index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]),
        )

        weekly = resample_ohlcv(frame, "W-FRI")

        first_week = weekly.iloc[0]
        self.assertEqual(float(first_week["open"]), 10.0)
        self.assertEqual(float(first_week["high"]), 14.0)
        self.assertEqual(float(first_week["low"]), 9.0)
        self.assertEqual(float(first_week["close"]), 13.2)
        self.assertEqual(float(first_week["volume"]), 460.0)

    def test_offline_loader_reads_existing_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            loader = MarketDataLoader(cache_dir, offline=True)
            cached = pd.DataFrame(
                {
                    "open": [1.0],
                    "high": [2.0],
                    "low": [0.5],
                    "close": [1.5],
                    "volume": [100.0],
                },
                index=pd.to_datetime(["2024-01-02"]),
            )
            cached.to_parquet(loader.cache_path("AAPL", "1d"))

            loaded = loader.load("AAPL", "1d", refresh=False)

        self.assertEqual(len(loaded), 1)
        self.assertEqual(float(loaded.iloc[0]["close"]), 1.5)


class AlpacaSymbolTests(unittest.TestCase):
    def _run_fetch(self, ticker: str) -> str:
        """Return the outbound Alpaca request symbol captured for `ticker`."""
        captured = {}

        class _FakeResponse:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *exc):
                return False

            def read(self_inner) -> bytes:
                payload = {
                    "bars": [
                        {
                            "t": "2024-01-02T14:30:00Z",
                            "o": 10.0,
                            "h": 11.0,
                            "l": 9.5,
                            "c": 10.5,
                            "v": 1000,
                        }
                    ],
                    "next_page_token": None,
                }
                return json.dumps(payload).encode("utf-8")

        def _fake_urlopen(request, timeout=None):
            captured["url"] = request.full_url
            return _FakeResponse()

        with tempfile.TemporaryDirectory() as tmpdir:
            loader = MarketDataLoader(Path(tmpdir), provider="alpaca")
            env = {"ALPACA_API_KEY": "key", "ALPACA_SECRET_KEY": "secret"}
            with mock.patch.dict("os.environ", env), mock.patch("data.loader.urllib.request.urlopen", _fake_urlopen):
                frame = loader.load(ticker, "5m", refresh=True)
            # cache/frame identity stays the yfinance-style ticker
            self.assertTrue(loader.cache_path(ticker, "5m").exists())
            self.assertFalse(frame.empty)
        return captured["url"]

    def test_class_share_ticker_is_translated_to_dot(self) -> None:
        url = self._run_fetch("BRK-B")
        self.assertIn("/stocks/BRK.B/bars", url)
        self.assertNotIn("BRK-B", url)

    def test_plain_ticker_is_unchanged(self) -> None:
        url = self._run_fetch("AAPL")
        self.assertIn("/stocks/AAPL/bars", url)


class PolygonSecurityTests(unittest.TestCase):
    def test_http_error_traceback_does_not_expose_api_key(self) -> None:
        api_key = "polygon-secret-value"
        provider_error = urllib.error.HTTPError(
            f"https://api.polygon.io/v2/aggs?apiKey={api_key}",
            401,
            "Unauthorized",
            None,
            None,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            loader = MarketDataLoader(Path(tmpdir), provider="polygon")
            with (
                mock.patch.dict("os.environ", {"POLYGON_API_KEY": api_key}, clear=True),
                mock.patch("data.loader.urllib.request.urlopen", side_effect=provider_error),
                self.assertRaises(DataProviderError) as raised,
            ):
                loader.load("AAPL", "1d", refresh=True)

        rendered_traceback = "".join(traceback.format_exception(raised.exception))
        self.assertNotIn(api_key, str(raised.exception))
        self.assertNotIn(api_key, rendered_traceback)
        self.assertIsNone(raised.exception.__cause__)
        self.assertIsNone(raised.exception.__context__)
        self.assertTrue(raised.exception.__suppress_context__)


if __name__ == "__main__":
    unittest.main()
