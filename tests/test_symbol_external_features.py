from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from data.cboe_loader import parse_cboe_csv
from data.finra_loader import (
    ShortInterestLoader,
    match_tickers,
    normalize_alias_entries,
    normalize_short_interest_frame,
)
from data.macro_loader import MacroDataError, load_external_feature_set
from features.pipeline import (
    ExternalFeatureSet,
    build_symbol_external_feature_set,
    merge_external_features,
)
from main import apply_start_date
from tests.helpers import synthetic_ohlcv


def make_symbol_data(values: pd.Series, *, lag_business_days: int = 10) -> pd.DataFrame:
    frame = pd.DataFrame({"value": values.astype(float)})
    frame.index = pd.DatetimeIndex(values.index, name="settlement_date")
    frame["available_from"] = frame.index + pd.offsets.BusinessDay(lag_business_days)
    return frame


def biweekly_settlements(start: str, periods: int) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([pd.Timestamp(start) + pd.Timedelta(days=14 * step) for step in range(periods)])


class SymbolMergeTests(unittest.TestCase):
    def test_symbol_release_lag_blocks_unpublished_values(self) -> None:
        """A bar at t must only see settlements whose publication lag has elapsed."""
        settlements = biweekly_settlements("2024-01-05", 8)
        observations = pd.Series(np.arange(len(settlements), dtype=float), index=settlements)
        symbol_data = {"AAPL": make_symbol_data(observations, lag_business_days=10)}
        spec = {"release_lag_business_days": 10, "max_staleness_days": 35, "transforms": ["level"]}
        frames, staleness = build_symbol_external_feature_set(symbol_data, spec)
        feature_set = ExternalFeatureSet(symbol_frames=frames, symbol_staleness=staleness)

        bar_index = pd.date_range("2024-01-05", periods=90, freq="B")
        bars = synthetic_ohlcv(len(bar_index))
        bars.index = bar_index
        merged = merge_external_features(bars, feature_set, symbol="AAPL")

        available = symbol_data["AAPL"]["available_from"]
        for position, timestamp in enumerate(bar_index):
            published = available[available <= timestamp]
            if published.empty:
                self.assertTrue(pd.isna(merged["si_dtc_level"].iloc[position]))
            else:
                expected = observations[published.index.max()]
                self.assertEqual(merged["si_dtc_level"].iloc[position], expected)

        # Leak canary: making settlements available same-day must change the merge.
        leaked = symbol_data["AAPL"].copy()
        leaked["available_from"] = leaked.index
        leaked_frames, leaked_staleness = build_symbol_external_feature_set({"AAPL": leaked}, spec)
        leaked_set = ExternalFeatureSet(symbol_frames=leaked_frames, symbol_staleness=leaked_staleness)
        leaked_merge = merge_external_features(bars, leaked_set, symbol="AAPL")
        self.assertFalse(leaked_merge["si_dtc_level"].equals(merged["si_dtc_level"]))

    def test_cross_symbol_isolation(self) -> None:
        """Ticker A's short interest must never appear in ticker B's frame."""
        settlements = biweekly_settlements("2024-01-05", 6)
        data = {
            "AAPL": make_symbol_data(pd.Series(np.full(len(settlements), 1.0), index=settlements)),
            "MSFT": make_symbol_data(pd.Series(np.full(len(settlements), 2.0), index=settlements)),
        }
        spec = {"max_staleness_days": 35, "transforms": ["level"]}
        frames, staleness = build_symbol_external_feature_set(data, spec)
        feature_set = ExternalFeatureSet(symbol_frames=frames, symbol_staleness=staleness)

        bar_index = pd.date_range("2024-03-01", periods=20, freq="B")
        bars = synthetic_ohlcv(len(bar_index))
        bars.index = bar_index

        merged_aapl = merge_external_features(bars, feature_set, symbol="AAPL")
        merged_msft = merge_external_features(bars, feature_set, symbol="MSFT")
        self.assertTrue((merged_aapl["si_dtc_level"].dropna() == 1.0).all())
        self.assertTrue((merged_msft["si_dtc_level"].dropna() == 2.0).all())

        # A ticker without its own series gets no per-ticker columns at all.
        merged_other = merge_external_features(bars, feature_set, symbol="NVDA")
        self.assertNotIn("si_dtc_level", merged_other.columns)
        # And omitting the symbol argument must not broadcast anyone's series.
        merged_none = merge_external_features(bars, feature_set)
        self.assertNotIn("si_dtc_level", merged_none.columns)

    def test_biweekly_gap_stays_nan_beyond_staleness(self) -> None:
        settlements = biweekly_settlements("2024-01-05", 4)
        observations = pd.Series(np.arange(4, dtype=float), index=settlements)
        symbol_data = {"AAPL": make_symbol_data(observations, lag_business_days=10)}
        spec = {"max_staleness_days": 35, "transforms": ["level"]}
        frames, staleness = build_symbol_external_feature_set(symbol_data, spec)
        feature_set = ExternalFeatureSet(symbol_frames=frames, symbol_staleness=staleness)

        bar_index = pd.date_range("2024-01-05", periods=160, freq="B")
        bars = synthetic_ohlcv(len(bar_index))
        bars.index = bar_index
        merged = merge_external_features(bars, feature_set, symbol="AAPL")

        last_available = symbol_data["AAPL"]["available_from"].max()
        stale_cutoff = last_available + pd.Timedelta(days=35)
        within = merged.index[(merged.index >= last_available) & (merged.index <= stale_cutoff)]
        beyond = merged.index[merged.index > stale_cutoff]
        self.assertTrue(merged.loc[within, "si_dtc_level"].notna().all())
        self.assertTrue(merged.loc[beyond, "si_dtc_level"].isna().all(), "stale short interest must stay NaN")


class ShortInterestLoaderTests(unittest.TestCase):
    def make_loader(self, tmp: str, symbols: dict[str, list[str]] | None = None) -> ShortInterestLoader:
        return ShortInterestLoader(Path(tmp), symbols=symbols or {"AAPL": ["AAPL"]}, offline=True)

    def test_cache_roundtrip_and_lag_annotation(self) -> None:
        settlements = biweekly_settlements("2024-01-05", 5)
        raw = pd.DataFrame(
            {
                "short_interest": np.arange(5, dtype=float) * 1e6,
                "avg_daily_volume": np.full(5, 2e6),
                "days_to_cover": np.linspace(1.0, 3.0, 5),
            },
            index=settlements,
        )
        with tempfile.TemporaryDirectory() as tmp:
            loader = self.make_loader(tmp)
            raw.to_parquet(loader.cache_path("AAPL"))
            loaded = loader.load("AAPL", release_lag_business_days=10)
            self.assertEqual(list(loaded.columns), ["value", "available_from"])
            self.assertEqual(loaded["available_from"].iloc[0], settlements[0] + pd.offsets.BusinessDay(10))
            self.assertAlmostEqual(loaded["value"].iloc[-1], 3.0)
            with self.assertRaises(MacroDataError):
                loader.load("AAPL", release_lag_business_days=0)
            with self.assertRaises(MacroDataError):
                loader.load("MSFT", release_lag_business_days=10)  # no cache

    def test_missing_days_to_cover_dropped_not_filled(self) -> None:
        settlements = biweekly_settlements("2024-01-05", 4)
        raw = pd.DataFrame(
            {
                "short_interest": np.full(4, 1e6),
                "avg_daily_volume": np.full(4, 2e6),
                "days_to_cover": [1.0, np.nan, 2.0, 3.0],
            },
            index=settlements,
        )
        with tempfile.TemporaryDirectory() as tmp:
            loader = self.make_loader(tmp)
            raw.to_parquet(loader.cache_path("AAPL"))
            loaded = loader.load("AAPL", release_lag_business_days=10)
            self.assertEqual(len(loaded), 3)
            self.assertNotIn(settlements[1], loaded.index)

    def test_windowed_alias_rejects_recycled_code(self) -> None:
        """FB belongs to META only until the 2022 rename; the recycled FB (an ETF) must not match."""
        symbols = {
            "META": normalize_alias_entries(
                "META", [{"code": "FB", "until": "2022-06-08"}, {"code": "META", "from": "2022-06-09"}]
            ),
            "AAPL": normalize_alias_entries("AAPL", ["AAPL"]),
        }
        self.assertEqual(match_tickers(symbols, "FB", pd.Timestamp("2022-05-31")), ["META"])
        self.assertEqual(match_tickers(symbols, "FB", pd.Timestamp("2026-06-30")), [])
        self.assertEqual(match_tickers(symbols, "META", pd.Timestamp("2022-05-31")), [])
        self.assertEqual(match_tickers(symbols, "META", pd.Timestamp("2026-06-30")), ["META"])
        self.assertEqual(match_tickers(symbols, "AAPL", pd.Timestamp("2020-01-15")), ["AAPL"])
        self.assertEqual(match_tickers(symbols, "ZZZZ", pd.Timestamp("2020-01-15")), [])
        with self.assertRaises(MacroDataError):
            normalize_alias_entries("BAD", [{"from": "2020-01-01"}])

    def test_normalize_dedups_and_sorts_alias_concat(self) -> None:
        """FB + META rows for the same issuer must merge into one sorted series."""
        fb_rows = pd.DataFrame(
            {"short_interest": [1.0], "avg_daily_volume": [1.0], "days_to_cover": [1.0]},
            index=pd.DatetimeIndex([pd.Timestamp("2022-05-13")]),
        )
        meta_rows = pd.DataFrame(
            {"short_interest": [2.0, 3.0], "avg_daily_volume": [1.0, 1.0], "days_to_cover": [2.0, 3.0]},
            index=pd.DatetimeIndex([pd.Timestamp("2022-06-15"), pd.Timestamp("2022-05-13")]),
        )
        combined = normalize_short_interest_frame(pd.concat([fb_rows, meta_rows]))
        self.assertTrue(combined.index.is_monotonic_increasing)
        self.assertEqual(len(combined), 2)
        # duplicate settlement (2022-05-13) keeps the later alias row
        self.assertEqual(combined.loc[pd.Timestamp("2022-05-13"), "days_to_cover"], 3.0)


class CboeParserTests(unittest.TestCase):
    def test_parse_cboe_csv(self) -> None:
        payload = "DATE,SKEW\n01/02/1990,126.09\n01/03/1990,\n01/04/1990,127.50\n"
        frame = parse_cboe_csv(payload, "SKEW")
        self.assertEqual(len(frame), 2)
        self.assertEqual(frame.index[0], pd.Timestamp("1990-01-02"))
        self.assertAlmostEqual(frame["value"].iloc[1], 127.50)
        with self.assertRaises(MacroDataError):
            parse_cboe_csv("bogus,header\n1,2\n", "SKEW")


class StartDateGateTests(unittest.TestCase):
    def test_absent_start_date_is_identity(self) -> None:
        bars = synthetic_ohlcv(50)
        data = {("AAPL", "1d"): bars}
        for gate_off in (None, ""):
            result = apply_start_date(data, gate_off)
            assert_frame_equal(result[("AAPL", "1d")], bars)

    def test_start_date_slices_bars(self) -> None:
        bar_index = pd.date_range("2017-01-02", periods=600, freq="B")
        bars = synthetic_ohlcv(len(bar_index))
        bars.index = bar_index
        result = apply_start_date({("AAPL", "1d"): bars}, "2018-01-01")
        sliced = result[("AAPL", "1d")]
        self.assertGreaterEqual(sliced.index.min(), pd.Timestamp("2018-01-01"))
        self.assertEqual(sliced.index.max(), bars.index.max())
        self.assertLess(len(sliced), len(bars))


class ComposedExternalConfigTests(unittest.TestCase):
    def test_enabled_without_any_source_raises(self) -> None:
        config = {"features": {"external": {"enabled": True}}}
        with self.assertRaises(MacroDataError):
            load_external_feature_set(config, "configs/config.yaml", offline=True)

    def test_finra_only_offline_composition(self) -> None:
        settlements = biweekly_settlements("2024-01-05", 8)
        raw = pd.DataFrame(
            {
                "short_interest": np.full(8, 1e6),
                "avg_daily_volume": np.full(8, 5e5),
                "days_to_cover": np.linspace(1.0, 2.0, 8),
            },
            index=settlements,
        )
        with tempfile.TemporaryDirectory() as tmp:
            loader = ShortInterestLoader(Path(tmp), symbols={"AAPL": ["AAPL"]}, offline=True)
            raw.to_parquet(loader.cache_path("AAPL"))
            config = {
                "features": {
                    "external": {
                        "enabled": True,
                        "finra_short_interest": {
                            "cache_dir": tmp,
                            "start": "2017-12-01",
                            "release_lag_business_days": 10,
                            "max_staleness_days": 35,
                            "transforms": ["level", "diff_1"],
                            "symbols": {"AAPL": ["AAPL"]},
                        },
                    }
                }
            }
            feature_set = load_external_feature_set(config, "configs/config.yaml", offline=True)
            self.assertEqual(feature_set.frames, {})
            self.assertIn("AAPL", feature_set.symbol_frames)
            columns = list(feature_set.symbol_frames["AAPL"].columns)
            self.assertEqual(columns, ["si_dtc_level", "si_dtc_diff_1"])


if __name__ == "__main__":
    unittest.main()
