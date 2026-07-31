from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from data.macro_loader import MacroDataError, MacroDataLoader, parse_fred_csv
from features.pipeline import (
    build_external_feature_set,
    external_transforms,
    merge_external_features,
)
from features.targets import build_model_frame
from tests.helpers import synthetic_ohlcv


def make_series_data(values: pd.Series, *, lag_business_days: int = 1) -> pd.DataFrame:
    frame = pd.DataFrame({"value": values.astype(float)})
    frame.index = pd.DatetimeIndex(values.index, name="observation_date")
    frame["available_from"] = frame.index + pd.offsets.BusinessDay(lag_business_days)
    return frame


class ReleaseLagAlignmentTests(unittest.TestCase):
    def test_release_lag_blocks_same_day_and_future_values(self) -> None:
        """A bar at date t must only ever see observations released at or before t."""
        index = pd.date_range("2024-01-01", periods=40, freq="B")
        observations = pd.Series(np.arange(len(index), dtype=float), index=index)
        series_data = {"TESTX": make_series_data(observations, lag_business_days=1)}
        spec = [{"id": "TESTX", "release_lag_business_days": 1, "max_staleness_days": 10, "transforms": ["level"]}]
        feature_set = build_external_feature_set(series_data, spec)

        bars = synthetic_ohlcv(len(index))
        bars.index = index
        merged = merge_external_features(bars, feature_set)

        # With a one-business-day release lag, bar t sees the observation of t-1B,
        # never its own same-day observation (which would be lookahead).
        for position in range(1, len(index)):
            self.assertEqual(
                merged["macro_testx_level"].iloc[position],
                observations.iloc[position - 1],
                msg=f"bar {index[position].date()} must see the previous observation only",
            )
        self.assertTrue(pd.isna(merged["macro_testx_level"].iloc[0]))

        # Leak canary: shift the series availability back by one day and the merge
        # WOULD expose same-day values — assert the honest merge differs from that.
        leaked = series_data["TESTX"].copy()
        leaked["available_from"] = leaked.index
        leaked_set = build_external_feature_set({"TESTX": leaked}, spec)
        leaked_merge = merge_external_features(bars, leaked_set)
        self.assertTrue((leaked_merge["macro_testx_level"].iloc[1:] != merged["macro_testx_level"].iloc[1:]).all())

    def test_available_from_crosses_weekends(self) -> None:
        friday = pd.Timestamp("2024-01-05")
        observations = pd.Series([1.23], index=pd.DatetimeIndex([friday]))
        series_data = make_series_data(observations, lag_business_days=1)
        self.assertEqual(series_data["available_from"].iloc[0], pd.Timestamp("2024-01-08"))  # Monday

        loader_spec = {"id": "TESTX", "release_lag_business_days": 0}
        with tempfile.TemporaryDirectory() as tmp:
            loader = MacroDataLoader(Path(tmp), series=[loader_spec], offline=False)
            loader._write_cache(
                loader.cache_path("TESTX"), pd.DataFrame({"value": [1.0]}, index=pd.DatetimeIndex([friday]))
            )
            with self.assertRaises(MacroDataError):
                loader.load(loader_spec, refresh=False)  # lag 0 would allow same-bar availability


class MissingDataTests(unittest.TestCase):
    def test_series_gap_keeps_feature_nan_beyond_staleness(self) -> None:
        obs_index = pd.date_range("2024-01-01", periods=10, freq="B")
        observations = pd.Series(np.linspace(1.0, 2.0, len(obs_index)), index=obs_index)
        series_data = {"TESTX": make_series_data(observations, lag_business_days=1)}
        spec = [{"id": "TESTX", "release_lag_business_days": 1, "max_staleness_days": 5, "transforms": ["level"]}]
        feature_set = build_external_feature_set(series_data, spec)

        bar_index = pd.date_range("2024-01-01", periods=60, freq="B")
        bars = synthetic_ohlcv(len(bar_index))
        bars.index = bar_index
        merged = merge_external_features(bars, feature_set)

        last_available = series_data["TESTX"]["available_from"].max()
        stale_cutoff = last_available + pd.Timedelta(days=5)
        within = merged.index[(merged.index >= last_available) & (merged.index <= stale_cutoff)]
        beyond = merged.index[merged.index > stale_cutoff]
        self.assertTrue(merged.loc[within, "macro_testx_level"].notna().all())
        self.assertTrue(
            merged.loc[beyond, "macro_testx_level"].isna().all(), "stale values must stay NaN, not forward-fill"
        )

    def test_transform_warmup_stays_nan_after_merge(self) -> None:
        obs_index = pd.date_range("2024-01-01", periods=30, freq="B")
        observations = pd.Series(np.arange(len(obs_index), dtype=float), index=obs_index)
        transformed = external_transforms(observations, ["level", "diff_5", "pct_5", "z_10"])
        self.assertTrue(transformed["diff_5"].iloc[:5].isna().all())
        self.assertTrue(transformed["pct_5"].iloc[:5].isna().all())
        self.assertTrue(transformed["z_10"].iloc[:9].isna().all())
        self.assertEqual(transformed["diff_5"].iloc[5], 5.0)
        self.assertAlmostEqual(transformed["pct_5"].iloc[10], (10.0 - 5.0) / 5.0)

        with self.assertRaises(ValueError):
            external_transforms(observations, ["median_5"])


class GateOffTests(unittest.TestCase):
    def test_gate_off_keeps_model_frame_unchanged(self) -> None:
        bars = synthetic_ohlcv(280)
        base_config = {"target": {"mode": "binary"}, "execution": {"entry_lag_bars": 1}}
        gated_config = dict(base_config)
        gated_config["features"] = {"external": {"enabled": False, "series": [{"id": "DGS10"}]}}

        baseline = build_model_frame(bars, timeframe="1d", config=base_config)
        gated_off = build_model_frame(bars, timeframe="1d", config=gated_config, external_features=None)

        assert_frame_equal(baseline, gated_off)
        self.assertFalse([column for column in gated_off.columns if column.startswith("macro_")])

    def test_disabled_gate_loads_nothing(self) -> None:
        from data.macro_loader import load_external_feature_set

        config = {"features": {"external": {"enabled": False, "series": [{"id": "DGS10"}]}}}
        self.assertIsNone(load_external_feature_set(config, "configs/config.yaml", offline=True))
        self.assertIsNone(load_external_feature_set({}, "configs/config.yaml", offline=True))


class MacroLoaderTests(unittest.TestCase):
    def test_offline_without_cache_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            loader = MacroDataLoader(Path(tmp), series=[{"id": "TESTX"}], offline=True)
            with self.assertRaises(MacroDataError):
                loader.load({"id": "TESTX"})

    def test_cache_roundtrip_and_lag_annotation(self) -> None:
        obs_index = pd.date_range("2024-01-01", periods=5, freq="B")
        raw = pd.DataFrame({"value": np.arange(5, dtype=float)}, index=obs_index)
        with tempfile.TemporaryDirectory() as tmp:
            loader = MacroDataLoader(Path(tmp), series=[{"id": "TESTX"}], offline=True, start="2024-01-01")
            loader._write_cache(loader.cache_path("TESTX"), raw)
            loaded = loader.load({"id": "TESTX", "release_lag_business_days": 2})
            self.assertEqual(list(loaded.columns), ["value", "available_from"])
            self.assertEqual(loaded["available_from"].iloc[0], obs_index[0] + pd.offsets.BusinessDay(2))
            self.assertEqual(len(loaded), 5)

    def test_parse_fred_csv_drops_missing_observations(self) -> None:
        payload = "observation_date,DGS10\n2024-01-01,.\n2024-01-02,4.10\n2024-01-03,4.05\n"
        frame = parse_fred_csv(payload, "DGS10")
        self.assertEqual(len(frame), 2)
        self.assertEqual(frame["value"].iloc[0], 4.10)
        with self.assertRaises(MacroDataError):
            parse_fred_csv("bogus,header\n", "DGS10")


if __name__ == "__main__":
    unittest.main()
