from __future__ import annotations

import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from features.pipeline import build_feature_frame
from features.targets import add_target_and_execution, build_direction_target, label_uniqueness_from_intervals
from tests.helpers import synthetic_ohlcv


class FeatureTargetTests(unittest.TestCase):
    def test_feature_pipeline_has_no_lookahead_when_tail_removed(self) -> None:
        bars = synthetic_ohlcv(280)
        full = build_feature_frame(bars, timeframe="1d")
        truncated = build_feature_frame(bars.iloc[:-20], timeframe="1d")

        common = truncated.index
        assert_frame_equal(full.loc[common], truncated, check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)

    def test_binary_target_uses_next_bar_close_direction(self) -> None:
        frame = pd.DataFrame(
            {
                "open": [100, 100, 100],
                "high": [101, 102, 101],
                "low": [99, 99, 98],
                "close": [100, 101, 99],
                "volume": [1, 1, 1],
            },
            index=pd.date_range("2024-01-01", periods=3),
        )
        target = build_direction_target(frame, config={"mode": "binary"}, horizon=1)

        self.assertEqual(int(target.iloc[0]), 1)
        self.assertEqual(int(target.iloc[1]), -1)
        self.assertTrue(pd.isna(target.iloc[2]))

    def test_binary_label_uniqueness_is_one_for_valid_targets(self) -> None:
        frame = pd.DataFrame(
            {
                "open": [100, 101, 102],
                "high": [101, 102, 103],
                "low": [99, 100, 101],
                "close": [100, 101, 100],
                "volume": [1, 1, 1],
            },
            index=pd.date_range("2024-01-01", periods=3),
        )
        labeled = add_target_and_execution(
            frame,
            target_config={"mode": "binary"},
            execution_config={"entry_lag_bars": 1},
        )

        self.assertEqual(labeled["label_uniqueness"].iloc[:2].tolist(), [1.0, 1.0])
        self.assertTrue(pd.isna(labeled["label_uniqueness"].iloc[2]))

    def test_execution_price_occurs_after_signal_bar(self) -> None:
        frame = pd.DataFrame(
            {
                "open": [100.0, 110.0, 120.0],
                "high": [101.0, 112.0, 123.0],
                "low": [99.0, 109.0, 119.0],
                "close": [100.5, 111.0, 121.0],
                "volume": [1, 1, 1],
            },
            index=pd.date_range("2024-01-01", periods=3),
        )
        labeled = add_target_and_execution(
            frame,
            target_config={"mode": "binary", "horizon_bars": 1},
            execution_config={"entry_lag_bars": 1},
        )

        self.assertEqual(labeled["entry_open"].iloc[0], frame["open"].iloc[1])
        self.assertEqual(labeled["exit_close"].iloc[0], frame["close"].iloc[1])
        self.assertAlmostEqual(
            labeled["execution_return"].iloc[0],
            frame["close"].iloc[1] / frame["open"].iloc[1] - 1.0,
        )

    def test_label_uniqueness_from_fully_overlapping_intervals(self) -> None:
        uniqueness = label_uniqueness_from_intervals(2, [(0, 1), (0, 1)])

        self.assertEqual(uniqueness.tolist(), [0.5, 0.5])

    def test_triple_barrier_labels_barrier_and_expiry_events(self) -> None:
        frame = triple_barrier_frame()
        labeled = add_target_and_execution(
            frame,
            target_config=triple_barrier_config(),
            execution_config={"entry_lag_bars": 1},
        )

        self.assertEqual(int(labeled["target"].iloc[0]), 1)
        self.assertEqual(labeled["target_event"].iloc[0], "tp")
        self.assertAlmostEqual(labeled["execution_return"].iloc[0], 0.01)
        self.assertEqual(int(labeled["target"].iloc[1]), -1)
        self.assertEqual(labeled["target_event"].iloc[1], "sl")
        self.assertAlmostEqual(labeled["execution_return"].iloc[1], -0.01)
        self.assertEqual(int(labeled["target"].iloc[2]), -1)
        self.assertEqual(labeled["target_event"].iloc[2], "sl")
        self.assertEqual(int(labeled["target"].iloc[3]), 1)
        self.assertEqual(labeled["target_event"].iloc[3], "expiry_up")
        self.assertEqual(labeled["holding_bars"].iloc[3], 2)

    def test_triple_barrier_ignores_bars_after_max_holding(self) -> None:
        frame = triple_barrier_frame()
        changed_tail = frame.copy()
        changed_tail.iloc[4:, changed_tail.columns.get_loc("high")] = 500.0
        changed_tail.iloc[4:, changed_tail.columns.get_loc("low")] = 1.0

        original = add_target_and_execution(
            frame,
            target_config=triple_barrier_config(),
            execution_config={"entry_lag_bars": 1},
        )
        changed = add_target_and_execution(
            changed_tail,
            target_config=triple_barrier_config(),
            execution_config={"entry_lag_bars": 1},
        )

        pd.testing.assert_series_equal(
            original.loc[original.index[:3], "target_event"], changed.loc[changed.index[:3], "target_event"]
        )
        pd.testing.assert_series_equal(
            original.loc[original.index[:3], "execution_return"],
            changed.loc[changed.index[:3], "execution_return"],
            check_names=False,
        )


def triple_barrier_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [100, 100, 100, 100, 100, 100, 100],
            "high": [100, 101.2, 100.5, 102.0, 100.4, 100.6, 100.2],
            "low": [100, 99.5, 98.8, 98.0, 99.7, 99.8, 99.8],
            "close": [100, 100.8, 99.0, 100.0, 100.2, 100.5, 100.0],
            "volume": [1] * 7,
            "atr_14": [1.0] * 7,
        },
        index=pd.date_range("2024-01-01", periods=7),
    )


def triple_barrier_config() -> dict:
    return {
        "mode": "triple_barrier",
        "triple_barrier": {"tp_atr_mult": 1.0, "sl_atr_mult": 1.0, "max_holding_bars": 2},
        "flat_band": {"method": "fixed_bps", "fixed_bps": 0},
    }


if __name__ == "__main__":
    unittest.main()
