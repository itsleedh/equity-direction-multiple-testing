from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from backtest.engine import BacktestEngine
from backtest.splitter import WalkForwardSplitter
from features.pipeline import feature_columns
from features.targets import build_model_frame
from main import run_research_pipeline
from metrics.classification import wilson_interval, win_rate_stats
from strategies.factory import build_strategy_suite
from tests.helpers import synthetic_ohlcv


class BacktestMetricsPipelineTests(unittest.TestCase):
    def test_walk_forward_split_enforces_purge_and_embargo_gap(self) -> None:
        frame = pd.DataFrame(index=pd.date_range("2024-01-01", periods=100))
        splitter = WalkForwardSplitter(
            initial_train_bars=40,
            test_bars=10,
            purge_lookback_bars=5,
            embargo_bars=2,
            min_train_bars=20,
            adaptive=False,
        )
        folds = splitter.split(frame)

        self.assertTrue(folds)
        first = folds[0]
        self.assertEqual(first.train_positions[-1], 39)
        self.assertEqual(first.test_positions[0], 47)
        self.assertEqual(first.test_positions[0] - first.train_positions[-1] - 1, 7)

    def test_cpcv_split_generates_combinations_without_overlap(self) -> None:
        frame = pd.DataFrame(index=pd.date_range("2024-01-01", periods=80))
        splitter = WalkForwardSplitter(
            mode="cpcv",
            purge_lookback_bars=2,
            embargo_bars=1,
            min_train_bars=10,
            cpcv_n_groups=4,
            cpcv_n_test_groups=2,
        )

        folds = splitter.split(frame)

        self.assertEqual(len(folds), 6)
        for fold in folds:
            train = set(fold.train_positions.tolist())
            test = set(fold.test_positions.tolist())
            self.assertFalse(train & test)
            for test_position in test:
                blocked_start = max(0, test_position - 2)
                blocked_end = min(len(frame), test_position + 2)
                self.assertFalse(any(position in train for position in range(blocked_start, blocked_end)))

    def test_triple_barrier_config_expands_purge_and_embargo(self) -> None:
        splitter = WalkForwardSplitter.from_config(
            {
                "target": {
                    "mode": "triple_barrier",
                    "triple_barrier": {"max_holding_bars": 7},
                },
                "walk_forward": {
                    "mode": "expanding",
                    "initial_train_bars": 40,
                    "test_bars": 10,
                    "purge_lookback_bars": 2,
                    "embargo_bars": 1,
                    "min_train_bars": 20,
                    "adaptive": False,
                },
            }
        )

        self.assertEqual(splitter.purge_lookback_bars, 7)
        self.assertEqual(splitter.embargo_bars, 7)

    def test_triple_barrier_purge_accounts_for_entry_lag(self) -> None:
        splitter = WalkForwardSplitter.from_config(
            {
                "target": {
                    "mode": "triple_barrier",
                    "triple_barrier": {"max_holding_bars": 10},
                },
                "execution": {"entry_lag_bars": 2},
                "walk_forward": {
                    "mode": "expanding",
                    "initial_train_bars": 40,
                    "test_bars": 10,
                    "purge_lookback_bars": 2,
                    "embargo_bars": 1,
                    "min_train_bars": 20,
                    "adaptive": False,
                },
            }
        )

        frame = pd.DataFrame(index=pd.date_range("2024-01-01", periods=100))
        fold = splitter.split(frame)[0]
        last_train_label_end = int(fold.train_positions[-1]) + 2 + 10 - 1

        self.assertEqual(splitter.purge_lookback_bars, 11)
        self.assertEqual(splitter.embargo_bars, 11)
        self.assertLess(last_train_label_end, int(fold.test_positions[0]))

    def test_wilson_interval_bounds_known_win_rate(self) -> None:
        low, high = wilson_interval(55, 100)

        self.assertLess(low, 0.55)
        self.assertGreater(high, 0.55)
        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 1.0)

    def test_backtest_engine_produces_oos_predictions_without_ml(self) -> None:
        config = small_config()
        frame = build_model_frame(synthetic_ohlcv(320), timeframe="1d", config=config)
        engine = BacktestEngine.from_config(config)
        result = engine.run(
            frame,
            ticker="TEST",
            timeframe="1d",
            strategies=build_strategy_suite(config, include_ml=False),
            feature_columns=feature_columns(frame),
            config=config,
        )

        self.assertFalse(result.predictions.empty)
        self.assertIn("test", set(result.predictions["sample_type"]))
        stats = win_rate_stats(result.predictions[result.predictions["sample_type"] == "test"])
        self.assertGreater(stats["predictions"], 0)
        test_rows = result.predictions[result.predictions["sample_type"] == "test"]
        self.assertTrue((pd.to_datetime(test_rows["date"]) > pd.to_datetime(test_rows["train_end"])).all())

    def test_transaction_cost_is_subtracted_once_in_fractional_bps(self) -> None:
        config = small_config()
        frame = build_model_frame(synthetic_ohlcv(320), timeframe="1d", config=config)
        result = BacktestEngine.from_config(config).run(
            frame,
            ticker="TEST",
            timeframe="1d",
            strategies=build_strategy_suite(config, include_ml=False),
            feature_columns=feature_columns(frame),
            config=config,
        )
        active = result.predictions[
            (result.predictions["sample_type"] == "test")
            & (result.predictions["strategy"] == "B1_always_up")
            & result.predictions["active"]
        ]

        self.assertFalse(active.empty)
        np.testing.assert_allclose(active["round_trip_cost_bps"], 5.0)
        np.testing.assert_allclose(active["gross_return"] - active["net_return"], 5.0 / 10_000.0)

    def test_random_baseline_repeats_with_the_same_seed(self) -> None:
        config = small_config()
        data = {("TEST", "1d"): synthetic_ohlcv(320)}

        first, _ = run_research_pipeline(
            data=data,
            config=config,
            include_ml=False,
            intraday_enabled=False,
        )
        second, _ = run_research_pipeline(
            data=data,
            config=config,
            include_ml=False,
            intraday_enabled=False,
        )
        first_random = first[(first["sample_type"] == "test") & (first["strategy"] == "B2_random_50_50")]
        second_random = second[(second["sample_type"] == "test") & (second["strategy"] == "B2_random_50_50")]

        pd.testing.assert_series_equal(
            first_random["signal"].reset_index(drop=True),
            second_random["signal"].reset_index(drop=True),
        )

    def test_persist_train_predictions_flag_controls_train_rows(self) -> None:
        base = small_config()
        frame = build_model_frame(synthetic_ohlcv(320), timeframe="1d", config=base)
        strategies = build_strategy_suite(base, include_ml=False)
        columns = feature_columns(frame)

        def _run(cfg: dict) -> pd.DataFrame:
            return (
                BacktestEngine.from_config(cfg)
                .run(frame, ticker="TEST", timeframe="1d", strategies=strategies, feature_columns=columns, config=cfg)
                .predictions
            )

        default_rows = _run(base)
        self.assertIn("train", set(default_rows["sample_type"]))

        no_train = dict(base)
        no_train["backtest"] = {"persist_train_predictions": False}
        trimmed = _run(no_train)
        self.assertEqual(set(trimmed["sample_type"]), {"test"})
        # dropping train rows must not change the out-of-sample records
        self.assertEqual(
            len(trimmed),
            int((default_rows["sample_type"] == "test").sum()),
        )

    def test_full_pipeline_runs_on_cached_like_data_without_ml(self) -> None:
        config = small_config()
        predictions, skipped = run_research_pipeline(
            data={("TEST", "1d"): synthetic_ohlcv(320)},
            config=config,
            include_ml=False,
            intraday_enabled=False,
        )

        self.assertFalse(predictions.empty)
        self.assertIsInstance(skipped, list)


def small_config() -> dict:
    return {
        "seed": 42,
        "target": {"mode": "binary", "horizon_bars": 1, "flat_band": {"method": "fixed_bps", "fixed_bps": 5}},
        "execution": {"entry_lag_bars": 1, "round_trip_cost_bps": 5},
        "walk_forward": {
            "mode": "expanding",
            "initial_train_bars": 80,
            "test_bars": 30,
            "purge_lookback_bars": 20,
            "embargo_bars": 1,
            "min_train_bars": 60,
            "adaptive": False,
        },
        "models": {},
        "strategies": {"ma_crossover": {"fast": 20, "slow": 50}},
    }


if __name__ == "__main__":
    unittest.main()
