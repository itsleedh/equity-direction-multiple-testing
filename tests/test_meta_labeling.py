from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from strategies.base import Strategy, StrategyContext, StrategyUnavailable
from strategies.factory import build_strategy_suite
from strategies.meta import MetaLabelingStrategy


class MetaLabelingTests(unittest.TestCase):
    def test_meta_labeling_fits_primary_on_train_only(self) -> None:
        train = meta_frame(60)
        test = meta_frame(20, start="2024-04-01")
        context = meta_context()

        strategy = MetaLabelingStrategy(base_strategy=RecordingPrimaryStrategy())
        strategy.fit(train, context)
        fitted_primary = strategy.base_strategy
        self.assertIsInstance(fitted_primary, RecordingPrimaryStrategy)

        self.assertEqual(fitted_primary.fit_rows, len(train))
        self.assertEqual(fitted_primary.fit_last_index, train.index[-1])
        self.assertLess(fitted_primary.fit_last_index, test.index[0])

    def test_meta_labeling_does_not_search_grid_on_test(self) -> None:
        train = meta_frame(80)
        test = meta_frame(20, start="2024-04-01")
        context = meta_context(confidence_grid=[0.50, 0.55, 0.60, 0.65])

        strategy = MetaLabelingStrategy(base_strategy=RecordingPrimaryStrategy())
        strategy.fit(train, context)
        evaluations_after_fit = strategy.threshold_evaluations
        predictions = strategy.predict(test, context)

        self.assertEqual(evaluations_after_fit, 4)
        self.assertEqual(strategy.threshold_evaluations, evaluations_after_fit)
        self.assertEqual(len(predictions), len(test))
        self.assertIn(strategy.selected_threshold, [0.50, 0.55, 0.60, 0.65])

    def test_meta_labeling_skips_single_class_training_labels(self) -> None:
        train = meta_frame(60)
        train["target_event"] = "tp"

        strategy = MetaLabelingStrategy(base_strategy=RecordingPrimaryStrategy())

        with self.assertRaisesRegex(StrategyUnavailable, "one class"):
            strategy.fit(train, meta_context())

    def test_meta_labeling_skips_too_small_training_sample(self) -> None:
        train = meta_frame(29)

        strategy = MetaLabelingStrategy(base_strategy=RecordingPrimaryStrategy())

        with self.assertRaisesRegex(StrategyUnavailable, "at least 30"):
            strategy.fit(train, meta_context())

    def test_meta_labeling_enabled_in_binary_mode_raises_clear_error(self) -> None:
        config = {
            "target": {"mode": "binary"},
            "ml": {"meta_labeling": {"enabled": True}},
        }

        with self.assertRaisesRegex(ValueError, "target.mode='triple_barrier'"):
            build_strategy_suite(config, include_ml=True)


class RecordingPrimaryStrategy(Strategy):
    name = "M1_logistic_regression"
    requires_training = True
    fit_requires_complete_features = True

    def __init__(self) -> None:
        self.fit_rows = 0
        self.fit_last_index = None

    def fit(self, frame: pd.DataFrame, context: StrategyContext) -> None:
        self.fit_rows = len(frame)
        self.fit_last_index = frame.index[-1]

    def predict(self, frame: pd.DataFrame, context: StrategyContext) -> pd.Series:
        return frame["primary_signal"].astype(int)

    def predict_proba_up(self, frame: pd.DataFrame, context: StrategyContext) -> pd.Series:
        return frame["primary_probability"].astype(float)


def meta_frame(rows: int, *, start: str = "2024-01-01") -> pd.DataFrame:
    index = pd.date_range(start, periods=rows)
    pattern = np.arange(rows) % 4
    feature = np.where(pattern < 2, 1.0, -1.0)
    target_event = np.where(pattern < 2, "tp", "sl")
    execution_return = np.where(pattern < 2, 0.02, -0.015)
    return pd.DataFrame(
        {
            "feature_a": feature,
            "feature_b": np.linspace(-1.0, 1.0, rows),
            "target": np.where(target_event == "tp", 1, -1),
            "target_event": target_event,
            "holding_bars": [1] * rows,
            "execution_return": execution_return,
            "entry_open": [100.0] * rows,
            "exit_close": [101.0] * rows,
            "primary_signal": [1] * rows,
            "primary_probability": np.where(pattern < 2, 0.8, 0.6),
        },
        index=index,
    )


def meta_context(confidence_grid: list[float] | None = None) -> StrategyContext:
    return StrategyContext(
        ticker="TEST",
        timeframe="1d",
        feature_columns=["feature_a", "feature_b"],
        config={
            "seed": 42,
            "target": {"mode": "triple_barrier"},
            "execution": {"entry_lag_bars": 1, "round_trip_cost_bps": 0},
            "ml": {
                "meta_labeling": {
                    "enabled": True,
                    "base_strategy": "M1_logistic_regression",
                    "label": "tp",
                    "confidence_grid": confidence_grid or [0.50, 0.55, 0.60, 0.65],
                    "model": {"type": "logistic", "C": 1.0, "max_iter": 500},
                },
                "feature_selection": {"method": "off"},
                "calibration": {"enabled": False},
                "logistic": {"C": 1.0, "max_iter": 500},
            },
        },
    )


if __name__ == "__main__":
    unittest.main()
