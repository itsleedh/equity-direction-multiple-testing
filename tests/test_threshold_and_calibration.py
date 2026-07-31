from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from backtest.engine import BacktestEngine
from backtest.splitter import WalkForwardSplitter
from backtest.thresholds import select_nested_threshold
from reports.performance import threshold_sensitivity
from strategies.base import Strategy, StrategyContext
from strategies.ml import LogisticRegressionStrategy


class ThresholdCalibrationTests(unittest.TestCase):
    def test_logistic_probability_with_calibration_off_and_on(self) -> None:
        frame = classification_frame()
        context_off = context({"enabled": False})
        context_on = context({"enabled": True, "method": "sigmoid", "cv": 3})

        off = LogisticRegressionStrategy()
        off.fit(frame, context_off)
        off_probability = off.predict_proba_up(frame.tail(12), context_off)

        on = LogisticRegressionStrategy()
        on.fit(frame, context_on)
        on_probability = on.predict_proba_up(frame.tail(12), context_on)

        self.assertEqual(len(off_probability), 12)
        self.assertEqual(len(on_probability), 12)
        self.assertTrue(((off_probability >= 0) & (off_probability <= 1)).all())
        self.assertTrue(((on_probability >= 0) & (on_probability <= 1)).all())

    def test_confidence_threshold_reduces_trades(self) -> None:
        frame = classification_frame()
        no_threshold = context({"enabled": False})
        with_threshold = context({"enabled": False}, confidence_threshold=0.90)

        base = LogisticRegressionStrategy()
        base.fit(frame, no_threshold)
        all_signals = base.predict(frame, no_threshold)

        selective = LogisticRegressionStrategy()
        selective.fit(frame, with_threshold)
        selective_signals = selective.predict(frame, with_threshold)

        self.assertLessEqual(int((selective_signals != 0).sum()), int((all_signals != 0).sum()))

    def test_threshold_sensitivity_trade_count_is_monotone(self) -> None:
        predictions = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=5),
                "ticker": ["TEST"] * 5,
                "timeframe": ["1d"] * 5,
                "strategy": ["M1_logistic_regression"] * 5,
                "sample_type": ["test"] * 5,
                "probability_up": [0.49, 0.51, 0.54, 0.58, 0.61],
                "target": [-1, 1, 1, 1, 1],
                "execution_return": [0.01] * 5,
                "round_trip_cost_bps": [5] * 5,
            }
        )

        sensitivity = threshold_sensitivity(predictions, thresholds=[0.50, 0.55, 0.60])
        counts = sensitivity.sort_values("threshold")["n_trades"].tolist()

        self.assertEqual(counts, sorted(counts, reverse=True))

    def test_nested_threshold_selection_uses_train_objective(self) -> None:
        train = pd.DataFrame(
            {
                "target": [1, 1, 1, 1],
                "execution_return": [0.01, -0.02, -0.03, 0.04],
            },
            index=pd.date_range("2024-01-01", periods=4),
        )
        probability_up = pd.Series([0.51, 0.56, 0.59, 0.61], index=train.index)

        threshold, objective = select_nested_threshold(
            train,
            probability_up,
            grid=[0.50, 0.55, 0.60],
            objective="net_cumulative_return",
            round_trip_cost=0.0,
        )

        self.assertEqual(threshold, 0.60)
        self.assertAlmostEqual(objective, 0.04)

    def test_engine_applies_nested_threshold_to_test_once(self) -> None:
        dates = pd.date_range("2024-01-01", periods=8)
        frame = pd.DataFrame(
            {
                "probability_feature": [0.51, 0.56, 0.59, 0.61, 0.59, 0.61, 0.52, 0.62],
                "target": [1] * 8,
                "entry_open": [100.0] * 8,
                "exit_close": [101.0] * 8,
                "execution_return": [0.01, -0.02, -0.03, 0.04, 0.01, 0.01, 0.01, 0.01],
                "can_enter": [True] * 8,
            },
            index=dates,
        )
        engine = BacktestEngine(
            WalkForwardSplitter(
                mode="expanding",
                initial_train_bars=4,
                test_bars=2,
                purge_lookback_bars=0,
                embargo_bars=0,
                min_train_bars=3,
                adaptive=False,
            ),
            round_trip_cost_bps=0,
        )

        result = engine.run(
            frame,
            ticker="TEST",
            timeframe="1d",
            strategies=[ProbabilityOnlyStrategy()],
            feature_columns=["probability_feature"],
            config={
                "ml": {
                    "threshold_selection": {
                        "method": "nested",
                        "grid": [0.50, 0.55, 0.60],
                        "objective": "net_cumulative_return",
                    }
                }
            },
        )
        fold_one_test = result.predictions[
            (result.predictions["fold"] == 1) & (result.predictions["sample_type"] == "test")
        ]

        self.assertEqual(fold_one_test["selected_threshold"].unique().tolist(), [0.60])
        self.assertEqual(fold_one_test["signal"].tolist(), [0, 1])


def classification_frame(rows: int = 90) -> pd.DataFrame:
    rng = np.random.default_rng(4)
    feature_a = rng.normal(size=rows)
    feature_b = rng.normal(size=rows)
    frame = pd.DataFrame({"feature_a": feature_a, "feature_b": feature_b})
    frame["target"] = np.where(feature_a + 0.5 * feature_b > 0, 1, -1)
    return frame


def context(calibration: dict, *, confidence_threshold: float | None = None) -> StrategyContext:
    return StrategyContext(
        ticker="TEST",
        timeframe="1d",
        feature_columns=["feature_a", "feature_b"],
        config={
            "seed": 42,
            "execution": {"confidence_threshold": confidence_threshold},
            "ml": {
                "calibration": calibration,
                "logistic": {"C": 1.0, "max_iter": 500},
            },
        },
    )


class ProbabilityOnlyStrategy(Strategy):
    name = "M_test_probability_only"
    requires_training = True
    fit_requires_complete_features = False

    def fit(self, frame: pd.DataFrame, context: StrategyContext) -> None:
        return None

    def predict(self, frame: pd.DataFrame, context: StrategyContext) -> pd.Series:
        return pd.Series(1, index=frame.index, dtype=int)

    def predict_proba_up(self, frame: pd.DataFrame, context: StrategyContext) -> pd.Series:
        return frame["probability_feature"].astype(float)


if __name__ == "__main__":
    unittest.main()
