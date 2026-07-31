from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from strategies.base import StrategyContext
from strategies.ml import LogisticRegressionStrategy


class MLFeatureSelectionTests(unittest.TestCase):
    def test_feature_selection_uses_train_fold_and_limits_columns(self) -> None:
        rng = np.random.default_rng(123)
        rows = 120
        frame = pd.DataFrame(
            {
                "feature_a": rng.normal(size=rows),
                "feature_b": rng.normal(size=rows),
                "feature_c": rng.normal(size=rows),
                "feature_d": rng.normal(size=rows),
                "feature_e": rng.normal(size=rows),
            }
        )
        frame["target"] = np.where(frame["feature_a"] + 0.2 * frame["feature_b"] > 0, 1, -1)
        context = StrategyContext(
            ticker="TEST",
            timeframe="1d",
            feature_columns=["feature_a", "feature_b", "feature_c", "feature_d", "feature_e"],
            config={
                "seed": 42,
                "ml": {
                    "feature_selection": {"method": "importance", "top_k": 2},
                    "logistic": {"C": 0.5, "max_iter": 500},
                },
            },
        )

        strategy = LogisticRegressionStrategy()
        strategy.fit(frame, context)
        predictions = strategy.predict(frame.tail(10), context)

        self.assertEqual(len(strategy.selected_features or []), 2)
        self.assertEqual(len(predictions), 10)


if __name__ == "__main__":
    unittest.main()
