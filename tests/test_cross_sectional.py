from __future__ import annotations

import unittest

import pandas as pd

from reports.performance import cross_sectional_portfolio
from strategies.cross_sectional import CrossSectionalMomentumRanker


class CrossSectionalTests(unittest.TestCase):
    def test_momentum_ranker_assigns_top_and_bottom_each_timestamp(self) -> None:
        index = pd.date_range("2024-01-01", periods=4)
        frames = {
            "AAA": pd.DataFrame({"close": [100.0, 110.0, 120.0, 130.0]}, index=index),
            "BBB": pd.DataFrame({"close": [100.0, 99.0, 98.0, 97.0]}, index=index),
            "CCC": pd.DataFrame({"close": [100.0, 105.0, 106.0, 107.0]}, index=index),
        }
        ranker = CrossSectionalMomentumRanker(lookback_bars=1, top_n=1, bottom_n=1)

        signals = ranker.signals(frames)

        self.assertEqual(signals["AAA"].iloc[1:].tolist(), [1, 1, 1])
        self.assertEqual(signals["BBB"].iloc[1:].tolist(), [-1, -1, -1])
        self.assertEqual(signals["CCC"].iloc[1:].tolist(), [0, 0, 0])

    def test_cross_sectional_portfolio_uses_active_equal_weight_returns(self) -> None:
        predictions = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
                "ticker": ["AAA", "BBB", "AAA"],
                "timeframe": ["1d", "1d", "1d"],
                "strategy": ["CS1_momentum_rank"] * 3,
                "sample_type": ["test"] * 3,
                "active": [True, True, False],
                "net_return": [0.02, -0.01, 0.03],
            }
        )

        portfolio = cross_sectional_portfolio(predictions)

        self.assertEqual(int(portfolio.iloc[0]["bars"]), 2)
        self.assertAlmostEqual(float(portfolio.iloc[0]["mean_bar_return"]), 0.0025)


if __name__ == "__main__":
    unittest.main()
