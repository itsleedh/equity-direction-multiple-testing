from __future__ import annotations

import unittest

import pandas as pd

from reports.performance import (
    breakeven_survival,
    cpcv_bar_summary,
    label_distribution,
    sequential_summary,
    sequential_trades,
    summarize_predictions,
)


class CPCVReportingTests(unittest.TestCase):
    def test_cpcv_bar_summary_collapses_duplicate_test_folds(self) -> None:
        predictions = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]),
                "ticker": ["TEST"] * 4,
                "timeframe": ["1d"] * 4,
                "strategy": ["M1"] * 4,
                "fold": [1, 2, 1, 2],
                "split_mode": ["cpcv"] * 4,
                "sample_type": ["test"] * 4,
                "signal": [1, 1, -1, -1],
                "target": [1, 1, -1, 1],
                "active": [True] * 4,
                "hit": [True, True, True, False],
                "net_return": [0.01, 0.02, 0.03, -0.01],
                "gross_return": [0.01, 0.02, 0.03, -0.01],
                "round_trip_cost_bps": [5.0] * 4,
                "execution_return": [0.01, 0.02, 0.03, -0.01],
            }
        )

        summary = cpcv_bar_summary(predictions)

        self.assertEqual(int(summary.iloc[0]["unique_test_bars"]), 2)
        self.assertEqual(int(summary.iloc[0]["fold_prediction_rows"]), 4)
        self.assertAlmostEqual(float(summary.iloc[0]["mean_fold_predictions_per_bar"]), 2.0)
        self.assertAlmostEqual(float(summary.iloc[0]["bar_mean_hit_rate"]), 0.75)

    def test_cpcv_performance_summary_invalidates_overlap_sensitive_stats(self) -> None:
        predictions = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=4),
                "ticker": ["TEST"] * 4,
                "timeframe": ["1d"] * 4,
                "strategy": ["M1"] * 4,
                "fold": [1, 1, 2, 2],
                "split_mode": ["cpcv"] * 4,
                "sample_type": ["test"] * 4,
                "signal": [1, 1, -1, -1],
                "target": [1, -1, -1, 1],
                "active": [True] * 4,
                "hit": [True, False, True, False],
                "net_return": [0.01, -0.01, 0.02, -0.02],
                "gross_return": [0.01, -0.01, 0.02, -0.02],
            }
        )

        summary = summarize_predictions(predictions)

        self.assertEqual(summary.iloc[0]["stats_validity"], "cpcv_overlap_invalid")
        self.assertTrue(pd.isna(summary.iloc[0]["cumulative_return"]))
        self.assertTrue(pd.isna(summary.iloc[0]["binom_pvalue_0_5"]))
        self.assertTrue(pd.isna(summary.iloc[0]["wilson_low"]))

    def test_triple_barrier_summary_invalidates_compounding_but_keeps_weighted_inference(self) -> None:
        predictions = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=4),
                "ticker": ["TEST"] * 4,
                "timeframe": ["1d"] * 4,
                "strategy": ["B1"] * 4,
                "fold": [1] * 4,
                "split_mode": ["expanding"] * 4,
                "sample_type": ["test"] * 4,
                "signal": [1, 1, 1, 1],
                "target": [1, -1, 1, -1],
                "active": [True] * 4,
                "hit": [True, False, True, False],
                "net_return": [0.01, -0.01, 0.02, -0.02],
                "gross_return": [0.01, -0.01, 0.02, -0.02],
                "target_event": ["tp", "sl", "tp", "sl"],
                "holding_bars": [2, 2, 2, 2],
                "label_uniqueness": [0.5, 0.5, 1.0, 1.0],
            }
        )

        summary = summarize_predictions(predictions)
        row = summary.iloc[0]

        self.assertIn("overlap_compounding_invalid", row["stats_validity"])
        self.assertIn("uniqueness_weighted", row["stats_validity"])
        self.assertTrue(pd.isna(row["cumulative_return"]))
        self.assertTrue(pd.isna(row["sharpe"]))
        self.assertAlmostEqual(float(row["avg_label_uniqueness"]), 0.75)
        self.assertAlmostEqual(float(row["effective_sample_size"]), 3.0)
        self.assertAlmostEqual(float(row["win_rate_weighted"]), 0.5)
        self.assertFalse(pd.isna(row["wilson_low"]))

    def test_binary_uniqueness_summary_regresses_to_raw_stats(self) -> None:
        predictions = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=4),
                "ticker": ["TEST"] * 4,
                "timeframe": ["1d"] * 4,
                "strategy": ["B1"] * 4,
                "fold": [1] * 4,
                "split_mode": ["expanding"] * 4,
                "sample_type": ["test"] * 4,
                "signal": [1, 1, 1, 1],
                "target": [1, -1, 1, -1],
                "active": [True] * 4,
                "hit": [True, False, True, False],
                "net_return": [0.01, -0.01, 0.02, -0.02],
                "gross_return": [0.01, -0.01, 0.02, -0.02],
                "label_uniqueness": [1.0] * 4,
            }
        )

        summary = summarize_predictions(predictions)
        row = summary.iloc[0]

        self.assertEqual(row["stats_validity"], "valid")
        self.assertEqual(int(row["predictions"]), 4)
        self.assertAlmostEqual(float(row["effective_sample_size"]), 4.0)
        self.assertAlmostEqual(float(row["win_rate_weighted"]), float(row["win_rate"]))

    def test_label_distribution_deduplicates_strategy_and_fold_rows(self) -> None:
        predictions = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]),
                "ticker": ["TEST"] * 4,
                "timeframe": ["1d"] * 4,
                "strategy": ["A", "B", "A", "B"],
                "fold": [1, 1, 1, 1],
                "target": [1, 1, -1, -1],
                "target_event": ["tp", "tp", "sl", "sl"],
                "holding_bars": [1, 1, 2, 2],
            }
        )

        summary = label_distribution(predictions)

        self.assertEqual(int(summary.iloc[0]["rows"]), 2)
        self.assertEqual(int(summary.iloc[0]["tp_count"]), 1)
        self.assertEqual(int(summary.iloc[0]["sl_count"]), 1)
        self.assertAlmostEqual(float(summary.iloc[0]["avg_holding_bars"]), 1.5)

    def test_sequential_trades_skip_during_hold_and_reenter_after_exit(self) -> None:
        predictions = sequential_fixture(signals=[1, 1, 1, 1, 0, 0], holding_bars=[2, 2, 2, 1, 1, 1])

        trades = sequential_trades(predictions)
        summary = sequential_summary(trades)

        self.assertEqual(trades["signal_date"].dt.strftime("%Y-%m-%d").tolist(), ["2024-01-01", "2024-01-04"])
        self.assertEqual(int(summary.iloc[0]["trades"]), 2)
        self.assertEqual(int(summary.iloc[0]["skipped_signals"]), 2)

    def test_sequential_cumulative_matches_row_compounding_without_overlap(self) -> None:
        returns = [0.01, -0.02, 0.03]
        predictions = sequential_fixture(signals=[1, 1, 1], holding_bars=[1, 1, 1], returns=returns, entry_lag_bars=0)

        trades = sequential_trades(predictions)
        summary = sequential_summary(trades)

        expected = (1.01 * 0.98 * 1.03) - 1.0
        self.assertEqual(int(summary.iloc[0]["trades"]), 3)
        self.assertAlmostEqual(float(summary.iloc[0]["sequential_cumulative_return"]), expected)

    def test_breakeven_survival_counts_cost_thresholds(self) -> None:
        summary = pd.DataFrame(
            {
                "ticker": ["TEST", "TEST", "TEST"],
                "timeframe": ["1d", "1d", "1d"],
                "sample_type": ["test", "test", "test"],
                "strategy": ["A", "B", "C"],
                "breakeven_cost_bps": [0.5, 5.0, 12.0],
            }
        )

        survival = breakeven_survival(summary, costs_bps=[0, 1, 10])

        self.assertEqual(survival["surviving_strategies"].tolist(), [3, 2, 1])
        self.assertEqual(survival["total_strategies"].tolist(), [3, 3, 3])


def sequential_fixture(
    *,
    signals: list[int],
    holding_bars: list[int],
    returns: list[float] | None = None,
    entry_lag_bars: int = 1,
) -> pd.DataFrame:
    returns = returns or [0.01] * len(signals)
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=len(signals)),
            "ticker": ["TEST"] * len(signals),
            "timeframe": ["1d"] * len(signals),
            "strategy": ["B1"] * len(signals),
            "fold": [1] * len(signals),
            "split_mode": ["expanding"] * len(signals),
            "sample_type": ["test"] * len(signals),
            "signal": signals,
            "target": [1] * len(signals),
            "active": [signal != 0 for signal in signals],
            "hit": [signal == 1 for signal in signals],
            "net_return": returns,
            "gross_return": returns,
            "target_event": ["tp"] * len(signals),
            "holding_bars": holding_bars,
            "entry_lag_bars": [entry_lag_bars] * len(signals),
            "label_uniqueness": [1.0] * len(signals),
        }
    )


if __name__ == "__main__":
    unittest.main()
