from __future__ import annotations

import unittest

import pandas as pd

from reports.performance import significance_summary, summarize_predictions


def strategy_block(name: str, *, n_up: int, n_down: int, n_hit: int) -> pd.DataFrame:
    """Deterministic prediction rows: target majority = n_up/(n_up+n_down), win rate = n_hit/n."""
    n = n_up + n_down
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=n),
            "ticker": ["TEST"] * n,
            "timeframe": ["1d"] * n,
            "strategy": [name] * n,
            "fold": [1] * n,
            "split_mode": ["expanding"] * n,
            "sample_type": ["test"] * n,
            "signal": [1] * n,
            "target": [1] * n_up + [-1] * n_down,
            "active": [True] * n,
            "hit": [True] * n_hit + [False] * (n - n_hit),
            "net_return": [0.001] * n,
            "gross_return": [0.001] * n,
        }
    )


class MajorityFdrTests(unittest.TestCase):
    """The 0.5 null is passed by pure drift; the majority-rate gate must not be."""

    def build_summary(self) -> pd.DataFrame:
        predictions = pd.concat(
            [
                # win rate 0.60 == majority 0.60: pure drift, no edge over majority
                strategy_block("DRIFT", n_up=240, n_down=160, n_hit=240),
                # win rate 0.75 > majority 0.60: genuine edge on both nulls
                strategy_block("EDGE", n_up=240, n_down=160, n_hit=300),
                # win rate 0.40 < majority 0.60: significantly worse, not an edge
                strategy_block("WORSE", n_up=240, n_down=160, n_hit=160),
            ],
            ignore_index=True,
        )
        return summarize_predictions(predictions)

    def test_drift_passes_half_null_but_not_majority_gate(self) -> None:
        summary = self.build_summary()
        row = summary[summary["strategy"] == "DRIFT"].iloc[0]

        self.assertTrue(bool(row["significant_after_fdr"]))
        self.assertFalse(bool(row["significant_after_fdr_majority"]))
        self.assertFalse(bool(row["beats_majority_after_fdr"]))

    def test_genuine_edge_passes_both_nulls_and_beats_majority(self) -> None:
        summary = self.build_summary()
        row = summary[summary["strategy"] == "EDGE"].iloc[0]

        self.assertTrue(bool(row["significant_after_fdr"]))
        self.assertTrue(bool(row["significant_after_fdr_majority"]))
        self.assertTrue(bool(row["beats_majority_after_fdr"]))

    def test_significantly_worse_rejects_majority_null_without_beating_it(self) -> None:
        summary = self.build_summary()
        row = summary[summary["strategy"] == "WORSE"].iloc[0]

        self.assertTrue(bool(row["significant_after_fdr_majority"]))
        self.assertFalse(bool(row["beats_majority_after_fdr"]))

    def test_significance_summary_reports_majority_columns(self) -> None:
        table = significance_summary(self.build_summary()).set_index("strategy")

        self.assertEqual(int(table.loc["DRIFT", "fdr_pass_count"]), 1)
        self.assertEqual(int(table.loc["DRIFT", "fdr_pass_majority_count"]), 0)
        self.assertEqual(int(table.loc["DRIFT", "beats_majority_after_fdr_count"]), 0)
        self.assertEqual(int(table.loc["EDGE", "beats_majority_after_fdr_count"]), 1)
        self.assertEqual(int(table.loc["WORSE", "fdr_pass_majority_count"]), 1)
        self.assertEqual(int(table.loc["WORSE", "beats_majority_after_fdr_count"]), 0)

    def test_cpcv_rows_invalidate_majority_fdr(self) -> None:
        predictions = strategy_block("M1", n_up=6, n_down=4, n_hit=7)
        predictions["split_mode"] = "cpcv"
        predictions["fold"] = [1] * 5 + [2] * 5

        summary = summarize_predictions(predictions)
        row = summary.iloc[0]

        self.assertTrue(pd.isna(row["pvalue_fdr_bh_majority"]))
        self.assertFalse(bool(row["significant_after_fdr_majority"]))
        self.assertFalse(bool(row["beats_majority_after_fdr"]))


if __name__ == "__main__":
    unittest.main()
