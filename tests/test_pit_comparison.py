from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from reports.pit_comparison import compare_runs, portfolio_stats


def write_run(run_dir: Path, tickers: list[str], dates: pd.DatetimeIndex, returns: dict[str, float]) -> None:
    rows = []
    for ticker in tickers:
        for date in dates:
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "strategy": "B1_always_up",
                    "sample_type": "test",
                    "execution_return": returns[ticker],
                }
            )
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(run_dir / "predictions.csv", index=False)


class PitComparisonTests(unittest.TestCase):
    def test_portfolio_stats_compounds_returns(self) -> None:
        returns = pd.Series([0.01, -0.005, 0.02], index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))
        stats = portfolio_stats(returns)
        self.assertEqual(stats["days"], 3)
        self.assertAlmostEqual(stats["cumulative_return"], 1.01 * 0.995 * 1.02 - 1.0)

    def test_compare_runs_uses_common_window_and_equal_weight(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            long_dates = pd.bdate_range("2024-01-02", periods=10)
            short_dates = long_dates[2:8]
            write_run(base / "a", ["AAA", "BBB"], long_dates, {"AAA": 0.02, "BBB": 0.0})
            write_run(base / "b", ["CCC"], short_dates, {"CCC": 0.01})

            comparison = compare_runs({"a": base / "a", "b": base / "b"})
            self.assertEqual(set(comparison["run"]), {"a", "b"})
            self.assertEqual(set(comparison["days"]), {6})
            row_a = comparison[comparison["run"] == "a"].iloc[0]
            self.assertAlmostEqual(row_a["cumulative_return"], 1.01**6 - 1.0)


if __name__ == "__main__":
    unittest.main()
