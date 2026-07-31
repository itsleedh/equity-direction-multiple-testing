from __future__ import annotations

import unittest

import pandas as pd

from features.intraday import build_intraday_feature_frame, mark_intraday_entry_rules
from main import apply_intraday_cost_schedule


class IntradayTests(unittest.TestCase):
    def test_regular_hours_filter_handles_market_timezone(self) -> None:
        index = pd.to_datetime(
            [
                "2024-03-11 13:29:00+00:00",
                "2024-03-11 13:30:00+00:00",
                "2024-03-11 19:59:00+00:00",
                "2024-03-11 20:00:00+00:00",
            ]
        )
        bars = pd.DataFrame(
            {
                "open": [1.0, 1.0, 1.0, 1.0],
                "high": [1.1, 1.1, 1.1, 1.1],
                "low": [0.9, 0.9, 0.9, 0.9],
                "close": [1.0, 1.0, 1.0, 1.0],
                "volume": [100, 100, 100, 100],
            },
            index=index,
        )

        features = build_intraday_feature_frame(bars, timezone="America/New_York")

        self.assertEqual(len(features), 2)
        self.assertEqual(features.index[0].hour, 9)
        self.assertEqual(features.index[0].minute, 30)
        self.assertEqual(features.index[-1].hour, 15)
        self.assertEqual(features.index[-1].minute, 59)

    def test_intraday_entry_rules_block_close_and_cross_session(self) -> None:
        index = pd.to_datetime(["2024-01-02 15:54:00-05:00", "2024-01-02 15:55:00-05:00", "2024-01-03 09:30:00-05:00"])
        bars = pd.DataFrame(
            {
                "open": [1.0, 1.0, 1.0],
                "high": [1.1, 1.1, 1.1],
                "low": [0.9, 0.9, 0.9],
                "close": [1.0, 1.0, 1.0],
                "volume": [100, 100, 100],
            },
            index=index,
        )

        marked = mark_intraday_entry_rules(bars, no_new_entries_after="15:55")

        self.assertFalse(bool(marked.iloc[0]["can_enter"]))
        self.assertFalse(bool(marked.iloc[1]["can_enter"]))
        self.assertFalse(bool(marked.iloc[2]["can_enter"]))

    def test_intraday_cost_schedule_applies_time_multipliers(self) -> None:
        index = pd.to_datetime(["2024-01-02 09:35:00-05:00", "2024-01-02 10:05:00-05:00"])
        frame = pd.DataFrame({"close": [1.0, 1.0]}, index=index)

        adjusted = apply_intraday_cost_schedule(
            frame,
            base_round_trip_cost_bps=4.0,
            config={
                "execution": {
                    "cost_schedule": {
                        "enabled": True,
                        "rules": [{"start": "09:30", "end": "10:00", "multiplier": 1.5}],
                    }
                }
            },
        )

        self.assertEqual(adjusted["round_trip_cost_bps"].tolist(), [6.0, 4.0])


if __name__ == "__main__":
    unittest.main()
