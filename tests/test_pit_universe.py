from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from core.config import ConfigError
from core.universe import PitUniverse, load_pit_universe_from_config


def sample_universe() -> PitUniverse:
    return PitUniverse(
        {
            2020: ("AAA", "BBB"),
            2021: ("AAA", "CCC"),
            2022: ("CCC", "DDD"),
        }
    )


class PitUniverseTests(unittest.TestCase):
    def test_union_and_member_years(self) -> None:
        universe = sample_universe()
        self.assertEqual(universe.union_tickers(), ["AAA", "BBB", "CCC", "DDD"])
        self.assertEqual(universe.member_years("AAA"), [2020, 2021])
        self.assertEqual(universe.member_years("DDD"), [2022])
        self.assertTrue(universe.is_member("bbb", 2020))
        self.assertFalse(universe.is_member("BBB", 2021))

    def test_rejects_year_gap(self) -> None:
        with self.assertRaises(ConfigError):
            PitUniverse({2020: ("AAA",), 2022: ("AAA",)})

    def test_rejects_size_mismatch_and_duplicates(self) -> None:
        with self.assertRaises(ConfigError):
            PitUniverse({2020: ("AAA", "BBB"), 2021: ("AAA",)})
        with self.assertRaises(ConfigError):
            PitUniverse({2020: ("AAA", "AAA")})

    def test_member_mask_respects_year_boundaries(self) -> None:
        universe = sample_universe()
        frame = pd.DataFrame(
            {
                "ticker": ["BBB", "BBB", "CCC", "CCC", "AAA"],
                "date": pd.to_datetime(["2020-12-31", "2021-01-04", "2020-12-31", "2021-01-04", "2019-06-01"]),
            }
        )
        mask = universe.member_mask(frame["ticker"], frame["date"])
        self.assertEqual(list(mask), [True, False, False, True, False])

    def test_annotate_filter_and_coverage(self) -> None:
        universe = sample_universe()
        predictions = pd.DataFrame(
            {
                "ticker": ["AAA"] * 4 + ["DDD"] * 2,
                "timeframe": ["1d"] * 6,
                "sample_type": ["train", "test", "test", "test", "test", "test"],
                "date": pd.to_datetime(
                    ["2020-03-02", "2020-06-01", "2021-06-01", "2022-06-01", "2021-06-01", "2022-06-01"]
                ),
            }
        )
        annotated = universe.annotate_predictions(predictions)
        self.assertEqual(list(annotated["pit_member"]), [True, True, True, False, False, True])

        filtered = universe.filter_predictions(annotated)
        self.assertEqual(len(filtered), 4)
        self.assertTrue(filtered["pit_member"].all())

        coverage = universe.coverage_summary(annotated)
        aaa = coverage[coverage["ticker"] == "AAA"].iloc[0]
        self.assertEqual(aaa["membership_years"], "2020,2021")
        self.assertEqual(int(aaa["rows_total"]), 4)
        self.assertEqual(int(aaa["rows_member"]), 3)
        self.assertEqual(int(aaa["test_rows_total"]), 3)
        self.assertEqual(int(aaa["test_rows_member"]), 2)

    def test_from_file_and_config_loading(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            membership_path = Path(tmpdir) / "universe_pit.yaml"
            membership_path.write_text(
                json.dumps(
                    {
                        "selection": "test_top2",
                        "notes": "year Y uses the ranking as of year-end Y-1",
                        "memberships": {"2020": ["aaa", "BBB"], "2021": ["AAA", "CCC"]},
                    }
                ),
                encoding="utf-8",
            )
            universe = PitUniverse.from_file(membership_path)
            self.assertEqual(universe.memberships[2020], ("AAA", "BBB"))
            self.assertEqual(universe.selection, "test_top2")

            config_path = Path(tmpdir) / "config.yaml"
            config = {"universe_membership": {"file": "universe_pit.yaml", "mode": "evaluation_mask"}}
            loaded, mode = load_pit_universe_from_config(config, config_path)
            self.assertIsNotNone(loaded)
            self.assertEqual(mode, "evaluation_mask")
            self.assertEqual(loaded.union_tickers(), ["AAA", "BBB", "CCC"])

            none_universe, none_mode = load_pit_universe_from_config({}, config_path)
            self.assertIsNone(none_universe)
            self.assertEqual(none_mode, "")

            with self.assertRaises(ConfigError):
                load_pit_universe_from_config(
                    {"universe_membership": {"file": "universe_pit.yaml", "mode": "bogus"}}, config_path
                )


if __name__ == "__main__":
    unittest.main()
