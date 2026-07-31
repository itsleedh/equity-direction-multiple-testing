from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from scripts.generate_synthetic_sample import DEFAULT_ROWS, DEFAULT_SEED, generate_synthetic_ohlcv
from scripts.run_experiment import load_synthetic_sample


class PublicInterfaceTests(unittest.TestCase):
    def test_synthetic_generator_is_deterministic(self) -> None:
        first = generate_synthetic_ohlcv(rows=DEFAULT_ROWS, seed=DEFAULT_SEED)
        second = generate_synthetic_ohlcv(rows=DEFAULT_ROWS, seed=DEFAULT_SEED)

        pd.testing.assert_frame_equal(first, second)
        self.assertTrue(first["synthetic"].all())

    def test_source_controlled_sample_is_explicitly_synthetic(self) -> None:
        sample = Path(__file__).resolve().parents[1] / "data" / "sample" / "synthetic_ohlcv.csv"
        loaded = load_synthetic_sample(sample)

        self.assertEqual(len(loaded), DEFAULT_ROWS)
        self.assertEqual(list(loaded.columns), ["open", "high", "low", "close", "volume"])


if __name__ == "__main__":
    unittest.main()
