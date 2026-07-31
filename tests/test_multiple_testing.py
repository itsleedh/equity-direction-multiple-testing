from __future__ import annotations

import unittest

import numpy as np

from metrics.multiple_testing import benjamini_hochberg, deflated_sharpe_ratio, probabilistic_sharpe_ratio


class MultipleTestingTests(unittest.TestCase):
    def test_benjamini_hochberg_known_equal_adjusted_values(self) -> None:
        adjusted, reject = benjamini_hochberg([0.01, 0.02, 0.03, 0.04], alpha=0.05)

        np.testing.assert_allclose(adjusted, [0.04, 0.04, 0.04, 0.04])
        self.assertEqual(reject.tolist(), [True, True, True, True])

    def test_benjamini_hochberg_preserves_unsorted_input_order(self) -> None:
        adjusted, reject = benjamini_hochberg([0.04, 0.01, 0.03, 0.02], alpha=0.05)

        np.testing.assert_allclose(adjusted, [0.04, 0.04, 0.04, 0.04])
        self.assertEqual(reject.tolist(), [True, True, True, True])

    def test_deflated_sharpe_reduces_to_psr_for_single_trial(self) -> None:
        dsr = deflated_sharpe_ratio(0.4, n_trials=1, n_obs=100, skew=0.0, kurt=3.0)
        psr = probabilistic_sharpe_ratio(0.4, n_obs=100, skew=0.0, kurt=3.0, benchmark_sharpe=0.0)

        self.assertAlmostEqual(dsr, psr, places=12)


if __name__ == "__main__":
    unittest.main()
