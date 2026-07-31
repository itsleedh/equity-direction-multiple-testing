from __future__ import annotations

import math
from statistics import NormalDist
from typing import Iterable

import numpy as np


def benjamini_hochberg(pvalues: Iterable[float], alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR correction preserving input order."""
    values = np.asarray(list(pvalues), dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    reject = np.full(values.shape, False, dtype=bool)

    valid_mask = np.isfinite(values)
    valid = values[valid_mask]
    if valid.size == 0:
        return adjusted, reject

    order = np.argsort(valid)
    sorted_p = valid[order]
    m = float(len(sorted_p))
    raw_adjusted = sorted_p * m / np.arange(1, len(sorted_p) + 1)
    monotone = np.minimum.accumulate(raw_adjusted[::-1])[::-1]
    monotone = np.clip(monotone, 0.0, 1.0)

    valid_adjusted = np.empty_like(monotone)
    valid_adjusted[order] = monotone
    adjusted[valid_mask] = valid_adjusted
    reject[valid_mask] = valid_adjusted <= alpha
    return adjusted, reject


def probabilistic_sharpe_ratio(
    sharpe: float, n_obs: int, skew: float = 0.0, kurt: float = 3.0, benchmark_sharpe: float = 0.0
) -> float:
    """Probabilistic Sharpe Ratio from Bailey and Lopez de Prado."""
    variance = sharpe_variance(sharpe, n_obs, skew, kurt)
    if not math.isfinite(variance) or variance <= 0:
        return float("nan")
    statistic = (sharpe - benchmark_sharpe) / math.sqrt(variance)
    return NormalDist().cdf(statistic)


def deflated_sharpe_ratio(sharpe: float, n_trials: int, n_obs: int, skew: float = 0.0, kurt: float = 3.0) -> float:
    """Deflated Sharpe Ratio accounting for multiple trials on the same data.

    For a single trial this reduces to PSR against a zero Sharpe benchmark. That
    edge case is explicit because the expected maximum Sharpe formula is not
    defined for N=1.
    """
    if n_trials <= 1:
        return probabilistic_sharpe_ratio(sharpe, n_obs, skew=skew, kurt=kurt, benchmark_sharpe=0.0)

    variance = sharpe_variance(sharpe, n_obs, skew, kurt)
    if not math.isfinite(variance) or variance <= 0:
        return float("nan")

    euler_gamma = 0.5772156649015329
    normal = NormalDist()
    n = float(n_trials)
    expected_max = math.sqrt(variance) * (
        (1.0 - euler_gamma) * normal.inv_cdf(1.0 - 1.0 / n) + euler_gamma * normal.inv_cdf(1.0 - 1.0 / (n * math.e))
    )
    return probabilistic_sharpe_ratio(sharpe, n_obs, skew=skew, kurt=kurt, benchmark_sharpe=expected_max)


def sharpe_variance(sharpe: float, n_obs: int, skew: float = 0.0, kurt: float = 3.0) -> float:
    if n_obs <= 1 or not math.isfinite(sharpe):
        return float("nan")
    skew = 0.0 if not math.isfinite(skew) else skew
    kurt = 3.0 if not math.isfinite(kurt) else kurt
    numerator = 1.0 - skew * sharpe + ((kurt - 1.0) / 4.0) * sharpe * sharpe
    return numerator / (n_obs - 1.0)
