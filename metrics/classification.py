from __future__ import annotations

import math

import pandas as pd


def win_rate_stats(predictions: pd.DataFrame) -> dict[str, float | int]:
    active = predictions[predictions["active"]]
    n = int(len(active))
    wins = int(active["hit"].sum()) if n else 0
    win_rate = wins / n if n else float("nan")
    ci_low, ci_high = wilson_interval(wins, n)
    majority_rate = target_majority_rate(active["target"]) if n else float("nan")
    return {
        "predictions": n,
        "wins": wins,
        "win_rate": win_rate,
        "wilson_low": ci_low,
        "wilson_high": ci_high,
        "target_majority_rate": majority_rate,
        "binom_pvalue_0_5": binomial_two_sided_pvalue(wins, n, 0.5) if n else float("nan"),
        "binom_pvalue_majority": binomial_two_sided_pvalue(wins, n, majority_rate) if n else float("nan"),
    }


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    return wilson_interval_from_rate(successes / n, n, z=z)


def wilson_interval_from_rate(success_rate: float, n_eff: float, z: float = 1.959963984540054) -> tuple[float, float]:
    if n_eff <= 0 or math.isnan(success_rate) or math.isnan(n_eff):
        return float("nan"), float("nan")
    phat = success_rate
    n = n_eff
    denominator = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def target_majority_rate(target: pd.Series) -> float:
    non_flat = target[target != 0]
    if non_flat.empty:
        return 0.5
    counts = non_flat.value_counts(normalize=True)
    return float(counts.max())


def confusion_counts(predictions: pd.DataFrame) -> dict[str, int]:
    labels = [-1, 0, 1]
    output: dict[str, int] = {}
    for actual in labels:
        for predicted in labels:
            output[f"actual_{actual}_pred_{predicted}"] = int(
                ((predictions["target"] == actual) & (predictions["signal"] == predicted)).sum()
            )
    return output


def precision_recall(predictions: pd.DataFrame) -> dict[str, float]:
    output: dict[str, float] = {}
    for label in (-1, 1):
        tp = int(((predictions["target"] == label) & (predictions["signal"] == label)).sum())
        fp = int(((predictions["target"] != label) & (predictions["signal"] == label)).sum())
        fn = int(((predictions["target"] == label) & (predictions["signal"] != label)).sum())
        output[f"precision_{label}"] = tp / (tp + fp) if tp + fp else float("nan")
        output[f"recall_{label}"] = tp / (tp + fn) if tp + fn else float("nan")
    return output


def binomial_two_sided_pvalue(successes: int, n: int, p: float) -> float:
    if n == 0 or not 0.0 < p < 1.0:
        return float("nan")
    if n > 5000:
        mean = n * p
        variance = n * p * (1.0 - p)
        if variance == 0:
            return float("nan")
        z = abs(successes - mean) / math.sqrt(variance)
        return min(1.0, math.erfc(z / math.sqrt(2.0)) * 2.0)

    log_probs = [_log_binomial_pmf(k, n, p) for k in range(n + 1)]
    observed = log_probs[successes]
    eligible = [value for value in log_probs if value <= observed + 1e-12]
    max_log = max(eligible)
    return min(1.0, math.exp(max_log) * sum(math.exp(value - max_log) for value in eligible))


def effective_binomial_two_sided_pvalue(success_rate: float, n_eff: float, p: float) -> float:
    if n_eff <= 0 or not 0.0 < p < 1.0 or math.isnan(success_rate) or math.isnan(n_eff):
        return float("nan")
    variance = p * (1.0 - p) / n_eff
    if variance <= 0:
        return float("nan")
    z = abs(success_rate - p) / math.sqrt(variance)
    return min(1.0, math.erfc(z / math.sqrt(2.0)))


def _log_binomial_pmf(k: int, n: int, p: float) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * math.log(p) + (n - k) * math.log1p(-p)
