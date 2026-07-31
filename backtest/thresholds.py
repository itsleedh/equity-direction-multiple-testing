from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ThresholdSelection:
    method: str
    grid: list[float]
    objective: str


def threshold_selection_from_config(config: dict) -> ThresholdSelection | None:
    selection = config.get("ml", {}).get("threshold_selection", {})
    method = str(selection.get("method", "off")).lower()
    if method in {"off", "none", "false"}:
        return None
    if method != "nested":
        raise ValueError(f"Unsupported ml.threshold_selection.method: {method}")
    grid = [float(value) for value in selection.get("grid", [0.50, 0.52, 0.55, 0.58, 0.60])]
    if not grid:
        raise ValueError("ml.threshold_selection.grid must not be empty")
    if any(value < 0.5 or value > 1.0 for value in grid):
        raise ValueError("ml.threshold_selection.grid values must be between 0.5 and 1.0")
    objective = str(selection.get("objective", "net_cumulative_return")).lower()
    if objective != "net_cumulative_return":
        raise ValueError(f"Unsupported ml.threshold_selection.objective: {objective}")
    return ThresholdSelection(method=method, grid=sorted(set(grid)), objective=objective)


def signals_from_probability(probability_up: pd.Series, threshold: float) -> pd.Series:
    signals = pd.Series(0, index=probability_up.index, dtype=int)
    signals[probability_up >= threshold] = 1
    signals[probability_up <= 1.0 - threshold] = -1
    return signals


def select_nested_threshold(
    frame: pd.DataFrame,
    probability_up: pd.Series,
    *,
    grid: list[float],
    objective: str,
    round_trip_cost: float,
) -> tuple[float, float]:
    if objective != "net_cumulative_return":
        raise ValueError(f"Unsupported nested threshold objective: {objective}")
    aligned_probability = probability_up.reindex(frame.index)
    candidate = frame.assign(probability_up=aligned_probability)
    candidate = candidate.dropna(subset=["probability_up", "execution_return", "target"])
    if candidate.empty:
        return float("nan"), float("nan")

    best_threshold = float("nan")
    best_value = float("-inf")
    for threshold in sorted(float(value) for value in grid):
        value = evaluate_threshold_objective(
            candidate,
            candidate["probability_up"],
            threshold=threshold,
            round_trip_cost=round_trip_cost,
            objective=objective,
        )
        if value > best_value:
            best_threshold = float(threshold)
            best_value = float(value)
    return best_threshold, best_value


def evaluate_threshold_objective(
    frame: pd.DataFrame,
    probability_up: pd.Series,
    *,
    threshold: float,
    round_trip_cost: float,
    objective: str,
) -> float:
    if objective != "net_cumulative_return":
        raise ValueError(f"Unsupported nested threshold objective: {objective}")
    signals = signals_from_probability(probability_up.reindex(frame.index), threshold)
    if "can_enter" in frame.columns:
        signals = signals.where(frame["can_enter"].astype(bool), 0)
    if "round_trip_cost_bps" in frame.columns:
        round_trip_costs = (
            frame["round_trip_cost_bps"].reindex(frame.index).fillna(round_trip_cost * 10_000.0) / 10_000.0
        )
    else:
        round_trip_costs = round_trip_cost
    gross_return = signals * frame["execution_return"]
    net_return = gross_return - (signals.abs() * round_trip_costs)
    return float((1.0 + net_return.fillna(0.0)).prod() - 1.0)
