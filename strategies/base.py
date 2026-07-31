from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import pandas as pd


class StrategyUnavailable(RuntimeError):
    """Raised when optional model dependencies are not installed."""


@dataclass(frozen=True)
class StrategyContext:
    ticker: str
    timeframe: str
    feature_columns: list[str]
    config: dict[str, Any]


class Strategy:
    name = "strategy"
    requires_training = False
    fit_requires_complete_features = False
    # True when predict() advances internal state that a later predict() call depends on
    # (e.g. a per-call RNG counter). Such strategies must be predicted in the original
    # train-then-test order even when train rows are not persisted.
    predict_advances_state = False

    def clone(self) -> "Strategy":
        return copy.deepcopy(self)

    def fit(self, frame: pd.DataFrame, context: StrategyContext) -> None:
        return None

    def predict(self, frame: pd.DataFrame, context: StrategyContext) -> pd.Series:
        raise NotImplementedError

    def predict_proba_up(self, frame: pd.DataFrame, context: StrategyContext) -> pd.Series:
        return pd.Series(float("nan"), index=frame.index)


def coerce_signal(series: pd.Series) -> pd.Series:
    return series.fillna(0).clip(lower=-1, upper=1).astype(int)
