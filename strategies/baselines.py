from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.base import Strategy, StrategyContext, coerce_signal


class AlwaysUpBaseline(Strategy):
    name = "B1_always_up"

    def predict(self, frame: pd.DataFrame, context: StrategyContext) -> pd.Series:
        return pd.Series(1, index=frame.index, dtype=int)


class RandomBaseline(Strategy):
    name = "B2_random_50_50"
    predict_advances_state = True  # per-call RNG counter: test seed depends on prior predict() calls

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self._predict_calls = 0

    def predict(self, frame: pd.DataFrame, context: StrategyContext) -> pd.Series:
        seed = self.seed + self._predict_calls
        self._predict_calls += 1
        rng = np.random.default_rng(seed)
        values = rng.choice([-1, 1], size=len(frame))
        return pd.Series(values, index=frame.index, dtype=int)


class BuyAndHoldBaseline(Strategy):
    name = "B3_buy_and_hold"

    def predict(self, frame: pd.DataFrame, context: StrategyContext) -> pd.Series:
        return pd.Series(1, index=frame.index, dtype=int)


class TrainMajorityBaseline(Strategy):
    name = "B1_train_majority"
    requires_training = True

    def __init__(self) -> None:
        self.majority_signal = 1

    def fit(self, frame: pd.DataFrame, context: StrategyContext) -> None:
        counts = frame["target"].dropna().astype(int).value_counts()
        if not counts.empty:
            self.majority_signal = int(counts.sort_values(ascending=False).index[0])
            if self.majority_signal == 0:
                self.majority_signal = 1

    def predict(self, frame: pd.DataFrame, context: StrategyContext) -> pd.Series:
        return coerce_signal(pd.Series(self.majority_signal, index=frame.index))
