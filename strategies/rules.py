from __future__ import annotations

import pandas as pd

from strategies.base import Strategy, StrategyContext, coerce_signal


class MACrossoverStrategy(Strategy):
    name = "S1_ma_crossover"

    def __init__(self, fast: int = 20, slow: int = 50) -> None:
        if fast >= slow:
            raise ValueError("fast MA window must be smaller than slow MA window")
        self.fast = fast
        self.slow = slow

    def predict(self, frame: pd.DataFrame, context: StrategyContext) -> pd.Series:
        fast_column = f"sma_{self.fast}"
        slow_column = f"sma_{self.slow}"
        signal = pd.Series(0, index=frame.index, dtype=float)
        valid = frame[fast_column].notna() & frame[slow_column].notna()
        signal[valid & (frame[fast_column] > frame[slow_column])] = 1
        signal[valid & (frame[fast_column] <= frame[slow_column])] = -1
        return coerce_signal(signal)


class RSIBollingerMeanReversionStrategy(Strategy):
    name = "S2_rsi_bollinger_mean_reversion"

    def __init__(
        self, oversold: float = 30.0, overbought: float = 70.0, lower_b: float = 0.2, upper_b: float = 0.8
    ) -> None:
        self.oversold = oversold
        self.overbought = overbought
        self.lower_b = lower_b
        self.upper_b = upper_b

    def predict(self, frame: pd.DataFrame, context: StrategyContext) -> pd.Series:
        signal = pd.Series(0, index=frame.index, dtype=float)
        long_condition = (frame["rsi_14"] <= self.oversold) & (frame["bb_pct_b_20"] <= self.lower_b)
        short_condition = (frame["rsi_14"] >= self.overbought) & (frame["bb_pct_b_20"] >= self.upper_b)
        signal[long_condition] = 1
        signal[short_condition] = -1
        return coerce_signal(signal)
