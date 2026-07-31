from __future__ import annotations

import pandas as pd

from strategies.base import Strategy, StrategyContext, coerce_signal


class OpeningRangeBreakoutStrategy(Strategy):
    name = "I1_opening_range_breakout"

    def predict(self, frame: pd.DataFrame, context: StrategyContext) -> pd.Series:
        signal = pd.Series(0, index=frame.index, dtype=float)
        signal[frame["orb_breakout_up"] > 0] = 1
        signal[frame["orb_breakout_down"] > 0] = -1
        return coerce_signal(signal)


class VWAPMeanReversionStrategy(Strategy):
    name = "I2_vwap_mean_reversion"

    def __init__(self, threshold: float = 0.002) -> None:
        self.threshold = threshold

    def predict(self, frame: pd.DataFrame, context: StrategyContext) -> pd.Series:
        signal = pd.Series(0, index=frame.index, dtype=float)
        signal[frame["price_to_vwap"] <= -self.threshold] = 1
        signal[frame["price_to_vwap"] >= self.threshold] = -1
        return coerce_signal(signal)


class IntradayMomentumContinuationStrategy(Strategy):
    name = "I3_intraday_momentum_continuation"

    def __init__(self, threshold: float = 0.001) -> None:
        self.threshold = threshold

    def predict(self, frame: pd.DataFrame, context: StrategyContext) -> pd.Series:
        signal = pd.Series(0, index=frame.index, dtype=float)
        signal[(frame["intraday_cum_return"] > self.threshold) & (frame["price_to_vwap"] > 0)] = 1
        signal[(frame["intraday_cum_return"] < -self.threshold) & (frame["price_to_vwap"] < 0)] = -1
        return coerce_signal(signal)
