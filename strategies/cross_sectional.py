from __future__ import annotations

import pandas as pd

from strategies.base import Strategy, StrategyContext, coerce_signal


class CrossSectionalMomentumRanker:
    def __init__(self, *, lookback_bars: int = 20, top_n: int = 1, bottom_n: int = 1) -> None:
        if lookback_bars < 1:
            raise ValueError("cross_sectional.lookback_bars must be at least 1")
        if top_n < 0 or bottom_n < 0:
            raise ValueError("cross_sectional top_n and bottom_n must be non-negative")
        self.lookback_bars = lookback_bars
        self.top_n = top_n
        self.bottom_n = bottom_n

    @classmethod
    def from_config(cls, config: dict) -> "CrossSectionalMomentumRanker":
        cs = config.get("cross_sectional", {})
        return cls(
            lookback_bars=int(cs.get("lookback_bars", 20)),
            top_n=int(cs.get("top_n", 1)),
            bottom_n=int(cs.get("bottom_n", 1)),
        )

    def signals(self, frames: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
        scores = {
            ticker: frame["close"] / frame["close"].shift(self.lookback_bars) - 1.0 for ticker, frame in frames.items()
        }
        score_frame = pd.DataFrame(scores).sort_index()
        signal_frame = pd.DataFrame(0, index=score_frame.index, columns=score_frame.columns, dtype=int)
        for timestamp, row in score_frame.iterrows():
            valid = row.dropna()
            if valid.empty:
                continue
            if self.top_n:
                top = valid.nlargest(min(self.top_n, len(valid))).index
                signal_frame.loc[timestamp, top] = 1
            if self.bottom_n:
                bottom = valid.nsmallest(min(self.bottom_n, len(valid))).index
                signal_frame.loc[timestamp, bottom] = -1
        return {
            ticker: signal_frame[ticker].reindex(frame.index).fillna(0).astype(int) for ticker, frame in frames.items()
        }


class CrossSectionalMomentumRankStrategy(Strategy):
    name = "CS1_momentum_rank"

    def predict(self, frame: pd.DataFrame, context: StrategyContext) -> pd.Series:
        if "_cross_sectional_signal" not in frame.columns:
            return pd.Series(0, index=frame.index, dtype=int)
        return coerce_signal(frame["_cross_sectional_signal"])
