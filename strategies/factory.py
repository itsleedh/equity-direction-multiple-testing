from __future__ import annotations

from strategies.base import Strategy
from strategies.baselines import AlwaysUpBaseline, BuyAndHoldBaseline, RandomBaseline, TrainMajorityBaseline
from strategies.intraday import (
    IntradayMomentumContinuationStrategy,
    OpeningRangeBreakoutStrategy,
    VWAPMeanReversionStrategy,
)
from strategies.meta import MetaLabelingStrategy, meta_labeling_enabled, validate_meta_labeling_config
from strategies.ml import GradientBoostingStrategy, LogisticRegressionStrategy, RandomForestStrategy
from strategies.rules import MACrossoverStrategy, RSIBollingerMeanReversionStrategy


def build_strategy_suite(config: dict, *, include_ml: bool = True, intraday: bool = False) -> list[Strategy]:
    validate_meta_labeling_config(config)
    seed = int(config.get("seed", 42))
    strategies: list[Strategy] = [
        AlwaysUpBaseline(),
        TrainMajorityBaseline(),
        RandomBaseline(seed=seed),
        BuyAndHoldBaseline(),
    ]

    if intraday:
        strategies.extend(
            [
                OpeningRangeBreakoutStrategy(),
                VWAPMeanReversionStrategy(),
                IntradayMomentumContinuationStrategy(),
            ]
        )
    else:
        strategy_config = config.get("strategies", {})
        ma_config = strategy_config.get("ma_crossover", {})
        strategies.extend(
            [
                MACrossoverStrategy(fast=int(ma_config.get("fast", 20)), slow=int(ma_config.get("slow", 50))),
                RSIBollingerMeanReversionStrategy(),
            ]
        )

    if include_ml:
        strategies.extend([LogisticRegressionStrategy(), RandomForestStrategy(), GradientBoostingStrategy()])
        if meta_labeling_enabled(config):
            strategies.append(MetaLabelingStrategy())
    enabled = config.get("strategies", {}).get("enabled")
    if enabled:
        enabled_names = {str(name) for name in enabled}
        strategies = [strategy for strategy in strategies if strategy.name in enabled_names]
    return strategies
