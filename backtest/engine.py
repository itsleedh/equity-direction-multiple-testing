from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from backtest.splitter import Fold, WalkForwardSplitter
from backtest.thresholds import select_nested_threshold, signals_from_probability, threshold_selection_from_config
from strategies.base import Strategy, StrategyContext, StrategyUnavailable

LOGGER = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    predictions: pd.DataFrame
    skipped: list[dict[str, str]] = field(default_factory=list)
    folds: list[Fold] = field(default_factory=list)


class BacktestEngine:
    def __init__(self, splitter: WalkForwardSplitter, *, round_trip_cost_bps: float = 5.0) -> None:
        self.splitter = splitter
        self.round_trip_cost = round_trip_cost_bps / 10_000.0

    @classmethod
    def from_config(cls, config: dict) -> "BacktestEngine":
        execution = config.get("execution", {})
        return cls(
            WalkForwardSplitter.from_config(config),
            round_trip_cost_bps=float(execution.get("round_trip_cost_bps", 5)),
        )

    def run(
        self,
        frame: pd.DataFrame,
        *,
        ticker: str,
        timeframe: str,
        strategies: list[Strategy],
        feature_columns: list[str],
        config: dict,
    ) -> BacktestResult:
        usable = frame.dropna(subset=["target", "entry_open", "exit_close", "execution_return"]).copy()
        if "can_enter" not in usable.columns:
            usable["can_enter"] = True
        folds = self.splitter.split(usable)
        skipped: list[dict[str, str]] = []
        records: list[pd.DataFrame] = []
        if not folds:
            skipped.append(
                {
                    "ticker": ticker,
                    "timeframe": timeframe,
                    "strategy": "*",
                    "reason": "not enough rows for walk-forward split",
                }
            )
            return BacktestResult(pd.DataFrame(), skipped=skipped, folds=[])

        context = StrategyContext(ticker=ticker, timeframe=timeframe, feature_columns=feature_columns, config=config)
        threshold_selection = threshold_selection_from_config(config)
        entry_lag_bars = int(config.get("execution", {}).get("entry_lag_bars", 1))
        # In an expanding walk-forward the train slice grows every fold, so persisting train-sample
        # prediction rows is quadratic in fold count — fine for daily history, but it explodes on
        # multi-year intraday runs. Gate it (default on = unchanged behaviour); test rows (the only
        # out-of-sample records the significance/survival reports read) are always kept.
        persist_train = bool(config.get("backtest", {}).get("persist_train_predictions", True))
        for fold in folds:
            train = usable.iloc[fold.train_positions]
            test = usable.iloc[fold.test_positions]
            for template in strategies:
                strategy = template.clone()
                selected_threshold = float("nan")
                threshold_train_objective = float("nan")
                try:
                    use_nested_threshold = (
                        threshold_selection is not None
                        and strategy.name.startswith("M")
                        and bool(getattr(strategy, "uses_external_threshold_selection", True))
                    )
                    if strategy.requires_training:
                        required_columns = (
                            feature_columns + ["target"] if strategy.fit_requires_complete_features else ["target"]
                        )
                        fit_frame = train.dropna(subset=required_columns)
                        if fit_frame.empty:
                            raise StrategyUnavailable("no complete training rows")
                        strategy.fit(fit_frame, context)
                    if strategy.requires_training and use_nested_threshold:
                        train_probability_up = strategy.predict_proba_up(train, context)
                        test_probability_up = strategy.predict_proba_up(test, context)
                        selected_threshold, threshold_train_objective = select_nested_threshold(
                            train,
                            train_probability_up,
                            grid=threshold_selection.grid,
                            objective=threshold_selection.objective,
                            round_trip_cost=self.round_trip_cost,
                        )
                        if pd.isna(selected_threshold):
                            raise StrategyUnavailable(
                                "nested threshold selection requires non-empty training probabilities"
                            )
                        train_signals = signals_from_probability(train_probability_up, selected_threshold)
                        test_signals = signals_from_probability(test_probability_up, selected_threshold)
                    else:
                        # Predicting over the (growing) train slice is only needed to emit train
                        # records, so skip it when they are not persisted — except for strategies
                        # whose predict() advances state their test prediction depends on, which
                        # must still run first, in order, to stay bit-identical.
                        if persist_train or strategy.predict_advances_state:
                            train_signals = strategy.predict(train, context)
                        else:
                            train_signals = None
                        test_signals = strategy.predict(test, context)
                        train_probability_up = strategy.predict_proba_up(train, context) if persist_train else None
                        test_probability_up = strategy.predict_proba_up(test, context)
                except StrategyUnavailable as exc:
                    LOGGER.warning("Skipping %s %s %s: %s", ticker, timeframe, strategy.name, exc)
                    skipped.append(
                        {"ticker": ticker, "timeframe": timeframe, "strategy": strategy.name, "reason": str(exc)}
                    )
                    continue
                except Exception as exc:
                    LOGGER.exception("Skipping %s %s %s after strategy failure", ticker, timeframe, strategy.name)
                    skipped.append(
                        {"ticker": ticker, "timeframe": timeframe, "strategy": strategy.name, "reason": str(exc)}
                    )
                    continue

                if pd.isna(selected_threshold) and hasattr(strategy, "selected_threshold"):
                    selected_threshold = float(getattr(strategy, "selected_threshold"))
                if pd.isna(threshold_train_objective) and hasattr(strategy, "threshold_train_objective"):
                    threshold_train_objective = float(getattr(strategy, "threshold_train_objective"))

                if persist_train:
                    records.append(
                        self._records(
                            train,
                            train_signals,
                            train_probability_up,
                            ticker,
                            timeframe,
                            strategy.name,
                            fold,
                            "train",
                            selected_threshold=selected_threshold,
                            threshold_train_objective=threshold_train_objective,
                            entry_lag_bars=entry_lag_bars,
                        )
                    )
                records.append(
                    self._records(
                        test,
                        test_signals,
                        test_probability_up,
                        ticker,
                        timeframe,
                        strategy.name,
                        fold,
                        "test",
                        selected_threshold=selected_threshold,
                        threshold_train_objective=threshold_train_objective,
                        entry_lag_bars=entry_lag_bars,
                    )
                )

        if not records:
            return BacktestResult(pd.DataFrame(), skipped=skipped, folds=folds)
        return BacktestResult(pd.concat(records, axis=0).reset_index(drop=True), skipped=skipped, folds=folds)

    def _records(
        self,
        frame: pd.DataFrame,
        signals: pd.Series,
        probability_up: pd.Series,
        ticker: str,
        timeframe: str,
        strategy_name: str,
        fold: Fold,
        sample_type: str,
        selected_threshold: float = float("nan"),
        threshold_train_objective: float = float("nan"),
        entry_lag_bars: int = 1,
    ) -> pd.DataFrame:
        aligned = signals.reindex(frame.index).fillna(0).astype(int)
        aligned = aligned.where(frame["can_enter"].astype(bool), 0)
        probability_up = probability_up.reindex(frame.index)
        if "round_trip_cost_bps" in frame.columns:
            round_trip_cost_bps = (
                frame["round_trip_cost_bps"].reindex(frame.index).fillna(self.round_trip_cost * 10_000.0)
            )
        else:
            round_trip_cost_bps = pd.Series(self.round_trip_cost * 10_000.0, index=frame.index)
        round_trip_cost = round_trip_cost_bps / 10_000.0
        gross_return = aligned * frame["execution_return"]
        net_return = gross_return - (aligned.abs() * round_trip_cost)
        active = aligned != 0
        hit = (aligned == frame["target"].astype(int)) & active & (frame["target"].astype(int) != 0)
        data = {
            "date": frame.index,
            "ticker": ticker,
            "timeframe": timeframe,
            "strategy": strategy_name,
            "fold": fold.number,
            "split_mode": self.splitter.mode,
            "sample_type": sample_type,
            "signal": aligned.values,
            "target": frame["target"].astype(int).values,
            "active": active.values,
            "hit": hit.values,
            "probability_up": probability_up.values,
            "entry_open": frame["entry_open"].values,
            "exit_close": frame["exit_close"].values,
            "execution_return": frame["execution_return"].values,
            "gross_return": gross_return.values,
            "net_return": net_return.values,
            "round_trip_cost_bps": round_trip_cost_bps.values,
            "selected_threshold": selected_threshold,
            "threshold_train_objective": threshold_train_objective,
            "entry_lag_bars": entry_lag_bars,
            "train_start": fold.train_start,
            "train_end": fold.train_end,
            "test_start": fold.test_start,
            "test_end": fold.test_end,
            "purge_bars": fold.purge_bars,
            "embargo_bars": fold.embargo_bars,
        }
        if "target_event" in frame.columns:
            data["target_event"] = frame["target_event"].values
        if "holding_bars" in frame.columns:
            data["holding_bars"] = frame["holding_bars"].values
        if "label_uniqueness" in frame.columns:
            data["label_uniqueness"] = frame["label_uniqueness"].values
        return pd.DataFrame(data)
