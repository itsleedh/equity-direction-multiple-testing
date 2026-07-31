from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from strategies.base import Strategy, StrategyContext, StrategyUnavailable, coerce_signal
from strategies.ml import GradientBoostingStrategy, LogisticRegressionStrategy, RandomForestStrategy


@dataclass(frozen=True)
class MetaLabelingSettings:
    enabled: bool
    base_strategy: str
    label: str
    confidence_grid: list[float]
    model: dict


class MetaLabelingStrategy(Strategy):
    name = "M1_meta_labeled"
    requires_training = True
    fit_requires_complete_features = True
    uses_external_threshold_selection = False

    def __init__(self, base_strategy: Strategy | None = None) -> None:
        self.base_strategy = base_strategy
        self.meta_model = None
        self.selected_threshold = float("nan")
        self.threshold_train_objective = float("nan")
        self.threshold_evaluations = 0
        self.meta_feature_columns: list[str] | None = None

    def fit(self, frame: pd.DataFrame, context: StrategyContext) -> None:
        settings = meta_labeling_settings(context.config)
        if not settings.enabled:
            raise StrategyUnavailable("meta-labeling is disabled")
        if str(context.config.get("target", {}).get("mode", "binary")).lower() != "triple_barrier":
            raise StrategyUnavailable("meta-labeling requires target.mode='triple_barrier'")

        primary = self._primary_strategy(settings)
        primary.fit(frame, context)
        primary_probability = primary.predict_proba_up(frame, context)
        primary_signal = primary.predict(frame, context)

        candidate_mask = primary_signal.reindex(frame.index).fillna(0).astype(int) != 0
        candidate = frame.loc[candidate_mask].copy()
        if len(candidate) < 30:
            raise StrategyUnavailable("meta-labeling requires at least 30 primary-signal training rows")

        y = self._meta_target(candidate, primary_signal.loc[candidate.index], settings, context)
        valid = y.notna()
        candidate = candidate.loc[valid].copy()
        y = y.loc[valid].astype(int)
        if len(candidate) < 30:
            raise StrategyUnavailable("meta-labeling requires at least 30 labeled training rows")
        if y.nunique() < 2:
            raise StrategyUnavailable("meta-labeling training labels contain one class")

        X = self._meta_features(candidate, primary_probability.loc[candidate.index], context)
        self.meta_feature_columns = list(X.columns)
        self.meta_model = self._build_meta_model(settings, context)
        self.meta_model.fit(X, y)
        self.base_strategy = primary

        train_meta_probability = self._predict_meta_probability(frame, context)
        self.selected_threshold, self.threshold_train_objective = self._select_threshold(
            frame,
            primary_signal,
            train_meta_probability,
            settings,
            context,
        )

    def predict(self, frame: pd.DataFrame, context: StrategyContext) -> pd.Series:
        primary_signal = self._primary_signal(frame, context)
        meta_probability = self._predict_meta_probability(frame, context)
        threshold = self.selected_threshold
        if pd.isna(threshold):
            raise StrategyUnavailable("meta-labeling threshold was not selected")
        filtered = primary_signal.where(meta_probability >= threshold, 0)
        return coerce_signal(filtered)

    def predict_proba_up(self, frame: pd.DataFrame, context: StrategyContext) -> pd.Series:
        primary_signal = self._primary_signal(frame, context)
        probability = self._predict_meta_probability(frame, context)
        return probability.where(primary_signal != 0)

    def _primary_strategy(self, settings: MetaLabelingSettings) -> Strategy:
        if self.base_strategy is not None:
            return self.base_strategy.clone()
        mapping: dict[str, Strategy] = {
            "M1_logistic_regression": LogisticRegressionStrategy(),
            "M2_random_forest": RandomForestStrategy(),
            "M3_gradient_boosting": GradientBoostingStrategy(),
        }
        try:
            return mapping[settings.base_strategy].clone()
        except KeyError as exc:
            raise StrategyUnavailable(f"Unsupported meta-labeling base_strategy: {settings.base_strategy}") from exc

    def _primary_signal(self, frame: pd.DataFrame, context: StrategyContext) -> pd.Series:
        if self.base_strategy is None:
            raise StrategyUnavailable("meta-labeling primary strategy was not fitted")
        return self.base_strategy.predict(frame, context).reindex(frame.index).fillna(0).astype(int)

    def _primary_probability(self, frame: pd.DataFrame, context: StrategyContext) -> pd.Series:
        if self.base_strategy is None:
            raise StrategyUnavailable("meta-labeling primary strategy was not fitted")
        return self.base_strategy.predict_proba_up(frame, context).reindex(frame.index)

    def _meta_features(
        self, frame: pd.DataFrame, primary_probability: pd.Series, context: StrategyContext
    ) -> pd.DataFrame:
        features = frame.loc[:, context.feature_columns].copy()
        features["primary_probability"] = primary_probability.reindex(frame.index)
        return features.replace([np.inf, -np.inf], np.nan)

    def _meta_target(
        self,
        frame: pd.DataFrame,
        primary_signal: pd.Series,
        settings: MetaLabelingSettings,
        context: StrategyContext,
    ) -> pd.Series:
        label = settings.label
        if label == "tp":
            if "target_event" not in frame.columns:
                raise StrategyUnavailable("meta-labeling label='tp' requires target_event")
            return (frame["target_event"].astype(str) == "tp").astype(int)
        if label == "net_positive":
            if "execution_return" not in frame.columns:
                raise StrategyUnavailable("meta-labeling label='net_positive' requires execution_return")
            net_return = primary_signal.reindex(frame.index).fillna(0).astype(int) * frame["execution_return"].astype(
                float
            )
            net_return = net_return - primary_signal.reindex(frame.index).fillna(0).abs() * round_trip_costs(
                frame, context
            )
            return (net_return > 0).astype(int)
        raise StrategyUnavailable(f"Unsupported meta-labeling label: {label}")

    def _build_meta_model(self, settings: MetaLabelingSettings, context: StrategyContext):
        model_config = settings.model
        model_type = str(model_config.get("type", "logistic")).lower()
        if model_type != "logistic":
            raise StrategyUnavailable(f"Unsupported meta-labeling model type: {model_type}")
        try:
            from sklearn.impute import SimpleImputer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
        except ModuleNotFoundError as exc:
            raise StrategyUnavailable("scikit-learn is required for meta-labeling") from exc
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=float(model_config.get("C", 0.5)),
                        max_iter=int(model_config.get("max_iter", 1000)),
                        random_state=int(context.config.get("seed", 42)),
                    ),
                ),
            ]
        )

    def _predict_meta_probability(self, frame: pd.DataFrame, context: StrategyContext) -> pd.Series:
        if self.meta_model is None:
            raise StrategyUnavailable("meta-labeling model was not fitted")
        primary_probability = self._primary_probability(frame, context)
        X = self._meta_features(frame, primary_probability, context)
        probabilities = self.meta_model.predict_proba(X)
        classes = list(getattr(self.meta_model, "classes_", []))
        if 1 not in classes:
            return pd.Series(float("nan"), index=frame.index)
        positive_index = classes.index(1)
        return pd.Series(probabilities[:, positive_index], index=frame.index, dtype=float)

    def _select_threshold(
        self,
        frame: pd.DataFrame,
        primary_signal: pd.Series,
        meta_probability: pd.Series,
        settings: MetaLabelingSettings,
        context: StrategyContext,
    ) -> tuple[float, float]:
        best_threshold = float("nan")
        best_value = float("-inf")
        self.threshold_evaluations = 0
        for threshold in sorted(settings.confidence_grid):
            self.threshold_evaluations += 1
            filtered = primary_signal.reindex(frame.index).where(meta_probability.reindex(frame.index) >= threshold, 0)
            value = sequential_cumulative_return_objective(frame, filtered, context)
            if value > best_value:
                best_threshold = float(threshold)
                best_value = float(value)
        return best_threshold, best_value


def meta_labeling_settings(config: dict) -> MetaLabelingSettings:
    raw = dict(config.get("ml", {}).get("meta_labeling", {}))
    enabled = bool(raw.get("enabled", False))
    grid = [float(value) for value in raw.get("confidence_grid", [0.50, 0.55, 0.60, 0.65])]
    if not grid:
        raise StrategyUnavailable("ml.meta_labeling.confidence_grid must not be empty")
    if any(value < 0.0 or value > 1.0 for value in grid):
        raise StrategyUnavailable("ml.meta_labeling.confidence_grid values must be between 0 and 1")
    label = str(raw.get("label", "tp")).lower()
    if label not in {"tp", "net_positive"}:
        raise StrategyUnavailable("ml.meta_labeling.label must be 'tp' or 'net_positive'")
    return MetaLabelingSettings(
        enabled=enabled,
        base_strategy=str(raw.get("base_strategy", "M1_logistic_regression")),
        label=label,
        confidence_grid=sorted(set(grid)),
        model=dict(raw.get("model", {"type": "logistic", "C": 0.5, "max_iter": 1000})),
    )


def meta_labeling_enabled(config: dict) -> bool:
    return bool(config.get("ml", {}).get("meta_labeling", {}).get("enabled", False))


def validate_meta_labeling_config(config: dict) -> None:
    if (
        meta_labeling_enabled(config)
        and str(config.get("target", {}).get("mode", "binary")).lower() != "triple_barrier"
    ):
        raise ValueError("ml.meta_labeling.enabled=true requires target.mode='triple_barrier'")


def round_trip_costs(frame: pd.DataFrame, context: StrategyContext) -> pd.Series:
    default_bps = float(context.config.get("execution", {}).get("round_trip_cost_bps", 5))
    if "round_trip_cost_bps" in frame.columns:
        return frame["round_trip_cost_bps"].astype(float).fillna(default_bps) / 10_000.0
    return pd.Series(default_bps / 10_000.0, index=frame.index)


def sequential_cumulative_return_objective(frame: pd.DataFrame, signals: pd.Series, context: StrategyContext) -> float:
    if frame.empty:
        return 0.0
    ordered = frame.sort_index().copy()
    aligned = signals.reindex(ordered.index).fillna(0).astype(int)
    costs = round_trip_costs(ordered, context)
    entry_lag = int(context.config.get("execution", {}).get("entry_lag_bars", 1))
    next_available_pos = 0
    realized_returns: list[float] = []
    for position, (_, row) in enumerate(ordered.iterrows()):
        signal = int(aligned.iloc[position])
        if signal == 0 or position < next_available_pos:
            continue
        holding = int(row.get("holding_bars", 1))
        if holding < 1 or pd.isna(row.get("execution_return", float("nan"))):
            continue
        net_return = signal * float(row["execution_return"]) - abs(signal) * float(costs.iloc[position])
        realized_returns.append(net_return)
        next_available_pos = position + entry_lag + holding
    if not realized_returns:
        return 0.0
    return float((1.0 + pd.Series(realized_returns)).prod() - 1.0)
