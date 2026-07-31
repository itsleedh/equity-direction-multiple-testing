from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from strategies.base import Strategy, StrategyContext, StrategyUnavailable, coerce_signal


@dataclass(frozen=True)
class MLSettings:
    params: dict
    feature_selection: dict
    calibration: dict


class SklearnClassifierStrategy(Strategy):
    requires_training = True
    fit_requires_complete_features = True

    def __init__(self) -> None:
        self.model = None
        self.constant_signal: int | None = None
        self.selected_features: list[str] | None = None

    def fit(self, frame: pd.DataFrame, context: StrategyContext) -> None:
        feature_columns = self._select_features(frame, context)
        self.selected_features = feature_columns
        X = frame[feature_columns]
        y = frame["target"].astype(int)
        valid = y.notna()
        X = X.loc[valid]
        y = y.loc[valid]
        if y.empty:
            raise StrategyUnavailable(f"{self.name} has no training labels")
        unique = sorted(y.unique())
        if len(unique) == 1:
            self.constant_signal = int(unique[0])
            return
        self.model = self._maybe_calibrated_model(self._build_model(context), y, context)
        self.model.fit(X, y)

    def predict(self, frame: pd.DataFrame, context: StrategyContext) -> pd.Series:
        if self.constant_signal is not None:
            return pd.Series(self.constant_signal, index=frame.index, dtype=int)
        if self.model is None:
            raise StrategyUnavailable(f"{self.name} was not fitted")
        probability_up = self.predict_proba_up(frame, context)
        threshold = confidence_threshold(context.config)
        if threshold is not None and probability_up.notna().any():
            predictions = pd.Series(0, index=frame.index, dtype=int)
            predictions[probability_up >= threshold] = 1
            predictions[probability_up <= 1.0 - threshold] = -1
            return coerce_signal(predictions)
        predictions = self.model.predict(frame[self._active_features(context)])
        return coerce_signal(pd.Series(predictions, index=frame.index))

    def predict_proba_up(self, frame: pd.DataFrame, context: StrategyContext) -> pd.Series:
        if self.constant_signal is not None:
            return pd.Series(1.0 if self.constant_signal == 1 else 0.0, index=frame.index, dtype=float)
        if self.model is None or not hasattr(self.model, "predict_proba"):
            return pd.Series(float("nan"), index=frame.index)
        probabilities = self.model.predict_proba(frame[self._active_features(context)])
        classes = list(getattr(self.model, "classes_", []))
        if 1 not in classes:
            return pd.Series(float("nan"), index=frame.index)
        up_index = classes.index(1)
        return pd.Series(probabilities[:, up_index], index=frame.index, dtype=float)

    def _build_model(self, context: StrategyContext):
        raise NotImplementedError

    def _maybe_calibrated_model(self, model, y: pd.Series, context: StrategyContext):
        settings = self._settings(context)
        calibration = settings.calibration
        if not bool(calibration.get("enabled", False)):
            return model
        method = str(calibration.get("method", "isotonic")).lower()
        if method not in {"isotonic", "sigmoid"}:
            raise StrategyUnavailable(f"Unsupported calibration method: {method}")
        class_counts = y.value_counts()
        cv = int(calibration.get("cv", 3))
        max_cv = int(class_counts.min()) if not class_counts.empty else 0
        if max_cv < 2:
            return model
        cv = max(2, min(cv, max_cv))
        try:
            from sklearn.calibration import CalibratedClassifierCV
        except ModuleNotFoundError as exc:
            raise StrategyUnavailable("scikit-learn is required for probability calibration") from exc
        return CalibratedClassifierCV(estimator=model, method=method, cv=cv)

    def _active_features(self, context: StrategyContext) -> list[str]:
        return self.selected_features or context.feature_columns

    def _select_features(self, frame: pd.DataFrame, context: StrategyContext) -> list[str]:
        settings = self._settings(context)
        selection = settings.feature_selection
        method = str(selection.get("method", "off")).lower()
        if method in {"off", "none", "false"}:
            return list(context.feature_columns)
        top_k = int(selection.get("top_k", 15))
        if top_k <= 0 or top_k >= len(context.feature_columns):
            return list(context.feature_columns)
        if method != "importance":
            raise StrategyUnavailable(f"Unsupported feature selection method: {method}")
        return select_top_features_by_importance(
            frame,
            context.feature_columns,
            top_k=top_k,
            random_state=int(context.config.get("seed", 42)),
        )

    def _settings(self, context: StrategyContext) -> MLSettings:
        return ml_settings(context.config, self.name)


class LogisticRegressionStrategy(SklearnClassifierStrategy):
    name = "M1_logistic_regression"

    def _build_model(self, context: StrategyContext):
        try:
            from sklearn.impute import SimpleImputer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
        except ModuleNotFoundError as exc:
            raise StrategyUnavailable("scikit-learn is required for Logistic Regression") from exc

        params = self._settings(context).params
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=float(params.get("C", 0.5)),
                        max_iter=int(params.get("max_iter", 1000)),
                        random_state=int(context.config.get("seed", 42)),
                    ),
                ),
            ]
        )


class RandomForestStrategy(SklearnClassifierStrategy):
    name = "M2_random_forest"

    def _build_model(self, context: StrategyContext):
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.impute import SimpleImputer
            from sklearn.pipeline import Pipeline
        except ModuleNotFoundError as exc:
            raise StrategyUnavailable("scikit-learn is required for Random Forest") from exc

        params = self._settings(context).params
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=int(params.get("n_estimators", 200)),
                        max_depth=params.get("max_depth", 5),
                        min_samples_leaf=int(params.get("min_samples_leaf", 50)),
                        max_features=params.get("max_features", "sqrt"),
                        n_jobs=1,
                        random_state=int(context.config.get("seed", 42)),
                    ),
                ),
            ]
        )


class GradientBoostingStrategy(SklearnClassifierStrategy):
    name = "M3_gradient_boosting"

    def _build_model(self, context: StrategyContext):
        params = self._settings(context).params
        seed = int(context.config.get("seed", 42))
        model = (
            self._lightgbm_model(params, seed) or self._xgboost_model(params, seed) or self._sklearn_model(params, seed)
        )

        try:
            from sklearn.impute import SimpleImputer
            from sklearn.pipeline import Pipeline
        except ModuleNotFoundError as exc:
            raise StrategyUnavailable("scikit-learn is required for Gradient Boosting preprocessing") from exc

        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])

    def _lightgbm_model(self, params: dict, seed: int):
        try:
            from lightgbm import LGBMClassifier  # type: ignore
        except Exception:
            return None
        return LGBMClassifier(
            n_estimators=int(params.get("n_estimators", 150)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            max_depth=int(params.get("max_depth", 3)),
            subsample=float(params.get("subsample", 0.8)),
            random_state=seed,
            verbose=-1,
        )

    def _xgboost_model(self, params: dict, seed: int):
        try:
            from xgboost import XGBClassifier  # type: ignore
        except Exception:
            return None
        return XGBClassifier(
            n_estimators=int(params.get("n_estimators", 150)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            max_depth=int(params.get("max_depth", 3)),
            subsample=float(params.get("subsample", 0.8)),
            random_state=seed,
            eval_metric="logloss",
        )

    def _sklearn_model(self, params: dict, seed: int):
        try:
            from sklearn.ensemble import GradientBoostingClassifier
        except ModuleNotFoundError as exc:
            raise StrategyUnavailable("scikit-learn, LightGBM, or XGBoost is required for Gradient Boosting") from exc
        return GradientBoostingClassifier(
            n_estimators=int(params.get("n_estimators", 150)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            max_depth=int(params.get("max_depth", 3)),
            subsample=float(params.get("subsample", 0.8)),
            random_state=seed,
        )


def finite_feature_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame.replace([np.inf, -np.inf], np.nan).dropna(subset=columns + ["target"])


def confidence_threshold(config: dict) -> float | None:
    value = config.get("execution", {}).get("confidence_threshold")
    if value is None:
        return None
    threshold = float(value)
    if threshold < 0.5 or threshold > 1.0:
        raise StrategyUnavailable("execution.confidence_threshold must be between 0.5 and 1.0")
    return threshold


def ml_settings(config: dict, strategy_name: str) -> MLSettings:
    legacy = config.get("models", {})
    modern = config.get("ml", {})
    strategy_key, legacy_key = {
        "M1_logistic_regression": ("logistic", "logistic_regression"),
        "M2_random_forest": ("random_forest", "random_forest"),
        "M3_gradient_boosting": ("gradient_boosting", "gradient_boosting"),
    }[strategy_name]
    params = {}
    params.update(legacy.get(legacy_key, {}))
    params.update(modern.get(strategy_key, {}))
    return MLSettings(
        params=params,
        feature_selection=dict(modern.get("feature_selection", {"method": "off"})),
        calibration=dict(modern.get("calibration", {"enabled": False})),
    )


def select_top_features_by_importance(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    top_k: int,
    random_state: int,
) -> list[str]:
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
    except ModuleNotFoundError as exc:
        raise StrategyUnavailable("scikit-learn is required for feature selection") from exc

    training = finite_feature_frame(frame, feature_columns)
    if training.empty:
        raise StrategyUnavailable("no complete training rows for feature selection")
    selector = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=64,
                    max_depth=4,
                    min_samples_leaf=max(10, min(50, len(training) // 20)),
                    max_features="sqrt",
                    n_jobs=1,
                    random_state=random_state,
                ),
            ),
        ]
    )
    selector.fit(training[feature_columns], training["target"].astype(int))
    importances = selector.named_steps["model"].feature_importances_
    ranked = sorted(zip(feature_columns, importances), key=lambda item: (-item[1], item[0]))
    selected = [name for name, _ in ranked[:top_k]]
    return selected or list(feature_columns)
