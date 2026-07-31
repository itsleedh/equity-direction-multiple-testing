from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data.loader import normalize_cached_frame

DEFAULT_FEATURE_LOOKBACK = 200
DEFAULT_MACRO_STALENESS_DAYS = 10


def build_feature_frame(
    bars: pd.DataFrame,
    *,
    timeframe: str,
    spy_bars: pd.DataFrame | None = None,
    include_cross_sectional: bool = False,
) -> pd.DataFrame:
    """Build strictly backward-looking technical features.

    Every rolling/EMA operation is evaluated at timestamp t using observations
    through t only. Labels and execution returns are added elsewhere.
    """
    base = normalize_cached_frame(bars)
    close = base["close"]
    high = base["high"]
    low = base["low"]
    volume = base["volume"]

    features = pd.DataFrame(index=base.index)

    for lag in (1, 5, 10, 20):
        features[f"log_ret_{lag}"] = np.log(close / close.shift(lag))

    for window in (10, 20, 50, 200):
        sma = close.rolling(window, min_periods=window).mean()
        ema = close.ewm(span=window, adjust=False, min_periods=window).mean()
        features[f"sma_{window}"] = sma
        features[f"ema_{window}"] = ema
        features[f"price_to_sma_{window}"] = close / sma - 1.0
        features[f"price_to_ema_{window}"] = close / ema - 1.0

    ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    features["macd"] = macd
    features["macd_signal"] = macd_signal
    features["macd_hist"] = macd - macd_signal

    features["rsi_14"] = rsi(close, 14)
    lowest_14 = low.rolling(14, min_periods=14).min()
    highest_14 = high.rolling(14, min_periods=14).max()
    stoch_k = 100.0 * (close - lowest_14) / (highest_14 - lowest_14)
    features["stoch_k_14"] = stoch_k
    features["stoch_d_3"] = stoch_k.rolling(3, min_periods=3).mean()
    features["roc_10"] = close / close.shift(10) - 1.0

    atr_14 = atr(base, 14)
    features["atr_14"] = atr_14
    features["atr_14_pct"] = atr_14 / close
    features["rolling_std_20"] = features["log_ret_1"].rolling(20, min_periods=20).std()
    bb_mid = close.rolling(20, min_periods=20).mean()
    bb_std = close.rolling(20, min_periods=20).std()
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    bb_range = bb_upper - bb_lower
    features["bb_width_20"] = bb_range / bb_mid
    features["bb_pct_b_20"] = (close - bb_lower) / bb_range

    direction = np.sign(close.diff()).fillna(0.0)
    features["obv"] = (direction * volume).cumsum()
    volume_mean = volume.rolling(20, min_periods=20).mean()
    volume_std = volume.rolling(20, min_periods=20).std()
    features["volume_z_20"] = (volume - volume_mean) / volume_std
    features["dollar_volume"] = close * volume

    if timeframe in {"1d", "1wk"}:
        index = pd.DatetimeIndex(base.index)
        features["day_of_week"] = index.dayofweek
        features["month"] = index.month

    if include_cross_sectional and spy_bars is not None:
        spy = normalize_cached_frame(spy_bars)
        spy_close = spy["close"].reindex(base.index).ffill()
        features["rel_strength_spy_20"] = (close / close.shift(20)) / (spy_close / spy_close.shift(20)) - 1.0
        features["rel_strength_spy_60"] = (close / close.shift(60)) / (spy_close / spy_close.shift(60)) - 1.0

    output = pd.concat([base, features], axis=1)
    return output.replace([np.inf, -np.inf], np.nan)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def true_range(bars: pd.DataFrame) -> pd.Series:
    high = bars["high"]
    low = bars["low"]
    previous_close = bars["close"].shift(1)
    ranges = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(bars).rolling(period, min_periods=period).mean()


@dataclass(frozen=True)
class ExternalFeatureSet:
    """Transformed external series keyed by series id, indexed by available_from.

    Each frame holds one column per transform, already computed on observation
    order, with the index shifted to the release timestamp. Merging therefore
    only ever exposes values that were published at or before each bar.

    `frames` are market-level series broadcast to every ticker. `symbol_frames`
    are per-ticker series (e.g. FINRA short interest) merged only into the
    matching ticker's bar frame — never across tickers.
    """

    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    staleness: dict[str, pd.Timedelta] = field(default_factory=dict)
    symbol_frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    symbol_staleness: dict[str, pd.Timedelta] = field(default_factory=dict)


def build_external_feature_set(series_data: dict[str, pd.DataFrame], series_specs: list[dict]) -> ExternalFeatureSet:
    frames: dict[str, pd.DataFrame] = {}
    staleness: dict[str, pd.Timedelta] = {}
    for spec in series_specs:
        series_id = str(spec["id"]).upper().strip()
        data = series_data[series_id]
        if "available_from" not in data.columns:
            raise ValueError(f"External series {series_id} is missing the required available_from column.")
        transforms = list(spec.get("transforms", ["level"]))
        prefix = str(spec.get("prefix", "macro")).strip().lower()
        transformed = external_transforms(data["value"], transforms)
        transformed = transformed.add_prefix(f"{prefix}_{series_id.lower()}_")
        transformed.index = pd.DatetimeIndex(data["available_from"], name="available_from")
        if not transformed.index.is_monotonic_increasing:
            transformed = transformed.sort_index()
        frames[series_id] = transformed
        staleness[series_id] = pd.Timedelta(days=int(spec.get("max_staleness_days", DEFAULT_MACRO_STALENESS_DAYS)))
    return ExternalFeatureSet(frames=frames, staleness=staleness)


def build_symbol_external_feature_set(
    symbol_data: dict[str, pd.DataFrame],
    spec: dict,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.Timedelta]]:
    """Transform per-ticker external series (same column names for every ticker).

    Transforms run on observation (settlement) order per ticker, then the index
    shifts to available_from — identical no-lookahead semantics to the
    market-level path. Column names carry no ticker so each model sees a
    consistent feature space; the merge keys frames by ticker instead.
    """
    transforms = list(spec.get("transforms", ["level"]))
    prefix = str(spec.get("prefix", "si_dtc")).strip().lower()
    staleness_days = int(spec.get("max_staleness_days", DEFAULT_MACRO_STALENESS_DAYS))
    frames: dict[str, pd.DataFrame] = {}
    staleness: dict[str, pd.Timedelta] = {}
    for symbol, data in symbol_data.items():
        key = str(symbol).upper().strip()
        if "available_from" not in data.columns:
            raise ValueError(f"External symbol series {key} is missing the required available_from column.")
        transformed = external_transforms(data["value"], transforms)
        transformed = transformed.add_prefix(f"{prefix}_")
        transformed.index = pd.DatetimeIndex(data["available_from"], name="available_from")
        if not transformed.index.is_monotonic_increasing:
            transformed = transformed.sort_index()
        frames[key] = transformed
        staleness[key] = pd.Timedelta(days=staleness_days)
    return frames, staleness


def external_transforms(values: pd.Series, transforms: list[str]) -> pd.DataFrame:
    """Strictly backward-looking transforms evaluated on observation order."""
    output: dict[str, pd.Series] = {}
    for name in transforms:
        key = str(name).strip().lower()
        if key == "level":
            output[key] = values
            continue
        match = re.fullmatch(r"(diff|pct|z)_(\d+)", key)
        if not match:
            raise ValueError(f"Unsupported external transform '{name}'. Use level, diff_N, pct_N, or z_N.")
        kind, window = match.group(1), int(match.group(2))
        if window < 1:
            raise ValueError(f"External transform '{name}' requires a positive window.")
        if kind == "diff":
            output[key] = values.diff(window)
        elif kind == "pct":
            output[key] = values.pct_change(window)
        else:
            mean = values.rolling(window, min_periods=window).mean()
            std = values.rolling(window, min_periods=window).std()
            output[key] = (values - mean) / std
    return pd.DataFrame(output, index=values.index).replace([np.inf, -np.inf], np.nan)


def merge_external_features(
    frame: pd.DataFrame, external: ExternalFeatureSet, *, symbol: str | None = None
) -> pd.DataFrame:
    """As-of merge external features into a bar frame without lookahead.

    For each bar timestamp t the merge takes the latest transformed value whose
    available_from <= t (backward direction only). Values older than the series'
    max staleness stay NaN — gaps are surfaced, never forward-filled away.
    Per-ticker series merge only when `symbol` matches; a ticker without its own
    series gets no per-ticker columns rather than another ticker's values.
    """
    output = frame.copy()
    bar_index = pd.DatetimeIndex(output.index)
    left_key = bar_index.tz_localize(None) if bar_index.tz is not None else bar_index
    left = pd.DataFrame({"_bar_time": left_key.as_unit("ns")})

    def merge_one(transformed: pd.DataFrame, tolerance: pd.Timedelta) -> None:
        right_index = pd.DatetimeIndex(transformed.index)
        if right_index.tz is not None:
            right_index = right_index.tz_localize(None)
        right_index = right_index.as_unit("ns")
        right = transformed.reset_index(drop=True)
        right.insert(0, "_bar_time", right_index)
        merged = pd.merge_asof(
            left,
            right,
            on="_bar_time",
            direction="backward",
            tolerance=tolerance,
        )
        for column in transformed.columns:
            output[column] = merged[column].to_numpy()

    for series_id, transformed in external.frames.items():
        merge_one(transformed, external.staleness[series_id])
    if symbol is not None and external.symbol_frames:
        key = str(symbol).upper().strip()
        transformed = external.symbol_frames.get(key)
        if transformed is not None:
            merge_one(transformed, external.symbol_staleness[key])
    return output.replace([np.inf, -np.inf], np.nan)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "target",
        "future_close",
        "target_return",
        "entry_open",
        "exit_close",
        "execution_return",
        "can_enter",
        "target_event",
        "holding_bars",
        "label_uniqueness",
        "_cross_sectional_signal",
    }
    return [column for column in frame.columns if column not in excluded]
