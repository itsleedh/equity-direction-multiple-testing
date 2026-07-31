from __future__ import annotations

import numpy as np
import pandas as pd

from features.pipeline import ExternalFeatureSet, atr, build_feature_frame, merge_external_features


def build_model_frame(
    bars: pd.DataFrame,
    *,
    timeframe: str,
    config: dict,
    spy_bars: pd.DataFrame | None = None,
    external_features: ExternalFeatureSet | None = None,
    symbol: str | None = None,
) -> pd.DataFrame:
    """Return OHLCV, features, target, and one-bar execution return columns."""
    target_config = config.get("target", {})
    execution_config = config.get("execution", {})
    horizon = int(target_config.get("horizon_bars", 1))
    entry_lag = int(execution_config.get("entry_lag_bars", 1))

    frame = build_feature_frame(
        bars,
        timeframe=timeframe,
        spy_bars=spy_bars,
        include_cross_sectional=bool(target_config.get("include_cross_sectional", False)),
    )
    if external_features is not None:
        frame = merge_external_features(frame, external_features, symbol=symbol)
    return add_target_and_execution(
        frame, target_config=target_config, execution_config=execution_config, horizon=horizon, entry_lag=entry_lag
    )


def add_target_and_execution(
    frame: pd.DataFrame,
    *,
    target_config: dict,
    execution_config: dict,
    horizon: int | None = None,
    entry_lag: int | None = None,
) -> pd.DataFrame:
    horizon = int(target_config.get("horizon_bars", 1) if horizon is None else horizon)
    entry_lag = int(execution_config.get("entry_lag_bars", 1) if entry_lag is None else entry_lag)
    if str(target_config.get("mode", "binary")).lower() == "triple_barrier":
        return add_triple_barrier_target_and_execution(
            frame,
            target_config=target_config,
            entry_lag=entry_lag,
        )
    frame = frame.copy()
    frame["target"] = build_direction_target(frame, config=target_config, horizon=horizon)
    frame["future_close"] = frame["close"].shift(-horizon)
    frame["target_return"] = frame["future_close"] / frame["close"] - 1.0
    frame["entry_open"] = frame["open"].shift(-entry_lag)
    frame["exit_close"] = frame["close"].shift(-entry_lag)
    frame["execution_return"] = frame["exit_close"] / frame["entry_open"] - 1.0
    frame["label_uniqueness"] = np.where(frame["target"].notna(), 1.0, np.nan)
    return frame.replace([np.inf, -np.inf], np.nan)


def build_direction_target(frame: pd.DataFrame, *, config: dict, horizon: int = 1) -> pd.Series:
    mode = str(config.get("mode", "binary")).lower()
    close = frame["close"]
    future_return = close.shift(-horizon) / close - 1.0

    if mode == "binary":
        values = np.where(future_return > 0.0, 1, -1)
        target = pd.Series(values, index=frame.index, dtype="float")
        target[future_return.isna()] = np.nan
        return target

    if mode not in {"ternary", "3-class", "three_class"}:
        raise ValueError(f"Unsupported target mode: {mode}")

    threshold = flat_threshold(frame, config)
    target = pd.Series(0, index=frame.index, dtype="float")
    target[future_return > threshold] = 1
    target[future_return < -threshold] = -1
    target[future_return.isna()] = np.nan
    return target


def add_triple_barrier_target_and_execution(
    frame: pd.DataFrame,
    *,
    target_config: dict,
    entry_lag: int,
) -> pd.DataFrame:
    barrier_config = target_config.get("triple_barrier", {})
    tp_mult = float(barrier_config.get("tp_atr_mult", 1.5))
    sl_mult = float(barrier_config.get("sl_atr_mult", 1.0))
    max_holding_bars = int(barrier_config.get("max_holding_bars", 10))
    atr_period = int(barrier_config.get("atr_period", target_config.get("flat_band", {}).get("atr_period", 14)))
    if max_holding_bars < 1:
        raise ValueError("target.triple_barrier.max_holding_bars must be at least 1")
    if entry_lag < 0:
        raise ValueError("execution.entry_lag_bars must be non-negative")

    output = frame.copy()
    atr_values = output[f"atr_{atr_period}"] if f"atr_{atr_period}" in output.columns else atr(output, atr_period)
    expiry_threshold = flat_threshold(output, target_config)
    n = len(output)
    opens = output["open"].to_numpy(dtype=float)
    highs = output["high"].to_numpy(dtype=float)
    lows = output["low"].to_numpy(dtype=float)
    closes = output["close"].to_numpy(dtype=float)

    targets = np.full(n, np.nan)
    future_close = np.full(n, np.nan)
    target_return = np.full(n, np.nan)
    entry_open = np.full(n, np.nan)
    exit_close = np.full(n, np.nan)
    execution_return = np.full(n, np.nan)
    holding_bars = np.full(n, np.nan)
    events: list[str | float] = [np.nan] * n

    for row in range(n):
        entry_pos = row + entry_lag
        last_pos = entry_pos + max_holding_bars - 1
        if entry_pos >= n or last_pos >= n:
            continue
        entry_price = opens[entry_pos]
        atr_value = float(atr_values.iloc[row])
        if not np.isfinite(entry_price) or not np.isfinite(atr_value) or entry_price <= 0 or atr_value <= 0:
            continue

        upper = entry_price + tp_mult * atr_value
        lower = entry_price - sl_mult * atr_value
        exit_price = float("nan")
        event = ""
        target = float("nan")
        held = float("nan")

        for position in range(entry_pos, last_pos + 1):
            held = float(position - entry_pos + 1)
            if lows[position] <= lower:
                exit_price = lower
                event = "sl"
                target = -1.0
                break
            if highs[position] >= upper:
                exit_price = upper
                event = "tp"
                target = 1.0
                break

        if not event:
            exit_price = closes[last_pos]
            held = float(max_holding_bars)
            realized_return = exit_price / entry_price - 1.0
            threshold = float(expiry_threshold.iloc[row])
            if realized_return > threshold:
                event = "expiry_up"
                target = 1.0
            elif realized_return < -threshold:
                event = "expiry_down"
                target = -1.0
            else:
                event = "expiry_flat"
                target = 0.0

        entry_open[row] = entry_price
        exit_close[row] = exit_price
        future_close[row] = exit_price
        target_return[row] = exit_price / entry_price - 1.0
        execution_return[row] = target_return[row]
        holding_bars[row] = held
        targets[row] = target
        events[row] = event

    output["target"] = targets
    output["future_close"] = future_close
    output["target_return"] = target_return
    output["entry_open"] = entry_open
    output["exit_close"] = exit_close
    output["execution_return"] = execution_return
    output["target_event"] = events
    output["holding_bars"] = holding_bars
    output["label_uniqueness"] = label_uniqueness_from_holding_bars(
        pd.Series(holding_bars, index=output.index), entry_lag=entry_lag
    )
    return output.replace([np.inf, -np.inf], np.nan)


def label_uniqueness_from_holding_bars(holding_bars: pd.Series, *, entry_lag: int) -> pd.Series:
    n = len(holding_bars)
    intervals: list[tuple[int, int] | None] = []
    for row, value in enumerate(holding_bars.to_numpy(dtype=float)):
        if not np.isfinite(value) or value < 1:
            intervals.append(None)
            continue
        start = row + entry_lag
        end = start + int(value) - 1
        if start >= n or end >= n:
            intervals.append(None)
            continue
        intervals.append((start, end))
    return pd.Series(label_uniqueness_from_intervals(n, intervals), index=holding_bars.index)


def label_uniqueness_from_intervals(n_bars: int, intervals: list[tuple[int, int] | None]) -> np.ndarray:
    concurrency = np.zeros(n_bars, dtype=float)
    for interval in intervals:
        if interval is None:
            continue
        start, end = interval
        concurrency[start : end + 1] += 1.0
    uniqueness = np.full(n_bars, np.nan)
    for row, interval in enumerate(intervals[:n_bars]):
        if interval is None:
            continue
        start, end = interval
        active_concurrency = concurrency[start : end + 1]
        if len(active_concurrency) == 0 or (active_concurrency <= 0).any():
            continue
        uniqueness[row] = float((1.0 / active_concurrency).mean())
    return uniqueness


def flat_threshold(frame: pd.DataFrame, config: dict) -> pd.Series:
    flat_config = config.get("flat_band", {})
    method = str(flat_config.get("method", "atr")).lower()
    if method == "fixed_bps":
        fixed_bps = float(flat_config.get("fixed_bps", 5))
        return pd.Series(fixed_bps / 10_000.0, index=frame.index)
    if method == "atr":
        period = int(flat_config.get("atr_period", 14))
        if "atr_14_pct" in frame.columns and period == 14:
            return frame["atr_14_pct"].fillna(0.0)
        return (atr(frame, period) / frame["close"]).fillna(0.0)
    raise ValueError(f"Unsupported flat band method: {method}")
