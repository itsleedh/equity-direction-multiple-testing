from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd

from data.loader import normalize_cached_frame
from features.pipeline import atr

MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)


def build_intraday_feature_frame(
    bars: pd.DataFrame,
    *,
    timezone: str = "America/New_York",
    opening_range_minutes: int = 30,
) -> pd.DataFrame:
    frame = normalize_cached_frame(bars)
    frame = to_market_timezone(frame, timezone)
    frame = regular_trading_hours(frame)

    grouped = frame.groupby(frame.index.date, group_keys=False)
    session_open = grouped["open"].transform("first")
    cumulative_volume = grouped["volume"].cumsum()
    cumulative_dollar = (frame["close"] * frame["volume"]).groupby(frame.index.date).cumsum()

    features = pd.DataFrame(index=frame.index)
    features["vwap"] = cumulative_dollar / cumulative_volume.replace(0, np.nan)
    features["price_to_vwap"] = frame["close"] / features["vwap"] - 1.0
    features["intraday_cum_return"] = frame["close"] / session_open - 1.0
    features["intraday_atr_14"] = atr(frame, 14)

    volume_mean = frame["volume"].rolling(20, min_periods=20).mean()
    volume_std = frame["volume"].rolling(20, min_periods=20).std()
    features["intraday_volume_z_20"] = (frame["volume"] - volume_mean) / volume_std
    features["volume_imbalance"] = signed_volume_imbalance(frame)

    features = pd.concat([features, opening_range_features(frame, opening_range_minutes)], axis=1)
    features = pd.concat([features, time_of_day_dummies(frame)], axis=1)

    return pd.concat([frame, features], axis=1).replace([np.inf, -np.inf], np.nan)


def to_market_timezone(frame: pd.DataFrame, timezone: str) -> pd.DataFrame:
    output = frame.copy()
    index = pd.DatetimeIndex(output.index)
    if index.tz is None:
        index = index.tz_localize(timezone, nonexistent="shift_forward", ambiguous="NaT")
    else:
        index = index.tz_convert(timezone)
    output.index = index
    output = output[~output.index.isna()]
    return output


def regular_trading_hours(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.between_time(MARKET_OPEN, MARKET_CLOSE, inclusive="left")


def opening_range_features(frame: pd.DataFrame, minutes: int) -> pd.DataFrame:
    index = pd.DatetimeIndex(frame.index)
    minutes_from_open = (index.hour * 60 + index.minute) - (9 * 60 + 30)
    output = pd.DataFrame(index=frame.index)
    output["minutes_from_open"] = minutes_from_open

    in_opening_range = output["minutes_from_open"].between(0, minutes - 1)
    opening_high = frame["high"].where(in_opening_range).groupby(index.date).transform("max")
    opening_low = frame["low"].where(in_opening_range).groupby(index.date).transform("min")
    output["opening_range_high"] = opening_high.groupby(index.date).ffill()
    output["opening_range_low"] = opening_low.groupby(index.date).ffill()
    output["opening_range_width"] = (output["opening_range_high"] - output["opening_range_low"]) / frame["close"]
    output["orb_breakout_up"] = (frame["close"] > output["opening_range_high"]).astype(float)
    output["orb_breakout_down"] = (frame["close"] < output["opening_range_low"]).astype(float)
    return output


def time_of_day_dummies(frame: pd.DataFrame) -> pd.DataFrame:
    index = pd.DatetimeIndex(frame.index)
    minutes_from_open = (index.hour * 60 + index.minute) - (9 * 60 + 30)
    output = pd.DataFrame(index=frame.index)
    output["tod_open_30m"] = ((minutes_from_open >= 0) & (minutes_from_open <= 29)).astype(float)
    output["tod_midday"] = ((minutes_from_open >= 120) & (minutes_from_open <= 299)).astype(float)
    output["tod_close_30m"] = ((minutes_from_open >= 360) & (minutes_from_open <= 389)).astype(float)
    return output


def signed_volume_imbalance(frame: pd.DataFrame, window: int = 20) -> pd.Series:
    signed_volume = np.sign(frame["close"].diff()).fillna(0.0) * frame["volume"]
    numerator = signed_volume.rolling(window, min_periods=window).sum()
    denominator = frame["volume"].rolling(window, min_periods=window).sum()
    return numerator / denominator.replace(0, np.nan)


def mark_intraday_entry_rules(
    frame: pd.DataFrame,
    *,
    no_new_entries_after: str = "15:55",
) -> pd.DataFrame:
    output = frame.copy()
    cutoff_hour, cutoff_minute = [int(part) for part in no_new_entries_after.split(":")]
    cutoff_minutes = cutoff_hour * 60 + cutoff_minute
    index = pd.DatetimeIndex(output.index)
    next_index = pd.Series(index, index=output.index).shift(-1)
    next_valid = pd.DatetimeIndex(next_index.dropna())
    next_minutes = pd.Series(next_valid.hour * 60 + next_valid.minute, index=next_index.dropna().index)
    next_session = pd.Series(next_valid.date, index=next_index.dropna().index).reindex(output.index)
    current_session = pd.Series(index.date, index=output.index)
    same_session_next_bar = current_session == next_session
    output["can_enter"] = (next_minutes.reindex(output.index) < cutoff_minutes) & same_session_next_bar.fillna(False)
    return output
