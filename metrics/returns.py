from __future__ import annotations

import math

import numpy as np
import pandas as pd


PERIODS_PER_YEAR = {
    "1m": 252 * 390,
    "5m": 252 * 78,
    "15m": 252 * 26,
    "1h": 252 * 6.5,
    "1d": 252,
    "1wk": 52,
}

HOLDING_MINUTES = {
    "1m": 1.0,
    "5m": 5.0,
    "15m": 15.0,
    "1h": 60.0,
    "1d": 390.0,
    "1wk": 1950.0,
}


def return_stats(predictions: pd.DataFrame, *, timeframe: str) -> dict[str, float]:
    returns = predictions["net_return"].fillna(0.0)
    gross_returns = predictions["gross_return"].fillna(0.0)
    active_returns = returns[predictions["active"]]
    periods = PERIODS_PER_YEAR.get(timeframe, 252)
    cumulative = float((1.0 + returns).prod() - 1.0)
    return {
        "cumulative_return": cumulative,
        "gross_cumulative_return": float((1.0 + gross_returns).prod() - 1.0),
        "sharpe": sharpe_ratio(returns, periods),
        "sortino": sortino_ratio(returns, periods),
        "max_drawdown": max_drawdown(returns),
        "profit_factor": profit_factor(active_returns),
        "avg_win_loss_ratio": average_win_loss_ratio(active_returns),
        "return_skew": return_skew(returns),
        "return_kurtosis": return_kurtosis(returns),
        "return_observations": int(returns.notna().sum()),
    }


def sharpe_ratio(returns: pd.Series, periods_per_year: float) -> float:
    std = returns.std(ddof=1)
    if std == 0 or math.isnan(std):
        return float("nan")
    return float(returns.mean() / std * math.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, periods_per_year: float) -> float:
    downside = returns[returns < 0]
    downside_std = downside.std(ddof=1)
    if downside_std == 0 or math.isnan(downside_std):
        return float("nan")
    return float(returns.mean() / downside_std * math.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min()) if not drawdown.empty else float("nan")


def profit_factor(returns: pd.Series) -> float:
    wins = returns[returns > 0].sum()
    losses = returns[returns < 0].sum()
    if losses == 0:
        return float("inf") if wins > 0 else float("nan")
    return float(wins / abs(losses))


def average_win_loss_ratio(returns: pd.Series) -> float:
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    if wins.empty or losses.empty:
        return float("nan")
    return float(wins.mean() / abs(losses.mean()))


def return_skew(returns: pd.Series) -> float:
    clean = returns.dropna()
    if len(clean) < 3:
        return float("nan")
    return float(clean.skew())


def return_kurtosis(returns: pd.Series) -> float:
    clean = returns.dropna()
    if len(clean) < 4:
        return float("nan")
    # pandas returns excess kurtosis, DSR expects ordinary kurtosis.
    return float(clean.kurtosis() + 3.0)


def breakeven_cost_bps(predictions: pd.DataFrame) -> float:
    active_count = int(predictions["active"].sum())
    if active_count == 0:
        return float("nan")
    gross_sum = float(predictions.loc[predictions["active"], "gross_return"].sum())
    return gross_sum / active_count * 10_000.0


def daily_pnl(predictions: pd.DataFrame) -> pd.Series:
    dated = predictions.copy()
    dated["session"] = pd.to_datetime(dated["date"]).dt.date
    return dated.groupby("session")["net_return"].sum()


def trade_frequency_stats(predictions: pd.DataFrame, *, timeframe: str) -> dict[str, float]:
    active = predictions[predictions["active"]].copy()
    if active.empty:
        return {"trades_per_day": 0.0, "avg_holding_minutes": float("nan")}
    sessions = pd.to_datetime(predictions["date"]).dt.date.nunique()
    trades_per_day = len(active) / sessions if sessions else float("nan")
    return {
        "trades_per_day": float(trades_per_day),
        "avg_holding_minutes": HOLDING_MINUTES.get(timeframe, float("nan")),
    }


def block_bootstrap_mean_pvalue(values: pd.Series, *, seed: int = 42, samples: int = 2000) -> float:
    clean = values.dropna().to_numpy()
    if len(clean) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    observed = clean.mean()
    centered = clean - observed
    boot = []
    for _ in range(samples):
        sample = rng.choice(centered, size=len(centered), replace=True)
        boot.append(sample.mean())
    boot_abs = np.abs(np.asarray(boot))
    return float((boot_abs >= abs(observed)).mean())
