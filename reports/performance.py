from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pandas as pd

from metrics.classification import (
    confusion_counts,
    effective_binomial_two_sided_pvalue,
    precision_recall,
    wilson_interval_from_rate,
    win_rate_stats,
)
from metrics.multiple_testing import benjamini_hochberg, deflated_sharpe_ratio
from metrics.returns import (
    block_bootstrap_mean_pvalue,
    breakeven_cost_bps,
    daily_pnl,
    max_drawdown,
    return_stats,
    trade_frequency_stats,
)

LOGGER = logging.getLogger(__name__)


def summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    rows = []
    groups = predictions.groupby(["ticker", "timeframe", "strategy", "sample_type"], dropna=False)
    for (ticker, timeframe, strategy, sample_type), group in groups:
        row = {
            "ticker": ticker,
            "timeframe": timeframe,
            "strategy": strategy,
            "sample_type": sample_type,
            "split_mode": split_mode(group),
            "target_mode": target_mode(group),
        }
        row.update(win_rate_stats(group))
        row.update(label_uniqueness_stats(group))
        if row["target_mode"] == "triple_barrier":
            apply_uniqueness_weighted_inference(row)
        row.update(precision_recall(group))
        row.update(return_stats(group, timeframe=timeframe))
        row.update(trade_frequency_stats(group, timeframe=timeframe))
        row["confusion_matrix"] = json.dumps(confusion_counts(group), sort_keys=True)
        row["breakeven_cost_bps"] = breakeven_cost_bps(group)
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(["sample_type", "timeframe", "ticker", "strategy"]).reset_index(drop=True)
    return add_statistical_corrections(summary)


def split_mode(group: pd.DataFrame) -> str:
    if "split_mode" not in group.columns:
        return "unknown"
    modes = sorted(str(value) for value in group["split_mode"].dropna().unique())
    return modes[0] if len(modes) == 1 else "mixed"


def target_mode(group: pd.DataFrame) -> str:
    if "target_event" in group.columns and group["target_event"].notna().any():
        return "triple_barrier"
    return "binary"


def label_uniqueness_stats(group: pd.DataFrame) -> dict[str, float]:
    active = group[group["active"]].copy()
    if active.empty:
        return {
            "avg_label_uniqueness": float("nan"),
            "effective_sample_size": float("nan"),
            "kish_effective_sample_size": float("nan"),
            "win_rate_weighted": float("nan"),
        }
    if "label_uniqueness" in active.columns:
        weights = active["label_uniqueness"].astype(float)
    else:
        weights = pd.Series(1.0, index=active.index)
    weights = weights.replace([float("inf"), float("-inf")], float("nan")).fillna(0.0).clip(lower=0.0)
    weight_sum = float(weights.sum())
    weight_square_sum = float((weights * weights).sum())
    hits = active["hit"].astype(float)
    weighted_win_rate = float((weights * hits).sum() / weight_sum) if weight_sum > 0 else float("nan")
    kish_ess = float(weight_sum * weight_sum / weight_square_sum) if weight_square_sum > 0 else float("nan")
    return {
        "avg_label_uniqueness": float(weights.mean()),
        "effective_sample_size": weight_sum if weight_sum > 0 else float("nan"),
        "kish_effective_sample_size": kish_ess,
        "win_rate_weighted": weighted_win_rate,
    }


def apply_uniqueness_weighted_inference(row: dict[str, object]) -> None:
    ess = float(row.get("effective_sample_size", float("nan")))
    weighted_win_rate = float(row.get("win_rate_weighted", float("nan")))
    if pd.isna(ess) or ess <= 0 or pd.isna(weighted_win_rate):
        return
    row["wilson_low"], row["wilson_high"] = wilson_interval_from_rate(weighted_win_rate, ess)
    row["binom_pvalue_0_5"] = effective_binomial_two_sided_pvalue(weighted_win_rate, ess, 0.5)
    majority_rate = float(row.get("target_majority_rate", float("nan")))
    row["binom_pvalue_majority"] = effective_binomial_two_sided_pvalue(weighted_win_rate, ess, majority_rate)


def add_statistical_corrections(summary: pd.DataFrame, *, alpha: float = 0.05) -> pd.DataFrame:
    output = summary.copy()
    output["pvalue_fdr_bh"] = float("nan")
    output["significant_after_fdr"] = False
    output["pvalue_fdr_bh_majority"] = float("nan")
    output["significant_after_fdr_majority"] = False
    output["beats_majority_after_fdr"] = False
    output["deflated_sharpe"] = float("nan")
    output["stats_validity"] = "valid"
    if output.empty:
        return output

    test_mask = output["sample_type"] == "test"
    adjusted, reject = benjamini_hochberg(output.loc[test_mask, "binom_pvalue_0_5"], alpha=alpha)
    output.loc[test_mask, "pvalue_fdr_bh"] = adjusted
    output.loc[test_mask, "significant_after_fdr"] = reject

    # Drift-aware gate: the 0.5 null is passed by any long bias in a rising
    # market, so the edge claim must clear the target majority rate instead.
    # The two-sided rejection alone also flags "significantly worse" rows;
    # beats_majority_after_fdr additionally requires the tested win rate
    # (uniqueness-weighted when that is what the p-value used) above majority.
    adjusted_majority, reject_majority = benjamini_hochberg(output.loc[test_mask, "binom_pvalue_majority"], alpha=alpha)
    output.loc[test_mask, "pvalue_fdr_bh_majority"] = adjusted_majority
    output.loc[test_mask, "significant_after_fdr_majority"] = reject_majority
    tested_win_rate = effective_tested_win_rate(output)
    output.loc[test_mask, "beats_majority_after_fdr"] = (
        reject_majority
        & (tested_win_rate.loc[test_mask] > output.loc[test_mask, "target_majority_rate"]).fillna(False).to_numpy()
    )

    trial_counts = output.groupby(["ticker", "timeframe", "sample_type"])["strategy"].transform("nunique")
    for idx, row in output.iterrows():
        output.at[idx, "deflated_sharpe"] = deflated_sharpe_ratio(
            float(row["sharpe"]),
            int(trial_counts.loc[idx]),
            int(row.get("return_observations", row.get("predictions", 0))),
            float(row.get("return_skew", 0.0)),
            float(row.get("return_kurtosis", 3.0)),
        )
    triple_mask = output.get("target_mode", pd.Series("", index=output.index)) == "triple_barrier"
    if triple_mask.any():
        invalid_columns = [
            "cumulative_return",
            "gross_cumulative_return",
            "sharpe",
            "sortino",
            "max_drawdown",
            "profit_factor",
        ]
        for column in invalid_columns:
            if column in output.columns:
                output.loc[triple_mask, column] = float("nan")
        append_stats_validity(output, triple_mask, "overlap_compounding_invalid")
        append_stats_validity(output, triple_mask, "uniqueness_weighted")
    cpcv_mask = output["split_mode"] == "cpcv"
    if cpcv_mask.any():
        invalid_columns = [
            "cumulative_return",
            "gross_cumulative_return",
            "binom_pvalue_0_5",
            "binom_pvalue_majority",
            "wilson_low",
            "wilson_high",
            "pvalue_fdr_bh",
            "pvalue_fdr_bh_majority",
            "deflated_sharpe",
            "sharpe",
            "sortino",
            "max_drawdown",
            "profit_factor",
        ]
        for column in invalid_columns:
            if column in output.columns:
                output.loc[cpcv_mask, column] = float("nan")
        output.loc[cpcv_mask, "significant_after_fdr"] = False
        output.loc[cpcv_mask, "significant_after_fdr_majority"] = False
        output.loc[cpcv_mask, "beats_majority_after_fdr"] = False
        append_stats_validity(output, cpcv_mask, "cpcv_overlap_invalid")
    return output


def effective_tested_win_rate(output: pd.DataFrame) -> pd.Series:
    """Win rate the binomial p-values were computed on, row by row.

    Triple-barrier rows run inference on the uniqueness-weighted win rate
    (apply_uniqueness_weighted_inference); every other row uses the raw one.
    """
    tested = output["win_rate"].astype(float).copy()
    if "win_rate_weighted" in output.columns and "target_mode" in output.columns:
        weighted_mask = (output["target_mode"] == "triple_barrier") & output["win_rate_weighted"].notna()
        tested.loc[weighted_mask] = output.loc[weighted_mask, "win_rate_weighted"].astype(float)
    return tested


def append_stats_validity(output: pd.DataFrame, mask: pd.Series, reason: str) -> None:
    for idx in output.index[mask]:
        current = str(output.at[idx, "stats_validity"])
        if current == "valid" or current == "" or current == "nan":
            output.at[idx, "stats_validity"] = reason
        elif reason not in current.split(";"):
            output.at[idx, "stats_validity"] = f"{current};{reason}"


def overfitting_gap(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    metric_columns = ["win_rate", "cumulative_return", "sharpe"]
    train = summary[summary["sample_type"] == "train"].set_index(["ticker", "timeframe", "strategy"])
    test = summary[summary["sample_type"] == "test"].set_index(["ticker", "timeframe", "strategy"])
    joined = (
        train[metric_columns].join(test[metric_columns], lsuffix="_train", rsuffix="_test", how="inner").reset_index()
    )
    for metric in metric_columns:
        joined[f"{metric}_gap"] = joined[f"{metric}_train"] - joined[f"{metric}_test"]
    return joined


def write_performance_reports(
    *,
    predictions: pd.DataFrame,
    summary: pd.DataFrame,
    skipped: pd.DataFrame,
    output_dir: str | Path,
    intraday_cost_bps: list[float] | None = None,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    paths["predictions"] = output / "predictions.csv"
    predictions.to_csv(paths["predictions"], index=False)

    paths["performance_summary"] = output / "performance_summary.csv"
    summary.to_csv(paths["performance_summary"], index=False)

    gap = overfitting_gap(summary)
    paths["overfitting_gap"] = output / "overfitting_gap.csv"
    gap.to_csv(paths["overfitting_gap"], index=False)

    paths["significance_summary"] = output / "significance_summary.csv"
    significance_summary(summary).to_csv(paths["significance_summary"], index=False)

    paths["breakeven_survival"] = output / "breakeven_survival.csv"
    breakeven_survival(summary).to_csv(paths["breakeven_survival"], index=False)

    if is_cpcv_predictions(predictions):
        paths["cpcv_fold_summary"] = output / "cpcv_fold_summary.csv"
        cpcv_fold_summary(predictions).to_csv(paths["cpcv_fold_summary"], index=False)
        paths["cpcv_bar_summary"] = output / "cpcv_bar_summary.csv"
        cpcv_bar_summary(predictions).to_csv(paths["cpcv_bar_summary"], index=False)

    triple_predictions = is_triple_barrier_predictions(predictions)
    if triple_predictions and not is_cpcv_predictions(predictions):
        sequential_trades_frame = sequential_trades(predictions)
        paths["sequential_trades"] = output / "sequential_trades.csv"
        sequential_trades_frame.to_csv(paths["sequential_trades"], index=False)
        paths["sequential_summary"] = output / "sequential_summary.csv"
        sequential_summary(sequential_trades_frame).to_csv(paths["sequential_summary"], index=False)

    labels = label_distribution(predictions)
    if not labels.empty:
        paths["label_distribution"] = output / "label_distribution.csv"
        labels.to_csv(paths["label_distribution"], index=False)

    cross_portfolio = cross_sectional_portfolio(predictions)
    if not cross_portfolio.empty:
        paths["cross_sectional_portfolio"] = output / "cross_sectional_portfolio.csv"
        cross_portfolio.to_csv(paths["cross_sectional_portfolio"], index=False)

    cross_b1 = cross_sectional_b1_comparison(summary)
    if not cross_b1.empty:
        paths["cross_sectional_b1_comparison"] = output / "cross_sectional_b1_comparison.csv"
        cross_b1.to_csv(paths["cross_sectional_b1_comparison"], index=False)

    paths["skipped"] = output / "skipped_strategies.csv"
    skipped.to_csv(paths["skipped"], index=False)

    matrix = strategy_matrix(summary)
    paths["win_rate_matrix"] = output / "win_rate_matrix.csv"
    matrix.to_csv(paths["win_rate_matrix"])

    paths["limitations"] = output / "limitations.md"
    paths["limitations"].write_text(
        limitations_text(
            cpcv=is_cpcv_predictions(predictions),
            triple_barrier=triple_predictions,
            cpcv_triple_barrier=triple_predictions and is_cpcv_predictions(predictions),
        ),
        encoding="utf-8",
    )

    heatmap_path = output / "win_rate_heatmap.png"
    if write_heatmap(matrix, heatmap_path):
        paths["heatmap"] = heatmap_path

    threshold_frame = threshold_sensitivity(predictions)
    if not threshold_frame.empty:
        paths["threshold_sensitivity"] = output / "threshold_sensitivity.csv"
        threshold_frame.to_csv(paths["threshold_sensitivity"], index=False)

    nested_selections = nested_threshold_selections(predictions)
    if not nested_selections.empty:
        paths["nested_threshold_selections"] = output / "nested_threshold_selections.csv"
        nested_selections.to_csv(paths["nested_threshold_selections"], index=False)

    intraday = (
        predictions[predictions["timeframe"].isin(["1m", "5m", "15m"])] if not predictions.empty else pd.DataFrame()
    )
    if not intraday.empty:
        paths["intraday_cost_sensitivity"] = output / "intraday_cost_sensitivity.csv"
        intraday_cost_sensitivity(intraday, intraday_cost_bps or [0, 1, 2, 5]).to_csv(
            paths["intraday_cost_sensitivity"], index=False
        )

        paths["intraday_daily_pnl"] = output / "intraday_daily_pnl.csv"
        daily_rows = []
        for (ticker, timeframe, strategy), group in intraday[intraday["sample_type"] == "test"].groupby(
            ["ticker", "timeframe", "strategy"]
        ):
            pnl = daily_pnl(group)
            for session, value in pnl.items():
                daily_rows.append(
                    {
                        "ticker": ticker,
                        "timeframe": timeframe,
                        "strategy": strategy,
                        "session": session,
                        "daily_pnl": value,
                    }
                )
        daily_frame = pd.DataFrame(daily_rows)
        daily_frame.to_csv(paths["intraday_daily_pnl"], index=False)

        paths["intraday_worst_days"] = output / "intraday_worst_days.csv"
        if not daily_frame.empty:
            daily_frame.sort_values("daily_pnl").groupby(["ticker", "timeframe", "strategy"]).head(5).to_csv(
                paths["intraday_worst_days"], index=False
            )
            paths["intraday_daily_significance"] = output / "intraday_daily_significance.csv"
            significance_rows = []
            for (ticker, timeframe, strategy), group in daily_frame.groupby(["ticker", "timeframe", "strategy"]):
                significance_rows.append(
                    {
                        "ticker": ticker,
                        "timeframe": timeframe,
                        "strategy": strategy,
                        "days": int(len(group)),
                        "mean_daily_pnl": float(group["daily_pnl"].mean()),
                        "block_bootstrap_pvalue": block_bootstrap_mean_pvalue(group["daily_pnl"]),
                    }
                )
            pd.DataFrame(significance_rows).to_csv(paths["intraday_daily_significance"], index=False)
            histogram_path = output / "intraday_daily_pnl_histogram.png"
            if write_daily_pnl_histogram(daily_frame, histogram_path):
                paths["intraday_daily_pnl_histogram"] = histogram_path
        else:
            pd.DataFrame().to_csv(paths["intraday_worst_days"], index=False)

    return paths


def is_cpcv_predictions(predictions: pd.DataFrame) -> bool:
    return "split_mode" in predictions.columns and (predictions["split_mode"] == "cpcv").any()


def is_triple_barrier_predictions(predictions: pd.DataFrame) -> bool:
    return (
        "target_event" in predictions.columns
        and "holding_bars" in predictions.columns
        and predictions["target_event"].notna().any()
    )


def sequential_trades(predictions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "ticker",
        "timeframe",
        "strategy",
        "sample_type",
        "signal_date",
        "entry_date",
        "exit_date",
        "signal",
        "holding_bars",
        "net_return",
        "target_event",
        "skipped_signals_before_entry",
        "group_skipped_signals",
    ]
    if predictions.empty or not is_triple_barrier_predictions(predictions):
        return pd.DataFrame(columns=columns)
    rows = []
    candidate = predictions[predictions["target_event"].notna() & predictions["holding_bars"].notna()].copy()
    for (ticker, timeframe, strategy, sample_type), group in candidate.groupby(
        ["ticker", "timeframe", "strategy", "sample_type"], dropna=False
    ):
        ordered = group.sort_values(["date", "fold"]).drop_duplicates(["date"], keep="first").reset_index(drop=True)
        if ordered.empty:
            continue
        dates = pd.to_datetime(ordered["date"])
        next_available_pos = 0
        skipped_since_last_trade = 0
        skipped_signals = 0
        group_rows = []
        for position, row in ordered.iterrows():
            signal = int(row["signal"])
            if signal == 0:
                continue
            if position < next_available_pos:
                skipped_signals += 1
                skipped_since_last_trade += 1
                continue
            holding_bars = int(row["holding_bars"])
            entry_lag = int(row.get("entry_lag_bars", 1))
            entry_pos = min(len(ordered) - 1, position + entry_lag)
            exit_pos = min(len(ordered) - 1, entry_pos + holding_bars - 1)
            group_rows.append(
                {
                    "ticker": ticker,
                    "timeframe": timeframe,
                    "strategy": strategy,
                    "sample_type": sample_type,
                    "signal_date": dates.iloc[position],
                    "entry_date": dates.iloc[entry_pos],
                    "exit_date": dates.iloc[exit_pos],
                    "signal": signal,
                    "holding_bars": holding_bars,
                    "net_return": float(row["net_return"]),
                    "target_event": row["target_event"],
                    "skipped_signals_before_entry": int(skipped_since_last_trade),
                    "group_skipped_signals": 0,
                }
            )
            skipped_since_last_trade = 0
            next_available_pos = exit_pos + 1
        for trade_row in group_rows:
            trade_row["group_skipped_signals"] = int(skipped_signals)
        rows.extend(group_rows)
    return pd.DataFrame(rows, columns=columns)


def sequential_summary(trades: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "ticker",
        "timeframe",
        "strategy",
        "sample_type",
        "trades",
        "wins",
        "win_rate",
        "sequential_cumulative_return",
        "max_drawdown",
        "skipped_signals",
    ]
    if trades.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for (ticker, timeframe, strategy, sample_type), group in trades.groupby(
        ["ticker", "timeframe", "strategy", "sample_type"], dropna=False
    ):
        returns = group["net_return"].fillna(0.0)
        trades_count = int(len(group))
        wins = int((returns > 0).sum())
        rows.append(
            {
                "ticker": ticker,
                "timeframe": timeframe,
                "strategy": strategy,
                "sample_type": sample_type,
                "trades": trades_count,
                "wins": wins,
                "win_rate": float(wins / trades_count) if trades_count else float("nan"),
                "sequential_cumulative_return": float((1.0 + returns).prod() - 1.0),
                "max_drawdown": max_drawdown(returns),
                "skipped_signals": int(group["group_skipped_signals"].max())
                if "group_skipped_signals" in group.columns
                else int(group["skipped_signals_before_entry"].sum()),
            }
        )
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["sample_type", "timeframe", "ticker", "strategy"])
        .reset_index(drop=True)
    )


def cpcv_fold_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    test = predictions[(predictions["sample_type"] == "test") & (predictions.get("split_mode") == "cpcv")].copy()
    if test.empty:
        return pd.DataFrame()
    rows = []
    for (ticker, timeframe, strategy, fold), group in test.groupby(
        ["ticker", "timeframe", "strategy", "fold"], dropna=False
    ):
        stats = win_rate_stats(group)
        rows.append(
            {
                "ticker": ticker,
                "timeframe": timeframe,
                "strategy": strategy,
                "fold": int(fold),
                "fold_predictions": int(stats["predictions"]),
                "fold_win_rate": stats["win_rate"],
                "fold_net_return_sum": float(group["net_return"].fillna(0.0).sum()),
            }
        )
    fold_frame = pd.DataFrame(rows)
    aggregate = (
        fold_frame.groupby(["ticker", "timeframe", "strategy"], dropna=False)
        .agg(
            fold_win_rate_mean=("fold_win_rate", "mean"),
            fold_win_rate_std=("fold_win_rate", "std"),
            fold_win_rate_gt_0_5_ratio=("fold_win_rate", lambda values: float((values > 0.5).mean())),
        )
        .reset_index()
    )
    return fold_frame.merge(aggregate, on=["ticker", "timeframe", "strategy"], how="left")


def cpcv_bar_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    test = predictions[(predictions["sample_type"] == "test") & (predictions.get("split_mode") == "cpcv")].copy()
    if test.empty:
        return pd.DataFrame()
    bar_rows = []
    for (ticker, timeframe, strategy, date), group in test.groupby(
        ["ticker", "timeframe", "strategy", "date"], dropna=False
    ):
        active = group[group["active"]]
        bar_rows.append(
            {
                "ticker": ticker,
                "timeframe": timeframe,
                "strategy": strategy,
                "date": date,
                "fold_prediction_rows": int(len(group)),
                "active_fold_prediction_rows": int(len(active)),
                "bar_hit_rate": float(active["hit"].mean()) if not active.empty else float("nan"),
                "bar_active_rate": float(group["active"].mean()),
                "bar_net_return_mean": float(group["net_return"].fillna(0.0).mean()),
            }
        )
    bars = pd.DataFrame(bar_rows)
    rows = []
    for (ticker, timeframe, strategy), group in bars.groupby(["ticker", "timeframe", "strategy"], dropna=False):
        active_bars = group[group["active_fold_prediction_rows"] > 0]
        rows.append(
            {
                "ticker": ticker,
                "timeframe": timeframe,
                "strategy": strategy,
                "unique_test_bars": int(len(group)),
                "active_test_bars": int(len(active_bars)),
                "fold_prediction_rows": int(group["fold_prediction_rows"].sum()),
                "mean_fold_predictions_per_bar": float(group["fold_prediction_rows"].mean()),
                "bar_mean_hit_rate": float(active_bars["bar_hit_rate"].mean())
                if not active_bars.empty
                else float("nan"),
                "bar_net_return_sum_mean": float(group["bar_net_return_mean"].sum()),
            }
        )
    return pd.DataFrame(rows)


def significance_summary(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "strategy",
        "test_rows",
        "raw_p_lt_0_05",
        "fdr_pass_count",
        "raw_p_majority_lt_0_05",
        "fdr_pass_majority_count",
        "beats_majority_after_fdr_count",
    ]
    if summary.empty:
        return pd.DataFrame(columns=columns)
    test_rows = summary[summary["sample_type"] == "test"].copy()
    if test_rows.empty:
        return pd.DataFrame(columns=columns)
    grouped = test_rows.groupby("strategy", dropna=False)
    rows = []
    for strategy, group in grouped:
        rows.append(
            {
                "strategy": strategy,
                "test_rows": int(len(group)),
                "raw_p_lt_0_05": int((group["binom_pvalue_0_5"] < 0.05).sum()),
                "fdr_pass_count": int(group["significant_after_fdr"].sum()),
                "raw_p_majority_lt_0_05": int((group["binom_pvalue_majority"] < 0.05).sum()),
                "fdr_pass_majority_count": int(group["significant_after_fdr_majority"].sum()),
                "beats_majority_after_fdr_count": int(group["beats_majority_after_fdr"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["fdr_pass_count", "raw_p_lt_0_05", "strategy"], ascending=[False, False, True]
    )


def breakeven_survival(summary: pd.DataFrame, costs_bps: list[float] | None = None) -> pd.DataFrame:
    columns = [
        "ticker",
        "timeframe",
        "sample_type",
        "cost_bps",
        "surviving_strategies",
        "total_strategies",
        "survival_rate",
    ]
    if summary.empty or "breakeven_cost_bps" not in summary.columns:
        return pd.DataFrame(columns=columns)
    costs_bps = costs_bps or [0, 1, 2, 5, 10]
    rows = []
    for (ticker, timeframe, sample_type), group in summary.groupby(
        ["ticker", "timeframe", "sample_type"], dropna=False
    ):
        breakeven = group["breakeven_cost_bps"]
        total = int(breakeven.notna().sum())
        for cost in costs_bps:
            surviving = int((breakeven >= float(cost)).sum())
            rows.append(
                {
                    "ticker": ticker,
                    "timeframe": timeframe,
                    "sample_type": sample_type,
                    "cost_bps": float(cost),
                    "surviving_strategies": surviving,
                    "total_strategies": total,
                    "survival_rate": float(surviving / total) if total else float("nan"),
                }
            )
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["sample_type", "timeframe", "ticker", "cost_bps"])
        .reset_index(drop=True)
    )


def threshold_sensitivity(predictions: pd.DataFrame, thresholds: list[float] | None = None) -> pd.DataFrame:
    columns = [
        "threshold",
        "ticker",
        "timeframe",
        "strategy",
        "n_trades",
        "win_rate",
        "wilson_low",
        "wilson_high",
        "net_cumulative_return",
        "sharpe",
    ]
    if predictions.empty or "probability_up" not in predictions.columns:
        return pd.DataFrame(columns=columns)
    if is_triple_barrier_predictions(predictions) or is_cpcv_predictions(predictions):
        return pd.DataFrame(columns=columns)
    thresholds = thresholds or [0.50, 0.52, 0.55, 0.58, 0.60]
    candidate = predictions[(predictions["sample_type"] == "test") & predictions["probability_up"].notna()].copy()
    if candidate.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for threshold in thresholds:
        for (ticker, timeframe, strategy), group in candidate.groupby(["ticker", "timeframe", "strategy"]):
            signals = pd.Series(0, index=group.index, dtype=int)
            signals[group["probability_up"] >= threshold] = 1
            signals[group["probability_up"] <= 1.0 - threshold] = -1
            adjusted = group.copy()
            adjusted["signal"] = signals
            adjusted["active"] = adjusted["signal"] != 0
            adjusted["hit"] = (
                (adjusted["signal"] == adjusted["target"].astype(int))
                & adjusted["active"]
                & (adjusted["target"].astype(int) != 0)
            )
            cost = adjusted["round_trip_cost_bps"].fillna(0.0) / 10_000.0 if "round_trip_cost_bps" in adjusted else 0.0
            adjusted["gross_return"] = adjusted["signal"] * adjusted["execution_return"]
            adjusted["net_return"] = adjusted["gross_return"] - adjusted["signal"].abs() * cost
            stats = win_rate_stats(adjusted)
            returns = return_stats(adjusted, timeframe=timeframe)
            rows.append(
                {
                    "threshold": float(threshold),
                    "ticker": ticker,
                    "timeframe": timeframe,
                    "strategy": strategy,
                    "n_trades": int(stats["predictions"]),
                    "win_rate": stats["win_rate"],
                    "wilson_low": stats["wilson_low"],
                    "wilson_high": stats["wilson_high"],
                    "net_cumulative_return": returns["cumulative_return"],
                    "sharpe": returns["sharpe"],
                }
            )
    return pd.DataFrame(rows, columns=columns)


def nested_threshold_selections(predictions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "ticker",
        "timeframe",
        "strategy",
        "fold",
        "selected_threshold",
        "train_objective",
        "test_rows",
    ]
    if predictions.empty or "selected_threshold" not in predictions.columns:
        return pd.DataFrame(columns=columns)
    candidate = predictions[(predictions["sample_type"] == "test") & predictions["selected_threshold"].notna()].copy()
    if candidate.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for (ticker, timeframe, strategy, fold), group in candidate.groupby(
        ["ticker", "timeframe", "strategy", "fold"], dropna=False
    ):
        rows.append(
            {
                "ticker": ticker,
                "timeframe": timeframe,
                "strategy": strategy,
                "fold": int(fold),
                "selected_threshold": float(group["selected_threshold"].iloc[0]),
                "train_objective": float(group["threshold_train_objective"].iloc[0]),
                "test_rows": int(len(group)),
            }
        )
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["ticker", "timeframe", "strategy", "fold"])
        .reset_index(drop=True)
    )


def label_distribution(predictions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "ticker",
        "timeframe",
        "target_mode",
        "rows",
        "target_-1_count",
        "target_0_count",
        "target_1_count",
        "target_-1_ratio",
        "target_0_ratio",
        "target_1_ratio",
        "tp_count",
        "sl_count",
        "expiry_count",
        "tp_ratio",
        "sl_ratio",
        "expiry_ratio",
        "avg_holding_bars",
    ]
    if predictions.empty or "target_event" not in predictions.columns or "holding_bars" not in predictions.columns:
        return pd.DataFrame(columns=columns)
    unique = predictions.drop_duplicates(["ticker", "timeframe", "date"]).copy()
    unique = unique.dropna(subset=["target", "target_event", "holding_bars"])
    if unique.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for (ticker, timeframe), group in unique.groupby(["ticker", "timeframe"], dropna=False):
        total = len(group)
        events = group["target_event"].astype(str)
        target_counts = group["target"].astype(int).value_counts()
        tp_count = int((events == "tp").sum())
        sl_count = int((events == "sl").sum())
        expiry_count = int(events.str.startswith("expiry").sum())
        rows.append(
            {
                "ticker": ticker,
                "timeframe": timeframe,
                "target_mode": "triple_barrier",
                "rows": int(total),
                "target_-1_count": int(target_counts.get(-1, 0)),
                "target_0_count": int(target_counts.get(0, 0)),
                "target_1_count": int(target_counts.get(1, 0)),
                "target_-1_ratio": float(target_counts.get(-1, 0) / total),
                "target_0_ratio": float(target_counts.get(0, 0) / total),
                "target_1_ratio": float(target_counts.get(1, 0) / total),
                "tp_count": tp_count,
                "sl_count": sl_count,
                "expiry_count": expiry_count,
                "tp_ratio": float(tp_count / total),
                "sl_ratio": float(sl_count / total),
                "expiry_ratio": float(expiry_count / total),
                "avg_holding_bars": float(group["holding_bars"].mean()),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(["ticker", "timeframe"]).reset_index(drop=True)


def cross_sectional_portfolio(predictions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "timeframe",
        "strategy",
        "sample_type",
        "bars",
        "avg_active_positions",
        "cumulative_return",
        "mean_bar_return",
    ]
    if predictions.empty or "strategy" not in predictions.columns:
        return pd.DataFrame(columns=columns)
    candidate = predictions[predictions["strategy"].astype(str).str.startswith("CS")].copy()
    if candidate.empty:
        return pd.DataFrame(columns=columns)
    candidate = candidate.drop_duplicates(["ticker", "timeframe", "strategy", "sample_type", "date"])
    rows = []
    for (timeframe, strategy, sample_type), group in candidate.groupby(
        ["timeframe", "strategy", "sample_type"], dropna=False
    ):
        bar_returns = []
        active_counts = []
        for _, bar in group.groupby("date", dropna=False):
            active = bar[bar["active"]]
            active_counts.append(int(len(active)))
            bar_returns.append(float(active["net_return"].mean()) if not active.empty else 0.0)
        returns = pd.Series(bar_returns)
        rows.append(
            {
                "timeframe": timeframe,
                "strategy": strategy,
                "sample_type": sample_type,
                "bars": int(len(returns)),
                "avg_active_positions": float(pd.Series(active_counts).mean()) if active_counts else float("nan"),
                "cumulative_return": float((1.0 + returns.fillna(0.0)).prod() - 1.0),
                "mean_bar_return": float(returns.mean()) if not returns.empty else float("nan"),
            }
        )
    return (
        pd.DataFrame(rows, columns=columns).sort_values(["sample_type", "timeframe", "strategy"]).reset_index(drop=True)
    )


def cross_sectional_b1_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "ticker",
        "timeframe",
        "sample_type",
        "strategy",
        "cs_cumulative_return",
        "b1_cumulative_return",
        "cumulative_return_delta",
        "cs_win_rate",
        "b1_win_rate",
        "win_rate_delta",
    ]
    if summary.empty or not (summary["strategy"].astype(str).str.startswith("CS")).any():
        return pd.DataFrame(columns=columns)
    cs = summary[summary["strategy"].astype(str).str.startswith("CS")].copy()
    b1 = summary[summary["strategy"] == "B1_always_up"].copy()
    if cs.empty or b1.empty:
        return pd.DataFrame(columns=columns)
    joined = cs.merge(
        b1[["ticker", "timeframe", "sample_type", "cumulative_return", "win_rate"]],
        on=["ticker", "timeframe", "sample_type"],
        how="left",
        suffixes=("", "_b1"),
    )
    rows = []
    for _, row in joined.iterrows():
        rows.append(
            {
                "ticker": row["ticker"],
                "timeframe": row["timeframe"],
                "sample_type": row["sample_type"],
                "strategy": row["strategy"],
                "cs_cumulative_return": float(row["cumulative_return"]),
                "b1_cumulative_return": float(row["cumulative_return_b1"]),
                "cumulative_return_delta": float(row["cumulative_return"] - row["cumulative_return_b1"]),
                "cs_win_rate": float(row["win_rate"]),
                "b1_win_rate": float(row["win_rate_b1"]),
                "win_rate_delta": float(row["win_rate"] - row["win_rate_b1"]),
            }
        )
    return (
        pd.DataFrame(rows, columns=columns).sort_values(["sample_type", "timeframe", "ticker"]).reset_index(drop=True)
    )


def strategy_matrix(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    oos = summary[summary["sample_type"] == "test"].copy()
    oos["ticker_timeframe"] = oos["ticker"] + "_" + oos["timeframe"]
    return oos.pivot_table(index="ticker_timeframe", columns="strategy", values="win_rate", aggfunc="mean")


def write_heatmap(matrix: pd.DataFrame, path: Path) -> bool:
    if matrix.empty:
        return False
    os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".matplotlib"))
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        LOGGER.warning("matplotlib is not installed; skipping heatmap")
        return False

    fig_width = max(8, min(24, 0.55 * len(matrix.columns) + 4))
    fig_height = max(6, min(30, 0.28 * len(matrix.index) + 3))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(matrix.fillna(float("nan")).to_numpy(), aspect="auto", vmin=0.35, vmax=0.65, cmap="RdYlGn")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_title("Out-of-sample Win Rate")
    fig.colorbar(image, ax=ax, label="Win rate")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def intraday_cost_sensitivity(predictions: pd.DataFrame, costs_bps: list[float]) -> pd.DataFrame:
    rows = []
    oos = predictions[predictions["sample_type"] == "test"].copy()
    for cost in costs_bps:
        adjusted = oos.copy()
        adjusted["net_return"] = adjusted["gross_return"] - adjusted["active"].astype(float) * (float(cost) / 10_000.0)
        for (ticker, timeframe, strategy), group in adjusted.groupby(["ticker", "timeframe", "strategy"]):
            row = {"ticker": ticker, "timeframe": timeframe, "strategy": strategy, "round_trip_cost_bps": float(cost)}
            row.update(win_rate_stats(group))
            row.update(return_stats(group, timeframe=timeframe))
            row.update(trade_frequency_stats(group, timeframe=timeframe))
            row["breakeven_cost_bps"] = breakeven_cost_bps(group)
            rows.append(row)
    return pd.DataFrame(rows)


def write_daily_pnl_histogram(daily_frame: pd.DataFrame, path: Path) -> bool:
    os.environ.setdefault("MPLCONFIGDIR", str(path.parent / ".matplotlib"))
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        LOGGER.warning("matplotlib is not installed; skipping intraday P&L histogram")
        return False
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(daily_frame["daily_pnl"].dropna(), bins=50, color="#4c78a8", edgecolor="white")
    ax.set_title("Intraday Daily P&L Distribution")
    ax.set_xlabel("Daily net return")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def limitations_text(*, cpcv: bool = False, triple_barrier: bool = False, cpcv_triple_barrier: bool = False) -> str:
    text = """# Required Research Limitations

- Survivorship bias: the universe uses current mega-cap stocks, so historical backtests contain survivorship bias.
- yfinance intraday constraints: 1h bars cover roughly the most recent 730 days only, so direct comparison with daily/weekly windows requires caution.
- yfinance minute constraints: 1m data is very short-lived and 5m/15m data is limited, so intraday sample-size warnings should be interpreted seriously.
- Win rate alone cannot prove strategy superiority because payoff asymmetry, transaction costs, and class imbalance can dominate hit ratio.
- Bar-based fill assumptions are optimistic versus tick/quote data, especially for intraday scalping.
- This project is for research only and does not implement broker API integration, live streaming, order execution, or investment advice.
"""
    if cpcv:
        text += "\n## CPCV-Specific Warning\n\n"
        text += (
            "- CPCV test folds intentionally overlap at the bar level. In `performance_summary.csv`, "
            "overlap-sensitive statistics are set to NaN and marked with `stats_validity=cpcv_overlap_invalid`. "
            "Use `cpcv_fold_summary.csv` and `cpcv_bar_summary.csv` for CPCV interpretation.\n"
        )
    if triple_barrier:
        text += "\n## Triple-Barrier Label Overlap Warning\n\n"
        text += (
            "- Triple-barrier labels can remain open for multiple bars, so row-by-row compounded returns, Sharpe, "
            "Sortino, drawdown, and profit factor in `performance_summary.csv` are set to NaN and marked with "
            "`stats_validity=overlap_compounding_invalid`. Use `sequential_summary.csv` for single-position "
            "walk-forward return interpretation when it is generated.\n"
        )
        text += (
            "- Wilson intervals and binomial p-values for triple-barrier rows use label-uniqueness-adjusted "
            "effective sample size and are marked with `stats_validity=uniqueness_weighted`.\n"
        )
    if cpcv_triple_barrier:
        text += (
            "- `sequential_trades.csv` and `sequential_summary.csv` are not generated for triple-barrier CPCV runs "
            "because CPCV test folds overlap at the same bar timestamps.\n"
        )
    return text
