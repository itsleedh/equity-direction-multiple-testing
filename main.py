from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.engine import BacktestEngine
from core.config import load_config, resolve_path
from core.universe import load_pit_universe_from_config
from data.loader import DataProviderError, MarketDataLoader, TIMEFRAME_SPECS
from data.macro_loader import MacroDataError, load_external_feature_set
from features.intraday import build_intraday_feature_frame, mark_intraday_entry_rules
from features.pipeline import feature_columns
from features.targets import add_target_and_execution, build_model_frame
from reports.data_summary import build_data_summary, write_data_summary
from reports.performance import summarize_predictions, write_performance_reports
from strategies.baselines import AlwaysUpBaseline, BuyAndHoldBaseline
from strategies.cross_sectional import CrossSectionalMomentumRankStrategy, CrossSectionalMomentumRanker
from strategies.factory import build_strategy_suite
from strategies.meta import validate_meta_labeling_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Equity signal falsification research CLI")
    parser.add_argument(
        "--config", default="configs/config.yaml", help="Path to config YAML (default: configs/config.yaml)"
    )
    parser.add_argument("--offline", action="store_true", help="Use only local parquet cache")
    parser.add_argument("--no-refresh", action="store_true", help="Do not refresh provider data")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate config and print planned data loads without network access"
    )
    parser.add_argument("--data-only", action="store_true", help="Only load/cache data and write data_summary.csv")
    parser.add_argument("--no-ml", action="store_true", help="Skip ML strategies even when scikit-learn is installed")
    parser.add_argument(
        "--intraday", action="store_true", help="Run intraday scalping module instead of swing timeframes"
    )
    parser.add_argument("--cross-sectional", action="store_true", help="Run cross-sectional ranking module")
    parser.add_argument("--output-dir", help="Override reports.output_dir without editing the config")
    parser.add_argument("--tickers", nargs="*", help="Optional ticker override")
    parser.add_argument("--timeframes", nargs="*", help="Optional timeframe override")
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    config_path = Path(args.config)
    config = load_config(config_path)
    validate_meta_labeling_config(config)
    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)

    intraday_enabled = bool(args.intraday or config.get("intraday_scalping", {}).get("enabled", False))
    cross_sectional_enabled = bool(args.cross_sectional or config.get("cross_sectional", {}).get("enabled", False))
    pit_universe, pit_mode = load_pit_universe_from_config(config, config_path)
    tickers = args.tickers or config.get("universe", [])
    if pit_universe is not None and not tickers:
        tickers = pit_universe.union_tickers()
    elif pit_universe is not None and set(tickers) != set(pit_universe.union_tickers()):
        logging.warning(
            "Ticker list differs from PIT membership union; non-member predictions are masked/annotated only."
        )
    timeframes = args.timeframes or (
        config.get("intraday_scalping", {}).get("timeframes", []) if intraday_enabled else config.get("timeframes", [])
    )
    validate_timeframes(timeframes)

    data_config = dict(config.get("data", {}))
    cache_dir = resolve_path(config_path, data_config.get("cache_dir", "data/cache"))
    loader = MarketDataLoader.from_config(
        config, cache_dir=cache_dir, offline=args.offline or data_config.get("offline", False)
    )

    if args.dry_run:
        print("Dry run: no provider downloads will be attempted.")
        print(f"Cache directory: {cache_dir}")
        for ticker in tickers:
            for timeframe in timeframes:
                cache_path = loader.cache_path(ticker, timeframe)
                status = "cached" if cache_path.exists() else "missing"
                print(f"{ticker:>6} {timeframe:<3} {status:<7} {cache_path}")
        return 0

    try:
        data = loader.load_universe(tickers, timeframes, refresh=not args.no_refresh)
    except DataProviderError as exc:
        logging.error("%s", exc)
        return 2
    data = apply_start_date(data, data_config.get("start_date"))

    try:
        external_features = load_external_feature_set(
            config,
            config_path,
            offline=args.offline or data_config.get("offline", False),
        )
    except MacroDataError as exc:
        logging.error("%s", exc)
        return 2

    summary = build_data_summary(data)
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()  # CLI 경로는 실행 위치 기준
    else:
        output_dir = resolve_path(config_path, config.get("reports", {}).get("output_dir", "reports/output"))
    summary_path = write_data_summary(summary, output_dir)

    print(summary.to_string(index=False))
    print(f"\nWrote data summary: {summary_path}")
    if args.data_only:
        print("Data-only mode complete.")
        return 0

    if cross_sectional_enabled:
        predictions, skipped = run_cross_sectional_pipeline(data=data, config=config)
    else:
        predictions, skipped = run_research_pipeline(
            data=data,
            config=config,
            include_ml=(not args.no_ml) and bool(config.get("ml", {}).get("enabled", True)),
            intraday_enabled=intraday_enabled,
            external_features=external_features,
        )
    pit_coverage = pd.DataFrame()
    if pit_universe is not None and not predictions.empty:
        predictions = pit_universe.annotate_predictions(predictions)
        pit_coverage = pit_universe.coverage_summary(predictions)
        if pit_mode == "evaluation_mask":
            predictions = pit_universe.filter_predictions(predictions)
    performance_summary = summarize_predictions(predictions)
    report_paths = write_performance_reports(
        predictions=predictions,
        summary=performance_summary,
        skipped=pd.DataFrame(skipped).drop_duplicates()
        if skipped
        else pd.DataFrame(columns=["ticker", "timeframe", "strategy", "reason"]),
        output_dir=output_dir,
        intraday_cost_bps=[
            float(value) for value in config.get("intraday_scalping", {}).get("cost_sensitivity_bps", [0, 1, 2, 5])
        ],
    )
    if pit_universe is not None:
        coverage_path = Path(output_dir) / "pit_membership_coverage.csv"
        pit_coverage.to_csv(coverage_path, index=False)
        report_paths["pit_membership_coverage"] = coverage_path

    if performance_summary.empty:
        print("\nNo performance rows were produced. Check skipped_strategies.csv for the reason.")
    else:
        oos = performance_summary[performance_summary["sample_type"] == "test"]
        columns = [
            "ticker",
            "timeframe",
            "strategy",
            "predictions",
            "win_rate",
            "wilson_low",
            "wilson_high",
            "cumulative_return",
        ]
        print("\nOut-of-sample performance preview:")
        print(oos.loc[:, columns].head(30).to_string(index=False))
    print("\nWrote reports:")
    for name, path in report_paths.items():
        print(f"- {name}: {path}")
    print("Notes:")
    print("- 1h yfinance bars are limited to roughly the most recent 730 days.")
    print("- yfinance 1m/5m/15m intraday bars have short sample windows; report limitations include this.")
    print("- Missing OHLC bars are dropped; volume gaps are filled with 0; prices are not forward-filled.")
    print("- 1wk bars are resampled from adjusted daily data.")
    return 0


def apply_start_date(
    data: dict[tuple[str, str], pd.DataFrame], start_date: str | None
) -> dict[tuple[str, str], pd.DataFrame]:
    """Optionally restrict loaded bars to start_date and later (config `data.start_date`).

    Slicing happens after the cache read so read-only caches keep their full
    history; experiments whose external series begin late (e.g. FINRA short
    interest, 2017-12+) use this to give the on/off pair one common window.
    """
    if not start_date:
        return data
    cutoff = pd.Timestamp(str(start_date))
    sliced: dict[tuple[str, str], pd.DataFrame] = {}
    for key, bars in data.items():
        index = pd.DatetimeIndex(bars.index)
        start = cutoff.tz_localize(index.tz) if index.tz is not None else cutoff
        sliced[key] = bars.loc[index >= start]
    return sliced


def validate_timeframes(timeframes: list[str]) -> None:
    unsupported = sorted(set(timeframes) - set(TIMEFRAME_SPECS))
    if unsupported:
        supported = ", ".join(sorted(TIMEFRAME_SPECS))
        raise ValueError(f"Unsupported timeframes {unsupported}. Supported: {supported}")


def run_research_pipeline(
    *,
    data: dict[tuple[str, str], pd.DataFrame],
    config: dict,
    include_ml: bool,
    intraday_enabled: bool,
    external_features=None,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    all_predictions: list[pd.DataFrame] = []
    skipped: list[dict[str, str]] = []

    intraday_timeframes = set(config.get("intraday_scalping", {}).get("timeframes", ["1m", "5m", "15m"]))
    for (ticker, timeframe), bars in data.items():
        is_intraday = intraday_enabled or timeframe in intraday_timeframes
        try:
            model_frame = prepare_frame(
                bars,
                timeframe=timeframe,
                config=config,
                intraday=is_intraday,
                external_features=external_features,
                symbol=ticker,
            )
            columns = feature_columns(model_frame)
            strategies = build_strategy_suite(config, include_ml=include_ml, intraday=is_intraday)
            round_trip_cost_bps = effective_round_trip_cost_bps(config, intraday=is_intraday)
            if is_intraday:
                model_frame = apply_intraday_cost_schedule(
                    model_frame, base_round_trip_cost_bps=round_trip_cost_bps, config=config
                )
            engine = BacktestEngine(
                BacktestEngine.from_config(config).splitter,
                round_trip_cost_bps=round_trip_cost_bps,
            )
            result = engine.run(
                model_frame,
                ticker=ticker,
                timeframe=timeframe,
                strategies=strategies,
                feature_columns=columns,
                config=config,
            )
        except Exception as exc:
            logging.exception("Failed pipeline for %s %s", ticker, timeframe)
            skipped.append({"ticker": ticker, "timeframe": timeframe, "strategy": "*", "reason": str(exc)})
            continue
        skipped.extend(result.skipped)
        if not result.predictions.empty:
            all_predictions.append(result.predictions)

    predictions = pd.concat(all_predictions, axis=0, ignore_index=True) if all_predictions else pd.DataFrame()
    return predictions, skipped


def run_cross_sectional_pipeline(
    *,
    data: dict[tuple[str, str], pd.DataFrame],
    config: dict,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    all_predictions: list[pd.DataFrame] = []
    skipped: list[dict[str, str]] = []
    ranker = CrossSectionalMomentumRanker.from_config(config)

    by_timeframe: dict[str, dict[str, pd.DataFrame]] = {}
    for (ticker, timeframe), bars in data.items():
        try:
            by_timeframe.setdefault(timeframe, {})[ticker] = prepare_frame(
                bars, timeframe=timeframe, config=config, intraday=False
            )
        except Exception as exc:
            logging.exception("Failed cross-sectional frame preparation for %s %s", ticker, timeframe)
            skipped.append(
                {"ticker": ticker, "timeframe": timeframe, "strategy": "CS1_momentum_rank", "reason": str(exc)}
            )

    for timeframe, frames in by_timeframe.items():
        if len(frames) < 2:
            skipped.append(
                {
                    "ticker": "*",
                    "timeframe": timeframe,
                    "strategy": "CS1_momentum_rank",
                    "reason": "cross-sectional ranking requires at least two tickers",
                }
            )
            continue
        signals = ranker.signals(frames)
        for ticker, frame in frames.items():
            model_frame = frame.copy()
            model_frame["_cross_sectional_signal"] = signals[ticker]
            columns = feature_columns(model_frame)
            strategies = [AlwaysUpBaseline(), BuyAndHoldBaseline(), CrossSectionalMomentumRankStrategy()]
            try:
                engine = BacktestEngine.from_config(config)
                result = engine.run(
                    model_frame,
                    ticker=ticker,
                    timeframe=timeframe,
                    strategies=strategies,
                    feature_columns=columns,
                    config=config,
                )
            except Exception as exc:
                logging.exception("Failed cross-sectional backtest for %s %s", ticker, timeframe)
                skipped.append(
                    {"ticker": ticker, "timeframe": timeframe, "strategy": "CS1_momentum_rank", "reason": str(exc)}
                )
                continue
            skipped.extend(result.skipped)
            if not result.predictions.empty:
                all_predictions.append(result.predictions)

    predictions = pd.concat(all_predictions, axis=0, ignore_index=True) if all_predictions else pd.DataFrame()
    return predictions, skipped


def effective_round_trip_cost_bps(config: dict, *, intraday: bool) -> float:
    execution_cost = float(config.get("execution", {}).get("round_trip_cost_bps", 5))
    if not intraday:
        return execution_cost
    intraday_config = config.get("intraday_scalping", {})
    if "spread_cost_bps" in intraday_config or "slippage_bps" in intraday_config:
        return float(intraday_config.get("spread_cost_bps", 0)) + float(intraday_config.get("slippage_bps", 0))
    return execution_cost


def apply_intraday_cost_schedule(frame: pd.DataFrame, *, base_round_trip_cost_bps: float, config: dict) -> pd.DataFrame:
    schedule = config.get("execution", {}).get("cost_schedule", {})
    if not bool(schedule.get("enabled", False)):
        return frame
    rules = schedule.get("rules", [])
    if not rules:
        return frame
    output = frame.copy()
    index = pd.DatetimeIndex(output.index)
    multipliers = pd.Series(1.0, index=output.index)
    for rule in rules:
        start = pd.to_datetime(str(rule.get("start", "00:00"))).time()
        end = pd.to_datetime(str(rule.get("end", "23:59"))).time()
        multiplier = float(rule.get("multiplier", 1.0))
        times = pd.Series(index.time, index=output.index)
        if start <= end:
            mask = (times >= start) & (times < end)
        else:
            mask = (times >= start) | (times < end)
        multipliers.loc[mask] = multiplier
    output["round_trip_cost_bps"] = float(base_round_trip_cost_bps) * multipliers
    return output


def prepare_frame(
    bars: pd.DataFrame,
    *,
    timeframe: str,
    config: dict,
    intraday: bool,
    external_features=None,
    symbol: str | None = None,
) -> pd.DataFrame:
    if not intraday:
        return build_model_frame(
            bars, timeframe=timeframe, config=config, external_features=external_features, symbol=symbol
        )

    intraday_config = config.get("intraday_scalping", {})
    frame = build_intraday_feature_frame(
        bars,
        timezone=str(intraday_config.get("timezone", "America/New_York")),
        opening_range_minutes=int(intraday_config.get("opening_range_minutes", 30)),
    )
    frame = mark_intraday_entry_rules(
        frame,
        no_new_entries_after=str(intraday_config.get("no_new_entries_after", "15:55")),
    )
    return add_target_and_execution(
        frame,
        target_config=config.get("target", {}),
        execution_config=config.get("execution", {}),
    )


if __name__ == "__main__":
    raise SystemExit(main())
