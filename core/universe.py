from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.config import ConfigError, load_config


class PitUniverse:
    """Point-in-time universe membership keyed by application calendar year.

    Membership for calendar year Y must be decided with information available
    before Y starts (e.g., the market-cap ranking as of the last trading day of
    year Y-1), so applying it to year Y introduces no look-ahead.
    """

    def __init__(self, memberships: dict[int, tuple[str, ...]], *, selection: str = "", notes: str = "") -> None:
        if not memberships:
            raise ConfigError("PIT universe requires at least one membership year.")
        self.memberships = {int(year): tuple(tickers) for year, tickers in sorted(memberships.items())}
        self.selection = selection
        self.notes = notes
        self._validate()

    @classmethod
    def from_file(cls, path: str | Path) -> "PitUniverse":
        raw = load_config(path)
        memberships_raw = raw.get("memberships")
        if not isinstance(memberships_raw, dict) or not memberships_raw:
            raise ConfigError(f"PIT universe file {path} must define a non-empty 'memberships' mapping.")
        memberships: dict[int, tuple[str, ...]] = {}
        for year_key, tickers in memberships_raw.items():
            try:
                year = int(year_key)
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"PIT membership year '{year_key}' is not an integer.") from exc
            if not isinstance(tickers, list):
                raise ConfigError(f"PIT membership for year {year} must be a list of tickers.")
            memberships[year] = tuple(str(ticker).upper().strip() for ticker in tickers)
        return cls(
            memberships,
            selection=str(raw.get("selection", "")),
            notes=str(raw.get("notes", "")),
        )

    def _validate(self) -> None:
        years = sorted(self.memberships)
        for previous, current in zip(years, years[1:]):
            if current != previous + 1:
                raise ConfigError(f"PIT membership years must be contiguous; gap between {previous} and {current}.")
        sizes = {len(tickers) for tickers in self.memberships.values()}
        if len(sizes) != 1:
            raise ConfigError(f"PIT membership lists must all have the same size; found sizes {sorted(sizes)}.")
        for year, tickers in self.memberships.items():
            if len(set(tickers)) != len(tickers):
                raise ConfigError(f"PIT membership for year {year} contains duplicate tickers.")
            for ticker in tickers:
                if not ticker:
                    raise ConfigError(f"PIT membership for year {year} contains an empty ticker.")

    @property
    def years(self) -> list[int]:
        return sorted(self.memberships)

    def union_tickers(self) -> list[str]:
        union: set[str] = set()
        for tickers in self.memberships.values():
            union.update(tickers)
        return sorted(union)

    def member_years(self, ticker: str) -> list[int]:
        normalized = ticker.upper().strip()
        return [year for year, tickers in sorted(self.memberships.items()) if normalized in tickers]

    def is_member(self, ticker: str, year: int) -> bool:
        return ticker.upper().strip() in self.memberships.get(int(year), ())

    def member_mask(self, tickers: pd.Series, dates: pd.Series) -> pd.Series:
        """Vectorized membership test for (ticker, date) pairs.

        Dates outside every membership year are non-member (False) rather than an error,
        so bars before the first membership year fall out of the evaluation naturally.
        """
        years = pd.to_datetime(dates).dt.year
        pairs = set()
        for year, members in self.memberships.items():
            for ticker in members:
                pairs.add((ticker, year))
        keys = list(zip(tickers.astype(str).str.upper().str.strip(), years))
        return pd.Series([key in pairs for key in keys], index=tickers.index)

    def annotate_predictions(self, predictions: pd.DataFrame) -> pd.DataFrame:
        if predictions.empty:
            output = predictions.copy()
            output["pit_member"] = pd.Series(dtype=bool)
            return output
        output = predictions.copy()
        output["pit_member"] = self.member_mask(output["ticker"], output["date"])
        return output

    def filter_predictions(self, predictions: pd.DataFrame) -> pd.DataFrame:
        annotated = predictions if "pit_member" in predictions.columns else self.annotate_predictions(predictions)
        return annotated[annotated["pit_member"]].reset_index(drop=True)

    def coverage_summary(self, annotated_predictions: pd.DataFrame) -> pd.DataFrame:
        """Per ticker/timeframe accounting of how much of the sample survives the PIT mask."""
        columns = [
            "ticker",
            "timeframe",
            "membership_years",
            "rows_total",
            "rows_member",
            "test_rows_total",
            "test_rows_member",
            "first_member_date",
            "last_member_date",
        ]
        if annotated_predictions.empty:
            return pd.DataFrame(columns=columns)
        rows = []
        for (ticker, timeframe), group in annotated_predictions.groupby(["ticker", "timeframe"]):
            member = group[group["pit_member"]]
            test = group[group["sample_type"] == "test"]
            member_test = test[test["pit_member"]]
            member_dates = pd.to_datetime(member["date"]) if not member.empty else pd.Series(dtype="datetime64[ns]")
            rows.append(
                {
                    "ticker": ticker,
                    "timeframe": timeframe,
                    "membership_years": ",".join(str(year) for year in self.member_years(str(ticker))),
                    "rows_total": int(len(group)),
                    "rows_member": int(len(member)),
                    "test_rows_total": int(len(test)),
                    "test_rows_member": int(len(member_test)),
                    "first_member_date": member_dates.min() if not member_dates.empty else pd.NaT,
                    "last_member_date": member_dates.max() if not member_dates.empty else pd.NaT,
                }
            )
        return pd.DataFrame(rows, columns=columns).sort_values(["timeframe", "ticker"]).reset_index(drop=True)


def load_pit_universe_from_config(config: dict, config_path: str | Path) -> tuple[PitUniverse | None, str]:
    """Return (universe, mode) from the optional 'universe_membership' config block.

    mode: 'evaluation_mask' filters predictions to membership periods before any
    reporting; 'annotate_only' keeps all rows and only adds the pit_member column.
    """
    from core.config import resolve_path

    membership_config = config.get("universe_membership")
    if not membership_config:
        return None, ""
    if not isinstance(membership_config, dict) or "file" not in membership_config:
        raise ConfigError("'universe_membership' must be a mapping with a 'file' key.")
    mode = str(membership_config.get("mode", "evaluation_mask"))
    if mode not in {"evaluation_mask", "annotate_only"}:
        raise ConfigError(f"Unsupported universe_membership mode '{mode}'.")
    universe = PitUniverse.from_file(resolve_path(config_path, membership_config["file"]))
    return universe, mode
