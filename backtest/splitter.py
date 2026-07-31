from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    number: int
    train_positions: np.ndarray
    test_positions: np.ndarray
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    purge_bars: int
    embargo_bars: int


class WalkForwardSplitter:
    def __init__(
        self,
        *,
        mode: str = "expanding",
        initial_train_bars: int = 500,
        test_bars: int = 126,
        purge_lookback_bars: int = 200,
        embargo_bars: int = 1,
        min_train_bars: int = 60,
        adaptive: bool = True,
        cpcv_n_groups: int = 8,
        cpcv_n_test_groups: int = 2,
    ) -> None:
        self.mode = mode
        self.initial_train_bars = initial_train_bars
        self.test_bars = test_bars
        self.purge_lookback_bars = purge_lookback_bars
        self.embargo_bars = embargo_bars
        self.min_train_bars = min_train_bars
        self.adaptive = adaptive
        self.cpcv_n_groups = cpcv_n_groups
        self.cpcv_n_test_groups = cpcv_n_test_groups

    @classmethod
    def from_config(cls, config: dict) -> "WalkForwardSplitter":
        wf = config.get("walk_forward", {})
        purge_lookback_bars = int(wf.get("purge_lookback_bars", 200))
        embargo_bars = int(wf.get("embargo_bars", 1))
        target_config = config.get("target", {})
        if str(target_config.get("mode", "binary")).lower() == "triple_barrier":
            max_holding_bars = int(target_config.get("triple_barrier", {}).get("max_holding_bars", 10))
            entry_lag_bars = int(config.get("execution", {}).get("entry_lag_bars", 1))
            label_lookahead_bars = entry_lag_bars + max_holding_bars - 1
            purge_lookback_bars = max(purge_lookback_bars, label_lookahead_bars)
            embargo_bars = max(embargo_bars, label_lookahead_bars)
        return cls(
            mode=str(wf.get("mode", "expanding")),
            initial_train_bars=int(wf.get("initial_train_bars", 500)),
            test_bars=int(wf.get("test_bars", 126)),
            purge_lookback_bars=purge_lookback_bars,
            embargo_bars=embargo_bars,
            min_train_bars=int(wf.get("min_train_bars", 60)),
            adaptive=bool(wf.get("adaptive", True)),
            cpcv_n_groups=int(wf.get("cpcv", {}).get("n_groups", 8)),
            cpcv_n_test_groups=int(wf.get("cpcv", {}).get("n_test_groups", 2)),
        )

    def split(self, frame: pd.DataFrame) -> list[Fold]:
        if self.mode == "cpcv":
            return self._split_cpcv(frame)

        n = len(frame)
        if n < self.min_train_bars + 2:
            return []

        initial_train = self.initial_train_bars
        test_size = self.test_bars
        purge = self.purge_lookback_bars
        embargo = self.embargo_bars

        if self.adaptive and initial_train + purge + embargo + 1 >= n:
            test_size = max(1, min(self.test_bars, max(1, n // 5)))
            purge = max(0, min(purge, max(0, n // 10)))
            initial_train = max(self.min_train_bars, n - test_size - purge - embargo)

        folds: list[Fold] = []
        test_start = initial_train + purge + embargo
        fold_number = 1
        while test_start < n:
            train_end = test_start - purge - embargo
            if train_end < self.min_train_bars:
                break
            train_start = 0 if self.mode == "expanding" else max(0, train_end - initial_train)
            test_end = min(n, test_start + test_size)
            if test_end <= test_start:
                break

            train_positions = np.arange(train_start, train_end)
            test_positions = np.arange(test_start, test_end)
            index = pd.DatetimeIndex(frame.index)
            folds.append(
                Fold(
                    number=fold_number,
                    train_positions=train_positions,
                    test_positions=test_positions,
                    train_start=index[train_positions[0]],
                    train_end=index[train_positions[-1]],
                    test_start=index[test_positions[0]],
                    test_end=index[test_positions[-1]],
                    purge_bars=purge,
                    embargo_bars=embargo,
                )
            )
            test_start = test_end
            fold_number += 1
        return folds

    def _split_cpcv(self, frame: pd.DataFrame) -> list[Fold]:
        n = len(frame)
        if n < self.min_train_bars + 2:
            return []
        if self.cpcv_n_groups < 2:
            raise ValueError("cpcv n_groups must be at least 2")
        if not 1 <= self.cpcv_n_test_groups < self.cpcv_n_groups:
            raise ValueError("cpcv n_test_groups must be between 1 and n_groups - 1")

        positions = np.arange(n)
        groups = [group for group in np.array_split(positions, self.cpcv_n_groups) if len(group) > 0]
        index = pd.DatetimeIndex(frame.index)
        folds: list[Fold] = []
        fold_number = 1

        for test_group_ids in combinations(range(len(groups)), self.cpcv_n_test_groups):
            test_positions = np.sort(np.concatenate([groups[group_id] for group_id in test_group_ids]))
            blocked = np.zeros(n, dtype=bool)
            blocked[test_positions] = True

            for group_id in test_group_ids:
                group = groups[group_id]
                start = int(group[0])
                end_exclusive = int(group[-1]) + 1
                block_start = max(0, start - self.purge_lookback_bars)
                block_end = min(n, end_exclusive + self.embargo_bars)
                blocked[block_start:block_end] = True

            train_positions = positions[~blocked]
            if len(train_positions) < self.min_train_bars:
                continue
            folds.append(
                Fold(
                    number=fold_number,
                    train_positions=train_positions,
                    test_positions=test_positions,
                    train_start=index[train_positions[0]],
                    train_end=index[train_positions[-1]],
                    test_start=index[test_positions[0]],
                    test_end=index[test_positions[-1]],
                    purge_bars=self.purge_lookback_bars,
                    embargo_bars=self.embargo_bars,
                )
            )
            fold_number += 1
        return folds
