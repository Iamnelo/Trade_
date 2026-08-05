"""Time-series cross-validation splitters.

Two schemes:

- `walk_forward_splits`: expanding or rolling-window train/test pairs. The
  usual choice for financial models: each fold trains on a period ending at
  time T and tests on [T, T + test_bars). Trains never see future bars.
- `purged_embargo_folds`: adapted from Lopez de Prado. Splits the timeline
  into k equal test folds; the training set for a fold excludes a `purge`
  region around the test window and an `embargo` window AFTER the test to
  prevent label-leakage when horizons overlap.

Both return concrete index ranges into a sorted bar sequence. The runner
turns those into `MarketReplaySource` instances and calls `run_backtest`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Split:
    train_start: int
    train_end: int  # exclusive
    test_start: int
    test_end: int  # exclusive


def walk_forward_splits(
    n_bars: int,
    *,
    train_bars: int,
    test_bars: int,
    step_bars: int | None = None,
    expanding: bool = False,
) -> list[Split]:
    """Rolling (default) or expanding walk-forward splits.

    - Rolling: train window slides forward by `step_bars`; window size stays
      constant. Best for detecting concept drift.
    - Expanding: train window starts at 0 and grows; test window slides.
      More data per fold but slower reaction to regime change.
    """
    if train_bars < 1 or test_bars < 1:
        raise ValueError("train_bars and test_bars must be >= 1")
    if n_bars < train_bars + test_bars:
        return []
    step = step_bars if step_bars is not None else test_bars
    if step < 1:
        raise ValueError("step_bars must be >= 1")

    splits: list[Split] = []
    test_start = train_bars
    while test_start + test_bars <= n_bars:
        train_start = 0 if expanding else test_start - train_bars
        splits.append(
            Split(
                train_start=train_start,
                train_end=test_start,
                test_start=test_start,
                test_end=test_start + test_bars,
            )
        )
        test_start += step
    return splits


@dataclass(frozen=True, slots=True)
class PurgedFold:
    """A single fold in a purged/embargoed k-fold split.

    `train_ranges` is a list of (start, end) tuples: the training indices
    are the union of these ranges. `(test_start, test_end)` is the
    contiguous held-out region.
    """

    train_ranges: tuple[tuple[int, int], ...]
    test_start: int
    test_end: int  # exclusive


def purged_embargo_folds(
    n_bars: int,
    *,
    n_folds: int,
    purge_bars: int = 0,
    embargo_bars: int = 0,
) -> list[PurgedFold]:
    """Purged/embargoed k-fold time-series split.

    Splits the timeline into `n_folds` equal contiguous test regions. For
    each fold, the training set is everything OUTSIDE the test region and
    outside a purge zone (bars immediately before the test) and an embargo
    zone (bars immediately after the test).
    """
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    if n_bars < n_folds:
        raise ValueError("n_bars must be >= n_folds")
    if purge_bars < 0 or embargo_bars < 0:
        raise ValueError("purge_bars and embargo_bars must be >= 0")

    fold_size = n_bars // n_folds
    folds: list[PurgedFold] = []
    for i in range(n_folds):
        test_start = i * fold_size
        test_end = test_start + fold_size if i < n_folds - 1 else n_bars
        purged_start = max(0, test_start - purge_bars)
        embargoed_end = min(n_bars, test_end + embargo_bars)

        train_ranges: list[tuple[int, int]] = []
        if purged_start > 0:
            train_ranges.append((0, purged_start))
        if embargoed_end < n_bars:
            train_ranges.append((embargoed_end, n_bars))

        folds.append(
            PurgedFold(
                train_ranges=tuple(train_ranges),
                test_start=test_start,
                test_end=test_end,
            )
        )
    return folds
