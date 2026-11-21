"""Time intervals: the time window value object shared by the problem model and the solvers."""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

TIME_TOLERANCE = 1e-9
"""Tolerance in seconds for comparisons between instants, to absorb floating point rounding."""


@dataclass(frozen=True, slots=True, order=True)
class TimeWindow:
    """A closed time interval ``[start, end]`` in seconds, with ``start < end``.

    Used both for the HPC computing windows of a satellite and for the visibility
    windows of an arc (contact plan).
    """

    start: float
    end: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.start) and math.isfinite(self.end)):
            raise ValueError(
                f"time window bounds must be finite, got [{self.start}, {self.end}]"
            )
        if self.start < 0:
            raise ValueError(f"time window must start at t >= 0, got {self.start}")
        if self.start >= self.end:
            raise ValueError(
                f"time window must have start < end, got [{self.start}, {self.end}]"
            )

    @property
    def duration(self) -> float:
        """Length of the window in seconds."""
        return self.end - self.start

    def intersection(self, other: TimeWindow) -> TimeWindow | None:
        """Common part of the two windows, or ``None`` when they do not overlap.

        Windows that only touch at one instant also return ``None``, since their
        intersection has zero duration.
        """
        start = max(self.start, other.start)
        end = min(self.end, other.end)
        if start >= end:
            return None
        return TimeWindow(start, end)

    def contains(self, other: TimeWindow) -> bool:
        """Whether ``other`` lies inside this window, up to :data:`TIME_TOLERANCE`."""
        return (
            self.start - TIME_TOLERANCE <= other.start
            and other.end <= self.end + TIME_TOLERANCE
        )


class WindowSet:
    """An immutable set of time windows, kept sorted and pairwise disjoint.

    The constructor requires its input to be already sorted and pairwise disjoint, and
    raises ``ValueError`` otherwise. Use :meth:`merged` to build a set from arbitrary
    windows: overlapping or touching windows are then fused into one.
    """

    __slots__ = ("_windows",)

    def __init__(self, windows: Iterable[TimeWindow] = ()) -> None:
        items = tuple(windows)
        for previous, current in itertools.pairwise(items):
            if not previous.end < current.start:
                raise ValueError(
                    "windows must be sorted and disjoint: "
                    f"[{previous.start}, {previous.end}] then "
                    f"[{current.start}, {current.end}]"
                )
        self._windows: tuple[TimeWindow, ...] = items

    @classmethod
    def merged(cls, windows: Iterable[TimeWindow]) -> WindowSet:
        """Build the set from any windows, fusing those that overlap or touch."""
        return cls(_merge(windows))

    @property
    def windows(self) -> tuple[TimeWindow, ...]:
        """The windows of the set, sorted by start and pairwise disjoint."""
        return self._windows

    def __iter__(self) -> Iterator[TimeWindow]:
        return iter(self._windows)

    def __len__(self) -> int:
        return len(self._windows)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WindowSet):
            return NotImplemented
        return self._windows == other._windows

    def __hash__(self) -> int:
        return hash(self._windows)

    def __repr__(self) -> str:
        return f"WindowSet({list(self._windows)!r})"

    def first_gap(
        self, ready: float, length: float, eps: float = TIME_TOLERANCE
    ) -> float:
        """Return the earliest ``t >= ready`` such that ``[t, t + length)`` meets no window.

        Two windows that only touch do not overlap the candidate interval, up to ``eps``.
        """
        candidate = ready
        for window in self._windows:
            if window.end <= candidate + eps:
                continue
            if window.start >= candidate + length - eps:
                break
            candidate = max(candidate, window.end)
        return candidate

    def union_with(self, window: TimeWindow) -> WindowSet:
        """Return a new set that also contains ``window``, fused with its neighbours.

        The receiver is not modified.
        """
        return WindowSet.merged((*self._windows, window))


def _merge(windows: Iterable[TimeWindow]) -> tuple[TimeWindow, ...]:
    """Sort the windows and merge the ones that overlap or lie within the tolerance."""
    merged: list[TimeWindow] = []
    for window in sorted(windows, key=lambda w: w.start):
        if merged and window.start <= merged[-1].end + TIME_TOLERANCE:
            last = merged[-1]
            merged[-1] = TimeWindow(last.start, max(last.end, window.end))
        else:
            merged.append(window)
    return tuple(merged)
