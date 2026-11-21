"""Time intervals and the two kinds of sets of intervals used by the model and the solvers.

- :class:`WindowSet`: windows of *availability*, such as the computing windows of an HPC or
  the visibility windows of a link. An activity must lie *inside* one of them.
- :class:`BusyTimeline`: intervals in which a resource is already *taken*, such as the
  reservations on an HPC or an arc. An activity must lie *outside* all of them.
"""

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

    Used for the windows of a :class:`WindowSet` and for the intervals of a
    :class:`BusyTimeline`.
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
    """Windows of availability of a resource: an activity must lie inside one of them.

    The windows are given by the instance, sorted and pairwise disjoint: the constructor
    raises ``ValueError`` otherwise. Two windows that only touch are not disjoint.
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

    @property
    def windows(self) -> tuple[TimeWindow, ...]:
        """The windows, sorted by start and pairwise disjoint."""
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


class BusyTimeline:
    """Intervals in which a resource is already taken: an activity must avoid all of them.

    Reservations accumulate one after the other, so the constructor accepts any intervals
    and fuses those that overlap or touch: two adjacent reservations are one longer busy
    interval. The timeline is immutable; :meth:`with_interval` returns a new one.
    """

    __slots__ = ("_intervals",)

    def __init__(self, intervals: Iterable[TimeWindow] = ()) -> None:
        self._intervals: tuple[TimeWindow, ...] = _merge(intervals)

    @property
    def intervals(self) -> tuple[TimeWindow, ...]:
        """The busy intervals, sorted by start and pairwise disjoint."""
        return self._intervals

    def __iter__(self) -> Iterator[TimeWindow]:
        return iter(self._intervals)

    def __len__(self) -> int:
        return len(self._intervals)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BusyTimeline):
            return NotImplemented
        return self._intervals == other._intervals

    def __hash__(self) -> int:
        return hash(self._intervals)

    def __repr__(self) -> str:
        return f"BusyTimeline({list(self._intervals)!r})"

    def first_gap(
        self, ready: float, length: float, eps: float = TIME_TOLERANCE
    ) -> float:
        """Earliest ``t >= ready`` such that ``[t, t + length)`` meets no busy interval.

        An interval that only touches the candidate does not overlap it, up to ``eps``.
        """
        candidate = ready
        for interval in self._intervals:
            if interval.end <= candidate + eps:
                continue
            if interval.start >= candidate + length - eps:
                break
            candidate = max(candidate, interval.end)
        return candidate

    def with_interval(self, interval: TimeWindow) -> BusyTimeline:
        """A new timeline that also holds ``interval``. The receiver is not modified."""
        return BusyTimeline((*self._intervals, interval))


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
