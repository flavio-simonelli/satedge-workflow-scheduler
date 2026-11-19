"""Busy timelines of unary resources: the intervals in which an HPC or an arc is occupied."""

from __future__ import annotations

from collections.abc import Iterable

from satedge_workflow_scheduler.domain.window import TimeWindow


class BusyTimeline:
    """Sorted busy intervals of a unary resource, which serves one item at a time.

    The object is immutable: :meth:`with_reserved` returns a new timeline.
    """

    def __init__(self, intervals: Iterable[TimeWindow] = ()) -> None:
        self._intervals: tuple[TimeWindow, ...] = tuple(
            sorted(intervals, key=lambda iv: iv.start)
        )

    @property
    def intervals(self) -> tuple[TimeWindow, ...]:
        """The busy intervals, sorted by start."""
        return self._intervals

    def first_free(self, ready: float, length: float, eps: float) -> float:
        """Return the earliest ``t >= ready`` such that ``[t, t + length)`` is not busy.

        Two intervals that only touch (one ends at ``t``, the other starts at ``t``) do
        not overlap.
        """
        candidate = ready
        for interval in self._intervals:
            if interval.end <= candidate + eps:
                continue
            if interval.start >= candidate + length - eps:
                break
            candidate = max(candidate, interval.end)
        return candidate

    def with_reserved(self, interval: TimeWindow) -> BusyTimeline:
        """Return a copy of this timeline with ``interval`` added as busy."""
        return BusyTimeline((*self._intervals, interval))
