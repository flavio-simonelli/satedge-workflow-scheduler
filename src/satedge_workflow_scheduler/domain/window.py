"""Time window value objects."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

from satedge_workflow_scheduler.domain.errors import InstanceError

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
            raise InstanceError(
                f"time window bounds must be finite, got [{self.start}, {self.end}]"
            )
        if self.start < 0:
            raise InstanceError(f"time window must start at t >= 0, got {self.start}")
        if self.start >= self.end:
            raise InstanceError(
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


def validate_window_sequence(windows: tuple[TimeWindow, ...], owner: str) -> None:
    """Check that ``windows`` are sorted chronologically and pairwise disjoint.

    Args:
        windows: The window sequence to check.
        owner: Human readable owner of the sequence, used in the error message.

    Raises:
        InstanceError: If two consecutive windows are unsorted or overlap.
    """
    for previous, current in itertools.pairwise(windows):
        if not previous.end < current.start:
            raise InstanceError(
                f"windows of {owner} must be sorted and disjoint: "
                f"[{previous.start}, {previous.end}] then [{current.start}, {current.end}]"
            )
