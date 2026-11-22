from __future__ import annotations

import pytest

from satedge_workflow_scheduler.common.intervals import (
    BusyTimeline,
    TimeWindow,
    WindowSet,
)
from satedge_workflow_scheduler.domain.network import (
    AlwaysAvailable,
    WindowedAvailability,
)
from satedge_workflow_scheduler.scheduling.windows import (
    arc_windows,
    earliest_finish,
    latest_start,
)

# two windows, [10, 20] and [30, 50]
WINDOWS = WindowSet((TimeWindow(10.0, 20.0), TimeWindow(30.0, 50.0)))


@pytest.mark.parametrize(
    ("ready", "length", "expected"),
    [
        (0.0, 5.0, 15.0),  # waits for the first window to open at 10
        (12.0, 5.0, 17.0),  # starts at once inside the first window
        (16.0, 5.0, 35.0),  # would end at 21, after the first window: next one
        (0.0, 15.0, 45.0),  # longer than the first window
        (0.0, 20.0, 50.0),  # exactly as long as the second window
        (0.0, 25.0, None),  # longer than every window
        (60.0, 1.0, None),  # after the last window
    ],
)
def test_earliest_finish_in_windows(
    ready: float, length: float, expected: float | None
) -> None:
    assert earliest_finish(WINDOWS, ready, length) == expected


def test_without_windows_the_activity_starts_when_ready() -> None:
    assert earliest_finish(None, 7.0, 100.0) == 107.0


def test_busy_intervals_delay_the_activity() -> None:
    busy = BusyTimeline((TimeWindow(10.0, 14.0),))
    assert earliest_finish(WINDOWS, 0.0, 5.0, busy=busy) == 19.0


def test_busy_intervals_can_push_the_activity_to_the_next_window() -> None:
    # after the busy interval only 4 s are left in the first window
    busy = BusyTimeline((TimeWindow(10.0, 16.0),))
    assert earliest_finish(WINDOWS, 0.0, 5.0, busy=busy) == 35.0


def test_without_windows_busy_intervals_still_count() -> None:
    busy = BusyTimeline((TimeWindow(5.0, 9.0),))
    assert earliest_finish(None, 6.0, 3.0, busy=busy) == 12.0


def test_only_the_occupied_part_must_avoid_the_busy_intervals() -> None:
    # A transmission of 3 s with 1 s of latency must fit in the window for 4 s, but it
    # takes the arc only for 3 s: it may start 3 s before another busy interval.
    busy = BusyTimeline((TimeWindow(15.0, 20.0),))
    assert earliest_finish(WINDOWS, 12.0, 4.0, busy=busy, occupied=3.0) == 16.0
    assert earliest_finish(WINDOWS, 12.0, 4.0, busy=busy) == 34.0


@pytest.mark.parametrize(
    ("due", "length", "expected"),
    [
        (100.0, 5.0, 45.0),  # ends at the close of the last window
        (40.0, 5.0, 35.0),  # ends at the due instant
        (32.0, 5.0, 15.0),  # the second window is too short before 32: first one
        (100.0, 25.0, None),  # longer than every window
        (5.0, 1.0, None),  # before the first window
    ],
)
def test_latest_start_in_windows(
    due: float, length: float, expected: float | None
) -> None:
    assert latest_start(WINDOWS, due, length) == expected


def test_latest_start_without_windows() -> None:
    assert latest_start(None, 10.0, 4.0) == 6.0
    assert latest_start(None, 3.0, 4.0) is None


def test_arc_windows_of_a_windowed_arc_are_its_visibility_windows() -> None:
    assert arc_windows(WindowedAvailability(WINDOWS)) is WINDOWS


def test_arc_windows_of_an_always_available_arc_are_none() -> None:
    assert arc_windows(AlwaysAvailable()) is None
