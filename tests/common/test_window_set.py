from __future__ import annotations

import pytest

from satedge_workflow_scheduler.common.intervals import (
    BusyTimeline,
    TimeWindow,
    WindowSet,
)


def test_window_end_before_start_is_rejected() -> None:
    with pytest.raises(ValueError):
        TimeWindow(5.0, 4.0)


def test_constructor_accepts_sorted_disjoint_windows() -> None:
    windows = WindowSet([TimeWindow(0.0, 5.0), TimeWindow(10.0, 12.0)])
    assert windows.windows == (TimeWindow(0.0, 5.0), TimeWindow(10.0, 12.0))


def test_constructor_rejects_unsorted_windows() -> None:
    unsorted = [TimeWindow(10.0, 12.0), TimeWindow(0.0, 5.0)]
    with pytest.raises(ValueError):
        WindowSet(unsorted)


def test_constructor_rejects_overlapping_or_touching_windows() -> None:
    # windows of availability come from the instance: they are never fused
    overlapping = [TimeWindow(0.0, 6.0), TimeWindow(5.0, 9.0)]
    touching = [TimeWindow(0.0, 5.0), TimeWindow(5.0, 9.0)]
    with pytest.raises(ValueError):
        WindowSet(overlapping)
    with pytest.raises(ValueError):
        WindowSet(touching)


def test_window_sets_with_the_same_windows_are_equal_and_hash_alike() -> None:
    a = WindowSet([TimeWindow(0.0, 10.0), TimeWindow(20.0, 30.0)])
    b = WindowSet((TimeWindow(0.0, 10.0), TimeWindow(20.0, 30.0)))
    assert a == b
    assert hash(a) == hash(b)
    assert a != WindowSet([TimeWindow(0.0, 10.0)])


def test_a_window_set_iterates_over_its_windows_in_order() -> None:
    windows = (TimeWindow(0.0, 10.0), TimeWindow(20.0, 30.0))
    assert tuple(WindowSet(windows)) == windows
    assert len(WindowSet(windows)) == 2
    assert not WindowSet()


def test_windows_and_busy_intervals_are_never_equal() -> None:
    interval = [TimeWindow(0.0, 10.0)]
    assert WindowSet(interval) != BusyTimeline(interval)
