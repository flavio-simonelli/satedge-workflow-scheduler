from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from satedge_workflow_scheduler.common.intervals import TimeWindow, WindowSet

EPS = 1e-9


def test_empty_timeline_starts_at_ready() -> None:
    assert WindowSet().first_gap(ready=5.0, length=10.0, eps=EPS) == 5.0


def test_finds_gap_large_enough() -> None:
    timeline = WindowSet([TimeWindow(0.0, 10.0), TimeWindow(20.0, 30.0)])
    # gap [10, 20) fits a length-8 item
    assert timeline.first_gap(ready=0.0, length=8.0, eps=EPS) == pytest.approx(10.0)


def test_skips_gap_too_small() -> None:
    timeline = WindowSet([TimeWindow(0.0, 10.0), TimeWindow(12.0, 30.0)])
    # gap [10, 12) is too small for length 5: the item goes after 30
    assert timeline.first_gap(ready=0.0, length=5.0, eps=EPS) == pytest.approx(30.0)


def test_exact_adjacency_is_not_an_overlap() -> None:
    timeline = WindowSet([TimeWindow(0.0, 10.0), TimeWindow(15.0, 20.0)])
    # an item of length 5 starting exactly at 10 fills [10, 15) without overlapping
    assert timeline.first_gap(ready=10.0, length=5.0, eps=EPS) == pytest.approx(10.0)


def test_ready_inside_busy_interval_is_pushed_to_its_end() -> None:
    timeline = WindowSet([TimeWindow(0.0, 10.0)])
    assert timeline.first_gap(ready=4.0, length=2.0, eps=EPS) == pytest.approx(10.0)


def test_with_reserved_returns_a_copy() -> None:
    original = WindowSet([TimeWindow(0.0, 5.0)])
    extended = original.union_with(TimeWindow(10.0, 12.0))
    assert len(original.windows) == 1
    assert len(extended.windows) == 2
    assert extended.first_gap(ready=5.0, length=5.0, eps=EPS) == pytest.approx(5.0)


def test_window_end_before_start_is_rejected() -> None:
    with pytest.raises(ValueError):
        TimeWindow(5.0, 4.0)


_interval = st.tuples(
    st.floats(min_value=0.0, max_value=100.0),
    st.floats(min_value=0.1, max_value=20.0),
).map(lambda p: TimeWindow(p[0], p[0] + p[1]))


@given(
    busy=st.lists(_interval, max_size=10),
    ready=st.floats(min_value=0.0, max_value=100.0),
    length=st.floats(min_value=0.1, max_value=30.0),
)
def test_first_free_does_not_overlap_busy_and_respects_ready(
    busy: list[TimeWindow], ready: float, length: float
) -> None:
    timeline = WindowSet.merged(busy)
    t = timeline.first_gap(ready=ready, length=length, eps=EPS)
    assert t >= ready - EPS
    for iv in busy:
        assert iv.end <= t + EPS or iv.start >= t + length - EPS


def test_touching_reservation_is_merged_into_one_window() -> None:
    windows = WindowSet([TimeWindow(0.0, 10.0)]).union_with(TimeWindow(10.0, 15.0))
    assert windows.windows == (TimeWindow(0.0, 15.0),)


def test_overlapping_windows_given_to_the_constructor_are_merged() -> None:
    windows = WindowSet.merged([TimeWindow(5.0, 9.0), TimeWindow(0.0, 6.0)])
    assert windows.windows == (TimeWindow(0.0, 9.0),)


def test_constructor_accepts_sorted_disjoint_windows() -> None:
    windows = WindowSet([TimeWindow(0.0, 5.0), TimeWindow(10.0, 12.0)])
    assert windows.windows == (TimeWindow(0.0, 5.0), TimeWindow(10.0, 12.0))


def test_constructor_rejects_unsorted_windows() -> None:
    with pytest.raises(ValueError):
        WindowSet([TimeWindow(10.0, 12.0), TimeWindow(0.0, 5.0)])


def test_constructor_rejects_overlapping_or_touching_windows() -> None:
    with pytest.raises(ValueError):
        WindowSet([TimeWindow(0.0, 6.0), TimeWindow(5.0, 9.0)])
    with pytest.raises(ValueError):
        WindowSet([TimeWindow(0.0, 5.0), TimeWindow(5.0, 9.0)])


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
