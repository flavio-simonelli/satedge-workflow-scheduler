from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from satedge_workflow_scheduler.common.intervals import BusyTimeline, TimeWindow

EPS = 1e-9


def test_empty_timeline_starts_at_ready() -> None:
    assert BusyTimeline().first_gap(ready=5.0, length=10.0, eps=EPS) == 5.0


def test_finds_gap_large_enough() -> None:
    timeline = BusyTimeline([TimeWindow(0.0, 10.0), TimeWindow(20.0, 30.0)])
    # gap [10, 20) fits a length-8 item
    assert timeline.first_gap(ready=0.0, length=8.0, eps=EPS) == pytest.approx(10.0)


def test_skips_gap_too_small() -> None:
    timeline = BusyTimeline([TimeWindow(0.0, 10.0), TimeWindow(12.0, 30.0)])
    # gap [10, 12) is too small for length 5: the item goes after 30
    assert timeline.first_gap(ready=0.0, length=5.0, eps=EPS) == pytest.approx(30.0)


def test_exact_adjacency_is_not_an_overlap() -> None:
    timeline = BusyTimeline([TimeWindow(0.0, 10.0), TimeWindow(15.0, 20.0)])
    # an item of length 5 starting exactly at 10 fills [10, 15) without overlapping
    assert timeline.first_gap(ready=10.0, length=5.0, eps=EPS) == pytest.approx(10.0)


def test_ready_inside_busy_interval_is_pushed_to_its_end() -> None:
    timeline = BusyTimeline([TimeWindow(0.0, 10.0)])
    assert timeline.first_gap(ready=4.0, length=2.0, eps=EPS) == pytest.approx(10.0)


def test_with_interval_returns_a_new_timeline() -> None:
    original = BusyTimeline([TimeWindow(0.0, 5.0)])
    extended = original.with_interval(TimeWindow(10.0, 12.0))
    assert len(original) == 1
    assert len(extended) == 2
    assert extended.first_gap(ready=5.0, length=5.0, eps=EPS) == pytest.approx(5.0)


_interval = st.tuples(
    st.floats(min_value=0.0, max_value=100.0),
    st.floats(min_value=0.1, max_value=20.0),
).map(lambda p: TimeWindow(p[0], p[0] + p[1]))


@given(
    busy=st.lists(_interval, max_size=10),
    ready=st.floats(min_value=0.0, max_value=100.0),
    length=st.floats(min_value=0.1, max_value=30.0),
)
def test_first_gap_does_not_overlap_busy_and_respects_ready(
    busy: list[TimeWindow], ready: float, length: float
) -> None:
    t = BusyTimeline(busy).first_gap(ready=ready, length=length, eps=EPS)
    assert t >= ready - EPS
    for iv in busy:
        assert iv.end <= t + EPS or iv.start >= t + length - EPS


def test_touching_reservations_are_fused_into_one_interval() -> None:
    timeline = BusyTimeline([TimeWindow(0.0, 10.0)]).with_interval(
        TimeWindow(10.0, 15.0)
    )
    assert timeline.intervals == (TimeWindow(0.0, 15.0),)


def test_overlapping_unsorted_intervals_are_fused() -> None:
    timeline = BusyTimeline([TimeWindow(5.0, 9.0), TimeWindow(0.0, 6.0)])
    assert timeline.intervals == (TimeWindow(0.0, 9.0),)


def test_timelines_with_the_same_intervals_are_equal() -> None:
    a = BusyTimeline([TimeWindow(0.0, 5.0), TimeWindow(3.0, 8.0)])
    assert a == BusyTimeline([TimeWindow(0.0, 8.0)])
    assert hash(a) == hash(BusyTimeline([TimeWindow(0.0, 8.0)]))
