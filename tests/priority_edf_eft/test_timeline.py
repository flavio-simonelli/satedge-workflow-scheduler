from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from satedge_workflow_scheduler.domain.errors import InstanceError
from satedge_workflow_scheduler.domain.window import TimeWindow
from satedge_workflow_scheduler.solvers.priority_edf_eft.timeline import BusyTimeline

EPS = 1e-9


def test_empty_timeline_starts_at_ready() -> None:
    assert BusyTimeline().first_free(ready=5.0, length=10.0, eps=EPS) == 5.0


def test_finds_gap_large_enough() -> None:
    timeline = BusyTimeline([TimeWindow(0.0, 10.0), TimeWindow(20.0, 30.0)])
    # gap [10, 20) fits a length-8 item
    assert timeline.first_free(ready=0.0, length=8.0, eps=EPS) == pytest.approx(10.0)


def test_skips_gap_too_small() -> None:
    timeline = BusyTimeline([TimeWindow(0.0, 10.0), TimeWindow(12.0, 30.0)])
    # gap [10, 12) is too small for length 5: the item goes after 30
    assert timeline.first_free(ready=0.0, length=5.0, eps=EPS) == pytest.approx(30.0)


def test_exact_adjacency_is_not_an_overlap() -> None:
    timeline = BusyTimeline([TimeWindow(0.0, 10.0), TimeWindow(15.0, 20.0)])
    # an item of length 5 starting exactly at 10 fills [10, 15) without overlapping
    assert timeline.first_free(ready=10.0, length=5.0, eps=EPS) == pytest.approx(10.0)


def test_ready_inside_busy_interval_is_pushed_to_its_end() -> None:
    timeline = BusyTimeline([TimeWindow(0.0, 10.0)])
    assert timeline.first_free(ready=4.0, length=2.0, eps=EPS) == pytest.approx(10.0)


def test_with_reserved_returns_a_copy() -> None:
    original = BusyTimeline([TimeWindow(0.0, 5.0)])
    extended = original.with_reserved(TimeWindow(10.0, 12.0))
    assert len(original.intervals) == 1
    assert len(extended.intervals) == 2
    assert extended.first_free(ready=5.0, length=5.0, eps=EPS) == pytest.approx(5.0)


def test_window_end_before_start_is_rejected() -> None:
    with pytest.raises(InstanceError):
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
    timeline = BusyTimeline(busy)
    t = timeline.first_free(ready=ready, length=length, eps=EPS)
    assert t >= ready - EPS
    for iv in busy:
        assert iv.end <= t + EPS or iv.start >= t + length - EPS
