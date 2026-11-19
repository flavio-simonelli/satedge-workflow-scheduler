from __future__ import annotations

import pytest

from satedge_workflow_scheduler.domain.errors import InstanceError
from satedge_workflow_scheduler.domain.requests import Priority, Request, Task
from satedge_workflow_scheduler.scheduling.weights import (
    rejection_weights,
    tardiness_weights,
)


def _req(rid: str, priority: Priority, deadline: float) -> Request:
    return Request(
        id=rid,
        priority=priority,
        origin="S1",
        release=0.0,
        deadline=deadline,
        raw_size=1.0,
        tasks=(Task(duration=10.0, output_size=1.0),),
    )


U, N = Priority.URGENT, Priority.NON_URGENT


def test_rejection_weights_two_urgent_three_non_urgent() -> None:
    requests = [
        _req("u1", U, 500.0),
        _req("n1", N, 100.0),
        _req("n2", N, 200.0),
        _req("u2", U, 300.0),
        _req("n3", N, 400.0),
    ]
    weights = rejection_weights(requests)
    assert weights == {"u1": 4.0, "n1": 1.0, "n2": 1.0, "u2": 4.0, "n3": 1.0}


def test_one_urgent_rejection_costs_more_than_all_non_urgent_ones() -> None:
    requests = [_req("u", U, 10.0)] + [_req(f"n{i}", N, 10.0) for i in range(7)]
    weights = rejection_weights(requests)
    assert weights["u"] > sum(weights[f"n{i}"] for i in range(7))


def test_tardiness_weights_match_hand_computation() -> None:
    # H = 100; non urgent deadlines 40 and 120: slack max(0, 100-40) + max(0, 100-120) = 60.
    requests = [_req("u", U, 50.0), _req("n1", N, 40.0), _req("n2", N, 120.0)]
    weights = tardiness_weights(requests, horizon=100.0)
    assert weights == {"u": 61.0, "n1": 1.0, "n2": 1.0}


def test_tardiness_weights_resolution_scales_the_urgent_term() -> None:
    requests = [_req("u", U, 50.0), _req("n", N, 40.0)]
    weights = tardiness_weights(requests, horizon=100.0, resolution=10.0)
    assert weights["u"] == pytest.approx(1.0 + 60.0 / 10.0)


def test_tardiness_weights_reject_non_positive_resolution() -> None:
    with pytest.raises(InstanceError):
        tardiness_weights([_req("n", N, 1.0)], horizon=10.0, resolution=0.0)
