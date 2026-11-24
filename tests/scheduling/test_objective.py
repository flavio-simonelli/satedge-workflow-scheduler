from __future__ import annotations

import pytest

from satedge_workflow_scheduler.common.intervals import TimeWindow, WindowSet
from satedge_workflow_scheduler.domain.errors import InstanceError
from satedge_workflow_scheduler.domain.instance import ProblemInstance
from satedge_workflow_scheduler.domain.network import (
    AlwaysAvailable,
    Arc,
    Network,
    Plane,
)
from satedge_workflow_scheduler.domain.nodes import GroundStation, Satellite
from satedge_workflow_scheduler.domain.requests import Priority, Request, Task
from satedge_workflow_scheduler.domain.solution import RequestOutcome
from satedge_workflow_scheduler.scheduling.deadline import DeadlineMode
from satedge_workflow_scheduler.scheduling.objective import (
    evaluate,
    rejection_weights,
    tardiness,
    tardiness_weights,
)

U, N = Priority.URGENT, Priority.NON_URGENT


def _request(rid: str, priority: Priority, deadline: float) -> Request:
    return Request(
        id=rid,
        priority=priority,
        origin="S1",
        release=0.0,
        deadline=deadline,
        raw_size=1.0,
        tasks=(Task(execution_time=10.0, output_size=1.0),),
    )


def _instance(requests: tuple[Request, ...], horizon: float = 100.0) -> ProblemInstance:
    sat = Satellite(
        id="S1",
        plmm_capacity=10.0,
        hpc_rate=1.0,
        hpc_windows=WindowSet((TimeWindow(0.0, 1000.0),)),
    )
    network = Network(
        planes=(Plane(id=1, satellites=(sat,), com="S1"),),
        ground_station=GroundStation(id="G"),
        arcs=(
            Arc(
                source="S1",
                target="G",
                capacity=1.0,
                latency=1.0,
                availability=AlwaysAvailable(),
            ),
        ),
    )
    return ProblemInstance(
        network=network,
        requests=requests,
        horizon=horizon,
        plan_release={s: 0.0 for s in network.satellite_ids},
    )


def test_tardiness_of_accepted_late_request() -> None:
    request = _request("r", N, deadline=40.0)
    outcome = RequestOutcome("r", accepted=True, delivery_time=55.0)
    assert tardiness(outcome, request) == pytest.approx(15.0)


def test_tardiness_is_zero_when_on_time_or_rejected() -> None:
    request = _request("r", N, deadline=40.0)
    assert tardiness(RequestOutcome("r", True, 40.0), request) == 0.0
    assert tardiness(RequestOutcome("r", False), request) == 0.0


def test_hard_objective_sums_rejection_costs() -> None:
    requests = (
        _request("u", U, 50.0),
        _request("n1", N, 50.0),
        _request("n2", N, 50.0),
    )
    instance = _instance(requests)
    outcomes = [
        RequestOutcome("u", accepted=True, delivery_time=10.0),
        RequestOutcome("n1", accepted=False),
        RequestOutcome("n2", accepted=False),
    ]
    # κ_n1 = κ_n2 = 1 (|N| = 2 non urgent requests, urgent has |N|+1 = 3)
    assert evaluate(instance, outcomes, DeadlineMode.HARD).primary == pytest.approx(2.0)


def test_hard_objective_of_rejected_urgent_request() -> None:
    requests = (_request("u", U, 50.0), _request("n", N, 50.0))
    instance = _instance(requests)
    outcomes = [
        RequestOutcome("u", accepted=False),
        RequestOutcome("n", accepted=True, delivery_time=10.0),
    ]
    assert evaluate(instance, outcomes, DeadlineMode.HARD).primary == pytest.approx(2.0)


def test_soft_objective_is_weighted_tardiness() -> None:
    # H = 100; non urgent deadline 40 -> urgent weight = 1 + (100 - 40) = 61.
    requests = (_request("u", U, 50.0), _request("n", N, 40.0))
    instance = _instance(requests, horizon=100.0)
    outcomes = [
        RequestOutcome("u", accepted=True, delivery_time=60.0),  # tardiness 10
        RequestOutcome("n", accepted=True, delivery_time=50.0),  # tardiness 10
    ]
    expected = 61.0 * 10.0 + 1.0 * 10.0
    assert evaluate(instance, outcomes, DeadlineMode.SOFT).primary == pytest.approx(
        expected
    )


def test_evaluate_rejects_unknown_request() -> None:
    instance = _instance((_request("n", N, 50.0),))
    outcomes = [RequestOutcome("ghost", False)]
    with pytest.raises(InstanceError):
        evaluate(instance, outcomes, DeadlineMode.HARD)


def test_response_term_is_kept_apart_from_the_primary_term() -> None:
    requests = (_request("u", U, 50.0), _request("n", N, 50.0))
    instance = _instance(requests)
    outcomes = [
        RequestOutcome("u", accepted=True, delivery_time=10.0),
        RequestOutcome("n", accepted=True, delivery_time=30.0),
    ]
    value = evaluate(instance, outcomes, DeadlineMode.HARD)
    # no rejections: primary 0; response (10 - 0) + (30 - 0) = 40
    assert value.primary == pytest.approx(0.0)
    assert value.response == pytest.approx(40.0)
    assert value.total(eps=0.5) == pytest.approx(20.0)


def test_rejected_requests_contribute_no_response_time() -> None:
    requests = (_request("u", U, 50.0), _request("n", N, 50.0))
    instance = _instance(requests)
    outcomes = [
        RequestOutcome("u", accepted=False),
        RequestOutcome("n", accepted=True, delivery_time=30.0),
    ]
    value = evaluate(instance, outcomes, DeadlineMode.HARD)
    assert value.primary == pytest.approx(2.0)
    assert value.response == pytest.approx(30.0)


def test_rejection_weights_two_urgent_three_non_urgent() -> None:
    requests = [
        _request("u1", U, 500.0),
        _request("n1", N, 100.0),
        _request("n2", N, 200.0),
        _request("u2", U, 300.0),
        _request("n3", N, 400.0),
    ]
    weights = rejection_weights(requests)
    assert weights == {"u1": 4.0, "n1": 1.0, "n2": 1.0, "u2": 4.0, "n3": 1.0}


def test_one_urgent_rejection_costs_more_than_all_non_urgent_ones() -> None:
    requests = [_request("u", U, 10.0)] + [_request(f"n{i}", N, 10.0) for i in range(7)]
    weights = rejection_weights(requests)
    assert weights["u"] > sum(weights[f"n{i}"] for i in range(7))


def test_tardiness_weights_match_hand_computation() -> None:
    # H = 100; non urgent deadlines 40 and 120: slack max(0, 100-40) + max(0, 100-120) = 60.
    requests = [
        _request("u", U, 50.0),
        _request("n1", N, 40.0),
        _request("n2", N, 120.0),
    ]
    weights = tardiness_weights(requests, horizon=100.0)
    assert weights == {"u": 61.0, "n1": 1.0, "n2": 1.0}


def test_tardiness_weights_resolution_scales_the_urgent_term() -> None:
    requests = [_request("u", U, 50.0), _request("n", N, 40.0)]
    weights = tardiness_weights(requests, horizon=100.0, resolution=10.0)
    assert weights["u"] == pytest.approx(1.0 + 60.0 / 10.0)


def test_tardiness_weights_reject_non_positive_resolution() -> None:
    requests = [_request("n", N, 1.0)]
    with pytest.raises(InstanceError):
        tardiness_weights(requests, horizon=10.0, resolution=0.0)
