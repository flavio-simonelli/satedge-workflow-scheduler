from __future__ import annotations

import pytest

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
from satedge_workflow_scheduler.domain.window import TimeWindow
from satedge_workflow_scheduler.scheduling.deadline import DeadlineMode
from satedge_workflow_scheduler.scheduling.objective import evaluate, tardiness

U, N = Priority.URGENT, Priority.NON_URGENT


def _request(rid: str, priority: Priority, deadline: float) -> Request:
    return Request(
        id=rid,
        priority=priority,
        origin="S1",
        release=0.0,
        deadline=deadline,
        raw_size=1.0,
        tasks=(Task(duration=10.0, output_size=1.0),),
    )


def _instance(requests: tuple[Request, ...], horizon: float = 100.0) -> ProblemInstance:
    sat = Satellite(
        id="S1",
        plmm_capacity=10.0,
        hpc_write_rate=1.0,
        hpc_windows=(TimeWindow(0.0, 1000.0),),
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
    return ProblemInstance(network=network, requests=requests, horizon=horizon)


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
    assert evaluate(instance, outcomes, DeadlineMode.HARD) == pytest.approx(2.0)


def test_hard_objective_of_rejected_urgent_request() -> None:
    requests = (_request("u", U, 50.0), _request("n", N, 50.0))
    instance = _instance(requests)
    outcomes = [
        RequestOutcome("u", accepted=False),
        RequestOutcome("n", accepted=True, delivery_time=10.0),
    ]
    assert evaluate(instance, outcomes, DeadlineMode.HARD) == pytest.approx(2.0)


def test_soft_objective_is_weighted_tardiness() -> None:
    # H = 100; non urgent deadline 40 -> urgent weight = 1 + (100 - 40) = 61.
    requests = (_request("u", U, 50.0), _request("n", N, 40.0))
    instance = _instance(requests, horizon=100.0)
    outcomes = [
        RequestOutcome("u", accepted=True, delivery_time=60.0),  # tardiness 10
        RequestOutcome("n", accepted=True, delivery_time=50.0),  # tardiness 10
    ]
    expected = 61.0 * 10.0 + 1.0 * 10.0
    assert evaluate(instance, outcomes, DeadlineMode.SOFT) == pytest.approx(expected)


def test_evaluate_rejects_unknown_request() -> None:
    instance = _instance((_request("n", N, 50.0),))
    with pytest.raises(InstanceError):
        evaluate(instance, [RequestOutcome("ghost", False)], DeadlineMode.HARD)
