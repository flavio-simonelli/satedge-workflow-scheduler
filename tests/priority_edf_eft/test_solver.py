from __future__ import annotations

from collections.abc import Callable

import pytest

from satedge_workflow_scheduler.common.errors import InfeasibleModelError
from satedge_workflow_scheduler.common.intervals import TimeWindow
from satedge_workflow_scheduler.domain.instance import ProblemInstance
from satedge_workflow_scheduler.domain.network import (
    AlwaysAvailable,
    Arc,
    Network,
    Plane,
)
from satedge_workflow_scheduler.domain.nodes import GroundStation, Satellite
from satedge_workflow_scheduler.domain.requests import Priority, Request, Task
from satedge_workflow_scheduler.domain.solution import Solution
from satedge_workflow_scheduler.scheduling.deadline import DeadlineMode
from satedge_workflow_scheduler.solvers.priority_edf_eft import PriorityEdfEftSolver

U, N = Priority.URGENT, Priority.NON_URGENT

Checker = Callable[[ProblemInstance, Solution], None]


def _arc(source: str, target: str) -> Arc:
    # capacity 1 Gbps, latency 1 s
    return Arc(
        source=source,
        target=target,
        capacity=1.0,
        latency=1.0,
        availability=AlwaysAvailable(),
    )


def _request(
    rid: str,
    priority: Priority,
    deadline: float,
    release: float = 0.0,
    raw: float = 2.0,
    out: float = 3.0,
    duration: float = 10.0,
) -> Request:
    return Request(
        id=rid,
        priority=priority,
        origin="S1",
        release=release,
        deadline=deadline,
        raw_size=raw,
        tasks=(Task(duration=duration, output_size=out),),
    )


def _instance(
    requests: tuple[Request, ...],
    s2_windows: tuple[TimeWindow, ...] = (TimeWindow(0.0, 1000.0),),
    s1_windows: tuple[TimeWindow, ...] = (),
    horizon: float = 500.0,
    s2_capacity: float = 10.0,
    s1_capacity: float = 10.0,
) -> ProblemInstance:
    s1 = Satellite(
        id="S1", plmm_capacity=s1_capacity, hpc_write_rate=1.0, hpc_windows=s1_windows
    )
    s2 = Satellite(
        id="S2", plmm_capacity=s2_capacity, hpc_write_rate=1.0, hpc_windows=s2_windows
    )
    network = Network(
        planes=(
            Plane(id=1, satellites=(s1,), com="S1"),
            Plane(id=2, satellites=(s2,), com="S2"),
        ),
        ground_station=GroundStation(id="G"),
        arcs=(_arc("S1", "S2"), _arc("S2", "G")),
    )
    return ProblemInstance(network=network, requests=requests, horizon=horizon)


def _solve(instance: ProblemInstance, mode: DeadlineMode) -> Solution:
    return PriorityEdfEftSolver(mode).solve(instance)


def test_hard_mode_delivers_a_feasible_request(check_valid: Checker) -> None:
    # raw arrives at S2 at 3, the task runs 3..13, the output arrives at G at 17
    instance = _instance((_request("r", U, deadline=500.0),))
    solution = _solve(instance, DeadlineMode.HARD)

    assert solution.outcome("r").accepted
    assert solution.outcome("r").delivery_time == pytest.approx(17.0)
    assert solution.objective is not None
    assert solution.objective.primary == pytest.approx(0.0)
    check_valid(instance, solution)


def test_hard_mode_rejects_an_impossible_request_without_disturbing_the_others(
    check_valid: Checker,
) -> None:
    feasible = _request("a", U, deadline=500.0)
    impossible = _request("b", N, deadline=5.0)
    alone = _solve(_instance((feasible,)), DeadlineMode.HARD)
    together = _solve(_instance((feasible, impossible)), DeadlineMode.HARD)

    assert not together.outcome("b").accepted
    assert together.outcome("a") == alone.outcome("a")
    assert [e for e in together.executions if e.request_id == "a"] == list(
        alone.executions
    )
    assert [t for t in together.transfers if t.request_id == "a"] == list(
        alone.transfers
    )
    # rejecting the only non urgent request costs its κ, which is 1
    assert together.objective is not None
    assert together.objective.primary == pytest.approx(1.0)
    check_valid(_instance((feasible, impossible)), together)


def test_soft_mode_serves_a_late_request_and_charges_its_tardiness(
    check_valid: Checker,
) -> None:
    # delivered at 17 against a deadline of 5: tardiness 12, non urgent weight 1
    instance = _instance((_request("r", N, deadline=5.0),))
    solution = _solve(instance, DeadlineMode.SOFT)

    assert solution.outcome("r").accepted
    assert solution.objective is not None
    assert solution.objective.primary == pytest.approx(12.0)
    assert solution.objective.response == pytest.approx(17.0)
    check_valid(instance, solution)


def test_soft_mode_fails_when_a_request_cannot_meet_the_horizon() -> None:
    instance = _instance((_request("r", N, deadline=5.0),), horizon=10.0)
    with pytest.raises(InfeasibleModelError) as error:
        _solve(instance, DeadlineMode.SOFT)
    assert "r" in error.value.reasons


def test_urgent_request_gets_the_hpc_before_a_non_urgent_one_with_an_earlier_deadline(
    check_valid: Checker,
) -> None:
    # S2 has one window of 15 s: only one 10 s task fits in it.
    urgent = _request("u", U, deadline=500.0)
    non_urgent = _request("n", N, deadline=100.0)
    instance = _instance(
        (non_urgent, urgent),
        s2_windows=(TimeWindow(0.0, 15.0),),
    )
    solution = _solve(instance, DeadlineMode.HARD)

    assert solution.outcome("u").accepted
    assert not solution.outcome("n").accepted
    check_valid(instance, solution)


def test_solver_keeps_no_state_between_calls() -> None:
    instance = _instance((_request("r", N, deadline=5.0),))
    solver = PriorityEdfEftSolver(DeadlineMode.SOFT)
    first = solver.solve(instance)
    second = solver.solve(instance)
    assert first.objective == second.objective
    assert first.outcomes == second.outcomes


def _solver_with_retries(max_retries: int) -> PriorityEdfEftSolver:
    return PriorityEdfEftSolver(DeadlineMode.SOFT, max_retries=max_retries)


def test_soft_mode_retries_later_when_the_plmm_is_full(check_valid: Checker) -> None:
    # S2 holds 5 Gb. The urgent request fills it between 3 and 16. The non urgent one can
    # only be planned after its data is freed, so it needs a retry at a later instant.
    urgent = _request("u", U, deadline=500.0)
    non_urgent = _request("n", N, deadline=500.0)
    instance = _instance((urgent, non_urgent), horizon=500.0, s2_capacity=5.0)
    solution = _solver_with_retries(100).solve(instance)
    assert solution.outcome("u").accepted
    assert solution.outcome("n").accepted
    check_valid(instance, solution)


def test_soft_mode_without_retries_cannot_plan_the_second_request() -> None:
    urgent = _request("u", U, deadline=500.0)
    non_urgent = _request("n", N, deadline=500.0)
    instance = _instance((urgent, non_urgent), horizon=500.0, s2_capacity=5.0)
    with pytest.raises(InfeasibleModelError):
        _solver_with_retries(0).solve(instance)


def test_soft_mode_counts_the_raw_of_pending_requests_on_their_origin(
    check_valid: Checker,
) -> None:
    # Both raw data are 2 Gb on S1 from release 0, until they can leave (2 s). Together
    # they need 4 Gb, so a PLMM of 3 Gb cannot hold them: both requests fail on the PLMM.
    urgent = _request("u", U, deadline=500.0)
    non_urgent = _request("n", N, deadline=500.0)
    instance = _instance((urgent, non_urgent), horizon=500.0, s1_capacity=3.0)
    with pytest.raises(InfeasibleModelError) as error:
        _solver_with_retries(100).solve(instance)
    assert error.value.reasons == {
        "u": "PLMM capacity exceeded",
        "n": "PLMM capacity exceeded",
    }


def test_soft_mode_plans_both_raw_data_when_the_plmm_holds_them(
    check_valid: Checker,
) -> None:
    urgent = _request("u", U, deadline=500.0)
    non_urgent = _request("n", N, deadline=500.0)
    instance = _instance((urgent, non_urgent), horizon=500.0, s1_capacity=4.0)
    solution = _solver_with_retries(100).solve(instance)
    assert solution.outcome("u").accepted
    assert solution.outcome("n").accepted
    check_valid(instance, solution)


def test_hard_mode_records_why_an_impossible_request_is_rejected() -> None:
    instance = _instance((_request("b", N, deadline=5.0),))
    solution = _solve(instance, DeadlineMode.HARD)
    assert not solution.outcome("b").accepted
    assert (
        solution.outcome("b").rejection_reason
        == "no feasible placement before the limit"
    )
