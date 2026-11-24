from __future__ import annotations

from sample import sample_instance

from satedge_workflow_scheduler.domain.solution import (
    RequestOutcome,
    Solution,
    SolveStatus,
)
from satedge_workflow_scheduler.scheduling.bounds import request_bounds
from satedge_workflow_scheduler.scheduling.deadline import DeadlineMode
from satedge_workflow_scheduler.scheduling.feasibility import check
from satedge_workflow_scheduler.scheduling.objective import evaluate
from satedge_workflow_scheduler.scheduling.plmm_occupancy import PlmmOccupancyInterval
from satedge_workflow_scheduler.solvers.sequential.placement.eft import EftScheduler
from satedge_workflow_scheduler.solvers.sequential.placement.ports import (
    PlanFailure,
    RequestPlan,
)
from satedge_workflow_scheduler.solvers.sequential.reservations import Reservations
from satedge_workflow_scheduler.solvers.sequential.routing.safe_interval import (
    SafeIntervalRouter,
)

LIMIT = 500.0


def _plan(
    reservations: Reservations, limit: float = LIMIT
) -> RequestPlan | PlanFailure:
    instance = sample_instance()
    request = instance.request("r1")
    bounds = request_bounds(instance, request, limit)
    scheduler = EftScheduler(SafeIntervalRouter())
    return scheduler.schedule(instance, request, bounds, reservations)


def test_a_request_with_room_everywhere_is_planned_and_feasible() -> None:
    instance = sample_instance()
    plan = _plan(Reservations())
    assert isinstance(plan, RequestPlan)
    outcomes = (
        RequestOutcome("r1", accepted=True, delivery_time=plan.delivery_time),
        RequestOutcome("r2", accepted=False, rejection_reason="not planned here"),
    )
    solution = Solution(
        plan.schedule,
        outcomes,
        SolveStatus.FEASIBLE,
        objective=evaluate(instance, outcomes, DeadlineMode.HARD),
    )
    assert not check(instance, solution, DeadlineMode.HARD)


def test_each_task_of_the_plan_runs_once_in_pipeline_order() -> None:
    plan = _plan(Reservations())
    assert isinstance(plan, RequestPlan)
    executions = sorted(plan.schedule.executions, key=lambda e: e.task_index)
    assert [e.task_index for e in executions] == [0, 1]
    assert executions[0].end <= executions[1].start + 1e-9


def test_a_full_plmm_at_the_origin_fails_the_request_with_its_reason() -> None:
    # S1 holds 9 Gb for the whole horizon: the 2 Gb of the raw data of r1 cannot be kept there
    blocking = PlmmOccupancyInterval("other", 0, 9.0, start=0.0, end=LIMIT)
    reservations = Reservations().with_occupancies([("S1", blocking)])
    plan = _plan(reservations)
    assert isinstance(plan, PlanFailure)
    assert (
        plan.reason
        == "task 1: nessuno spazio nella PLMM per il suo input lungo il percorso"
    )


def test_without_backtracks_the_search_is_the_greedy_placement() -> None:
    instance = sample_instance()
    request = instance.request("r1")
    bounds = request_bounds(instance, request, LIMIT)
    greedy = EftScheduler(SafeIntervalRouter(), max_backtracks=0).schedule(
        instance, request, bounds, Reservations()
    )
    searching = EftScheduler(SafeIntervalRouter(), max_backtracks=5).schedule(
        instance, request, bounds, Reservations()
    )
    assert isinstance(greedy, RequestPlan)
    assert isinstance(searching, RequestPlan)
    assert searching.schedule == greedy.schedule
    assert searching.delivery_time == greedy.delivery_time
