from __future__ import annotations

import dataclasses

from satedge_workflow_scheduler.common.intervals import TimeWindow, WindowSet
from satedge_workflow_scheduler.domain.instance import ProblemInstance
from satedge_workflow_scheduler.domain.network import (
    AlwaysAvailable,
    Arc,
    Network,
    Plane,
    WindowedAvailability,
)
from satedge_workflow_scheduler.domain.nodes import GroundStation, NodeId, Satellite
from satedge_workflow_scheduler.domain.requests import Priority, Request, Task
from satedge_workflow_scheduler.domain.solution import (
    ObjectiveValue,
    RequestOutcome,
    Schedule,
    Solution,
    SolveStatus,
    TaskExecution,
    Transfer,
)
from satedge_workflow_scheduler.scheduling.deadline import DeadlineMode
from satedge_workflow_scheduler.scheduling.feasibility import Constraint, check

# S1 <-> S2 always available, S2 -> G only in [100, 200]; 1 Gbps, latency 1 s.
# HPC at 1 Gbps: S1 computes in [0, 50], S2 in [20, 60]. PLMM of 10 Gb.
# Request "r": raw 2 Gb on S1 from 0, one task of 10 s with a 1 Gb output: 13 s on an HPC.
#
# The feasible calendar used by the tests:
#   task on S1 in [0, 13]; the output is written from 12;
#   output S1 -> S2 in [13, 14], arrives at 15;
#   output S2 -> G in [100, 101], arrives at 102: delivered at 102.

TASK = TaskExecution("r", 0, "S1", 0.0, 13.0)
HOP_1 = Transfer("r", 1, "S1", "S2", 13.0, 14.0)
HOP_2 = Transfer("r", 1, "S2", "G", 100.0, 101.0)


def _request(
    rid: str = "r", release: float = 0.0, deadline: float = 150.0, tasks: int = 1
) -> Request:
    return Request(
        id=rid,
        priority=Priority.URGENT,
        origin="S1",
        release=release,
        deadline=deadline,
        raw_size=2.0,
        tasks=tuple(Task(execution_time=10.0, output_size=1.0) for _ in range(tasks)),
    )


def _instance(
    *requests: Request,
    plmm: float = 10.0,
    plan_release: dict[NodeId, float] | None = None,
    committed: Schedule | None = None,
) -> ProblemInstance:
    s1 = Satellite("S1", plmm, 1.0, WindowSet((TimeWindow(0.0, 50.0),)))
    s2 = Satellite("S2", plmm, 1.0, WindowSet((TimeWindow(20.0, 60.0),)))
    contact = WindowedAvailability(WindowSet((TimeWindow(100.0, 200.0),)))
    network = Network(
        planes=(Plane(1, (s1,), "S1"), Plane(2, (s2,), "S2")),
        ground_station=GroundStation("G"),
        arcs=(
            Arc("S1", "S2", 1.0, 1.0, AlwaysAvailable()),
            Arc("S2", "S1", 1.0, 1.0, AlwaysAvailable()),
            Arc("S2", "G", 1.0, 1.0, contact),
        ),
    )
    return ProblemInstance(
        network=network,
        requests=requests or (_request(),),
        horizon=300.0,
        plan_release={
            **{s: 0.0 for s in network.satellite_ids},
            **(plan_release or {}),
        },
        committed=committed or Schedule(),
    )


def _solution(
    executions: tuple[TaskExecution, ...] = (TASK,),
    transfers: tuple[Transfer, ...] = (HOP_1, HOP_2),
    outcomes: tuple[RequestOutcome, ...] = (
        RequestOutcome("r", accepted=True, delivery_time=102.0),
    ),
) -> Solution:
    return Solution(
        schedule=Schedule(executions, transfers),
        outcomes=outcomes,
        status=SolveStatus.FEASIBLE,
    )


def _violated(
    instance: ProblemInstance,
    solution: Solution,
    mode: DeadlineMode = DeadlineMode.HARD,
) -> set[Constraint]:
    return {v.constraint for v in check(instance, solution, mode)}


def test_a_feasible_calendar_has_no_violation() -> None:
    assert check(_instance(), _solution(), DeadlineMode.HARD) == ()


def test_a_rejected_request_is_allowed_only_with_hard_deadlines() -> None:
    rejected = _solution((), (), (RequestOutcome("r", accepted=False),))
    assert _violated(_instance(), rejected) == set()
    assert _violated(_instance(), rejected, DeadlineMode.SOFT) == {Constraint.OUTCOMES}


def test_every_request_needs_an_outcome() -> None:
    assert Constraint.OUTCOMES in _violated(_instance(), _solution((), (), ()))


def test_a_task_lasts_its_hpc_duration() -> None:
    short = dataclasses.replace(TASK, end=10.0)
    assert Constraint.TASK_DURATION in _violated(_instance(), _solution((short,)))


def test_a_task_lies_in_a_computing_window() -> None:
    # S1 computes until 50
    late = TaskExecution("r", 0, "S1", 40.0, 53.0)
    hops = (Transfer("r", 1, "S1", "S2", 53.0, 54.0), HOP_2)
    assert _violated(_instance(), _solution((late,), hops)) == {
        Constraint.COMPUTING_WINDOW
    }


def test_the_first_task_starts_after_the_release() -> None:
    instance = _instance(_request(release=5.0))
    assert _violated(instance, _solution()) == {Constraint.RELEASE}


def test_a_task_starts_after_the_previous_one() -> None:
    # two tasks on S1: the second lasts 1 + 10 + 1 = 12 s and starts before the first ends
    instance = _instance(_request(tasks=2))
    tasks = (TASK, TaskExecution("r", 1, "S1", 12.0, 24.0))
    hops = (Transfer("r", 2, "S1", "S2", 24.0, 25.0), HOP_2)
    violated = _violated(instance, _solution(tasks, hops))
    assert Constraint.PRECEDENCE in violated


def test_two_tasks_do_not_overlap_on_an_hpc() -> None:
    other = TaskExecution("o", 0, "S1", 10.0, 23.0)
    other_hops = (
        Transfer("o", 1, "S1", "S2", 23.0, 24.0),
        Transfer("o", 1, "S2", "G", 110.0, 111.0),
    )
    solution = _solution(
        (TASK, other),
        (HOP_1, HOP_2, *other_hops),
        (
            RequestOutcome("r", accepted=True, delivery_time=102.0),
            RequestOutcome("o", accepted=True, delivery_time=112.0),
        ),
    )
    instance = _instance(_request(), _request("o"))
    assert _violated(instance, solution) == {Constraint.HPC_OVERLAP}


def test_the_data_follow_a_path_to_their_consumer() -> None:
    # the output jumps from S1 to the ground without passing through S2
    assert Constraint.PATH in _violated(_instance(), _solution(transfers=(HOP_2,)))


def test_the_final_output_reaches_the_ground() -> None:
    assert Constraint.PATH in _violated(_instance(), _solution(transfers=(HOP_1,)))


def test_a_transfer_lasts_its_transmission_time() -> None:
    slow = dataclasses.replace(HOP_1, end=15.0)
    assert Constraint.TRANSMISSION_TIME in _violated(
        _instance(), _solution(transfers=(slow, HOP_2))
    )


def test_the_output_leaves_after_its_producer() -> None:
    early = Transfer("r", 1, "S1", "S2", 12.0, 13.0)
    assert _violated(_instance(), _solution(transfers=(early, HOP_2))) == {
        Constraint.AFTER_PRODUCER
    }


def test_the_raw_data_leaves_after_its_release() -> None:
    # the task runs on S2 in [20, 33]: the raw is sent at 0 although it exists from 5
    instance = _instance(_request(release=5.0))
    task = TaskExecution("r", 0, "S2", 20.0, 33.0)
    hops = (
        Transfer("r", 0, "S1", "S2", 0.0, 2.0),
        Transfer("r", 1, "S2", "G", 100.0, 101.0),
    )
    assert _violated(instance, _solution((task,), hops)) == {Constraint.AFTER_RELEASE}


def test_a_node_forwards_only_what_it_has_received() -> None:
    # the output arrives at S2 at 15 and cannot leave at 14
    instance = _instance()
    hop_2 = Transfer("r", 1, "S2", "G", 14.0, 15.0)
    solution = _solution(
        transfers=(HOP_1, hop_2),
        outcomes=(RequestOutcome("r", accepted=True, delivery_time=16.0),),
    )
    assert Constraint.STORE_AND_FORWARD in _violated(instance, solution)


def test_the_consumer_starts_after_its_input_arrives() -> None:
    # the raw arrives at S2 at 21, the task starts there at 20
    task = TaskExecution("r", 0, "S2", 20.0, 33.0)
    hops = (
        Transfer("r", 0, "S1", "S2", 18.0, 20.0),
        Transfer("r", 1, "S2", "G", 100.0, 101.0),
    )
    assert _violated(_instance(), _solution((task,), hops)) == {Constraint.ARRIVAL}


def test_the_output_reaches_the_ground_by_the_deadline() -> None:
    instance = _instance(_request(deadline=101.0))
    assert _violated(instance, _solution()) == {Constraint.LIMIT}
    # with soft deadlines the limit is the horizon of 300
    assert _violated(instance, _solution(), DeadlineMode.SOFT) == set()


def test_the_delivery_time_is_the_arrival_on_the_ground() -> None:
    wrong = (RequestOutcome("r", accepted=True, delivery_time=103.0),)
    assert _violated(_instance(), _solution(outcomes=wrong)) == {
        Constraint.DELIVERY_TIME
    }


def test_a_transfer_on_a_contact_lies_in_a_visibility_window() -> None:
    # the contact to the ground opens at 100
    early = Transfer("r", 1, "S2", "G", 99.0, 100.0)
    solution = _solution(
        transfers=(HOP_1, early),
        outcomes=(RequestOutcome("r", accepted=True, delivery_time=101.0),),
    )
    assert _violated(_instance(), solution) == {Constraint.VISIBILITY_WINDOW}


def test_two_transfers_do_not_overlap_on_an_arc() -> None:
    other = TaskExecution("o", 0, "S1", 13.0, 26.0)
    other_hops = (
        Transfer("o", 1, "S1", "S2", 26.0, 27.0),
        Transfer("o", 1, "S2", "G", 100.5, 101.5),
    )
    solution = _solution(
        (TASK, other),
        (HOP_1, HOP_2, *other_hops),
        (
            RequestOutcome("r", accepted=True, delivery_time=102.0),
            RequestOutcome("o", accepted=True, delivery_time=102.5),
        ),
    )
    instance = _instance(_request(), _request("o"))
    assert _violated(instance, solution) == {Constraint.LINK_CONTENTION}


def test_the_plmm_does_not_overflow() -> None:
    # On S1 the raw (2 Gb) stays until the task ends at 13, and the output (1 Gb) is
    # written from 12: 3 Gb in [12, 13], over a PLMM of 2.5 Gb.
    assert _violated(_instance(plmm=2.5), _solution()) == {Constraint.PLMM_CAPACITY}


def test_the_committed_calendar_is_kept_unchanged() -> None:
    committed = Schedule((TASK,), (HOP_1, HOP_2))
    instance = _instance(committed=committed)
    assert _violated(instance, _solution()) == set()
    moved = (TASK, HOP_1, Transfer("r", 1, "S2", "G", 110.0, 111.0))
    changed = _solution(
        transfers=moved[1:],
        outcomes=(RequestOutcome("r", accepted=True, delivery_time=112.0),),
    )
    assert _violated(instance, changed) == {Constraint.COMMITTED}


def test_new_requests_start_after_the_plan_reaches_the_node() -> None:
    instance = _instance(plan_release={"S1": 5.0})
    assert Constraint.PLAN_RELEASE in _violated(instance, _solution())


def test_committed_requests_ignore_the_plan_release() -> None:
    committed = Schedule((TASK,), (HOP_1, HOP_2))
    instance = _instance(plan_release={"S1": 5.0}, committed=committed)
    assert _violated(instance, _solution()) == set()


def test_a_violation_reads_as_constraint_request_and_message() -> None:
    (violation,) = check(
        _instance(_request(release=5.0)), _solution(), DeadlineMode.HARD
    )
    assert str(violation) == (
        "(4) release of the raw data: r: task 0 starts at 0.0, before the release 5.0"
    )


def test_a_task_outside_the_pipeline_is_reported_not_raised() -> None:
    extra = TaskExecution("r", 5, "S1", 20.0, 33.0)
    assert Constraint.ASSIGNMENT in _violated(_instance(), _solution((TASK, extra)))


def test_a_data_item_outside_the_pipeline_is_reported_not_raised() -> None:
    extra = Transfer("r", 7, "S1", "S2", 30.0, 31.0)
    assert Constraint.PATH in _violated(
        _instance(), _solution(transfers=(HOP_1, HOP_2, extra))
    )


def test_an_unknown_arc_is_reported_not_raised() -> None:
    # S1 has no arc to the ground: the PLMM cannot be checked, but nothing raises
    direct = Transfer("r", 1, "S1", "G", 100.0, 101.0)
    assert Constraint.PATH in _violated(_instance(), _solution(transfers=(direct,)))


def test_a_task_on_the_ground_station_is_reported_not_raised() -> None:
    on_ground = TaskExecution("r", 0, "G", 0.0, 13.0)
    assert Constraint.ASSIGNMENT in _violated(_instance(), _solution((on_ground,), ()))


def test_the_reported_objective_is_the_one_of_the_outcomes() -> None:
    # accepted and delivered at 102 from a release at 0: no rejection, response 102
    right = dataclasses.replace(_solution(), objective=ObjectiveValue(0.0, 102.0))
    assert _violated(_instance(), right) == set()
    wrong = dataclasses.replace(_solution(), objective=ObjectiveValue(0.0, 90.0))
    assert _violated(_instance(), wrong) == {Constraint.OBJECTIVE}
