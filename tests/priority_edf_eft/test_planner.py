from __future__ import annotations

import pytest

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
from satedge_workflow_scheduler.domain.solution import TaskExecution, Transfer
from satedge_workflow_scheduler.solvers.priority_edf_eft.ledger import (
    HpcKey,
    ResourceLedger,
)
from satedge_workflow_scheduler.solvers.priority_edf_eft.placement import EftPlacement
from satedge_workflow_scheduler.solvers.priority_edf_eft.planner import (
    PlanFailure,
    RequestPlan,
    RequestPlanner,
)
from satedge_workflow_scheduler.solvers.priority_edf_eft.routing import (
    EarliestArrivalRouter,
)

EPS = 1e-9


def _arc(source: str, target: str) -> Arc:
    # capacity 1 Gbps, latency 1 s
    return Arc(
        source=source,
        target=target,
        capacity=1.0,
        latency=1.0,
        availability=AlwaysAvailable(),
    )


def _setup(
    s1_windows: tuple[TimeWindow, ...] = (TimeWindow(0.0, 1000.0),),
    s2_capacity: float = 10.0,
    limit: float = 500.0,
) -> tuple[RequestPlanner, ResourceLedger, Request, ProblemInstance]:
    s1 = Satellite(
        id="S1", plmm_capacity=10.0, hpc_write_rate=1.0, hpc_windows=s1_windows
    )
    s2 = Satellite(
        id="S2",
        plmm_capacity=s2_capacity,
        hpc_write_rate=1.0,
        hpc_windows=(TimeWindow(0.0, 1000.0),),
    )
    network = Network(
        planes=(
            Plane(id=1, satellites=(s1,), com="S1"),
            Plane(id=2, satellites=(s2,), com="S2"),
        ),
        ground_station=GroundStation(id="G"),
        arcs=(_arc("S1", "S2"), _arc("S2", "G")),
    )
    request = Request(
        id="r",
        priority=Priority.URGENT,
        origin="S1",
        release=0.0,
        deadline=limit,
        raw_size=2.0,
        tasks=(Task(duration=10.0, output_size=3.0),),
    )
    instance = ProblemInstance(network=network, requests=(request,), horizon=1000.0)
    ledger = ResourceLedger(network)
    router = EarliestArrivalRouter(network, ledger, EPS)
    placement = EftPlacement(network, ledger, router, EPS)
    planner = RequestPlanner(instance, ledger, placement, router, EPS)
    return planner, ledger, request, instance


def test_planned_request_has_its_executions_transfers_and_delivery() -> None:
    # S1 has no HPC window, so the task runs on S2: the raw arrives at 3 (2 s + 1 s),
    # the task runs 3..13, and the 3 Gb output leaves at 13 and arrives at 17.
    planner, ledger, request, _ = _setup(s1_windows=())
    with ledger.transaction() as tx:
        result = planner.plan(
            request, limit=500.0, not_before=None, tx=tx, extra_plmm={}
        )
        tx.commit()

    assert isinstance(result, RequestPlan)
    assert result.executions == (TaskExecution("r", 0, "S2", 3.0, 13.0),)
    assert result.transfers == (
        Transfer("r", 0, "S1", "S2", 0.0, 2.0),
        Transfer("r", 1, "S2", "G", 13.0, 16.0),
    )
    assert result.delivery_time == pytest.approx(17.0)


def test_committed_request_reserves_the_hpc_and_the_plmm() -> None:
    planner, ledger, request, _ = _setup(s1_windows=())
    with ledger.transaction() as tx:
        planner.plan(request, limit=500.0, not_before=None, tx=tx, extra_plmm={})
        tx.commit()

    assert ledger.timeline(HpcKey("S2")).windows == (TimeWindow(3.0, 13.0),)
    assert len(ledger.plmm("S2")) == 2


def test_no_placement_before_the_limit_is_a_failure_not_caused_by_plmm() -> None:
    planner, ledger, request, _ = _setup(s1_windows=())
    with ledger.transaction() as tx:
        result = planner.plan(request, limit=5.0, not_before=None, tx=tx, extra_plmm={})
    assert result == PlanFailure("no feasible placement before the limit", plmm=False)


def test_delivery_after_the_limit_is_a_failure() -> None:
    planner, ledger, request, _ = _setup(s1_windows=())
    with ledger.transaction() as tx:
        result = planner.plan(
            request, limit=15.0, not_before=None, tx=tx, extra_plmm={}
        )
    assert result == PlanFailure(
        "no delivery to the ground station before the limit", plmm=False
    )


def test_plmm_overflow_is_a_failure_and_leaves_no_trace() -> None:
    # S2 can hold only 1 Gb, but the 3 Gb output must be written there.
    planner, ledger, request, _ = _setup(s1_windows=(), s2_capacity=1.0)
    with ledger.transaction() as tx:
        result = planner.plan(
            request, limit=500.0, not_before=None, tx=tx, extra_plmm={}
        )
    assert result == PlanFailure("PLMM capacity exceeded", plmm=True)
    assert ledger.timeline(HpcKey("S2")).windows == ()
    assert ledger.plmm("S2") == ()
