"""Tests of the online state: the committed calendar is fixed, and the plan has a release."""

from __future__ import annotations

import math
from typing import Any

import pytest
from sample import Built, build, online_instance

from satedge_workflow_scheduler.domain.instance import ProblemInstance
from satedge_workflow_scheduler.domain.solution import Solution
from satedge_workflow_scheduler.scheduling.deadline import DeadlineMode
from satedge_workflow_scheduler.solvers.errors import SolverError
from satedge_workflow_scheduler.solvers.milp.backends.fake_recording import (
    FakeRecordingBackend,
)
from satedge_workflow_scheduler.solvers.milp.blocks.base import ConstraintBlock
from satedge_workflow_scheduler.solvers.milp.blocks.link_window import LinkWindowBlock
from satedge_workflow_scheduler.solvers.milp.blocks.online import (
    CommittedTasksBlock,
    CommittedTransfersBlock,
)
from satedge_workflow_scheduler.solvers.milp.blocks.path import PathBlock
from satedge_workflow_scheduler.solvers.milp.blocks.store_and_forward import (
    StoreAndForwardBlock,
    TransmissionStarts,
)
from satedge_workflow_scheduler.solvers.milp.config import Level, MilpConfig
from satedge_workflow_scheduler.solvers.milp.expr import Sense
from satedge_workflow_scheduler.solvers.milp.model_data import DataKey, TaskKey
from satedge_workflow_scheduler.solvers.milp.solver import MilpSolver


def _built(release_s1: float = 20.0) -> Built:
    instance = online_instance({"S1": release_s1, "S2": 0.0})
    blocks: list[ConstraintBlock[Any]] = [
        PathBlock(),
        StoreAndForwardBlock(),
        LinkWindowBlock(),
        CommittedTasksBlock(),
        CommittedTransfersBlock(),
    ]
    return build(blocks, instance=instance)


def test_the_committed_task_is_fixed_on_its_satellite_window_and_start() -> None:
    built = _built()
    v = built.ctx.variables
    task = TaskKey("r2", 0)
    assert built.row(f"committed-u[{task},S2,0]").rhs == pytest.approx(1.0)
    assert built.row(f"committed-u[{task},S1,0]").rhs == pytest.approx(0.0)
    start = built.row(f"committed-t[{task}]")
    assert start.sense is Sense.EQ
    assert start.terms == {v.t(task): 1.0}
    assert start.rhs == pytest.approx(0.0)


def test_the_committed_transmission_is_fixed_on_its_arc_and_window() -> None:
    built = _built()
    starts = built.model.families.get(TransmissionStarts)
    output = DataKey("r2", 1)
    # arc 2 is S2 → G, with one window: ν = 1 there, and ξ = 8
    assert built.row(f"committed-nu[{output},2,0]").rhs == pytest.approx(1.0)
    xi = built.row(f"committed-xi[{output},2]")
    assert xi.terms == {starts.xi(output, 2): 1.0}
    assert xi.rhs == pytest.approx(8.0)
    # the arcs it does not cross have χ = 0 and ξ = 0
    assert built.row(f"committed-chi[{output},0]").rhs == pytest.approx(0.0)
    assert built.row(f"committed-xi[{output},0]").rhs == pytest.approx(0.0)


def test_a_new_task_starts_after_the_release_of_the_plan() -> None:
    built = _built()
    v = built.ctx.variables
    task = TaskKey("r1", 0)
    row = built.row(f"release-task[{task}]")
    # (30) t − 20 x_{S1} ≥ 0: the plan reaches S1 at 20 and S2 at 0
    assert row.sense is Sense.GE
    assert row.terms[v.t(task)] == pytest.approx(1.0)
    assert row.terms[v.u(task, "S1", 0)] == pytest.approx(-20.0)
    assert v.u(task, "S2", 0) not in row.terms


def test_a_satellite_the_plan_never_reaches_is_not_used_by_new_requests() -> None:
    built = _built(release_s1=math.inf)
    row = built.row(f"unreached-task[{TaskKey('r1', 0)},S1]")
    assert row.sense is Sense.EQ and row.rhs == pytest.approx(0.0)
    # no new data leaves S1: arc 0 is S1 → S2
    assert built.row(f"unreached-arc[{DataKey('r1', 0)},0]").rhs == pytest.approx(0.0)


def test_the_solver_adds_the_online_blocks_when_the_instance_is_online() -> None:
    instance = online_instance({"S1": 20.0, "S2": 0.0})
    backend = FakeRecordingBackend()
    with pytest.raises(SolverError):
        # the fake backend does not solve: what matters are the rows it received
        MilpSolver(MilpConfig(level=Level.ROUTING), lambda: backend).solve(instance)
    names = {row.name.split("[")[0] for row in backend.rows}
    assert {"committed-t", "committed-xi", "release-task"} <= names


# --- with Gurobi ------------------------------------------------------------------------


def _solve(
    release_s1: float, level: Level = Level.PLMM_CAPACITY
) -> tuple[ProblemInstance, Solution]:
    pytest.importorskip("gurobipy")
    from satedge_workflow_scheduler.solvers.milp.backends.gurobi import GurobiBackend

    instance = online_instance({"S1": release_s1, "S2": 0.0})
    config = MilpConfig(level=level, deadline=DeadlineMode.HARD)
    return instance, MilpSolver(config, GurobiBackend).solve(instance)


def test_the_online_level_keeps_the_committed_calendar_and_passes_the_checker() -> None:
    from satedge_workflow_scheduler.scheduling.feasibility import check

    instance, solution = _solve(20.0)
    violations = check(instance, solution, DeadlineMode.HARD)
    assert violations == (), [str(v) for v in violations]
    by_id = {o.request_id: o for o in solution.outcomes}
    # r2 is committed and delivered at 8 + 1 + 5 = 14; r1 waits for the plan on S1 until 20:
    # tasks on S1 in [20, 35] and [35, 59], then S1 → S2 at 59 and S2 → G at 61, so 66.5
    assert by_id["r2"].delivery_time == pytest.approx(14.0)
    assert by_id["r1"].delivery_time == pytest.approx(66.5)


def test_a_request_whose_origin_the_plan_never_reaches_is_rejected() -> None:
    from satedge_workflow_scheduler.scheduling.feasibility import check

    instance, solution = _solve(math.inf)
    assert check(instance, solution, DeadlineMode.HARD) == ()
    by_id = {o.request_id: o for o in solution.outcomes}
    assert not by_id["r1"].accepted
    assert by_id["r2"].accepted


@pytest.mark.parametrize("level", [Level.ROUTING, Level.LINK_CONTENTION])
def test_the_online_state_works_at_the_lower_levels_too(level: Level) -> None:
    from satedge_workflow_scheduler.scheduling.feasibility import check

    instance, solution = _solve(20.0, level)
    assert check(instance, solution, DeadlineMode.HARD) == ()
    by_id = {o.request_id: o for o in solution.outcomes}
    assert by_id["r1"].delivery_time == pytest.approx(66.5)
