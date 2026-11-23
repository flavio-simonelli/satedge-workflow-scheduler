"""Tests of the polish: it fixes the binaries, keeps the optimum, and never makes a solution worse."""

from __future__ import annotations

import pytest
from sample import build

from satedge_workflow_scheduler.solvers.milp.backends.fake_recording import (
    FakeRecordingBackend,
)
from satedge_workflow_scheduler.solvers.milp.backends.ports import (
    BackendStatus,
    SolveParams,
    SolveResult,
    VarKind,
)
from satedge_workflow_scheduler.solvers.milp.blocks.path import PathBlock
from satedge_workflow_scheduler.solvers.milp.polish import polish


def test_the_builder_records_every_binary_of_the_model() -> None:
    built = build([PathBlock()])
    binaries = set(built.model.binaries)
    assert binaries == {
        v.handle for v in built.backend.variables if v.kind is VarKind.BINARY
    }
    # ρ: 2; u: 3 tasks × 2 satellites; χ on the 2 arcs always available: 5 × 2; ν on the 1 windowed
    # arc: 5 × 1
    assert len(binaries) == 2 + 6 + 10 + 5


def test_set_bounds_changes_a_recorded_variable() -> None:
    backend = FakeRecordingBackend()
    var = backend.add_var(0.0, 1.0, VarKind.BINARY, "b")
    backend.set_bounds(var, 1.0, 1.0)
    assert (backend.variables[var].lb, backend.variables[var].ub) == (1.0, 1.0)


def test_the_polish_fixes_the_binaries_to_their_rounded_values() -> None:
    backend = FakeRecordingBackend()
    x = backend.add_var(0.0, 1.0, VarKind.BINARY, "x")
    y = backend.add_var(0.0, 1.0, VarKind.BINARY, "y")
    backend.scripted = SolveResult(
        BackendStatus.OPTIMAL, {x: 1.0, y: 0.0}, objective=2.0
    )
    first = SolveResult(BackendStatus.OPTIMAL, {x: 0.9999999, y: 1e-7}, objective=2.0)
    result = polish(backend, [x, y], first, SolveParams())
    assert result.status is BackendStatus.OPTIMAL
    assert (backend.variables[x].lb, backend.variables[x].ub) == (1.0, 1.0)
    assert (backend.variables[y].lb, backend.variables[y].ub) == (0.0, 0.0)


def test_a_polish_that_gets_worse_is_not_accepted() -> None:
    backend = FakeRecordingBackend()
    x = backend.add_var(0.0, 1.0, VarKind.BINARY, "x")
    backend.scripted = SolveResult(BackendStatus.OPTIMAL, {x: 1.0}, objective=5.0)
    first = SolveResult(BackendStatus.OPTIMAL, {x: 1.0}, objective=2.0)
    assert polish(backend, [x], first, SolveParams()) is first


def test_a_polish_that_is_infeasible_keeps_the_first_solution() -> None:
    backend = FakeRecordingBackend()
    x = backend.add_var(0.0, 1.0, VarKind.BINARY, "x")
    backend.scripted = SolveResult(BackendStatus.INFEASIBLE)
    first = SolveResult(BackendStatus.OPTIMAL, {x: 1.0}, objective=2.0)
    assert polish(backend, [x], first, SolveParams()) is first


def test_the_polish_with_gurobi_keeps_the_optimum_of_the_routing_level() -> None:
    pytest.importorskip("gurobipy")
    from sample import sample_instance

    from satedge_workflow_scheduler.scheduling.deadline import DeadlineMode
    from satedge_workflow_scheduler.scheduling.feasibility import check
    from satedge_workflow_scheduler.solvers.milp.backends.gurobi import GurobiBackend
    from satedge_workflow_scheduler.solvers.milp.config import Level, MilpConfig
    from satedge_workflow_scheduler.solvers.milp.solver import MilpSolver

    instance = sample_instance()
    plain = MilpSolver(MilpConfig(level=Level.ROUTING), GurobiBackend).solve(instance)
    polished = MilpSolver(
        MilpConfig(level=Level.ROUTING, polish=True), GurobiBackend
    ).solve(instance)
    assert check(instance, polished, DeadlineMode.HARD) == ()
    assert plain.objective is not None and polished.objective is not None
    assert polished.objective.response == pytest.approx(plain.objective.response)
