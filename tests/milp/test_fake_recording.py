"""Tests of the fake recording backend: it must keep exactly what the model declares."""

from __future__ import annotations

from pathlib import Path

import pytest

from satedge_workflow_scheduler.solvers.milp.backends.fake_recording import (
    FakeRecordingBackend,
)
from satedge_workflow_scheduler.solvers.milp.backends.ports import (
    BackendStatus,
    MilpBackend,
    ModelSink,
    SolveParams,
    SolveResult,
    VarKind,
)
from satedge_workflow_scheduler.solvers.milp.errors import ModelBuildError
from satedge_workflow_scheduler.solvers.milp.expr import LinExpr


def test_fake_recording_backend_satisfies_the_ports() -> None:
    # checked by mypy: the assignments fail to type-check if a method is missing
    backend: MilpBackend = FakeRecordingBackend()
    sink: ModelSink = backend
    assert sink is backend


def test_variables_get_consecutive_handles_and_keep_their_declaration() -> None:
    backend = FakeRecordingBackend()
    x = backend.add_var(0.0, 1.0, VarKind.BINARY, "x[1,2]")
    y = backend.add_var(0.0, 10.0, VarKind.CONTINUOUS)
    assert (x, y) == (0, 1)
    assert backend.variables[0].name == "x[1,2]"
    assert backend.variables[0].kind is VarKind.BINARY
    assert (backend.variables[1].lb, backend.variables[1].ub) == (0.0, 10.0)


def test_reversed_bounds_are_rejected() -> None:
    with pytest.raises(ModelBuildError, match="lower bound"):
        FakeRecordingBackend().add_var(2.0, 1.0, VarKind.CONTINUOUS)


def test_rows_are_kept_in_order_with_their_coefficients() -> None:
    backend = FakeRecordingBackend()
    x = backend.add_var(0.0, 1.0, VarKind.BINARY)
    y = backend.add_var(0.0, 1.0, VarKind.BINARY)
    first = LinExpr.var(x) + LinExpr.var(y) <= 1
    second = (LinExpr.var(x) - 2 * LinExpr.var(y)) >= 0
    backend.add_rows([first.named("cap")])
    backend.add_rows([second])
    assert [r.name for r in backend.rows] == ["cap", ""]
    assert dict(backend.rows[0].terms) == {x: 1.0, y: 1.0}
    assert backend.rows[1].rhs == 0.0


def test_a_row_on_an_unknown_variable_is_an_error() -> None:
    backend = FakeRecordingBackend()
    backend.add_var(0.0, 1.0, VarKind.BINARY)
    bad = (LinExpr.var(5) <= 1).named("bad")
    with pytest.raises(ModelBuildError, match="unknown variable 5"):
        backend.add_rows([bad])
    assert backend.rows == []


def test_objective_must_use_known_variables() -> None:
    backend = FakeRecordingBackend()
    x = backend.add_var(0.0, 1.0, VarKind.BINARY)
    backend.set_objective({x: 3.0})
    assert backend.objective == {x: 3.0}
    with pytest.raises(ModelBuildError, match="unknown variable"):
        backend.set_objective({7: 1.0})
    assert backend.objective == {x: 3.0}


def test_solve_without_script_reports_no_solution_and_keeps_params() -> None:
    backend = FakeRecordingBackend()
    params = SolveParams(time_limit=60.0, memory_limit_gb=4.0)
    result = backend.solve(params)
    assert result.status is BackendStatus.NO_SOLUTION
    assert backend.last_params == params


def test_solve_returns_the_scripted_result() -> None:
    backend = FakeRecordingBackend()
    backend.scripted = SolveResult(
        BackendStatus.OPTIMAL, {0: 1.0}, objective=3.0, gap=0.0
    )
    result = backend.solve(SolveParams())
    assert result.status is BackendStatus.OPTIMAL
    assert result.values == {0: 1.0}
    assert result.objective == 3.0


def test_write_dumps_variables_objective_and_rows(tmp_path: Path) -> None:
    backend = FakeRecordingBackend()
    x = backend.add_var(0.0, 1.0, VarKind.BINARY, "x")
    y = backend.add_var(0.0, 5.0, VarKind.CONTINUOUS, "y")
    backend.set_objective({y: 1.0})
    backend.add_rows([(LinExpr.var(x) + LinExpr.var(y) <= 4).named("cap")])
    path = tmp_path / "model.txt"
    backend.write(str(path))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "x0 [0.0, 1.0] binary x"
    assert lines[1].startswith("x1 [0.0, 5.0]")
    assert lines[2] == "minimise +1*x1"
    assert lines[3] == "cap: +1*x0 +1*x1 <= 4"


def test_set_start_records_the_starting_values() -> None:
    backend = FakeRecordingBackend()
    a = backend.add_var(0.0, 1.0, VarKind.BINARY, "a")
    b = backend.add_var(0.0, 5.0, VarKind.CONTINUOUS, "b")
    backend.set_start({a: 1.0})
    backend.set_start({b: 2.5})
    assert backend.starts == {a: 1.0, b: 2.5}


def test_set_start_on_an_unknown_variable_is_an_error() -> None:
    backend = FakeRecordingBackend()
    with pytest.raises(ModelBuildError):
        backend.set_start({0: 1.0})
