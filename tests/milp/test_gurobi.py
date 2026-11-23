"""Tests of the Gurobi adapter, on tiny models that fit the size-limited licence.

They are skipped when gurobipy is not installed, and they fail if a licence is installed but
its version does not match the library: that is a configuration error, to be seen at once.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("gurobipy")

from satedge_workflow_scheduler.solvers.milp.backends.gurobi import (
    GurobiBackend,
)
from satedge_workflow_scheduler.solvers.milp.backends.ports import (
    BackendStatus,
    MilpBackend,
    SolveParams,
    VarKind,
)
from satedge_workflow_scheduler.solvers.milp.errors import ModelBuildError
from satedge_workflow_scheduler.solvers.milp.expr import LinExpr, Row, Sense


def _backend() -> GurobiBackend:
    backend: MilpBackend = GurobiBackend()
    assert isinstance(backend, GurobiBackend)
    return backend


def test_solves_a_small_model_to_optimality() -> None:
    backend = _backend()
    x = backend.add_var(0.0, 1.0, VarKind.BINARY, "x")
    y = backend.add_var(0.0, 5.0, VarKind.CONTINUOUS, "y")
    backend.add_rows([(LinExpr.var(x) + 2 * LinExpr.var(y) <= 4).named("cap")])
    backend.set_objective({y: -1.0})  # maximise y

    result = backend.solve(SolveParams())

    assert result.status is BackendStatus.OPTIMAL
    assert result.values[y] == pytest.approx(2.0)
    assert result.values[x] == pytest.approx(0.0)
    assert result.objective == pytest.approx(-2.0)


def test_an_infeasible_model_is_reported_not_raised() -> None:
    backend = _backend()
    x = backend.add_var(0.0, 1.0, VarKind.BINARY)
    backend.add_rows([(LinExpr.var(x) >= 2).named("impossible")])
    backend.set_objective({x: 1.0})

    result = backend.solve(SolveParams())

    assert result.status is BackendStatus.INFEASIBLE
    assert result.values == {}
    assert result.objective is None


def test_equality_rows_are_enforced() -> None:
    backend = _backend()
    x = backend.add_var(0.0, 10.0, VarKind.CONTINUOUS)
    backend.add_rows([(LinExpr.var(x) == 3).named("fixed")])
    backend.set_objective({x: 1.0})

    result = backend.solve(SolveParams())

    assert result.values[x] == pytest.approx(3.0)


def test_gap_is_reported_only_for_integer_models() -> None:
    continuous = _backend()
    y = continuous.add_var(0.0, 1.0, VarKind.CONTINUOUS)
    continuous.set_objective({y: 1.0})
    assert continuous.solve(SolveParams()).gap is None

    integer = _backend()
    z = integer.add_var(0.0, 1.0, VarKind.BINARY)
    integer.set_objective({z: 1.0})
    assert integer.solve(SolveParams()).gap == pytest.approx(0.0)


def test_the_memory_limit_becomes_the_soft_limit_of_the_solver() -> None:
    backend = _backend()
    x = backend.add_var(0.0, 1.0, VarKind.CONTINUOUS)
    backend.set_objective({x: 1.0})

    backend.solve(SolveParams(memory_limit_gb=4.0))

    # the solver's own parameter, read back: the translation is what is under test
    assert backend._model.Params.SoftMemLimit == pytest.approx(4.0)


def test_time_limit_and_threads_reach_the_solver() -> None:
    backend = _backend()
    x = backend.add_var(0.0, 1.0, VarKind.CONTINUOUS)
    backend.set_objective({x: 1.0})

    backend.solve(SolveParams(time_limit=30.0, threads=1))

    assert backend._model.Params.TimeLimit == pytest.approx(30.0)
    assert backend._model.Params.Threads == 1


def test_a_row_on_an_unknown_variable_is_a_model_error() -> None:
    backend = _backend()
    backend.add_var(0.0, 1.0, VarKind.BINARY)
    with pytest.raises(ModelBuildError, match="unknown variable 4"):
        backend.add_rows([(LinExpr.var(4) <= 1).named("bad")])


def test_write_produces_a_readable_lp_file(tmp_path: Path) -> None:
    backend = _backend()
    x = backend.add_var(0.0, 1.0, VarKind.BINARY, "x")
    backend.add_rows([(LinExpr.var(x) <= 1).named("cap")])
    backend.set_objective({x: 1.0})

    path = tmp_path / "model.lp"
    backend.write(str(path))

    text = path.read_text(encoding="utf-8")
    assert "Minimize" in text
    assert "cap" in text


def test_a_nan_bound_is_a_model_error_not_a_solver_error() -> None:
    backend = _backend()
    with pytest.raises(ModelBuildError, match="NaN"):
        backend.add_var(float("nan"), 1.0, VarKind.CONTINUOUS, "x")


def test_a_nan_coefficient_or_rhs_is_a_model_error() -> None:
    backend = _backend()
    x = backend.add_var(0.0, 1.0, VarKind.CONTINUOUS)
    not_a_number = float("nan")
    bad_rhs = LinExpr.var(x) + 0.0
    with pytest.raises(ModelBuildError, match="NaN"):
        backend.add_rows([Row(dict(bad_rhs.terms), Sense.LE, not_a_number, "bad_rhs")])
    bad_coef = Row({x: not_a_number}, Sense.LE, 1.0, "bad_coef")
    with pytest.raises(ModelBuildError, match="NaN"):
        backend.add_rows([bad_coef])


def test_an_unbounded_model_is_reported_as_an_error() -> None:
    backend = _backend()
    x = backend.add_var(0.0, float("inf"), VarKind.CONTINUOUS)
    backend.set_objective({x: -1.0})
    assert backend.solve(SolveParams()).status is BackendStatus.ERROR


def test_a_starting_value_reaches_the_solver_and_an_unknown_variable_is_an_error() -> (
    None
):
    backend = _backend()
    x = backend.add_var(0.0, 1.0, VarKind.BINARY, "x")
    backend.set_start({x: 1.0})
    backend._model.update()  # Gurobi reads the attributes of the variables only after this
    assert backend._vars[x].Start == 1.0
    with pytest.raises(ModelBuildError):
        backend.set_start({x + 10: 0.0})
