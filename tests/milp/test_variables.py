"""Tests of the decision variables: their number, their domains and their lookup."""

from __future__ import annotations

import pytest
from sample import sample_instance

from satedge_workflow_scheduler.scheduling.deadline import DeadlineMode
from satedge_workflow_scheduler.solvers.milp.backends.fake_recording import (
    FakeRecordingBackend,
    RecordedVar,
)
from satedge_workflow_scheduler.solvers.milp.backends.ports import VarKind
from satedge_workflow_scheduler.solvers.milp.errors import ModelBuildError
from satedge_workflow_scheduler.solvers.milp.model_data import ModelData, TaskKey
from satedge_workflow_scheduler.solvers.milp.variables import (
    DecisionVariables,
    create_decision_variables,
)


def _create(mode: DeadlineMode) -> tuple[FakeRecordingBackend, DecisionVariables]:
    backend = FakeRecordingBackend()
    variables = create_decision_variables(backend, ModelData(sample_instance(), mode))
    return backend, variables


def _by_name(backend: FakeRecordingBackend, prefix: str) -> list[RecordedVar]:
    return [v for v in backend.variables if v.name.startswith(prefix)]


def test_each_family_has_one_variable_per_index() -> None:
    backend, _ = _create(DeadlineMode.HARD)
    # 2 requests, 3 tasks, 2 satellites, one computing window per satellite
    assert len(_by_name(backend, "rho[")) == 2
    assert _by_name(backend, "x[") == []  # x is an expression of the u, not a variable
    assert len(_by_name(backend, "t[")) == 3
    assert len(_by_name(backend, "u[")) == 3 * 2 * 1
    assert len(backend.variables) == 2 + 3 + 6


def test_the_auxiliary_families_are_not_created_with_the_decision_variables() -> None:
    backend, _ = _create(DeadlineMode.HARD)
    for prefix in ("chi[", "nu[", "xi[", "deliv[", "tard["):
        assert _by_name(backend, prefix) == []


def test_handles_are_consecutive_and_match_the_recorded_variables() -> None:
    backend, variables = _create(DeadlineMode.HARD)
    assert [v.handle for v in backend.variables] == list(range(len(backend.variables)))
    (rho,) = variables.rejection("r1").terms
    assert backend.variables[rho].name == "rho[r1]"


def test_rejection_is_binary_under_hard_deadlines() -> None:
    backend, variables = _create(DeadlineMode.HARD)
    (rho_handle,) = variables.rejection("r1").terms
    rho = backend.variables[rho_handle]
    assert (rho.lb, rho.ub, rho.kind) == (0.0, 1.0, VarKind.BINARY)


def test_under_soft_deadlines_no_rejection_variable_exists() -> None:
    backend, variables = _create(DeadlineMode.SOFT)
    assert _by_name(backend, "rho[") == []
    rejection = variables.rejection("r2")
    assert dict(rejection.terms) == {}
    assert rejection.constant == 0.0


def test_assignment_and_window_variables_are_binary() -> None:
    backend, variables = _create(DeadlineMode.HARD)
    u = backend.variables[variables.u(TaskKey("r1", 0), "S2", 0)]
    assert u.kind is VarKind.BINARY and (u.lb, u.ub) == (0.0, 1.0)


def test_x_is_the_sum_of_the_windows_of_its_satellite() -> None:
    _, variables = _create(DeadlineMode.HARD)
    expr = variables.x(TaskKey("r1", 0), "S2")
    assert set(expr.terms) == {variables.u(TaskKey("r1", 0), "S2", 0)}
    assert expr.constant == 0.0


def test_start_of_a_task_is_bounded_by_its_limit_minus_its_fastest_run() -> None:
    backend, variables = _create(DeadlineMode.HARD)
    # r1.0 lasts 15 s on either satellite and r1 must be delivered by 500 s
    start = backend.variables[variables.t(TaskKey("r1", 0))]
    assert start.kind is VarKind.CONTINUOUS
    assert start.lb == 0.0
    assert start.ub == pytest.approx(485.0)


def test_soft_deadlines_bound_the_start_by_the_horizon() -> None:
    backend, variables = _create(DeadlineMode.SOFT)
    start = backend.variables[variables.t(TaskKey("r1", 0))]
    assert start.ub == pytest.approx(600.0 - 15.0)


def test_rejection_of_an_unknown_request_is_a_model_error() -> None:
    _, variables = _create(DeadlineMode.HARD)
    with pytest.raises(ModelBuildError, match="not in the model"):
        variables.rejection("r9")


def test_a_missing_variable_is_a_model_error() -> None:
    _, variables = _create(DeadlineMode.HARD)
    with pytest.raises(ModelBuildError, match="does not exist"):
        variables.x(TaskKey("r9", 0), "S1")
    with pytest.raises(ModelBuildError, match="does not exist"):
        variables.u(TaskKey("r1", 0), "S1", 5)
