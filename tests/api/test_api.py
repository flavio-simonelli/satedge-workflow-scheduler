"""The public facade: solving from Python and from files, and planning the next epoch."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from sample import sample_instance

import satedge_workflow_scheduler as scheduler
from satedge_workflow_scheduler.domain.solution import SolveStatus
from satedge_workflow_scheduler.io import write_instance
from satedge_workflow_scheduler.io.results import read_result
from satedge_workflow_scheduler.solvers.milp.config import Level, MilpConfig
from satedge_workflow_scheduler.solvers.sequential.config import SequentialConfig


def test_solving_with_the_heuristic_gives_the_solution_metrics_and_statistics() -> None:
    instance = sample_instance()
    result = scheduler.solve(instance, SequentialConfig())

    assert result.solution.status is SolveStatus.FEASIBLE
    assert result.statistics.method == "sequential"
    assert [r.request_id for r in result.metrics.requests] == [
        r.id for r in instance.requests
    ]
    assert result.metrics.violations == ()


def test_the_saved_result_reads_back_as_the_same_solution(tmp_path: Path) -> None:
    result = scheduler.solve(sample_instance(), SequentialConfig())
    file = tmp_path / "out" / "result.json"
    result.save(file)

    back = read_result(file)
    assert back.outcomes == result.solution.outcomes
    assert back.schedule == result.solution.schedule


def test_solve_file_reads_the_instance_and_the_solver_and_saves_the_result(
    tmp_path: Path,
) -> None:
    write_instance(tmp_path / "instance.yaml", sample_instance())
    (tmp_path / "solver.yaml").write_text(
        'schema_version: "1.0.0"\nmethod: sequential\n', encoding="utf-8"
    )
    result = scheduler.solve_file(
        tmp_path / "instance.yaml",
        solver=tmp_path / "solver.yaml",
        output=tmp_path / "result.json",
    )

    assert result.statistics.method == "sequential"
    assert (tmp_path / "result.json").exists()


def test_the_next_epoch_keeps_the_undelivered_requests_as_committed() -> None:
    instance = sample_instance()
    result = scheduler.solve(instance, SequentialConfig())
    release = dict(instance.plan_release)

    epoch = scheduler.next_epoch(instance, result, now=0.0, plan_release=release)

    accepted = set(result.solution.accepted_ids)
    assert {r.id for r in epoch.requests} == accepted
    assert epoch.committed.request_ids == frozenset(accepted)
    assert dict(epoch.plan_release) == release


def test_the_next_epoch_drops_what_is_delivered_before_now() -> None:
    instance = sample_instance()
    result = scheduler.solve(instance, SequentialConfig())

    epoch = scheduler.next_epoch(
        instance, result, now=1e9, plan_release=dict(instance.plan_release)
    )

    assert epoch.requests == ()
    assert epoch.committed.request_ids == frozenset()


def test_a_new_request_joins_the_epoch_and_the_epoch_can_be_solved() -> None:
    instance = sample_instance()
    result = scheduler.solve(instance, SequentialConfig())
    newcomer = dataclasses.replace(
        next(r for r in instance.requests if r.id == "r2"), id="r3", release=0.0
    )

    epoch = scheduler.next_epoch(
        instance,
        result,
        now=0.0,
        new_requests=(newcomer,),
        plan_release=dict(instance.plan_release),
    )
    next_result = scheduler.solve(epoch, SequentialConfig())

    assert "r3" in {r.id for r in epoch.requests}
    assert not epoch.is_committed("r3")
    assert next_result.metrics.violations == ()


def test_compare_solves_once_per_label() -> None:
    results = scheduler.compare(
        sample_instance(),
        {"heuristic": SequentialConfig(), "again": SequentialConfig()},
    )
    assert set(results) == {"heuristic", "again"}
    assert all(r.statistics.method == "sequential" for r in results.values())


def test_the_milp_solves_the_sample_to_optimality_with_gurobi() -> None:
    pytest.importorskip("gurobipy")
    result = scheduler.solve(sample_instance(), MilpConfig(level=Level.PLMM_CAPACITY))

    assert result.solution.status is SolveStatus.OPTIMAL
    assert result.statistics.method == "milp"
    assert result.metrics.violations == ()
