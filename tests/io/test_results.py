"""The result JSON: it is written from a solve and read back into the same solution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sample import sample_instance

from satedge_workflow_scheduler.io import DocumentError
from satedge_workflow_scheduler.io.results import read_result, write_result
from satedge_workflow_scheduler.scheduling.deadline import DeadlineMode
from satedge_workflow_scheduler.solvers.run import SolverRun
from satedge_workflow_scheduler.solvers.sequential.config import SequentialConfig
from satedge_workflow_scheduler.solvers.sequential.solver import SequentialSolver


def _run() -> SolverRun:
    return SequentialSolver(SequentialConfig()).run(sample_instance())


def test_the_file_holds_the_schedule_the_outcomes_and_the_metrics(
    tmp_path: Path,
) -> None:
    run = _run()
    file = tmp_path / "out" / "result.json"
    write_result(file, sample_instance(), run, DeadlineMode.HARD)

    data = json.loads(file.read_text(encoding="utf-8"))
    assert data["status"] == "feasible"
    assert data["deadline"] == "hard"
    assert data["statistics"]["method"] == "sequential"
    assert len(data["requests"]) == len(sample_instance().requests)
    assert len(data["executions"]) == len(run.solution.schedule.executions)
    assert len(data["transfers"]) == len(run.solution.schedule.transfers)
    assert data["plmm"]
    assert data["hpc"]
    assert data["arcs"]
    assert data["violations"] == []


def test_reading_the_file_gives_back_the_same_solution(tmp_path: Path) -> None:
    run = _run()
    file = tmp_path / "result.json"
    write_result(file, sample_instance(), run, DeadlineMode.HARD)

    back = read_result(file)
    assert back.status is run.solution.status
    assert back.schedule == run.solution.schedule
    assert back.outcomes == run.solution.outcomes
    assert back.objective == run.solution.objective


def test_a_rejected_request_keeps_its_reason_through_the_file(tmp_path: Path) -> None:
    import dataclasses

    from satedge_workflow_scheduler.domain.requests import Task

    instance = sample_instance()
    endless = tuple(
        dataclasses.replace(r, tasks=(Task(execution_time=1e6, output_size=0.001),))
        if r.id == "r1"
        else r
        for r in instance.requests
    )
    impossible = dataclasses.replace(instance, requests=endless)
    run = SequentialSolver(SequentialConfig()).run(impossible)
    file = tmp_path / "result.json"
    write_result(file, impossible, run, DeadlineMode.HARD)

    back = read_result(file)
    outcome = next(o for o in back.outcomes if o.request_id == "r1")
    assert not outcome.accepted
    assert outcome.rejection_reason is not None
    assert set(back.accepted_ids) == set(back.schedule.request_ids)
    assert "r1" not in back.accepted_ids


def test_a_file_that_is_not_a_result_is_a_document_error(tmp_path: Path) -> None:
    file = tmp_path / "bad.json"
    file.write_text('{"instance": "x"}', encoding="utf-8")
    with pytest.raises(DocumentError):
        read_result(file)
