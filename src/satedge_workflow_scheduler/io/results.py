"""The result of a solve as a JSON file, and its reading back into a solution."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from satedge_workflow_scheduler.analysis import SolutionMetrics, analyze
from satedge_workflow_scheduler.domain.instance import ProblemInstance
from satedge_workflow_scheduler.domain.solution import (
    ObjectiveValue,
    RequestOutcome,
    Solution,
    SolveStatus,
)
from satedge_workflow_scheduler.io.documents import (
    ArcResultDoc,
    ExecutionDoc,
    HpcResultDoc,
    ObjectiveDoc,
    PlmmResultDoc,
    RequestResultDoc,
    ResultDoc,
    StatisticsDoc,
    TransferDoc,
    ViolationDoc,
)
from satedge_workflow_scheduler.io.errors import DocumentError
from satedge_workflow_scheduler.io.translate.schedule import schedule_from_documents
from satedge_workflow_scheduler.scheduling.deadline import DeadlineMode
from satedge_workflow_scheduler.solvers.run import SolverRun

GENERATOR = "satedge-workflow-scheduler"


def result_document(
    instance: ProblemInstance, run: SolverRun, mode: DeadlineMode
) -> ResultDoc:
    """The document of ``run``, the solve of ``instance`` under the deadline ``mode``."""
    solution = run.solution
    metrics = analyze(instance, solution, mode)
    stats = run.statistics
    return ResultDoc(
        instance=instance.name,
        deadline=mode.value,
        status=solution.status.value,
        objective=_objective(solution.objective),
        statistics=StatisticsDoc(
            method=stats.method,
            wall_time_s=stats.wall_time_s,
            solver_time_s=stats.solver_time_s,
            bound=stats.bound,
            gap=stats.gap,
            nodes=stats.nodes,
            rounds=stats.rounds,
            binary_variables=stats.binary_variables,
            constraints=stats.constraints,
        ),
        requests=_requests(metrics),
        executions=[
            ExecutionDoc(
                request=e.request_id,
                task=e.task_index,
                satellite=e.satellite,
                start_s=e.start,
                end_s=e.end,
            )
            for e in solution.schedule.executions
        ],
        transfers=[
            TransferDoc(
                request=t.request_id,
                data=t.data_index,
                source=t.source,
                target=t.target,
                start_s=t.start,
                end_s=t.end,
            )
            for t in solution.schedule.transfers
        ],
        plmm=[
            PlmmResultDoc(
                satellite=p.satellite,
                capacity_gb=p.capacity,
                peak_gb=p.peak,
                peak_time_s=p.peak_time,
            )
            for p in metrics.plmm
        ],
        hpc=[
            HpcResultDoc(
                satellite=h.satellite,
                available_s=h.available,
                busy_s=h.busy,
                utilisation=h.utilisation,
            )
            for h in metrics.hpc
        ],
        arcs=[
            ArcResultDoc(
                source=a.source,
                target=a.target,
                busy_s=a.busy,
                volume_gb=a.volume,
            )
            for a in metrics.arcs
        ],
        violations=[
            ViolationDoc(
                constraint=v.constraint.value,
                request=v.request_id,
                message=v.message,
            )
            for v in metrics.violations
        ],
    )


def write_result(
    path: str | Path, instance: ProblemInstance, run: SolverRun, mode: DeadlineMode
) -> None:
    """Write the result of ``run`` to the JSON file at ``path``, creating its folder."""
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    document = result_document(instance, run, mode)
    file.write_text(document.model_dump_json(indent=2) + "\n", encoding="utf-8")


def read_result(path: str | Path) -> Solution:
    """The solution stored in the JSON file at ``path``.

    Raises:
        DocumentError: If the file cannot be read or does not match the result format.
    """
    file = Path(path)
    try:
        text = file.read_text(encoding="utf-8")
    except OSError as error:
        raise DocumentError(f"cannot read {file}: {error.strerror}") from error
    try:
        document = ResultDoc.model_validate_json(text)
    except ValidationError as error:
        raise DocumentError(f"{file}: {error.errors()[0]['msg']}") from error
    return solution_from_document(document)


def solution_from_document(document: ResultDoc) -> Solution:
    """The solution described by a result document."""
    schedule = schedule_from_documents(document.executions, document.transfers)
    outcomes = tuple(
        RequestOutcome(
            r.id,
            r.accepted,
            delivery_time=r.delivery_s if r.accepted else None,
            rejection_reason=None if r.accepted else r.rejection_reason,
        )
        for r in document.requests
    )
    objective = None
    if document.objective is not None:
        objective = ObjectiveValue(
            document.objective.primary, document.objective.response
        )
    return Solution(
        schedule=schedule,
        outcomes=outcomes,
        status=SolveStatus(document.status),
        objective=objective,
    )


def _objective(value: ObjectiveValue | None) -> ObjectiveDoc | None:
    if value is None:
        return None
    return ObjectiveDoc(primary=value.primary, response=value.response)


def _requests(metrics: SolutionMetrics) -> list[RequestResultDoc]:
    return [
        RequestResultDoc(
            id=r.request_id,
            priority=r.priority,
            committed=r.committed,
            accepted=r.accepted,
            rejection_reason=r.rejection_reason,
            release_s=r.release,
            limit_s=r.limit,
            delivery_s=r.delivery,
            response_s=r.response,
            tardiness_s=r.tardiness,
        )
        for r in metrics.requests
    ]
