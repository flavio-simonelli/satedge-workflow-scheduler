"""Translation of calendar documents into a schedule."""

from __future__ import annotations

from satedge_workflow_scheduler.domain.solution import Schedule, TaskExecution, Transfer
from satedge_workflow_scheduler.io.documents import ExecutionDoc, TransferDoc


def schedule_from_documents(
    executions: list[ExecutionDoc], transfers: list[TransferDoc]
) -> Schedule:
    """The calendar described by execution and transfer documents."""
    return Schedule(
        executions=tuple(
            TaskExecution(
                request_id=e.request,
                task_index=e.task,
                satellite=e.satellite,
                start=e.start_s,
                end=e.end_s,
            )
            for e in executions
        ),
        transfers=tuple(
            Transfer(
                request_id=t.request,
                data_index=t.data,
                source=t.source,
                target=t.target,
                start=t.start_s,
                end=t.end_s,
            )
            for t in transfers
        ),
    )
