"""Documents of a calendar: task executions and transmissions."""

from __future__ import annotations

from satedge_workflow_scheduler.io.documents.base import Document


class ExecutionDoc(Document):
    """A task run on a satellite."""

    request: str
    task: int
    satellite: str
    start_s: float
    end_s: float


class TransferDoc(Document):
    """A transmission of one data item on an arc."""

    request: str
    data: int
    source: str
    target: str
    start_s: float
    end_s: float
