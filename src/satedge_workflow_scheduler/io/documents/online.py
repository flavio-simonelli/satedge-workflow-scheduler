"""Document of the state of a planning epoch."""

from __future__ import annotations

from pydantic import Field, field_validator, model_validator

from satedge_workflow_scheduler.io.documents.base import (
    SCHEMA_VERSION,
    Document,
    check_schema_version,
)
from satedge_workflow_scheduler.io.documents.schedule import ExecutionDoc, TransferDoc


class CommittedDoc(Document):
    """The committed calendar: a saved result, or the executions and transfers written here.

    Attributes:
        result: Path of a result file, relative to this file; its calendar is committed.
        requests: With ``result``, only these requests are committed; all of them if omitted.
        executions: Inline executions, instead of ``result``.
        transfers: Inline transfers, instead of ``result``.
    """

    result: str | None = None
    requests: list[str] | None = None
    executions: list[ExecutionDoc] = Field(default_factory=list)
    transfers: list[TransferDoc] = Field(default_factory=list)

    @model_validator(mode="after")
    def _one_source(self) -> CommittedDoc:
        inline = bool(self.executions or self.transfers)
        if self.result is not None and inline:
            raise ValueError("give either 'result' or inline executions and transfers")
        if self.requests is not None and self.result is None:
            raise ValueError("'requests' filters a 'result': give 'result' too")
        return self


class OnlineDoc(Document):
    """The state of a planning epoch: when the plan reaches each satellite, and what is fixed.

    Attributes:
        schema_version: Version of the format, ``major.minor.patch``.
        plan_release_s: Plan release of the satellites it lists; ``null`` means never.
        committed: The calendar of the requests fixed by the previous plan.
    """

    schema_version: str = SCHEMA_VERSION
    plan_release_s: dict[str, float | None] = Field(default_factory=dict)
    committed: CommittedDoc = Field(default_factory=CommittedDoc)

    @field_validator("schema_version")
    @classmethod
    def _same_major(cls, value: str) -> str:
        return check_schema_version(value)
