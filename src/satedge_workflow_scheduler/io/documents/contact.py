"""Document of a contact plan: the visibility windows of the links."""

from __future__ import annotations

from pydantic import Field, field_validator

from satedge_workflow_scheduler.io.documents.base import (
    SCHEMA_VERSION,
    Document,
    check_schema_version,
)
from satedge_workflow_scheduler.io.documents.topology import WindowDoc


class ContactLinkDoc(Document):
    """Visibility windows of the link between two nodes."""

    a: str
    b: str
    windows: list[WindowDoc]


class ContactPlanDoc(Document):
    """Visibility windows of the links of a topology, from a separate file."""

    schema_version: str = SCHEMA_VERSION
    links: list[ContactLinkDoc] = Field(default_factory=list)

    @field_validator("schema_version")
    @classmethod
    def _same_major(cls, value: str) -> str:
        return check_schema_version(value)
