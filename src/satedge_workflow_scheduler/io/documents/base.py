"""Base of the documents, and the version of the format."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


SCHEMA_VERSION = "1.0.0"
"""Version of the document format; the major number must match."""


def check_schema_version(value: str) -> str:
    """Return ``value`` if this reader understands its major version, otherwise raise."""
    if value.split(".")[0] != SCHEMA_VERSION.split(".")[0]:
        raise ValueError(
            f"unsupported schema version {value!r}: this reader understands "
            f"{SCHEMA_VERSION!r} and its minor versions"
        )
    return value
