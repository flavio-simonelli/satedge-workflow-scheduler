"""Reading of the YAML files: each file is parsed, validated and turned into domain objects."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError

from satedge_workflow_scheduler.common.errors import SchedulerError
from satedge_workflow_scheduler.domain.instance import ProblemInstance
from satedge_workflow_scheduler.io.documents import InstanceDoc
from satedge_workflow_scheduler.io.errors import DocumentError
from satedge_workflow_scheduler.io.translate import instance_from_document


def load_instance(path: str | Path) -> ProblemInstance:
    """The instance described by the YAML file at ``path``.

    The name of the instance is the name of the file, unless the file gives one.

    Raises:
        DocumentError: If the file cannot be read, does not match the format, or describes
            an instance that the domain rejects. The message names the file and the field.
    """
    file = Path(path)
    doc = read_document(file, InstanceDoc)
    if not doc.name:
        doc = doc.model_copy(update={"name": file.stem})
    try:
        return instance_from_document(doc)
    except SchedulerError as error:
        raise DocumentError(f"{file}: {error}") from error
    except ValueError as error:
        raise DocumentError(f"{file}: {error}") from error


def read_document[Doc: BaseModel](path: Path, model: type[Doc]) -> Doc:
    """The document of type ``model`` in the YAML file at ``path``.

    Raises:
        DocumentError: If the file cannot be read or parsed, or does not match ``model``.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise DocumentError(f"cannot read {path}: {error.strerror}") from error
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise DocumentError(f"{path}: invalid YAML: {error}") from error
    try:
        return model.model_validate(raw)
    except ValidationError as error:
        raise DocumentError(_describe(path, error)) from error


def _describe(path: Path, error: ValidationError) -> str:
    """One line per problem, with the path of the field, such as ``requests[0].tasks``."""
    lines = []
    for problem in error.errors():
        lines.append(f"{path}: {_location(problem['loc'])}: {problem['msg']}")
    return "\n".join(lines)


def _location(loc: tuple[int | str, ...]) -> str:
    text = ""
    for part in loc:
        text += f"[{part}]" if isinstance(part, int) else (f".{part}" if text else part)
    return text or "(document)"
