"""Reading of a YAML file into a document, with errors that name the file and the field."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError

from satedge_workflow_scheduler.io.errors import DocumentError


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
