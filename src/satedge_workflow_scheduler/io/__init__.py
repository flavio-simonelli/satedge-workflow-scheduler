"""Input and output files: YAML documents and their translation into domain objects."""

from satedge_workflow_scheduler.io.errors import DocumentError
from satedge_workflow_scheduler.io.files import load_instance

__all__ = ["DocumentError", "load_instance"]
