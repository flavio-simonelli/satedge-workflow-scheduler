"""Errors of the input and output files."""

from __future__ import annotations

from satedge_workflow_scheduler.common.errors import SchedulerError


class DocumentError(SchedulerError):
    """A file cannot be read, or its content does not describe a valid instance.

    The message names the file and, where possible, the field that is wrong.
    """
