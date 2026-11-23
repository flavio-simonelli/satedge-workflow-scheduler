"""Exceptions of the sequential heuristics."""

from __future__ import annotations

from satedge_workflow_scheduler.common.errors import SchedulerError


class SequentialConfigError(SchedulerError):
    """A configuration of the sequential heuristic is not valid."""
