"""Exceptions of the sequential heuristics."""

from __future__ import annotations

from satedge_workflow_scheduler.common.errors import SchedulerError


class SequentialConfigError(SchedulerError):
    """A configuration of the sequential heuristic is not valid."""


class CommittedCalendarError(SchedulerError):
    """The committed calendar of an instance does not deliver a committed request.

    Unlike a failure of the solver, this is a fault of the input: the committed calendar comes from
    a previous plan and must already be consistent with the instance.
    """
