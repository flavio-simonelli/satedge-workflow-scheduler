"""Exceptions raised by the domain: data that violates a condition of the model."""

from __future__ import annotations

from satedge_workflow_scheduler.common.errors import SchedulerError


class InstanceError(SchedulerError):
    """The problem instance is malformed or violates a data condition of the model."""
