"""Exceptions of the scheduling rules, shared by the solvers.

They derive from :class:`SchedulerError` directly: each solver translates them into its own
error, because the same fact (a calendar with no delivery) means a failure of the solver for a
solution and a bug for a committed calendar.
"""

from __future__ import annotations

from satedge_workflow_scheduler.common.errors import SchedulerError


class DeliveryError(SchedulerError):
    """The calendar delivers no final output of the request to the ground station."""
