"""Exceptions of the MILP solver.

They derive from :class:`SchedulerError` directly, not from ``SolverError``: they report a fault
of the model being built (a block that declared something inconsistent), not a failure of the
solver.
"""

from __future__ import annotations

from satedge_workflow_scheduler.common.errors import SchedulerError


class ModelBuildError(SchedulerError):
    """The model cannot be built: a declaration is inconsistent with the rest of the model."""
