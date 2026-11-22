"""Exceptions of the solvers, shared by the heuristic and the MILP."""

from __future__ import annotations

from collections.abc import Mapping

from satedge_workflow_scheduler.common.errors import SchedulerError


class SolverError(SchedulerError):
    """The optimizer failed or returned a solution that cannot be decoded."""


class InfeasibleModelError(SolverError):
    """The optimizer proved the model infeasible.

    Attributes:
        reasons: Reason of the infeasibility for each request that causes it, keyed by
            request identifier. Empty when the reasons are not known.
    """

    def __init__(self, message: str, reasons: Mapping[str, str] | None = None) -> None:
        super().__init__(message)
        self.reasons: dict[str, str] = dict(reasons or {})
