"""Exception hierarchy of the package."""

from __future__ import annotations

from collections.abc import Mapping


class SchedulerError(Exception):
    """Base class for all exceptions raised by the package."""


class InstanceError(SchedulerError):
    """The problem instance is malformed or violates a data condition of the model."""


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
