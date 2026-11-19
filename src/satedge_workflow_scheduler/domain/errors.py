"""Exception hierarchy of the package."""

from __future__ import annotations


class SchedulerError(Exception):
    """Base class for all exceptions raised by the package."""


class InstanceError(SchedulerError):
    """The problem instance is malformed or violates a data condition of the model."""


class SolverError(SchedulerError):
    """The optimizer failed or returned a solution that cannot be decoded."""


class InfeasibleModelError(SolverError):
    """The optimizer proved the model infeasible."""
