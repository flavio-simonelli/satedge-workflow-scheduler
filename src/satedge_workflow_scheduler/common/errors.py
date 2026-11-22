"""Exception base of the package: every error derives from :class:`SchedulerError`.

Each package declares its own errors in an ``errors`` module next to it, for example
:mod:`satedge_workflow_scheduler.domain.errors` and :mod:`satedge_workflow_scheduler.solvers.errors`.
"""

from __future__ import annotations


class SchedulerError(Exception):
    """Base class for all exceptions raised by the package."""
