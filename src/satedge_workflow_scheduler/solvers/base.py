"""Common interface of the solvers: MILP and heuristic return the same kind of solution."""

from __future__ import annotations

from typing import Protocol

from satedge_workflow_scheduler.domain.instance import ProblemInstance
from satedge_workflow_scheduler.domain.solution import Solution


class Solver(Protocol):
    """A solver that schedules an instance.

    Implementations must return a :class:`Solution` that the checker can validate
    against the instance, whatever their internal method is.
    """

    def solve(self, instance: ProblemInstance) -> Solution:
        """Return a schedule for ``instance``."""
        ...
