"""Assignment of the tasks to the satellites: where each task runs.

Equation (2) of the document: a task runs on exactly one satellite, unless its request is
rejected. The variable ``x_{i,s}`` is the sum of the windows of ``s`` (constraint (5)), so the
constraint is written on the windows directly.
"""

from __future__ import annotations

from collections.abc import Iterator

from satedge_workflow_scheduler.solvers.milp.blocks.base import (
    BuildContext,
    ConstraintBlock,
)
from satedge_workflow_scheduler.solvers.milp.expr import LinExpr, Row


class AssignmentBlock(ConstraintBlock):
    """``Σ_s x_{i,s} = 1 − ρ_r`` for every task ``i``, with ``x_{i,s} = Σ_w u_{i,s,w}``."""

    name = "assignment"
    equations = "(2)"
    requires = frozenset({"rejection", "u"})

    def rows(self, ctx: BuildContext) -> Iterator[Row]:
        variables = ctx.variables
        for i in ctx.data.tasks:
            placed = LinExpr.total(variables.x(i, s) for s in ctx.data.satellites)
            accepted = 1 - variables.rejection(i.request_id)
            yield (placed == accepted).named(f"assignment[{i}]")
