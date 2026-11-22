"""Timing of the tasks: when each task can start.

Two equations of the document, both lower bounds on a start:

- (3), the precedences: a task starts after its predecessor has finished. The predecessor
  occupies its HPC for ``L̄_{i,s}`` on the satellite where it runs, and ``Σ_s L̄_{i,s} x_{i,s}``
  is that duration for whichever satellite it is on. The factor ``1 − ρ_r`` is already inside
  the sum, because ``Σ_s x_{i,s} = 1 − ρ_r`` by (2).
- (4), the release: the first task of a request starts after the raw data is available, if the
  request is accepted, ``t_first ≥ b_r (1 − ρ_r)``.
"""

from __future__ import annotations

from collections.abc import Iterator

from satedge_workflow_scheduler.solvers.milp.blocks.base import (
    BuildContext,
    ConstraintBlock,
)
from satedge_workflow_scheduler.solvers.milp.expr import LinExpr, Row
from satedge_workflow_scheduler.solvers.milp.model_data import TaskKey


class TaskTimingBlock(ConstraintBlock):
    """Precedences (3) and release of the raw data (4)."""

    name = "task timing"
    equations = "(3), (4)"
    requires = frozenset({"rho", "t", "u"})

    def rows(self, ctx: BuildContext) -> Iterator[Row]:
        yield from self._precedences(ctx)
        yield from self._releases(ctx)

    def _precedences(self, ctx: BuildContext) -> Iterator[Row]:
        variables = ctx.variables
        for i, j in ctx.data.precedences:
            # t_j − t_i − Σ_s L̄_{i,s} x_{i,s} ≥ 0
            busy = LinExpr.total(
                ctx.data.duration(i, s) * variables.x(i, s) for s in ctx.data.satellites
            )
            yield (
                LinExpr.var(variables.t(j)) - LinExpr.var(variables.t(i)) - busy >= 0
            ).named(f"precedence[{i},{j}]")

    def _releases(self, ctx: BuildContext) -> Iterator[Row]:
        variables = ctx.variables
        for request_id in ctx.data.request_ids:
            first = TaskKey(request_id, 0)
            release = ctx.data.release(request_id)
            # t_first ≥ b_r (1 − ρ_r)
            available = release * (1 - variables.rejection(request_id))
            yield (LinExpr.var(variables.t(first)) >= available).named(
                f"release[{request_id}]"
            )
