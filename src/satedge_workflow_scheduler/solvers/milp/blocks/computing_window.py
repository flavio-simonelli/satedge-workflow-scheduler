"""Computing windows of the HPC: a task runs inside one window of its satellite.

Constraint (5) of the document is gone: ``x_{i,s}`` is the sum of the windows, so it defines
nothing to check. What remains:

- (6): the start of the task is not before the start of the window it runs in;
- (7): the task, its busy time included, ends before the end of its window.

The window of a task is chosen by the variables ``u_{i,s,w}``, which the assignment (2) makes
sum to one: the sums over the windows below pick the bounds of the window actually used.
"""

from __future__ import annotations

from collections.abc import Iterator

from satedge_workflow_scheduler.solvers.milp.blocks.base import (
    BuildContext,
    ConstraintBlock,
)
from satedge_workflow_scheduler.solvers.milp.blocks.terms import busy_time
from satedge_workflow_scheduler.solvers.milp.expr import LinExpr, Row


class ComputingWindowBlock(ConstraintBlock):
    """``t_i ≥ Σ hpcs u`` (6) and ``t_i + busy_i ≤ Σ hpce u`` (7), for every task."""

    name = "computing window"
    equations = "(6)-(7)"
    requires = frozenset({"t", "u"})

    def rows(self, ctx: BuildContext, aux: None) -> Iterator[Row]:
        variables = ctx.variables
        for i in ctx.data.tasks:
            starts = LinExpr()
            ends = LinExpr()
            for s in ctx.data.satellites:
                for w, window in enumerate(ctx.data.hpc_windows(s)):
                    chosen = variables.u(i, s, w)
                    starts += LinExpr.var(chosen, window.start)
                    ends += LinExpr.var(chosen, window.end)
            yield (LinExpr.var(variables.t(i)) >= starts).named(f"window_start[{i}]")
            yield (ends >= LinExpr.var(variables.t(i)) + busy_time(ctx, i)).named(
                f"window_end[{i}]"
            )
