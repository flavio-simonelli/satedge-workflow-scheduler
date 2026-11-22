"""Terms that several blocks use in the same way, written once."""

from __future__ import annotations

from satedge_workflow_scheduler.solvers.milp.blocks.base import BuildContext
from satedge_workflow_scheduler.solvers.milp.expr import LinExpr
from satedge_workflow_scheduler.solvers.milp.model_data import TaskKey


def busy_time(ctx: BuildContext, i: TaskKey) -> LinExpr:
    """``Σ_s L̄_{i,s} x_{i,s}``: the time the task occupies the HPC of the satellite it runs on.

    The factor ``1 − ρ_r`` is inside the sum, because ``Σ_s x_{i,s} = 1 − ρ_r`` by (2).
    """
    return LinExpr.total(
        ctx.data.duration(i, s) * ctx.variables.x(i, s) for s in ctx.data.satellites
    )
