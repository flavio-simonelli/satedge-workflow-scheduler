"""The objective of the model, as a linear expression over the variables the blocks created.

The coefficients are those of the evaluation of a schedule (``scheduling/objective.py``): the
rejection costs ``κ_r``, the tardiness weights ``ω_r`` and the weight ``ε`` of the response time.
The objective used by the model is the lexicographic one of the document, (1′) under hard
deadlines and (1″) under soft ones: the primary term first, and the response time, weighted by
ε, to break ties. The pure primary term (1) and the weighted tardiness alone are kept as
functions of their own, for the tests and for the proof that ε is small enough.
"""

from __future__ import annotations

from satedge_workflow_scheduler.scheduling.deadline import DeadlineMode
from satedge_workflow_scheduler.solvers.milp.blocks.base import BuildContext
from satedge_workflow_scheduler.solvers.milp.blocks.delivery import DeliveryTimes
from satedge_workflow_scheduler.solvers.milp.blocks.tardiness import TardinessTimes
from satedge_workflow_scheduler.solvers.milp.expr import LinExpr


def rejection_cost(ctx: BuildContext) -> LinExpr:
    """``Σ_r κ_r ρ_r``: the cost of the rejected requests (hard deadlines)."""
    costs = ctx.data.rejection_costs
    return LinExpr.total(
        costs[r] * ctx.variables.rejection(r) for r in ctx.data.request_ids
    )


def tardiness_cost(ctx: BuildContext) -> LinExpr:
    """``Σ_r ω_r τ_r``: the weighted tardiness of the requests (soft deadlines)."""
    weights = ctx.data.tardiness_weights
    tardy = ctx.families.get(TardinessTimes)
    return LinExpr.total(
        weights[r] * LinExpr.var(tardy.tard(r)) for r in ctx.data.request_ids
    )


def response_time(ctx: BuildContext) -> LinExpr:
    """``ε Σ_r (δ_r − b_r (1 − ρ_r))``: the response time, weighted by ε.

    Under soft deadlines ``ρ_r`` is the constant 0, so the term is ``δ_r − b_r``, as in (1″).
    """
    epsilon = ctx.data.epsilon
    deliveries = ctx.families.get(DeliveryTimes)
    return LinExpr.total(
        epsilon
        * (
            LinExpr.var(deliveries.deliv(r))
            - ctx.data.release(r) * (1 - ctx.variables.rejection(r))
        )
        for r in ctx.data.request_ids
    )


def objective(ctx: BuildContext) -> LinExpr:
    """The objective of the model: (1′) under hard deadlines, (1″) under soft ones."""
    if ctx.data.mode is DeadlineMode.HARD:
        primary = rejection_cost(ctx)
    else:
        primary = tardiness_cost(ctx)
    return primary + response_time(ctx)
