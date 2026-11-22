"""Terms that several blocks use in the same way, written once."""

from __future__ import annotations

from satedge_workflow_scheduler.domain.nodes import NodeId
from satedge_workflow_scheduler.solvers.milp.blocks.base import BuildContext
from satedge_workflow_scheduler.solvers.milp.expr import LinExpr
from satedge_workflow_scheduler.solvers.milp.model_data import DataKey, TaskKey


def busy_time(ctx: BuildContext, i: TaskKey) -> LinExpr:
    """``Σ_s L̄_{i,s} x_{i,s}``: the time the task occupies the HPC of the satellite it runs on.

    The factor ``1 − ρ_r`` is inside the sum, because ``Σ_s x_{i,s} = 1 − ρ_r`` by (2).
    """
    return LinExpr.total(
        ctx.data.duration(i, s) * ctx.variables.x(i, s) for s in ctx.data.satellites
    )


def produced_on(ctx: BuildContext, a: DataKey, node: NodeId) -> LinExpr:
    """``asg_{i,v}`` for the producer ``i`` of the data ``a``: the task ``i`` on the node, or
    for the raw data, the acquisition task on its origin.

    The acquisition task is on the origin satellite with probability ``1 − ρ_r``, and it is on
    no other node. A task on the ground station is 0 by definition.
    """
    producer = ctx.data.producer(a)
    if producer is None:
        return _fictitious(ctx, a, node, ctx.data.origin(a.request_id))
    return _on_satellite(ctx, producer, node)


def consumed_on(ctx: BuildContext, a: DataKey, node: NodeId) -> LinExpr:
    """``asg_{j,v}`` for the consumer ``j`` of the data ``a``: the task ``j`` on the node, or
    for the final output, the delivery task on the ground station."""
    consumer = ctx.data.consumer(a)
    if consumer is None:
        return _fictitious(ctx, a, node, ctx.data.ground_station)
    return _on_satellite(ctx, consumer, node)


def _fictitious(ctx: BuildContext, a: DataKey, node: NodeId, fixed: NodeId) -> LinExpr:
    if node != fixed:
        return LinExpr.number(0.0)
    return 1 - ctx.variables.rejection(a.request_id)


def _on_satellite(ctx: BuildContext, i: TaskKey, node: NodeId) -> LinExpr:
    if node not in ctx.data.satellites:
        return LinExpr.number(0.0)
    return ctx.variables.x(i, node)
