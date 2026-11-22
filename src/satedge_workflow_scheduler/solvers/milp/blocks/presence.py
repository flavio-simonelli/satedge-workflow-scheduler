"""The presence of a data item in a PLMM: ``q_{a,v}``, 1 when the data occupies the memory of ``v``.

Used by both formulations of the capacity of the PLMM. It is an expression, not a variable: the
data is in the PLMM when its producer runs on ``v``, or when it arrives on ``v``.
"""

from __future__ import annotations

from satedge_workflow_scheduler.domain.nodes import NodeId
from satedge_workflow_scheduler.solvers.milp.blocks.base import BuildContext
from satedge_workflow_scheduler.solvers.milp.blocks.path import RoutingVariables
from satedge_workflow_scheduler.solvers.milp.blocks.terms import produced_on
from satedge_workflow_scheduler.solvers.milp.expr import LinExpr
from satedge_workflow_scheduler.solvers.milp.model_data import DataKey


def presence(ctx: BuildContext, a: DataKey, node: NodeId) -> LinExpr:
    """``q_{a,v} = asg_{i,v} + Σ_{e ∈ δ⁻(v)} χ_{a,e}``: the data ``a`` is in the PLMM of ``v``.

    It is 1 when the producer of ``a`` runs on ``v``, or when ``a`` arrives on ``v``. Both cannot
    hold together, because the data arriving on ``v`` is not produced there.
    """
    routing = ctx.families.get(RoutingVariables)
    arriving = LinExpr.total(routing.chi(a, e) for e in ctx.data.incoming(node))
    return produced_on(ctx, a, node) + arriving
