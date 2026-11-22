"""The path of each data item: the arcs it crosses, from its producer to its consumer.

Equations (11)–(13) of the document. The variable ``χ_{a,e}`` is 1 when the data ``a`` crosses
the arc ``e``, and the three equations describe a path that starts where the producer is and
ends where the consumer is:

- (11): at every node, the data that leaves minus the data that enters is the departure of
  the producer minus the arrival of the consumer;
- (12): the data does not enter the node of its producer, nor leave the node of its consumer;
- (13): the data crosses no arc if its request is rejected.

Under soft deadlines no request is rejected, so (13) is ``χ ≤ 1``, which the binary domain
already gives: the block does not write it.
"""

from __future__ import annotations

from collections.abc import Iterator

from satedge_workflow_scheduler.solvers.milp.blocks.base import (
    BuildContext,
    ConstraintBlock,
)
from satedge_workflow_scheduler.solvers.milp.blocks.terms import (
    consumed_on,
    produced_on,
)
from satedge_workflow_scheduler.solvers.milp.expr import LinExpr, Row


class PathBlock(ConstraintBlock):
    """Flow conservation (11), ends of the path (12) and rejected data (13)."""

    name = "path"
    equations = "(11)-(13)"
    requires = frozenset({"rejection", "u", "chi"})

    def rows(self, ctx: BuildContext, aux: None) -> Iterator[Row]:
        data = ctx.data
        variables = ctx.variables
        for a in data.data:
            for v in data.nodes:
                leaving = LinExpr.total(variables.chi(a, e) for e in data.outgoing(v))
                entering = LinExpr.total(variables.chi(a, e) for e in data.incoming(v))
                # (11) Σ_out χ − Σ_in χ = asg_{i,v} − asg_{j,v}
                yield (
                    leaving - entering
                    == produced_on(ctx, a, v) - consumed_on(ctx, a, v)
                ).named(f"flow[{a},{v}]")
                # (12) Σ_in χ ≤ 1 − asg_{i,v} and Σ_out χ ≤ 1 − asg_{j,v}
                yield (entering <= 1 - produced_on(ctx, a, v)).named(
                    f"ends-in[{a},{v}]"
                )
                yield (leaving <= 1 - consumed_on(ctx, a, v)).named(
                    f"ends-out[{a},{v}]"
                )

        if not data.may_reject():
            return
        for a in data.data:
            for e in data.arc_ids:
                # (13) χ_{a,e} ≤ 1 − ρ_{reqof(a)}
                yield (
                    variables.chi(a, e) <= 1 - variables.rejection(a.request_id)
                ).named(f"rejected[{a},{e}]")
