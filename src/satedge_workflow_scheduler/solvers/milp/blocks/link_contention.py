"""Contention on the links: two data items of different requests do not share an arc at once.

Equations (18)–(19) of the document. For each pair ``(a, b)`` of data of different requests and
each arc ``e``, the binary ``σ_{a,b,e}`` says which of the two goes first on ``e``, when both
cross it: ``σ = 1`` if ``a`` precedes ``b``, ``σ = 0`` otherwise.

- (18): if ``a`` precedes ``b``, the transmission of ``b`` starts after ``a`` has been
  transmitted, ``ξ_b ≥ ξ_a + D_{a,e}``;
- (19): the converse, if ``b`` precedes ``a``.

Both constraints are relaxed by the big-M when one of the two data does not cross the arc, and
the big-M of the pair is the larger of the two time limits plus its transmission time.

Reduction 4: the triples are those of the reductions, where both data can cross the arc and can
occupy it at the same time. Elsewhere the two transmissions never overlap, and ``σ`` and the two
rows are not written.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from satedge_workflow_scheduler.solvers.milp.backends.ports import ModelSink, VarKind
from satedge_workflow_scheduler.solvers.milp.blocks.base import (
    BuildContext,
    ConstraintBlock,
)
from satedge_workflow_scheduler.solvers.milp.blocks.path import RoutingVariables
from satedge_workflow_scheduler.solvers.milp.blocks.store_and_forward import (
    TransmissionStarts,
)
from satedge_workflow_scheduler.solvers.milp.errors import ModelBuildError
from satedge_workflow_scheduler.solvers.milp.expr import LinExpr, Row, Var
from satedge_workflow_scheduler.solvers.milp.model_data import ArcId, DataKey

Triple = tuple[DataKey, DataKey, ArcId]


@dataclass(frozen=True)
class LinkOrder:
    """``σ_{a,b,e}``: 1 if ``a`` precedes ``b`` on the arc ``e``, when both cross it.

    Attributes:
        order: The variable of each pair of data and arc.
    """

    order: Mapping[Triple, Var]

    def sigma(self, a: DataKey, b: DataKey, arc: ArcId) -> Var:
        try:
            return self.order[(a, b, arc)]
        except KeyError:
            raise ModelBuildError(
                f"variable sigma[{a},{b},{arc}] does not exist"
            ) from None


class LinkContentionBlock(ConstraintBlock[LinkOrder]):
    """Two transmissions of different requests on the same arc do not overlap (18)–(19)."""

    name = "link contention"
    equations = "(18)-(19)"
    requires = frozenset({"xi", "chi"})

    def variables(self, ctx: BuildContext, sink: ModelSink) -> LinkOrder:
        order: dict[Triple, Var] = {
            (a, b, e): sink.add_var(0.0, 1.0, VarKind.BINARY, f"sigma[{a},{b},{e}]")
            for a, b, e in ctx.reductions.link_triples
        }
        return LinkOrder(order)

    def rows(self, ctx: BuildContext, aux: LinkOrder) -> Iterator[Row]:
        data = ctx.data
        starts = ctx.families.get(TransmissionStarts)
        routing = ctx.families.get(RoutingVariables)
        for a, b, e in ctx.reductions.link_triples:
            tx_a = data.transmission_time(a, e)
            tx_b = data.transmission_time(b, e)
            big_m = max(
                data.limit(a.request_id) + tx_a, data.limit(b.request_id) + tx_b
            )
            sigma = LinExpr.var(aux.sigma(a, b, e))
            both = 2 - routing.chi(a, e) - routing.chi(b, e)
            # (18) ξ_b − ξ_a − D_a + M(1 − σ) + M(2 − χ_a − χ_b) ≥ 0
            yield (
                LinExpr.var(starts.xi(b, e))
                - LinExpr.var(starts.xi(a, e))
                - tx_a
                + big_m * (1 - sigma)
                + big_m * both
                >= 0
            ).named(f"link-a[{a},{b},{e}]")
            # (19) ξ_a − ξ_b − D_b + M σ + M(2 − χ_a − χ_b) ≥ 0
            yield (
                LinExpr.var(starts.xi(a, e))
                - LinExpr.var(starts.xi(b, e))
                - tx_b
                + big_m * sigma
                + big_m * both
                >= 0
            ).named(f"link-b[{a},{b},{e}]")
