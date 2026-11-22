"""The instant of delivery of each request: when its final output reaches the ground station.

Equation (28) of the document: the delivery ``δ_r`` is at least the arrival of the final output
on any arc into the ground station that it crosses. The block creates ``δ_r``, in ``[0, τ̂_r]``,
and the objective and the tardiness read it. It is needed only when the objective is the
response time (1′) or the deadline is soft: with the objective (1) and hard deadlines the
document has no delivery.
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


@dataclass(frozen=True)
class DeliveryTimes:
    """``δ_r``: the instant of the delivery of the final output of each request."""

    delivered: Mapping[str, Var]

    def deliv(self, request_id: str) -> Var:
        try:
            return self.delivered[request_id]
        except KeyError:
            raise ModelBuildError(
                f"variable deliv[{request_id}] does not exist"
            ) from None


class DeliveryBlock(ConstraintBlock[DeliveryTimes]):
    """``δ_r ≥ ξ_{a,e} + (D_{a,e} + λ_e) χ_{a,e}`` for the final output ``a`` of ``r``, on
    every arc ``e`` into the ground station."""

    name = "delivery"
    equations = "(28)"
    requires = frozenset({"xi", "chi"})
    provides = frozenset({"deliv"})

    def variables(self, ctx: BuildContext, sink: ModelSink) -> DeliveryTimes:
        delivered = {
            request_id: sink.add_var(
                0.0,
                ctx.data.limit(request_id),
                VarKind.CONTINUOUS,
                f"deliv[{request_id}]",
            )
            for request_id in ctx.data.request_ids
        }
        return DeliveryTimes(delivered)

    def rows(self, ctx: BuildContext, aux: DeliveryTimes) -> Iterator[Row]:
        data = ctx.data
        starts = ctx.families.get(TransmissionStarts)
        routing = ctx.families.get(RoutingVariables)
        for request_id in data.request_ids:
            final = data.final_data(request_id)
            for e in data.incoming(data.ground_station):
                time_e = data.transmission_time(final, e) + data.latency(e)
                # δ_r − ξ_{a,e} − (D + λ) χ_{a,e} ≥ 0
                yield (
                    LinExpr.var(aux.deliv(request_id))
                    - LinExpr.var(starts.xi(final, e))
                    - time_e * routing.chi(final, e)
                    >= 0
                ).named(f"delivery[{request_id},{e}]")
