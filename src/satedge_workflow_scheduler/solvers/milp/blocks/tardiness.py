"""The tardiness of each request under soft deadlines: how far its delivery is past the due date.

Equation (29) of the document: ``τ_r ≥ δ_r − dl_r``. The tardiness is non-negative by its domain,
so a delivery before the due date costs nothing. The block creates ``τ_r``, in
``[0, max(0, H − dl_r)]``, and it is written only under soft deadlines, where the objective (1″)
weighs it. Under hard deadlines the document has no tardiness.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from satedge_workflow_scheduler.solvers.milp.backends.ports import ModelSink, VarKind
from satedge_workflow_scheduler.solvers.milp.blocks.base import (
    BuildContext,
    ConstraintBlock,
)
from satedge_workflow_scheduler.solvers.milp.blocks.delivery import DeliveryTimes
from satedge_workflow_scheduler.solvers.milp.errors import ModelBuildError
from satedge_workflow_scheduler.solvers.milp.expr import LinExpr, Row, Var


@dataclass(frozen=True)
class TardinessTimes:
    """``τ_r``: the tardiness of the delivery of each request."""

    tardy: Mapping[str, Var]

    def tard(self, request_id: str) -> Var:
        try:
            return self.tardy[request_id]
        except KeyError:
            raise ModelBuildError(
                f"variable tard[{request_id}] does not exist"
            ) from None


class TardinessBlock(ConstraintBlock[TardinessTimes]):
    """``τ_r ≥ δ_r − dl_r`` for every request ``r``."""

    name = "tardiness"
    equations = "(29)"
    requires = frozenset({"deliv"})
    provides = frozenset({"tard"})

    def variables(self, ctx: BuildContext, sink: ModelSink) -> TardinessTimes:
        data = ctx.data
        tardy = {
            request_id: sink.add_var(
                0.0,
                max(0.0, data.horizon - data.due(request_id)),
                VarKind.CONTINUOUS,
                f"tard[{request_id}]",
            )
            for request_id in data.request_ids
        }
        return TardinessTimes(tardy)

    def rows(self, ctx: BuildContext, aux: TardinessTimes) -> Iterator[Row]:
        deliveries = ctx.families.get(DeliveryTimes)
        for request_id in ctx.data.request_ids:
            # τ_r − δ_r ≥ −dl_r
            yield (
                LinExpr.var(aux.tard(request_id))
                - LinExpr.var(deliveries.deliv(request_id))
                >= -ctx.data.due(request_id)
            ).named(f"tardiness[{request_id}]")
