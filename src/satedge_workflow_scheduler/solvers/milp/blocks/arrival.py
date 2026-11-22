"""The arrival of the data at its consumer, and the delivery of the final output.

Equations (16a)–(16b) of the document, with the big-M. The consumer starts after the data has
arrived on the satellite where it runs (16a), and the final output reaches the ground station
before the time limit of its request (16b). Rule 7 of the optimizations replaces them with
constraints without big-M: see :mod:`.strengthening`.
"""

from __future__ import annotations

from collections.abc import Iterator

from satedge_workflow_scheduler.solvers.milp.blocks.base import (
    BuildContext,
    ConstraintBlock,
)
from satedge_workflow_scheduler.solvers.milp.blocks.path import RoutingVariables
from satedge_workflow_scheduler.solvers.milp.blocks.store_and_forward import (
    TransmissionStarts,
)
from satedge_workflow_scheduler.solvers.milp.expr import LinExpr, Row
from satedge_workflow_scheduler.solvers.milp.model_data import DataKey


class ArrivalBlock(ConstraintBlock):
    """Arrival at the consumer (16a) and delivery to the ground station before the limit (16b)."""

    name = "arrival"
    equations = "(16a)-(16b)"
    requires = frozenset({"rejection", "t", "u", "chi", "xi"})

    def rows(self, ctx: BuildContext, aux: None) -> Iterator[Row]:
        starts = ctx.families.get(TransmissionStarts)
        routing = ctx.families.get(RoutingVariables)
        for a in ctx.data.data:
            yield from self._arrival_at_consumer(ctx, starts, routing, a)
            yield from self._delivery_before_limit(ctx, starts, routing, a)

    def _arrival_at_consumer(
        self,
        ctx: BuildContext,
        starts: TransmissionStarts,
        routing: RoutingVariables,
        a: DataKey,
    ) -> Iterator[Row]:
        # (16a) t_j ≥ ξ_{a,e} + D_{a,e} + λ_e − M(2 − χ_{a,e} − x_{j,v}), for v satellite, e into v
        data, variables = ctx.data, ctx.variables
        consumer = data.consumer(a)
        if consumer is None:
            return
        for v in data.satellites:
            for e in data.incoming(v):
                time_e = data.transmission_time(a, e) + data.latency(e)
                big_m = data.limit(a.request_id) + time_e
                yield (
                    LinExpr.var(variables.t(consumer))
                    - LinExpr.var(starts.xi(a, e))
                    - time_e
                    + big_m * (2 - routing.chi(a, e) - variables.x(consumer, v))
                    >= 0
                ).named(f"arrival[{a},{e},{v}]")

    def _delivery_before_limit(
        self,
        ctx: BuildContext,
        starts: TransmissionStarts,
        routing: RoutingVariables,
        a: DataKey,
    ) -> Iterator[Row]:
        # (16b) ξ_{a,e} + D_{a,e} + λ_e ≤ τ̂_r + (D_{a,e} + λ_e)(1 − χ_{a,e}), for e into g
        data = ctx.data
        if not data.is_final(a):
            return
        ground = data.ground_station
        limit = data.limit(a.request_id)
        for e in data.incoming(ground):
            time_e = data.transmission_time(a, e) + data.latency(e)
            yield (
                (limit + time_e * (1 - routing.chi(a, e)))
                - LinExpr.var(starts.xi(a, e))
                - time_e
                >= 0
            ).named(f"deliver[{a},{e}]")
