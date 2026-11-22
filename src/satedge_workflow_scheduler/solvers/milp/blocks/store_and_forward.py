"""When each transmission starts: after the production, store-and-forward, and the arrival.

Equations (14a)–(16b) of the document. The variable ``ξ_{a,e}`` is the start of the transmission
of the data ``a`` on the arc ``e``, and ``χ_{a,e}`` says whether ``a`` crosses ``e`` at all: the
constraints of an arc not crossed are relaxed by the big-M, so they do not bind.

- (14a): a transmission starts after the producer has finished, if the data leaves through the
  arc. The producer occupies its HPC for ``Σ_s L̄_{i,s} x_{i,s}``, which is ``L̄_i (1 − ρ_r)``.
- (14b): the raw data is available from the release of its request.
- (15), store-and-forward: at a satellite, the data that arrives on an arc leaves on the next
  arc after the arrival, if it is stored there.
- (16a): the consumer starts after the data arrives on the satellite where it runs.
- (16b): the final output reaches the ground station before the time limit of its request.

The time of an arc is ``D_{a,e} + λ_e``: the transmission plus the latency. The big-M of
(14a), (15) and (16a) is the time limit of the request plus that time, as in the document.
"""

from __future__ import annotations

from collections.abc import Iterator

from satedge_workflow_scheduler.solvers.milp.blocks.base import (
    BuildContext,
    ConstraintBlock,
)
from satedge_workflow_scheduler.solvers.milp.blocks.terms import busy_time
from satedge_workflow_scheduler.solvers.milp.expr import LinExpr, Row
from satedge_workflow_scheduler.solvers.milp.model_data import DataKey


class StoreAndForwardBlock(ConstraintBlock):
    """Start of the transmissions (14a)–(14b), store-and-forward (15), arrival (16a) and
    delivery to the ground station before the limit (16b)."""

    name = "store and forward"
    equations = "(14a)-(16b)"
    requires = frozenset({"rejection", "t", "u", "chi", "xi"})

    def rows(self, ctx: BuildContext, aux: None) -> Iterator[Row]:
        data = ctx.data
        for a in data.data:
            yield from self._start_after_production(ctx, a)
        for a in data.data:
            yield from self._store_and_forward(ctx, a)
        for a in data.data:
            yield from self._arrival_at_consumer(ctx, a)
        for a in data.data:
            yield from self._delivery_before_limit(ctx, a)

    def _start_after_production(self, ctx: BuildContext, a: DataKey) -> Iterator[Row]:
        data, variables = ctx.data, ctx.variables
        producer = data.producer(a)
        for e in data.arc_ids:
            xi = LinExpr.var(variables.xi(a, e))
            chi = variables.chi(a, e)
            if producer is None:
                # (14b) ξ_{a,e} ≥ b_r · χ_{a,e}
                rel = data.release(a.request_id)
                yield (xi - rel * chi >= 0).named(f"raw[{a},{e}]")
            else:
                # (14a) ξ_{a,e} ≥ t_i + Σ_s L̄_{i,s} x_{i,s} − τ̂_r (1 − χ_{a,e})
                big_m = data.limit(a.request_id)
                yield (
                    xi
                    - LinExpr.var(variables.t(producer))
                    - busy_time(ctx, producer)
                    + big_m * (1 - chi)
                    >= 0
                ).named(f"after-producer[{a},{e}]")

    def _store_and_forward(self, ctx: BuildContext, a: DataKey) -> Iterator[Row]:
        # (15) ξ_{a,e'} ≥ ξ_{a,e} + D_{a,e} + λ_e − M(2 − χ_{a,e} − χ_{a,e'})
        data, variables = ctx.data, ctx.variables
        for v in data.satellites:
            for e in data.incoming(v):
                time_e = data.transmission_time(a, e) + data.latency(e)
                big_m = data.limit(a.request_id) + time_e
                for e_next in data.outgoing(v):
                    yield (
                        LinExpr.var(variables.xi(a, e_next))
                        - LinExpr.var(variables.xi(a, e))
                        - time_e
                        + big_m * (2 - variables.chi(a, e) - variables.chi(a, e_next))
                        >= 0
                    ).named(f"store[{a},{e},{e_next}]")

    def _arrival_at_consumer(self, ctx: BuildContext, a: DataKey) -> Iterator[Row]:
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
                    - LinExpr.var(variables.xi(a, e))
                    - time_e
                    + big_m * (2 - variables.chi(a, e) - variables.x(consumer, v))
                    >= 0
                ).named(f"arrival[{a},{e},{v}]")

    def _delivery_before_limit(self, ctx: BuildContext, a: DataKey) -> Iterator[Row]:
        # (16b) ξ_{a,e} + D_{a,e} + λ_e ≤ τ̂_r + (D_{a,e} + λ_e)(1 − χ_{a,e}), for e into g
        data, variables = ctx.data, ctx.variables
        if not data.is_final(a):
            return
        ground = data.ground_station
        limit = data.limit(a.request_id)
        for e in data.incoming(ground):
            time_e = data.transmission_time(a, e) + data.latency(e)
            yield (
                (limit + time_e * (1 - variables.chi(a, e)))
                - LinExpr.var(variables.xi(a, e))
                - time_e
                >= 0
            ).named(f"deliver[{a},{e}]")
