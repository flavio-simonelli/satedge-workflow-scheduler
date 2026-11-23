"""When each transmission starts: after the production, store-and-forward, and the arrival.

Equations (14a)–(16b) of the document. The variable ``ξ_{a,e}`` is the start of the transmission
of the data ``a`` on the arc ``e``, and the block creates it. ``χ_{a,e}`` says whether ``a``
crosses ``e`` at all, and it comes from the path block: the constraints of an arc not crossed
are relaxed by the big-M, so they do not bind.

- (14a): a transmission starts after the producer has finished, if the data leaves through the
  arc. The producer occupies its HPC for ``Σ_s L̄_{i,s} x_{i,s}``, which is ``L̄_i (1 − ρ_r)``.
- (14b): the raw data is available from the release of its request.
- (15), store-and-forward: at a satellite, the data that arrives on an arc leaves on the next
  arc after the arrival, if it is stored there.

The time of an arc is ``D_{a,e} + λ_e``: the transmission plus the latency. The big-M of
(14a) and (15) is the time limit of the request plus that time, as in the document. The arrival
at the consumer and the delivery, (16a)–(16b), are in :mod:`.arrival`.

``ξ_{a,e}`` exists only where ``χ_{a,e}`` does (reduction 3): on an arc that ``a`` never crosses
its constraints are relaxed by the big-M in every solution, so they and the variable go.
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
from satedge_workflow_scheduler.solvers.milp.blocks.terms import busy_time
from satedge_workflow_scheduler.solvers.milp.errors import ModelBuildError
from satedge_workflow_scheduler.solvers.milp.expr import LinExpr, Row, Var
from satedge_workflow_scheduler.solvers.milp.model_data import ArcId, DataKey


@dataclass(frozen=True)
class TransmissionStarts:
    """``ξ_{a,e}``: the start of the transmission of the data ``a`` on the arc ``e``."""

    start: Mapping[tuple[DataKey, ArcId], Var]

    def xi(self, a: DataKey, arc: ArcId) -> Var:
        try:
            return self.start[(a, arc)]
        except KeyError:
            raise ModelBuildError(f"variable xi[{a},{arc}] does not exist") from None


class StoreAndForwardBlock(ConstraintBlock[TransmissionStarts]):
    """Start of the transmissions (14a)–(14b) and store-and-forward (15)."""

    name = "store and forward"
    equations = "(14a)-(15)"
    requires = frozenset({"rejection", "t", "u", "chi"})
    provides = frozenset({"xi"})

    def variables(self, ctx: BuildContext, sink: ModelSink) -> TransmissionStarts:
        start: dict[tuple[DataKey, ArcId], Var] = {}
        for a in ctx.data.data:
            for e in ctx.reductions.arcs(a):
                # ξ_{a,e} ∈ [0, τ̂_r]: the transmission starts before the limit of its request
                start[(a, e)] = sink.add_var(
                    0.0,
                    ctx.data.limit(a.request_id),
                    VarKind.CONTINUOUS,
                    f"xi[{a},{e}]",
                )
        return TransmissionStarts(start)

    def rows(self, ctx: BuildContext, aux: TransmissionStarts) -> Iterator[Row]:
        routing = ctx.families.get(RoutingVariables)
        data = ctx.data
        for a in data.data:
            yield from self._start_after_production(ctx, aux, routing, a)
        for a in data.data:
            yield from self._store_and_forward(ctx, aux, routing, a)

    def _start_after_production(
        self,
        ctx: BuildContext,
        starts: TransmissionStarts,
        routing: RoutingVariables,
        a: DataKey,
    ) -> Iterator[Row]:
        data, variables = ctx.data, ctx.variables
        producer = data.producer(a)
        for e in ctx.reductions.arcs(a):
            xi = LinExpr.var(starts.xi(a, e))
            chi = routing.chi(a, e)
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

    def _store_and_forward(
        self,
        ctx: BuildContext,
        starts: TransmissionStarts,
        routing: RoutingVariables,
        a: DataKey,
    ) -> Iterator[Row]:
        # (15) ξ_{a,e'} ≥ ξ_{a,e} + D_{a,e} + λ_e − M(2 − χ_{a,e} − χ_{a,e'})
        data, reductions = ctx.data, ctx.reductions
        for v in data.satellites:
            for e in data.incoming(v):
                if not reductions.crosses(a, e):
                    continue
                time_e = data.transmission_time(a, e) + data.latency(e)
                big_m = data.limit(a.request_id) + time_e
                for e_next in data.outgoing(v):
                    if not reductions.crosses(a, e_next):
                        continue
                    yield (
                        LinExpr.var(starts.xi(a, e_next))
                        - LinExpr.var(starts.xi(a, e))
                        - time_e
                        + big_m * (2 - routing.chi(a, e) - routing.chi(a, e_next))
                        >= 0
                    ).named(f"store[{a},{e},{e_next}]")
