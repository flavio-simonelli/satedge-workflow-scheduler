"""Level 3: the data travel with a fixed delay between the satellites, and no routing.

Equations of the document's level 2, kept by level 3 (sections 2 and 3). A precedence between two
tasks of a request is a transfer of the output of the first to the second, and its delay depends
only on the satellites where the two tasks run: ``γ^{hk}(d)``, the least delay of ``d`` Gb from
``h`` to ``k`` (:meth:`ModelData.fixed_delay`). The pair of satellites is the product
``z^{hk}_{ij} = x_{ih} x_{jk}``, written as a linear system (the compact linearization of the
document, ``l0-lin-colonne`` and ``l0-lin-righe``):

- :class:`FixedRouteBlock` creates the ``z`` and writes the linearization;
- :class:`FixedDelayBlock` writes the precedences with their delays, the release of the raw data,
  the delivery of the final output and its deadline. It provides the delivery ``δ_r``, so the
  objective and the tardiness of the levels above work unchanged.

There are no transmissions at level 3, so no variable of a transfer and no visibility window:
the delays are the lower bounds of the transfers of the next levels.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from satedge_workflow_scheduler.domain.nodes import NodeId
from satedge_workflow_scheduler.solvers.milp.backends.ports import ModelSink, VarKind
from satedge_workflow_scheduler.solvers.milp.blocks.base import (
    BuildContext,
    ConstraintBlock,
)
from satedge_workflow_scheduler.solvers.milp.blocks.delivery import DeliveryTimes
from satedge_workflow_scheduler.solvers.milp.blocks.terms import busy_time
from satedge_workflow_scheduler.solvers.milp.errors import ModelBuildError
from satedge_workflow_scheduler.solvers.milp.expr import LinExpr, Row, Var
from satedge_workflow_scheduler.solvers.milp.model_data import DataKey, TaskKey

Route = tuple[TaskKey, TaskKey, NodeId, NodeId]
"""A precedence ``(i, j)`` and the satellites ``h`` of ``i`` and ``k`` of ``j``: the key of ``z``."""


@dataclass(frozen=True)
class FixedRoute:
    """``z^{hk}_{ij}``: 1 if ``i`` runs on ``h`` and ``j`` on ``k``.

    Attributes:
        z: The variable of each precedence and pair of satellites.
    """

    z: Mapping[Route, Var]

    def at(self, i: TaskKey, j: TaskKey, h: NodeId, k: NodeId) -> Var:
        try:
            return self.z[(i, j, h, k)]
        except KeyError:
            raise ModelBuildError(
                f"variable z[{i},{j},{h},{k}] does not exist"
            ) from None


class FixedRouteBlock(ConstraintBlock[FixedRoute]):
    """The pair of satellites of each precedence, linearized (level 2, eq. 32–33 of the paper)."""

    name = "fixed route"
    equations = "lin-colonne, lin-righe"
    requires = frozenset({"u", "rejection"})
    provides = frozenset({"z"})

    def variables(self, ctx: BuildContext, sink: ModelSink) -> FixedRoute:
        data = ctx.data
        z = {
            (i, j, h, k): sink.add_var(
                0.0, 1.0, VarKind.CONTINUOUS, f"z[{i},{j},{h},{k}]"
            )
            for i, j in data.precedences
            for h in data.satellites
            for k in data.satellites
        }
        return FixedRoute(z)

    def rows(self, ctx: BuildContext, aux: FixedRoute) -> Iterator[Row]:
        data, variables = ctx.data, ctx.variables
        for i, j in data.precedences:
            for k in data.satellites:
                # Σ_h z^{hk}_{ij} = x_{jk}
                placed = LinExpr.total(
                    LinExpr.var(aux.at(i, j, h, k)) for h in data.satellites
                )
                yield (placed == variables.x(j, k)).named(f"route-col[{i},{j},{k}]")
            for h in data.satellites:
                # Σ_k z^{hk}_{ij} = x_{ih}
                placed = LinExpr.total(
                    LinExpr.var(aux.at(i, j, h, k)) for k in data.satellites
                )
                yield (placed == variables.x(i, h)).named(f"route-row[{i},{j},{h}]")


class FixedDelayBlock(ConstraintBlock[DeliveryTimes]):
    """Precedences (3) with the fixed delays, the release of the raw data, the delivery and the
    deadline of the final output (level 2, eq. l2-comunicazione, l2-raw, l2-consegna, l2-scadenza).
    """

    name = "fixed delay"
    equations = "l2-comunicazione, l2-raw, l2-consegna, l2-scadenza"
    requires = frozenset({"rejection", "t", "u", "z"})
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
        routes = ctx.families.get(FixedRoute)
        yield from self._precedences(ctx, routes)
        for request_id in data.request_ids:
            yield from self._raw(ctx, request_id)
            yield from self._delivery(ctx, aux, request_id)

    def _precedences(self, ctx: BuildContext, routes: FixedRoute) -> Iterator[Row]:
        data, variables = ctx.data, ctx.variables
        for i, j in data.precedences:
            # the output of i is the data of index i + 1 of its request
            size = data.size(DataKey(i.request_id, i.task_index + 1))
            # t_j − t_i − Σ_s L̄_{i,s} x_{i,s} − Σ_{h,k} γ^{hk}(d) z^{hk}_{ij} ≥ 0
            delays = LinExpr.total(
                data.fixed_delay(h, k, size) * LinExpr.var(routes.at(i, j, h, k))
                for h in data.satellites
                for k in data.satellites
            )
            yield (
                LinExpr.var(variables.t(j))
                - LinExpr.var(variables.t(i))
                - busy_time(ctx, i)
                - delays
                >= 0
            ).named(f"precedence[{i},{j}]")

    def _raw(self, ctx: BuildContext, request_id: str) -> Iterator[Row]:
        # t_first ≥ b_r (1 − ρ_r) + Σ_s γ^{orig,s}(d_in) x_{first,s}
        data, variables = ctx.data, ctx.variables
        first = TaskKey(request_id, 0)
        origin = data.origin(request_id)
        raw_size = data.size(DataKey(request_id, 0))
        delays = LinExpr.total(
            data.fixed_delay(origin, s, raw_size) * variables.x(first, s)
            for s in data.satellites
        )
        available = data.release(request_id) * (1 - variables.rejection(request_id))
        yield (LinExpr.var(variables.t(first)) - available - delays >= 0).named(
            f"raw[{request_id}]"
        )

    def _delivery(
        self, ctx: BuildContext, aux: DeliveryTimes, request_id: str
    ) -> Iterator[Row]:
        # δ_r ≥ t_last + Σ_s L̄_{last,s} x_{last,s} + Σ_s γ^{s,g}(d_out) x_{last,s}, and ≤ τ̂_r
        data, variables = ctx.data, ctx.variables
        final = data.final_data(request_id)
        last = TaskKey(request_id, final.data_index - 1)
        size = data.size(final)
        to_ground = LinExpr.total(
            data.fixed_delay(s, data.ground_station, size) * variables.x(last, s)
            for s in data.satellites
        )
        finish = LinExpr.var(variables.t(last)) + busy_time(ctx, last) + to_ground
        yield (LinExpr.var(aux.deliv(request_id)) - finish >= 0).named(
            f"delivery[{request_id}]"
        )
        yield (finish <= data.limit(request_id)).named(f"deadline[{request_id}]")
