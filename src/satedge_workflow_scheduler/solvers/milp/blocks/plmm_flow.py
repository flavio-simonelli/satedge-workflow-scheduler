"""Capacity of the PLMM, flow formulation (FCT): the memory of each satellite is a flow of data.

Equations (20a)–(27) of the document. The occupation of the PLMM of ``v`` by the data ``a`` is an
interval ``[ζ⁺_{a,v}, ζ⁻_{a,v}]``: it starts when the data is written, and it ends when the data
leaves. The block bounds the interval by the times of the producer, of the arrival, and of the
consumer and of the departure (20)–(21).

The capacity is then a flow. Each unit of memory of ``v`` goes from a data item to the next one,
when the first leaves the memory before the second enters it (``π_{a,b,v} = 1``), and the flow
enters from the source ``fini`` and leaves to the sink ``ffin``. The chain of each unit of
capacity contains data that do not overlap in time (23), and each present data receives exactly
its size (25)–(26). The total capacity is ``K_v`` (27). This needs no event and no ordering of
the events: the big-M of the document, ``M^P``, is the only one.

The data present in the PLMM of ``v`` is ``q_{a,v}``, the presence of the data: the expression
of :func:`~satedge_workflow_scheduler.solvers.milp.blocks.terms.presence`.

Reduction 5: the PLMM of ``v`` holds only the data that can be in it, and the block is written
only on the satellites whose PLMM can be filled beyond its capacity. When the stay of ``a`` always
ends before the stay of ``b`` begins, ``π_{a,b,v}`` is the constant 1 and ``π_{b,a,v}`` the
constant 0: the pair keeps its flow ``f_{a,b,v}``, loses ``f_{b,a,v}``, and needs neither (22) nor
(23), because the order is already guaranteed by the bounds.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from itertools import combinations, permutations

from satedge_workflow_scheduler.domain.nodes import NodeId
from satedge_workflow_scheduler.solvers.milp.backends.ports import ModelSink, VarKind
from satedge_workflow_scheduler.solvers.milp.blocks.base import (
    BuildContext,
    ConstraintBlock,
    OnSubset,
)
from satedge_workflow_scheduler.solvers.milp.blocks.plmm_occupation import (
    PlmmOccupation,
)
from satedge_workflow_scheduler.solvers.milp.blocks.presence import presence
from satedge_workflow_scheduler.solvers.milp.errors import ModelBuildError
from satedge_workflow_scheduler.solvers.milp.expr import LinExpr, Row, Var
from satedge_workflow_scheduler.solvers.milp.model_data import DataKey

Pair = tuple[DataKey, DataKey]
Ordered = tuple[DataKey, DataKey, NodeId]
"""An ordered pair of data and a PLMM: the key of ``π`` and of ``f``."""
Slot = tuple[DataKey, NodeId]
"""A data item and a PLMM: the key of its occupation."""


@dataclass(frozen=True)
class FlowVariables:
    """The order and the flows of the PLMMs.

    Attributes:
        before: ``π_{a,b,v}``, by ordered pair of data and satellite: 1 if ``a`` leaves the PLMM
            of ``v`` before ``b`` enters it. Only the orders that the model decides.
        flow: ``f_{a,b,v}``, by ordered pair and satellite: capacity that passes from ``a`` to
            ``b`` in the PLMM of ``v``. Not on the pairs whose order is fixed to 0.
        from_source: ``fini_{b,v}``: capacity that ``b`` takes from the source.
        to_sink: ``ffin_{a,v}``: capacity that ``a`` gives back to the sink.
        unused: ``finifin_v``: capacity that goes from the source to the sink, never used.
        fixed: The orders fixed by the reductions: ``π_{a,b,v}`` is the constant 1 or 0.
    """

    before: Mapping[Ordered, Var]
    flow: Mapping[Ordered, Var]
    from_source: Mapping[Slot, Var]
    to_sink: Mapping[Slot, Var]
    unused: Mapping[NodeId, Var]
    fixed: Mapping[Ordered, bool] = field(default_factory=dict)

    def pi(self, a: DataKey, b: DataKey, v: NodeId) -> LinExpr:
        """``π_{a,b,v}``: the variable, or the constant of a fixed order."""
        if (a, b, v) in self.before:
            return LinExpr.var(self.before[(a, b, v)])
        if (a, b, v) in self.fixed:
            return LinExpr.number(1.0 if self.fixed[(a, b, v)] else 0.0)
        raise ModelBuildError(f"variable pi[{a},{b},{v}] does not exist")


class PlmmFlowBlock(OnSubset[NodeId], ConstraintBlock[FlowVariables]):
    """The flow of the capacity of the PLMM (22)–(27), on the occupations of (20)–(21)."""

    name = "plmm flow"
    equations = "(22)-(27)"
    requires = frozenset({"rejection", "u", "chi", "occ"})

    def variables(self, ctx: BuildContext, sink: ModelSink) -> FlowVariables:
        data, reductions = ctx.data, ctx.reductions
        satellites = self.chosen(reductions.plmm_satellites)
        slots = [
            (a, v) for a in data.data for v in satellites if reductions.in_plmm(a, v)
        ]
        ordered = [
            (a, b, v)
            for a, b in permutations(data.data, 2)
            for v in satellites
            if reductions.in_plmm(a, v) and reductions.in_plmm(b, v)
        ]
        fixed = {
            key: order
            for key in ordered
            if (order := reductions.plmm_order(*key)) is not None
        }
        before = {
            (a, b, v): sink.add_var(0.0, 1.0, VarKind.BINARY, f"pi[{a},{b},{v}]")
            for a, b, v in ordered
            if (a, b, v) not in fixed
        }
        flow = {
            (a, b, v): sink.add_var(
                0.0, float("inf"), VarKind.CONTINUOUS, f"f[{a},{b},{v}]"
            )
            for a, b, v in ordered
            if fixed.get((a, b, v), True)
        }
        from_source = {
            (b, v): sink.add_var(
                0.0, float("inf"), VarKind.CONTINUOUS, f"fini[{b},{v}]"
            )
            for b, v in slots
        }
        to_sink = {
            (a, v): sink.add_var(
                0.0, float("inf"), VarKind.CONTINUOUS, f"ffin[{a},{v}]"
            )
            for a, v in slots
        }
        unused = {
            v: sink.add_var(0.0, float("inf"), VarKind.CONTINUOUS, f"finifin[{v}]")
            for v in satellites
        }
        return FlowVariables(before, flow, from_source, to_sink, unused, fixed)

    def rows(self, ctx: BuildContext, aux: FlowVariables) -> Iterator[Row]:
        occupation = ctx.families.get(PlmmOccupation)
        yield from self._order(ctx, aux, occupation)
        yield from self._flow(ctx, aux)

    def _order(
        self, ctx: BuildContext, aux: FlowVariables, occupation: PlmmOccupation
    ) -> Iterator[Row]:
        data = ctx.data
        big_m = data.plmm_big_m
        satellites = self.chosen(ctx.reductions.plmm_satellites)
        # (22) π_{a,b,v} + π_{b,a,v} ≤ 1, for a before b, when the model decides the order
        for a, b in combinations(data.data, 2):
            for v in satellites:
                if (a, b, v) not in aux.before:
                    continue
                yield (
                    LinExpr.var(aux.before[(a, b, v)])
                    + LinExpr.var(aux.before[(b, a, v)])
                    <= 1
                ).named(f"order[{a},{b},{v}]")
        for a, b in permutations(data.data, 2):
            for v in satellites:
                if (a, b, v) not in aux.before:
                    continue
                # (23) ζ⁺_{b,v} ≥ ζ⁻_{a,v} − M^P(1 − π_{a,b,v})
                yield (
                    LinExpr.var(occupation.start[(b, v)])
                    - LinExpr.var(occupation.end[(a, v)])
                    + big_m * (1 - LinExpr.var(aux.before[(a, b, v)]))
                    >= 0
                ).named(f"after-leaving[{a},{b},{v}]")

    def _flow(self, ctx: BuildContext, aux: FlowVariables) -> Iterator[Row]:
        data = ctx.data
        satellites = self.chosen(ctx.reductions.plmm_satellites)
        for a, b in permutations(data.data, 2):
            for v in satellites:
                if (a, b, v) not in aux.flow:
                    continue
                # (24) f_{a,b,v} ≤ min(d_a, d_b) π_{a,b,v}
                limit = min(data.size(a), data.size(b))
                yield (
                    LinExpr.var(aux.flow[(a, b, v)]) <= limit * aux.pi(a, b, v)
                ).named(f"flow-order[{a},{b},{v}]")
        for a in data.data:
            for v in satellites:
                if (a, v) not in aux.to_sink:
                    continue
                # (25) Σ_{b≠a} f_{a,b,v} + ffin_{a,v} = d_a q_{a,v}
                outgoing = LinExpr.total(
                    LinExpr.var(aux.flow[(a, b, v)])
                    for b in data.data
                    if (a, b, v) in aux.flow
                )
                yield (
                    outgoing + LinExpr.var(aux.to_sink[(a, v)])
                    == data.size(a) * presence(ctx, a, v)
                ).named(f"outflow[{a},{v}]")
                # (26) Σ_{a≠b} f_{a,b,v} + fini_{b,v} = d_b q_{b,v}, for the data b = a of the loop
                incoming = LinExpr.total(
                    LinExpr.var(aux.flow[(b, a, v)])
                    for b in data.data
                    if (b, a, v) in aux.flow
                )
                yield (
                    incoming + LinExpr.var(aux.from_source[(a, v)])
                    == data.size(a) * presence(ctx, a, v)
                ).named(f"inflow[{a},{v}]")
        for v in satellites:
            # (27) Σ_b fini_{b,v} + finifin_v = K_v
            total = LinExpr.total(
                LinExpr.var(aux.from_source[(b, v)])
                for b in data.data
                if (b, v) in aux.from_source
            )
            yield (total + LinExpr.var(aux.unused[v]) == data.plmm_capacity(v)).named(
                f"capacity[{v}]"
            )
