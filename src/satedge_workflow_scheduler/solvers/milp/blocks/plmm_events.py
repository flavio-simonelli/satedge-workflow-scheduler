"""Capacity of the PLMM, event formulation (OOE): the memory is checked at each event.

Equations of the second version of the capacity (section 6 of the document, version B). Each
PLMM has one event per data item, ``k = 0, …, |D̄| − 1``, with dates ``τ_{v,k}`` that do not
decrease. Each data item ``a`` occupies the memory at the events ``k`` where ``o_{a,v,k} = 1``:

- the events cover the presence: the data is present in ``v`` if and only if it occupies some
  event, and the occupied events form one contiguous run, (presence) and (contiguity);
- the run starts at the start of the occupation, (start), and ends with the end of the
  occupation, (end): the date of the event where the run starts is the start ``ζ⁺``, and the
  date of the event after the run is at least the end ``ζ⁻``;
- at each event the sum of the sizes of the data that occupy the memory is at most ``K_v``.

The occupations ``ζ⁺`` and ``ζ⁻`` are those of the common block (20)–(21), read through ``occ``.
This formulation has no order variable and no flow, but it has big-M constraints for each triple
(data, satellite, event), so it is the one the document compares with the flow.

Reduction 5: the PLMM of ``v`` holds only the data that can be in it, so it has one event per such
data item, and the block is written only on the satellites whose PLMM can be filled beyond its
capacity.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

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
from satedge_workflow_scheduler.solvers.milp.expr import LinExpr, Row, Var
from satedge_workflow_scheduler.solvers.milp.model_data import DataKey


@dataclass(frozen=True)
class EventVariables:
    """The dates of the events and the occupation of each event, by PLMM.

    Attributes:
        date: ``τ_{v,k}``, by satellite and event.
        on: ``o_{a,v,k}``, by data, satellite and event: 1 if the data occupies the memory of
            ``v`` at the event ``k``.
    """

    date: Mapping[tuple[NodeId, int], Var]
    on: Mapping[tuple[DataKey, NodeId, int], Var]


class PlmmEventsBlock(OnSubset[NodeId], ConstraintBlock[EventVariables]):
    """The capacity of the PLMM at each event (version B of the document)."""

    name = "plmm events"
    equations = "OOE (version B)"
    requires = frozenset({"rejection", "u", "chi", "occ"})

    def variables(self, ctx: BuildContext, sink: ModelSink) -> EventVariables:
        data, reductions = ctx.data, ctx.reductions
        satellites = self.chosen(reductions.plmm_satellites)
        capacity = data.plmm_big_m
        date = {
            (v, k): sink.add_var(0.0, capacity, VarKind.CONTINUOUS, f"tau[{v},{k}]")
            for v in satellites
            for k in range(len(reductions.plmm_data(v)))
        }
        on = {
            (a, v, k): sink.add_var(0.0, 1.0, VarKind.BINARY, f"o[{a},{v},{k}]")
            for a in data.data
            for v in satellites
            if reductions.in_plmm(a, v)
            for k in range(len(reductions.plmm_data(v)))
        }
        return EventVariables(date, on)

    def rows(self, ctx: BuildContext, aux: EventVariables) -> Iterator[Row]:
        data, reductions = ctx.data, ctx.reductions
        occupation = ctx.families.get(PlmmOccupation)
        satellites = self.chosen(reductions.plmm_satellites)
        big_m = data.plmm_big_m

        def on(a: DataKey, v: NodeId, k: int) -> LinExpr:
            # o_{a,v,−1} := 0: the first event has no predecessor
            return LinExpr.number(0.0) if k < 0 else LinExpr.var(aux.on[(a, v, k)])

        def date(v: NodeId, k: int) -> LinExpr:
            return LinExpr.var(aux.date[(v, k)])

        for v in satellites:
            # dates of the events do not decrease
            for k in range(len(reductions.plmm_data(v)) - 1):
                yield (date(v, k + 1) - date(v, k) >= 0).named(f"event-order[{v},{k}]")

        for a in data.data:
            for v in satellites:
                if not reductions.in_plmm(a, v):
                    continue
                count = len(reductions.plmm_data(v))
                q = presence(ctx, a, v)
                start = LinExpr.var(occupation.start[(a, v)])
                end = LinExpr.var(occupation.end[(a, v)])
                occupied = LinExpr.total(on(a, v, k) for k in range(count))
                # (presence) o_{a,v,k} ≤ q_{a,v}, and Σ_k o_{a,v,k} ≥ q_{a,v}
                for k in range(count):
                    yield (on(a, v, k) <= q).named(f"present[{a},{v},{k}]")
                yield (occupied >= q).named(f"present-some[{a},{v}]")
                for k in range(1, count):
                    # d_k = o_{a,v,k} − o_{a,v,k−1}: 1 only at the event where the run starts
                    step = on(a, v, k) - on(a, v, k - 1)
                    before = LinExpr.total(on(a, v, j) for j in range(k))
                    after = LinExpr.total(on(a, v, j) for j in range(k, count))
                    # (contiguity) the events before the run are k·(1 − d_k) in all
                    yield (k * (1 - step) - before >= 0).named(
                        f"contiguous-before[{a},{v},{k}]"
                    )
                    # (contiguity) the events after the run are (|D| − k)·(1 + d_k) in all
                    yield ((count - k) * (1 + step) - after >= 0).named(
                        f"contiguous-after[{a},{v},{k}]"
                    )
                for k in range(count):
                    step = on(a, v, k) - (
                        on(a, v, k - 1) if k > 0 else LinExpr.number(0.0)
                    )
                    # (start) τ_{v,k} = ζ⁺_{a,v} at the event where the run starts
                    yield (start + big_m * (1 - step) - date(v, k) >= 0).named(
                        f"start-up[{a},{v},{k}]"
                    )
                    yield (date(v, k) - start + big_m * (1 - step) >= 0).named(
                        f"start-down[{a},{v},{k}]"
                    )
                for k in range(count - 1):
                    # (end) τ_{v,k+1} ≥ ζ⁻_{a,v} when the run ends at k
                    yield (
                        date(v, k + 1)
                        - end
                        + big_m * (1 - on(a, v, k) + on(a, v, k + 1))
                        >= 0
                    ).named(f"end[{a},{v},{k}]")

        for v in satellites:
            members = reductions.plmm_data(v)
            for k in range(len(members)):
                # (capacity) Σ_a d_a o_{a,v,k} ≤ K_v
                used = LinExpr.total(data.size(a) * on(a, v, k) for a in members)
                yield (used <= data.plmm_capacity(v)).named(f"capacity[{v},{k}]")
