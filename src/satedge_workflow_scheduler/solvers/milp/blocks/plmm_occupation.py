"""Occupation of the PLMM: when each data item starts and ends its stay in the memory of a satellite.

Equations (20a)–(21b) of the document, common to the two formulations of the PLMM capacity. The
occupation of the PLMM of ``v`` by the data ``a`` is an interval ``[ζ⁺_{a,v}, ζ⁻_{a,v}]``:

- (20a), (21a): the data starts when it is written on ``v`` by its producer, and ends when its
  consumer has read it;
- (20b): the raw data starts on its origin no later than the release of its request;
- (20c), (21b): the data starts when it arrives on ``v``, and ends when it leaves ``v``.

The formulation of the capacity follows: the flow (:mod:`.plmm_flow`) or the events
(:mod:`.plmm_events`). Both read the interval through the family ``occ``.
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
from satedge_workflow_scheduler.solvers.milp.blocks.path import RoutingVariables
from satedge_workflow_scheduler.solvers.milp.blocks.store_and_forward import (
    TransmissionStarts,
)
from satedge_workflow_scheduler.solvers.milp.expr import LinExpr, Row, Var
from satedge_workflow_scheduler.solvers.milp.model_data import DataKey

Occupation = tuple[DataKey, NodeId]
"""A data item and a PLMM: the key of its occupation."""


@dataclass(frozen=True)
class PlmmOccupation:
    """The start and the end of the occupation of each PLMM by each data item.

    Attributes:
        start: ``ζ⁺_{a,v}``, by data and satellite.
        end: ``ζ⁻_{a,v}``, by data and satellite.
    """

    start: Mapping[Occupation, Var]
    end: Mapping[Occupation, Var]


class PlmmOccupationBlock(OnSubset[NodeId], ConstraintBlock[PlmmOccupation]):
    """Occupation intervals of the PLMM (20a)–(21b)."""

    name = "plmm occupation"
    equations = "(20a)-(21b)"
    requires = frozenset({"rejection", "t", "u", "chi", "xi"})
    provides = frozenset({"occ"})

    def variables(self, ctx: BuildContext, sink: ModelSink) -> PlmmOccupation:
        capacity = ctx.data.plmm_big_m
        slots = [
            (a, v) for a in ctx.data.data for v in self.chosen(ctx.data.satellites)
        ]
        start = {
            (a, v): sink.add_var(0.0, capacity, VarKind.CONTINUOUS, f"occs[{a},{v}]")
            for a, v in slots
        }
        end = {
            (a, v): sink.add_var(0.0, capacity, VarKind.CONTINUOUS, f"occe[{a},{v}]")
            for a, v in slots
        }
        return PlmmOccupation(start, end)

    def rows(self, ctx: BuildContext, aux: PlmmOccupation) -> Iterator[Row]:
        data, variables = ctx.data, ctx.variables
        starts = ctx.families.get(TransmissionStarts)
        routing = ctx.families.get(RoutingVariables)
        big_m = data.plmm_big_m
        for a in data.data:
            producer = data.producer(a)
            consumer = data.consumer(a)
            for v in self.chosen(data.satellites):
                start = LinExpr.var(aux.start[(a, v)])
                end = LinExpr.var(aux.end[(a, v)])
                if producer is not None:
                    # (20a) ζ⁺_{a,v} ≤ t_i + L̄_{i,v} − w_{a,v} + M^P(1 − x_{i,v})
                    io = data.hpc_io_time(data.size(a), v)
                    yield (
                        LinExpr.var(variables.t(producer))
                        + data.duration(producer, v)
                        - io
                        + big_m * (1 - variables.x(producer, v))
                        - start
                        >= 0
                    ).named(f"start-produced[{a},{v}]")
                if consumer is not None:
                    # (21a) ζ⁻_{a,v} ≥ t_j + L̄_{j,v} − M^P(1 − x_{j,v})
                    yield (
                        end
                        - LinExpr.var(variables.t(consumer))
                        - data.duration(consumer, v)
                        + big_m * (1 - variables.x(consumer, v))
                        >= 0
                    ).named(f"end-consumed[{a},{v}]")
                for e in data.incoming(v):
                    # (20c) ζ⁺_{a,v} ≤ ξ_{a,e} + λ_e + M^P(1 − χ_{a,e})
                    yield (
                        LinExpr.var(starts.xi(a, e))
                        + data.latency(e)
                        + big_m * (1 - routing.chi(a, e))
                        - start
                        >= 0
                    ).named(f"start-received[{a},{e},{v}]")
                for e in data.outgoing(v):
                    # (21b) ζ⁻_{a,v} ≥ ξ_{a,e} + D_{a,e} − M^P(1 − χ_{a,e})
                    yield (
                        end
                        - LinExpr.var(starts.xi(a, e))
                        - data.transmission_time(a, e)
                        + big_m * (1 - routing.chi(a, e))
                        >= 0
                    ).named(f"end-sent[{a},{e},{v}]")
            # (20b) ζ⁺_{a,v} ≤ b_r for the raw data, on its origin
            origin = data.origin(a.request_id)
            if producer is None and origin in self.chosen(data.satellites):
                yield (
                    LinExpr.var(aux.start[(a, origin)]) <= data.release(a.request_id)
                ).named(f"start-raw[{a}]")
