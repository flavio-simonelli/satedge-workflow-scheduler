"""The strengthenings of the optimizations: constraints that tighten the relaxation.

Rules 7 and 8 of the document (section 6). They are optional: the configuration chooses them, and
they add or replace rows of the model, but they do not change its optimum.

- Rule 7, arrival without big-M (:class:`ArrivalWithoutBigMBlock`): the start of a transmission is
  zero on the arcs the data does not cross, and the arrival and the delivery are written without
  big-M. This replaces (16a)–(16b), which the configuration then leaves out.
- Rule 8, contact capacity (:class:`ContactCapacityBlock`): in each window of an arc, the
  transmissions chosen there last no longer than the window. It is a valid cut, implied by the
  disjunctive constraints of the arcs with windows.
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


class ArrivalWithoutBigMBlock(ConstraintBlock):
    """Rule 7: ``ξ_{a,e} ≤ τ̂ χ_{a,e}``, and the arrival and the delivery without big-M."""

    name = "arrival without big-m"
    equations = "rule 7"
    requires = frozenset({"rejection", "t", "u", "chi", "xi"})

    def rows(self, ctx: BuildContext, aux: None) -> Iterator[Row]:
        data, variables = ctx.data, ctx.variables
        starts = ctx.families.get(TransmissionStarts)
        routing = ctx.families.get(RoutingVariables)
        for a in data.data:
            limit = data.limit(a.request_id)
            arcs = ctx.reductions.arcs(a)
            for e in arcs:
                # ξ_{a,e} ≤ τ̂_r χ_{a,e}: the transmission starts at zero on an unused arc
                yield (
                    LinExpr.var(starts.xi(a, e)) - limit * routing.chi(a, e) <= 0
                ).named(f"unused[{a},{e}]")
            consumer = data.consumer(a)
            if consumer is not None:
                # t_j ≥ ξ_{a,e} + (D_{a,e} + λ_e) χ_{a,e}, on every arc: the last hop is the latest
                for e in arcs:
                    time_e = data.transmission_time(a, e) + data.latency(e)
                    yield (
                        LinExpr.var(variables.t(consumer))
                        - LinExpr.var(starts.xi(a, e))
                        - time_e * routing.chi(a, e)
                        >= 0
                    ).named(f"arrival-free[{a},{e}]")
            if data.is_final(a):
                # ξ_{a,e} + (D_{a,e} + λ_e) χ_{a,e} ≤ τ̂_r, on every arc
                for e in arcs:
                    time_e = data.transmission_time(a, e) + data.latency(e)
                    yield (
                        limit
                        - LinExpr.var(starts.xi(a, e))
                        - time_e * routing.chi(a, e)
                        >= 0
                    ).named(f"delivery-free[{a},{e}]")


class ContactCapacityBlock(ConstraintBlock):
    """Rule 8: in each window ``h`` of an arc, ``Σ_a D_{a,e} ν_{a,e,h} ≤ ℓ^e_{e,h} − ℓ^s_{e,h}``."""

    name = "contact capacity"
    equations = "rule 8"
    requires = frozenset({"nu"})

    def rows(self, ctx: BuildContext, aux: None) -> Iterator[Row]:
        data = ctx.data
        routing = ctx.families.get(RoutingVariables)
        for e in data.arc_ids:
            windows = data.link_windows(e)
            if windows is None:
                continue
            users = [a for a in data.data if ctx.reductions.crosses(a, e)]
            for h, window in enumerate(windows):
                used = LinExpr.total(
                    data.transmission_time(a, e) * LinExpr.var(routing.nu(a, e, h))
                    for a in users
                    if h in routing.link_windows(a, e)
                )
                yield (used <= window.end - window.start).named(f"contact[{e},{h}]")
