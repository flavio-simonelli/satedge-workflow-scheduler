"""The transmissions on the arcs with visibility windows: they start and end in a window.

Equations (17b)–(17c) of the document, for every data ``a`` and every arc ``e`` that has
visibility windows. Equation (17a), ``Σ_h ν_{a,e,h} = χ_{a,e}``, is not written here: on these
arcs ``χ_{a,e}`` is defined as that sum (see :class:`RoutingVariables`), so it holds by
construction.

- (17b): the transmission starts after the opening of the window it uses, ``Σ_h ℓ^s_{e,h} ν_{a,e,h}
  ≤ ξ_{a,e}``, where ``ℓ^s_{e,h}`` is the start of the window ``h``;
- (17c): the transmission ends before the closing of the window it uses, ``ξ_{a,e} + (D_{a,e} +
  λ_e) χ_{a,e} ≤ Σ_h ℓ^e_{e,h} ν_{a,e,h}``, where ``ℓ^e_{e,h}`` is the end of the window ``h``.

The window is the one with ``ν_{a,e,h} = 1``. Both sums have no big-M: the variables ``ν``
select the window, and each constraint refers only to the window chosen. The sums run on the
compatible windows, and the arcs are those that ``a`` can cross (reduction 3).
"""

from __future__ import annotations

from collections.abc import Iterator

from satedge_workflow_scheduler.common.intervals import WindowSet
from satedge_workflow_scheduler.solvers.milp.blocks.base import (
    BuildContext,
    ConstraintBlock,
)
from satedge_workflow_scheduler.solvers.milp.blocks.path import RoutingVariables
from satedge_workflow_scheduler.solvers.milp.blocks.store_and_forward import (
    TransmissionStarts,
)
from satedge_workflow_scheduler.solvers.milp.expr import LinExpr, Row
from satedge_workflow_scheduler.solvers.milp.model_data import ArcId, DataKey


class LinkWindowBlock(ConstraintBlock):
    """Start (17b) and end (17c) of each transmission inside the visibility window used."""

    name = "link window"
    equations = "(17b)-(17c)"
    requires = frozenset({"xi", "chi", "nu"})

    def rows(self, ctx: BuildContext, aux: None) -> Iterator[Row]:
        data = ctx.data
        starts = ctx.families.get(TransmissionStarts)
        routing = ctx.families.get(RoutingVariables)
        for a in data.data:
            for e in ctx.reductions.arcs(a):
                windows = data.link_windows(e)
                if windows is None:
                    continue
                yield from self._one_transmission(ctx, starts, routing, a, e, windows)

    def _one_transmission(
        self,
        ctx: BuildContext,
        starts: TransmissionStarts,
        routing: RoutingVariables,
        a: DataKey,
        e: ArcId,
        windows: WindowSet,
    ) -> Iterator[Row]:
        xi = LinExpr.var(starts.xi(a, e))
        chi = routing.chi(a, e)
        time_e = ctx.data.transmission_time(a, e) + ctx.data.latency(e)

        used = routing.link_windows(a, e)
        opening = LinExpr.total(
            windows.windows[h].start * LinExpr.var(routing.nu(a, e, h)) for h in used
        )
        closing = LinExpr.total(
            windows.windows[h].end * LinExpr.var(routing.nu(a, e, h)) for h in used
        )
        # (17b) Σ_h ℓ^s_{e,h} ν_{a,e,h} ≤ ξ_{a,e}
        yield (opening <= xi).named(f"open[{a},{e}]")
        # (17c) ξ_{a,e} + (D + λ) χ_{a,e} ≤ Σ_h ℓ^e_{e,h} ν_{a,e,h}
        yield (xi + time_e * chi <= closing).named(f"close[{a},{e}]")
