"""The path of each data item: the arcs it crosses, from its producer to its consumer.

Equations (11)–(13) of the document. The variable ``χ_{a,e}`` is 1 when the data ``a`` crosses
the arc ``e``, and the block creates it, with ``ν_{a,e,h}`` for the windows of the arcs that have
them: on those arcs ``χ`` is the sum of the windows (17a), so it is an expression. The three
equations describe a path that starts where the producer is and ends where the consumer is:

- (11): at every node, the data that leaves minus the data that enters is the departure of
  the producer minus the arrival of the consumer;
- (12): the data does not enter the node of its producer, nor leave the node of its consumer;
- (13): the data crosses no arc if its request is rejected.

Under soft deadlines no request is rejected, so (13) is ``χ ≤ 1``, which the binary domain
already gives: the block does not write it.

Reduction 3: ``χ_{a,e}`` exists only on the arcs that ``a`` can cross, and ``ν_{a,e,h}`` only on
the compatible windows. Elsewhere ``χ_{a,e}`` is the constant 0, and every block that writes a
constraint on the transmission of ``a`` on ``e`` writes it only where ``χ_{a,e}`` exists.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from satedge_workflow_scheduler.solvers.milp.backends.ports import ModelSink, VarKind
from satedge_workflow_scheduler.solvers.milp.blocks.base import (
    BuildContext,
    ConstraintBlock,
)
from satedge_workflow_scheduler.solvers.milp.blocks.terms import (
    consumed_on,
    produced_on,
)
from satedge_workflow_scheduler.solvers.milp.errors import ModelBuildError
from satedge_workflow_scheduler.solvers.milp.expr import LinExpr, Row, Var
from satedge_workflow_scheduler.solvers.milp.model_data import ArcId, DataKey


@dataclass(frozen=True)
class RoutingVariables:
    """``χ_{a,e}`` and ``ν_{a,e,h}``: the arcs each data item crosses, and the window of each.

    Attributes:
        crossed: ``χ_{a,e}`` as a variable, by data and arc, on the arcs always available.
        windows: ``ν_{a,e,h}``, by data and arc, then by the position of the window, on the arcs
            with visibility windows: only the compatible windows.
        removed: The pairs of data and arc that the reductions removed: ``χ_{a,e}`` is 0.
    """

    crossed: Mapping[tuple[DataKey, ArcId], Var]
    windows: Mapping[tuple[DataKey, ArcId], Mapping[int, Var]]
    removed: frozenset[tuple[DataKey, ArcId]] = frozenset()

    def chi(self, a: DataKey, arc: ArcId) -> LinExpr:
        """``χ_{a,e}``: the variable on an arc always available, the sum of the ``ν`` on an arc
        with windows (17a), the constant 0 on an arc that ``a`` cannot cross."""
        if (a, arc) in self.crossed:
            return LinExpr.var(self.crossed[(a, arc)])
        if (a, arc) in self.removed:
            return LinExpr.number(0.0)
        windows = self._windows(a, arc, "chi")
        return LinExpr.total(LinExpr.var(nu) for nu in windows.values())

    def link_windows(self, a: DataKey, arc: ArcId) -> tuple[int, ...]:
        """The windows of ``arc`` that have a variable ``ν_{a,e,h}``, in order."""
        return tuple(self._windows(a, arc, "nu"))

    def nu(self, a: DataKey, arc: ArcId, window: int) -> Var:
        """``ν_{a,e,h}``: 1 if the transmission of ``a`` on ``arc`` is in the window ``h``."""
        windows = self._windows(a, arc, "nu")
        if window not in windows:
            raise ModelBuildError(f"variable nu[{a},{arc},{window}] does not exist")
        return windows[window]

    def _windows(self, a: DataKey, arc: ArcId, family: str) -> Mapping[int, Var]:
        try:
            return self.windows[(a, arc)]
        except KeyError:
            raise ModelBuildError(
                f"variable {family}[{a},{arc}] does not exist"
            ) from None


class PathBlock(ConstraintBlock[RoutingVariables]):
    """Flow conservation (11), ends of the path (12) and rejected data (13)."""

    name = "path"
    equations = "(11)-(13)"
    requires = frozenset({"rejection", "u"})
    provides = frozenset({"chi", "nu"})

    def variables(self, ctx: BuildContext, sink: ModelSink) -> RoutingVariables:
        data, reductions = ctx.data, ctx.reductions
        crossed: dict[tuple[DataKey, ArcId], Var] = {}
        windows: dict[tuple[DataKey, ArcId], dict[int, Var]] = {}
        removed: set[tuple[DataKey, ArcId]] = set()
        for a in data.data:
            for e in data.arc_ids:
                if not reductions.crosses(a, e):
                    removed.add((a, e))
                elif data.link_windows(e) is None:
                    crossed[(a, e)] = sink.add_var(
                        0.0, 1.0, VarKind.BINARY, f"chi[{a},{e}]"
                    )
                else:
                    windows[(a, e)] = {
                        h: sink.add_var(0.0, 1.0, VarKind.BINARY, f"nu[{a},{e},{h}]")
                        for h in reductions.link_windows(a, e)
                    }
        return RoutingVariables(crossed, windows, frozenset(removed))

    def rows(self, ctx: BuildContext, aux: RoutingVariables) -> Iterator[Row]:
        data = ctx.data
        variables = ctx.variables
        for a in data.data:
            for v in data.nodes:
                leaving = LinExpr.total(aux.chi(a, e) for e in data.outgoing(v))
                entering = LinExpr.total(aux.chi(a, e) for e in data.incoming(v))
                # (11) Σ_out χ − Σ_in χ = asg_{i,v} − asg_{j,v}
                yield (
                    leaving - entering
                    == produced_on(ctx, a, v) - consumed_on(ctx, a, v)
                ).named(f"flow[{a},{v}]")
                # (12) Σ_in χ ≤ 1 − asg_{i,v} and Σ_out χ ≤ 1 − asg_{j,v}
                yield (entering <= 1 - produced_on(ctx, a, v)).named(
                    f"ends-in[{a},{v}]"
                )
                yield (leaving <= 1 - consumed_on(ctx, a, v)).named(
                    f"ends-out[{a},{v}]"
                )

        if not data.may_reject():
            return
        for a in data.data:
            for e in ctx.reductions.arcs(a):
                # (13) χ_{a,e} ≤ 1 − ρ_{reqof(a)}
                yield (aux.chi(a, e) <= 1 - variables.rejection(a.request_id)).named(
                    f"rejected[{a},{e}]"
                )
