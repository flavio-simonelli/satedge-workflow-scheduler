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
        windows: ``ν_{a,e,·}``, by data and arc, on the arcs with visibility windows.
    """

    crossed: Mapping[tuple[DataKey, ArcId], Var]
    windows: Mapping[tuple[DataKey, ArcId], tuple[Var, ...]]

    def chi(self, a: DataKey, arc: ArcId) -> LinExpr:
        """``χ_{a,e}``: the variable on an arc always available, the sum of the ``ν`` on an arc
        with windows (17a)."""
        if (a, arc) in self.crossed:
            return LinExpr.var(self.crossed[(a, arc)])
        windows = self._windows(a, arc, "chi")
        return LinExpr.total(LinExpr.var(nu) for nu in windows)

    def nu(self, a: DataKey, arc: ArcId, window: int) -> Var:
        """``ν_{a,e,h}``: 1 if the transmission of ``a`` on ``arc`` is in the window ``h``."""
        windows = self._windows(a, arc, "nu")
        if not 0 <= window < len(windows):
            raise ModelBuildError(f"variable nu[{a},{arc},{window}] does not exist")
        return windows[window]

    def _windows(self, a: DataKey, arc: ArcId, family: str) -> tuple[Var, ...]:
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
        crossed: dict[tuple[DataKey, ArcId], Var] = {}
        windows: dict[tuple[DataKey, ArcId], tuple[Var, ...]] = {}
        for a in ctx.data.data:
            for e in ctx.data.arc_ids:
                link_windows = ctx.data.link_windows(e)
                if link_windows is None:
                    crossed[(a, e)] = sink.add_var(
                        0.0, 1.0, VarKind.BINARY, f"chi[{a},{e}]"
                    )
                else:
                    windows[(a, e)] = tuple(
                        sink.add_var(0.0, 1.0, VarKind.BINARY, f"nu[{a},{e},{h}]")
                        for h in range(len(link_windows))
                    )
        return RoutingVariables(crossed, windows)

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
            for e in data.arc_ids:
                # (13) χ_{a,e} ≤ 1 − ρ_{reqof(a)}
                yield (aux.chi(a, e) <= 1 - variables.rejection(a.request_id)).named(
                    f"rejected[{a},{e}]"
                )
