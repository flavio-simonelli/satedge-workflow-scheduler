"""The online state: the committed calendar is fixed, and the new plan has a release on each node.

Section 7 of the document, equations (30)–(31). The online state belongs to the instance, not to
a level: the committed requests are in the model with their variables, and two blocks fix them
to the committed calendar and impose the release of the plan on the new requests.

- :class:`CommittedTasksBlock`, at every level: ``ρ_r = 0`` for a committed request, its tasks
  on the committed satellite and window (``u``) from the committed start (``t``), and the release
  (30) of the tasks of the new requests.
- :class:`CommittedTransfersBlock`, only where the transmissions are routed: the committed
  transmissions on their arcs (``χ``, ``ν``) from their committed start (``ξ``), and the
  release (31) of the transmissions of the new requests.

Where the plan never reaches a satellite (``b^plan = +∞``), the new requests do not use it: no
task runs there and no data leaves it. The ground station has no plan release.

Which variables of the committed requests are fixed is decided in the ``_fix_*`` methods only:
all of them today. A replanning that may change part of a committed calendar would change those
methods, and nothing else.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

from satedge_workflow_scheduler.solvers.milp.blocks.base import (
    BuildContext,
    ConstraintBlock,
)
from satedge_workflow_scheduler.solvers.milp.blocks.path import RoutingVariables
from satedge_workflow_scheduler.solvers.milp.blocks.store_and_forward import (
    TransmissionStarts,
)
from satedge_workflow_scheduler.solvers.milp.errors import ModelBuildError
from satedge_workflow_scheduler.solvers.milp.expr import LinExpr, Row
from satedge_workflow_scheduler.solvers.milp.model_data import DataKey, TaskKey


class CommittedTasksBlock(ConstraintBlock):
    """The committed requests accepted with their tasks fixed, and the release (30)."""

    name = "committed tasks"
    equations = "(30)"
    requires = frozenset({"rejection", "t", "u"})

    def rows(self, ctx: BuildContext, aux: None) -> Iterator[Row]:
        data, variables = ctx.data, ctx.variables
        for r in data.request_ids:
            if data.is_committed(r) and data.may_reject():
                yield (variables.rejection(r) == 0).named(f"committed-accepted[{r}]")
        for i in data.tasks:
            if data.is_committed(i.request_id):
                yield from self._fix_task(ctx, i)
            else:
                yield from self._release_task(ctx, i)

    def _fix_task(self, ctx: BuildContext, i: TaskKey) -> Iterator[Row]:
        data, variables = ctx.data, ctx.variables
        fixed = data.committed_execution(i)
        if fixed.window not in variables.windows(i, fixed.satellite):
            raise ModelBuildError(f"the committed window of {i} is not in the model")
        for s in data.satellites:
            for w in variables.windows(i, s):
                chosen = 1.0 if (s, w) == (fixed.satellite, fixed.window) else 0.0
                yield (LinExpr.var(variables.u(i, s, w)) == chosen).named(
                    f"committed-u[{i},{s},{w}]"
                )
        yield (LinExpr.var(variables.t(i)) == fixed.start).named(f"committed-t[{i}]")

    def _release_task(self, ctx: BuildContext, i: TaskKey) -> Iterator[Row]:
        data, variables = ctx.data, ctx.variables
        reached = []
        for s in data.satellites:
            if math.isinf(data.plan_release(s)):
                # the plan never reaches s: no new task runs there
                yield (variables.x(i, s) == 0).named(f"unreached-task[{i},{s}]")
            else:
                reached.append(s)
        # (30) t_i ≥ Σ_s b^plan_s x_{i,s}, on the satellites the plan reaches
        released = LinExpr.total(
            data.plan_release(s) * variables.x(i, s)
            for s in reached
            if data.plan_release(s) > 0
        )
        if released.terms:
            yield (LinExpr.var(variables.t(i)) >= released).named(f"release-task[{i}]")


class CommittedTransfersBlock(ConstraintBlock):
    """The committed transmissions fixed on their arcs, and the release (31)."""

    name = "committed transfers"
    equations = "(31)"
    requires = frozenset({"chi", "nu", "xi"})

    def rows(self, ctx: BuildContext, aux: None) -> Iterator[Row]:
        data = ctx.data
        for a in data.data:
            if data.is_committed(a.request_id):
                yield from self._fix_data(ctx, a)
            else:
                yield from self._release_data(ctx, a)

    def _fix_data(self, ctx: BuildContext, a: DataKey) -> Iterator[Row]:
        data = ctx.data
        routing = ctx.families.get(RoutingVariables)
        starts = ctx.families.get(TransmissionStarts)
        crossed = data.committed_transfers(a)
        arcs = ctx.reductions.arcs(a)
        for e, committed in crossed.items():
            if e not in arcs or (
                committed.window is not None
                and committed.window not in routing.link_windows(a, e)
            ):
                raise ModelBuildError(
                    f"the committed transmission of {a} on arc {e} is not in the model"
                )
        for e in arcs:
            hop = crossed.get(e)
            if data.link_windows(e) is None:
                used = 1.0 if hop is not None else 0.0
                yield (routing.chi(a, e) == used).named(f"committed-chi[{a},{e}]")
            else:
                for h in routing.link_windows(a, e):
                    chosen = 1.0 if hop is not None and hop.window == h else 0.0
                    yield (LinExpr.var(routing.nu(a, e, h)) == chosen).named(
                        f"committed-nu[{a},{e},{h}]"
                    )
            start = hop.start if hop is not None else 0.0
            yield (LinExpr.var(starts.xi(a, e)) == start).named(
                f"committed-xi[{a},{e}]"
            )

    def _release_data(self, ctx: BuildContext, a: DataKey) -> Iterator[Row]:
        data = ctx.data
        routing = ctx.families.get(RoutingVariables)
        starts = ctx.families.get(TransmissionStarts)
        for e in ctx.reductions.arcs(a):
            tail, _ = data.endpoints(e)
            if tail not in data.satellites:
                continue  # the ground station has no plan release
            release = data.plan_release(tail)
            if math.isinf(release):
                # the plan never reaches the tail: no new data leaves it
                yield (routing.chi(a, e) == 0).named(f"unreached-arc[{a},{e}]")
            elif release > 0:
                # (31) ξ_{a,e} ≥ b^plan_u χ_{a,e}
                yield (
                    LinExpr.var(starts.xi(a, e)) - release * routing.chi(a, e) >= 0
                ).named(f"release-transmission[{a},{e}]")
