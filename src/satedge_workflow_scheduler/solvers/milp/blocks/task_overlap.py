"""No overlap of the tasks of different requests on the same HPC window.

Equations (8)–(10) of the document. For each pair ``(i, j)`` of tasks of different requests:

- (8): ``same_{ij}`` is at least 1 if the two tasks use the same window of the same satellite;
- (9): if ``i`` precedes ``j`` (``order_{ij} = 1``) and they share a window, ``j`` starts after
  ``i`` has finished;
- (10): the converse, if ``j`` precedes ``i``.

The big-M of the pair is the larger time limit of the two requests, as in the document. The
auxiliary variables belong to this block: no other block reads them.

Reduction 2: the pairs and the windows are those of the reductions. (8) is written only on the
windows where the two tasks can be at the same time, and a pair with no such window has no
variable and no row: the two tasks can never overlap.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from satedge_workflow_scheduler.solvers.milp.backends.ports import ModelSink, VarKind
from satedge_workflow_scheduler.solvers.milp.blocks.base import (
    BuildContext,
    ConstraintBlock,
)
from satedge_workflow_scheduler.solvers.milp.blocks.terms import busy_time
from satedge_workflow_scheduler.solvers.milp.expr import LinExpr, Row, Var
from satedge_workflow_scheduler.solvers.milp.model_data import TaskKey

Pair = tuple[TaskKey, TaskKey]


@dataclass(frozen=True)
class OverlapVariables:
    """The auxiliary variables of the overlap, by pair of tasks.

    Attributes:
        same: ``same_{ij}``, continuous in ``[0, 1]``.
        order: ``order_{ij}``, binary: 1 if ``i`` precedes ``j`` when they share a window.
    """

    same: Mapping[Pair, Var]
    order: Mapping[Pair, Var]


class TaskOverlapBlock(ConstraintBlock[OverlapVariables]):
    """The tasks of different requests do not overlap in the same window."""

    name = "task overlap"
    equations = "(8)-(10)"
    requires = frozenset({"t", "u"})
    provides = frozenset({"same", "order"})

    def variables(self, ctx: BuildContext, sink: ModelSink) -> OverlapVariables:
        same: dict[Pair, Var] = {}
        order: dict[Pair, Var] = {}
        for i, j in ctx.reductions.task_pairs:
            same[(i, j)] = sink.add_var(0.0, 1.0, VarKind.CONTINUOUS, f"same[{i},{j}]")
            order[(i, j)] = sink.add_var(0.0, 1.0, VarKind.BINARY, f"order[{i},{j}]")
        return OverlapVariables(same, order)

    def rows(self, ctx: BuildContext, aux: OverlapVariables) -> Iterator[Row]:
        variables = ctx.variables
        for i, j in ctx.reductions.task_pairs:
            big_m = max(ctx.data.limit(i.request_id), ctx.data.limit(j.request_id))
            same = LinExpr.var(aux.same[(i, j)])
            order = LinExpr.var(aux.order[(i, j)])

            # (8) same ≥ u_{i,s,w} + u_{j,s,w} − 1, on the windows both can share
            for s, w in ctx.reductions.shared_windows(i, j):
                shared = LinExpr.var(variables.u(i, s, w)) + LinExpr.var(
                    variables.u(j, s, w)
                )
                yield (same >= shared - 1).named(f"same[{i},{j},{s},{w}]")

            # (9) t_j ≥ t_i + busy_i − M(1 − order) − M(1 − same)
            yield (
                LinExpr.var(variables.t(j))
                - LinExpr.var(variables.t(i))
                - busy_time(ctx, i)
                + big_m * (1 - order)
                + big_m * (1 - same)
                >= 0
            ).named(f"before[{i},{j}]")

            # (10) t_i ≥ t_j + busy_j − M·order − M(1 − same)
            yield (
                LinExpr.var(variables.t(i))
                - LinExpr.var(variables.t(j))
                - busy_time(ctx, j)
                + big_m * order
                + big_m * (1 - same)
                >= 0
            ).named(f"after[{i},{j}]")
