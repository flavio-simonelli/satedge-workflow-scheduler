"""The decision variables of the model: the ones that describe a solution, shared by the blocks.

They are created once, before the blocks, and read by several blocks, by the decoder and by the
warm start. Their domains follow the table of the variables of the document:

- ``ρ_r`` is binary, and it exists only under hard deadlines: under soft deadlines no request
  is rejected, and ``ρ_r`` is the constant 0;
- ``u_{i,s,w}`` is binary: the task ``i`` runs in the computing window ``w`` of ``s``;
- ``t_i`` is continuous in ``[0, τ̂_r − min_s L̄_{i,s}]``: the start of the task.

There is no variable ``x_{i,s}``: constraint (5) of the document defines it as the sum of the
``u_{i,s,w}`` over the windows of ``s``, so ``x_{i,s}`` is an expression, and it is integer
because its terms are binary (reduction 6).

The :class:`~satedge_workflow_scheduler.solvers.milp.reductions.Reductions` say which of them
exist. ``u_{i,s,w}`` exists only on the windows they keep (reduction 1), so ``x_{i,s}`` is 0 where
the task cannot run. A request they exclude, which can never be served, is rejected a priori:
its ``ρ_r`` is the constant 1, and the starts of its tasks are the constant 0, as for any
rejected request.

The auxiliary variables of the routing, of the transmissions and of the delivery are not here:
each is created by the block that needs it first, and the blocks that follow read it from
:class:`~satedge_workflow_scheduler.solvers.milp.blocks.base.Families`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from satedge_workflow_scheduler.solvers.milp.backends.ports import ModelSink, VarKind
from satedge_workflow_scheduler.solvers.milp.errors import ModelBuildError
from satedge_workflow_scheduler.solvers.milp.expr import LinExpr, Var
from satedge_workflow_scheduler.solvers.milp.model_data import ModelData, TaskKey
from satedge_workflow_scheduler.solvers.milp.reductions import Reductions

DECISION_FAMILIES = frozenset({"rejection", "t", "u"})
"""The families of decision variables, created before the blocks: every block can read them."""

WindowIndex = int
"""Position of a computing window in the windows of its satellite, in chronological order."""


@dataclass(frozen=True)
class DecisionVariables:
    """The handles of the decision variables, indexed by the keys of the document.

    Attributes:
        requests: Identifiers of the requests of the model.
        rejected: ``ρ_r``, by request. Only under hard deadlines, and not for the excluded
            requests: under soft deadlines no request is rejected and the variables do not exist.
        start: ``t_i``, by task.
        in_window: ``u_{i,s,w}``, by task and satellite, then by the position of the window: only
            the windows that the reductions keep, in chronological order.
        excluded: The requests rejected a priori, whose ``ρ_r`` is the constant 1.
    """

    requests: frozenset[str]
    rejected: Mapping[str, Var]
    start: Mapping[TaskKey, Var]
    in_window: Mapping[tuple[TaskKey, str], Mapping[WindowIndex, Var]]
    excluded: frozenset[str] = frozenset()

    def rejection(self, request_id: str) -> LinExpr:
        """``ρ_r`` as an expression: the variable under hard deadlines, the constant 0 under
        soft deadlines, where no request is rejected, and the constant 1 for a request rejected a
        priori. The blocks write ``1 - rejection(r)`` in every case, and the constant terms
        disappear from the rows."""
        if request_id not in self.requests:
            raise ModelBuildError(f"request {request_id!r} is not in the model")
        if request_id in self.excluded:
            return LinExpr.number(1.0)
        if request_id not in self.rejected:
            return LinExpr.number(0.0)
        return LinExpr.var(self.rejected[request_id])

    def windows(self, i: TaskKey, satellite: str) -> tuple[WindowIndex, ...]:
        """The windows of ``satellite`` that have a variable ``u_{i,s,w}``, in order."""
        return tuple(_lookup(self.in_window, (i, satellite), "u"))

    def x(self, i: TaskKey, satellite: str) -> LinExpr:
        """``x_{i,s}``, as the expression ``Σ_w u_{i,s,w}`` of constraint (5): 0 if the task has
        no window on ``satellite``."""
        windows = _lookup(self.in_window, (i, satellite), "x")
        return LinExpr.total(LinExpr.var(u) for u in windows.values())

    def t(self, i: TaskKey) -> Var:
        """``t_i``: start of the task ``i``."""
        return _lookup(self.start, i, "t")

    def u(self, i: TaskKey, satellite: str, window: WindowIndex) -> Var:
        """``u_{i,s,w}``: 1 if the task ``i`` runs in the window ``w`` of ``satellite``."""
        windows = _lookup(self.in_window, (i, satellite), "u")
        if window not in windows:
            raise ModelBuildError(
                f"variable u[{i},{satellite},{window}] does not exist"
            )
        return windows[window]


def create_decision_variables(
    sink: ModelSink, data: ModelData, reductions: Reductions | None = None
) -> DecisionVariables:
    """Create the decision variables of the model described by ``data``, on ``sink``.

    Args:
        sink: Where the variables go.
        data: The data of the model.
        reductions: Which variables exist; all of them by default.

    Raises:
        ModelBuildError: If a task of a request in the model cannot start in time on any
            satellite, so its start has no feasible domain.
    """
    if reductions is None:
        reductions = Reductions.none(data)

    rejected: dict[str, Var] = {}
    if data.may_reject():
        for request_id in data.request_ids:
            if not reductions.is_excluded(request_id):
                rejected[request_id] = sink.add_var(
                    0.0, 1.0, VarKind.BINARY, f"rho[{request_id}]"
                )

    start: dict[TaskKey, Var] = {}
    in_window: dict[tuple[TaskKey, str], dict[WindowIndex, Var]] = {}
    for i in data.tasks:
        if reductions.is_excluded(i.request_id):
            # rejected a priori: the start is 0, as for every rejected request
            start[i] = sink.add_var(0.0, 0.0, VarKind.CONTINUOUS, f"t[{i}]")
            for s in data.satellites:
                in_window[(i, s)] = {}
            continue
        limit = data.limit(i.request_id)
        fastest = min(data.duration(i, s) for s in data.satellites)
        if limit - fastest < 0:
            raise ModelBuildError(f"task {i} cannot finish by {limit} on any satellite")
        start[i] = sink.add_var(0.0, limit - fastest, VarKind.CONTINUOUS, f"t[{i}]")
        for s in data.satellites:
            in_window[(i, s)] = {
                w: sink.add_var(0.0, 1.0, VarKind.BINARY, f"u[{i},{s},{w}]")
                for w in reductions.task_windows(i, s)
            }
    return DecisionVariables(
        frozenset(data.request_ids), rejected, start, in_window, reductions.excluded
    )


def _lookup[K, V](table: Mapping[K, V], key: K, family: str) -> V:
    try:
        return table[key]
    except KeyError:
        raise ModelBuildError(f"variable {family}{key} does not exist") from None
