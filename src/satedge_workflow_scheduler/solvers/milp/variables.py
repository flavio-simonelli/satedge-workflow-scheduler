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

DECISION_FAMILIES = frozenset({"rejection", "t", "u"})
"""The families of decision variables, created before the blocks: every block can read them."""

WindowIndex = int
"""Position of a computing window in the windows of its satellite, in chronological order."""


@dataclass(frozen=True)
class DecisionVariables:
    """The handles of the decision variables, indexed by the keys of the document.

    Attributes:
        requests: Identifiers of the requests of the model.
        rejected: ``ρ_r``, by request. Only under hard deadlines: under soft deadlines no
            request is rejected and the variables do not exist.
        start: ``t_i``, by task.
        in_window: ``u_{i,s,·}``, by task and satellite: one variable per computing window of
            the satellite, in chronological order.
    """

    requests: frozenset[str]
    rejected: Mapping[str, Var]
    start: Mapping[TaskKey, Var]
    in_window: Mapping[tuple[TaskKey, str], tuple[Var, ...]]

    def rejection(self, request_id: str) -> LinExpr:
        """``ρ_r`` as an expression: the variable under hard deadlines, the constant 0 under
        soft deadlines, where no request is rejected. The blocks write ``1 - rejection(r)``
        in both cases, and the zero term disappears from the rows."""
        if request_id not in self.requests:
            raise ModelBuildError(f"request {request_id!r} is not in the model")
        if request_id not in self.rejected:
            return LinExpr.number(0.0)
        return LinExpr.var(self.rejected[request_id])

    def x(self, i: TaskKey, satellite: str) -> LinExpr:
        """``x_{i,s}``, as the expression ``Σ_w u_{i,s,w}`` of constraint (5)."""
        windows = _lookup(self.in_window, (i, satellite), "x")
        return LinExpr.total(LinExpr.var(u) for u in windows)

    def t(self, i: TaskKey) -> Var:
        """``t_i``: start of the task ``i``."""
        return _lookup(self.start, i, "t")

    def u(self, i: TaskKey, satellite: str, window: WindowIndex) -> Var:
        """``u_{i,s,w}``: 1 if the task ``i`` runs in the window ``w`` of ``satellite``."""
        windows = _lookup(self.in_window, (i, satellite), "u")
        if not 0 <= window < len(windows):
            raise ModelBuildError(
                f"variable u[{i},{satellite},{window}] does not exist"
            )
        return windows[window]


def create_decision_variables(sink: ModelSink, data: ModelData) -> DecisionVariables:
    """Create the decision variables of the model described by ``data``, on ``sink``.

    Raises:
        ModelBuildError: If a task cannot start in time on any satellite, so its start has no
            feasible domain.
    """
    rejected: dict[str, Var] = {}
    if data.may_reject():
        for request_id in data.request_ids:
            rejected[request_id] = sink.add_var(
                0.0, 1.0, VarKind.BINARY, f"rho[{request_id}]"
            )

    start: dict[TaskKey, Var] = {}
    in_window: dict[tuple[TaskKey, str], tuple[Var, ...]] = {}
    for i in data.tasks:
        limit = data.limit(i.request_id)
        fastest = min(data.duration(i, s) for s in data.satellites)
        if limit - fastest < 0:
            raise ModelBuildError(f"task {i} cannot finish by {limit} on any satellite")
        start[i] = sink.add_var(0.0, limit - fastest, VarKind.CONTINUOUS, f"t[{i}]")
        for s in data.satellites:
            in_window[(i, s)] = tuple(
                sink.add_var(0.0, 1.0, VarKind.BINARY, f"u[{i},{s},{w}]")
                for w in range(len(data.hpc_windows(s)))
            )
    return DecisionVariables(frozenset(data.request_ids), rejected, start, in_window)


def _lookup[K, V](table: Mapping[K, V], key: K, family: str) -> V:
    try:
        return table[key]
    except KeyError:
        raise ModelBuildError(f"variable {family}{key} does not exist") from None
