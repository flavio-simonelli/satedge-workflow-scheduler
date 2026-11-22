"""The decision variables of the model: the ones that describe a solution, shared by the blocks.

They are created once, before the blocks, and read by several blocks, by the decoder and by the
warm start. The auxiliary variables that a single block needs to write its constraints (for
example the order of two tasks) are not here: each block creates them.

Domains follow the table of the variables of the document:

- ``ρ_r`` is binary, and it exists only under hard deadlines: under soft deadlines no request
  is rejected, and ``ρ_r`` is the constant 0;
- ``u_{i,s,w}`` is binary: the task ``i`` runs in the computing window ``w`` of ``s``;
- ``t_i`` is continuous in ``[0, τ̂_r − min_s L̄_{i,s}]``: the start of the task;
- ``χ_{a,e}`` is binary: the data ``a`` crosses the arc ``e``. On an arc that is always
  available it is a variable; on an arc with visibility windows it is the sum of the
  ``ν_{a,e,h}`` over the windows ``h`` (constraint (17a) of the document), so it is an
  expression, and it is integer because its terms are binary (reduction 6 again).
- ``ν_{a,e,h}`` is binary: the transmission of ``a`` on ``e`` happens in the window ``h``.
  It exists only on the arcs with windows, and only when a block requires ``"chi"`` or
  ``"nu"``.

There is no variable ``x_{i,s}``: constraint (5) of the document defines it as the sum of the
``u_{i,s,w}`` over the windows of ``s``, so ``x_{i,s}`` is an expression, and it is integer
because its terms are binary (reduction 6).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from satedge_workflow_scheduler.solvers.milp.backends.ports import ModelSink, VarKind
from satedge_workflow_scheduler.solvers.milp.errors import ModelBuildError
from satedge_workflow_scheduler.solvers.milp.expr import LinExpr, Var
from satedge_workflow_scheduler.solvers.milp.model_data import (
    ArcId,
    DataKey,
    ModelData,
    TaskKey,
)

DECISION_FAMILIES = frozenset({"rejection", "t", "u", "chi", "nu", "xi"})
"""The families of decision variables: the ones the blocks can read. ``chi`` and ``nu`` exist
only when a block requires them, because the levels without routing do not have them."""

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
        arc_chi: ``χ_{a,e}`` as a variable, by data and arc, for the arcs that are always
            available. Empty when the formulation has no routing.
        arc_start: ``ξ_{a,e}``, by data and arc: the start of the transmission of ``a`` on ``e``.
            Empty when the formulation has no store-and-forward.
        arc_windows: ``ν_{a,e,·}``, by data and arc, for the arcs with visibility windows: one
            variable per window of the arc, in chronological order.
    """

    requests: frozenset[str]
    rejected: Mapping[str, Var]
    start: Mapping[TaskKey, Var]
    in_window: Mapping[tuple[TaskKey, str], tuple[Var, ...]]
    arc_chi: Mapping[tuple[DataKey, ArcId], Var]
    arc_windows: Mapping[tuple[DataKey, ArcId], tuple[Var, ...]]
    arc_start: Mapping[tuple[DataKey, ArcId], Var]

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

    def chi(self, a: DataKey, arc: ArcId) -> LinExpr:
        """``χ_{a,e}`` as an expression: the variable on an arc that is always available, and
        the sum of the ``ν_{a,e,h}`` on an arc with windows, by (17a)."""
        if (a, arc) in self.arc_chi:
            return LinExpr.var(self.arc_chi[(a, arc)])
        windows = _lookup(self.arc_windows, (a, arc), "chi")
        return LinExpr.total(LinExpr.var(nu) for nu in windows)

    def xi(self, a: DataKey, arc: ArcId) -> Var:
        """``ξ_{a,e}``: start of the transmission of ``a`` on ``arc``."""
        return _lookup(self.arc_start, (a, arc), "xi")

    def nu(self, a: DataKey, arc: ArcId, window: WindowIndex) -> Var:
        """``ν_{a,e,h}``: 1 if the transmission of ``a`` on ``arc`` is in the window ``h``."""
        windows = _lookup(self.arc_windows, (a, arc), "nu")
        if not 0 <= window < len(windows):
            raise ModelBuildError(f"variable nu[{a},{arc},{window}] does not exist")
        return windows[window]

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


def create_decision_variables(
    sink: ModelSink,
    data: ModelData,
    families: frozenset[str] = DECISION_FAMILIES,
) -> DecisionVariables:
    """Create the decision variables of the model described by ``data``, on ``sink``.

    The families ``rejection``, ``t`` and ``u`` are always created. The others, such as ``chi``,
    only when they are in ``families``: the builder passes the ones its blocks require.

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
    # χ on the arcs always available, and nu on the arcs with windows: created only when a block
    # requires "chi" or "nu" (the routing levels), so the levels without routing have none.
    arc_chi: dict[tuple[DataKey, ArcId], Var] = {}
    arc_windows: dict[tuple[DataKey, ArcId], tuple[Var, ...]] = {}
    arc_start: dict[tuple[DataKey, ArcId], Var] = {}
    if "xi" in families:
        # ξ_{a,e} ∈ [0, τ̂_r]: the transmission starts before the time limit of its request
        for a in data.data:
            for e in data.arc_ids:
                arc_start[(a, e)] = sink.add_var(
                    0.0, data.limit(a.request_id), VarKind.CONTINUOUS, f"xi[{a},{e}]"
                )
    if families & {"chi", "nu"}:
        for a in data.data:
            for e in data.arc_ids:
                link_windows = data.link_windows(e)
                if link_windows is None:
                    if "chi" in families:
                        arc_chi[(a, e)] = sink.add_var(
                            0.0, 1.0, VarKind.BINARY, f"chi[{a},{e}]"
                        )
                    continue
                arc_windows[(a, e)] = tuple(
                    sink.add_var(0.0, 1.0, VarKind.BINARY, f"nu[{a},{e},{h}]")
                    for h in range(len(link_windows))
                )
    return DecisionVariables(
        frozenset(data.request_ids),
        rejected,
        start,
        in_window,
        arc_chi,
        arc_windows,
        arc_start,
    )


def _lookup[K, V](table: Mapping[K, V], key: K, family: str) -> V:
    try:
        return table[key]
    except KeyError:
        raise ModelBuildError(f"variable {family}{key} does not exist") from None
