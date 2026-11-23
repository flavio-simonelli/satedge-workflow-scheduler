"""The builder: it writes the model from the blocks of a formulation, in order.

The builder creates the decision variables, checks that every block finds the families it
requires, and then lets each block write its rows. It does not know the levels, the blocks, or
the document: it receives a list of blocks and runs it. The checks come before any row is
written, so a formulation that is not consistent leaves nothing behind in the model.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from satedge_workflow_scheduler.solvers.milp.backends.ports import ModelSink, VarKind
from satedge_workflow_scheduler.solvers.milp.blocks.base import (
    BuildContext,
    ConstraintBlock,
    Families,
)
from satedge_workflow_scheduler.solvers.milp.errors import ModelBuildError
from satedge_workflow_scheduler.solvers.milp.expr import Row, Var
from satedge_workflow_scheduler.solvers.milp.model_data import ModelData
from satedge_workflow_scheduler.solvers.milp.reductions import Reductions
from satedge_workflow_scheduler.solvers.milp.variables import (
    DECISION_FAMILIES,
    DecisionVariables,
    create_decision_variables,
)


class _TrackingSink:
    """The sink the blocks write to: it passes everything on, and keeps the binary variables."""

    def __init__(self, sink: ModelSink) -> None:
        self._sink = sink
        self.binaries: list[Var] = []

    def add_var(self, lb: float, ub: float, kind: VarKind, name: str = "") -> Var:
        var = self._sink.add_var(lb, ub, kind, name)
        if kind is VarKind.BINARY:
            self.binaries.append(var)
        return var

    def add_rows(self, rows: Iterable[Row]) -> None:
        self._sink.add_rows(rows)


@dataclass(frozen=True)
class BuiltModel:
    """What the builder wrote: the variables and the rows of each block.

    Attributes:
        variables: The decision variables, shared by the blocks.
        reductions: Which variables and constraints the model contains.
        rows: Number of rows written by each block, keyed by the name of the block, in the
            order the blocks were built.
        families: The auxiliary variables created by the blocks, by type.
        binaries: Every binary variable of the model, the decision ones and the auxiliary ones.
    """

    variables: DecisionVariables
    reductions: Reductions
    rows: Mapping[str, int] = field(default_factory=dict)
    families: Families = field(default_factory=Families)
    binaries: tuple[Var, ...] = ()


class ModelBuilder:
    """Writes the model of a formulation on a sink.

    Args:
        data: The data of the model.
        sink: Where the variables and the rows go.
        reductions: Which variables and constraints the model contains; all of them by default.
    """

    def __init__(
        self, data: ModelData, sink: ModelSink, reductions: Reductions | None = None
    ) -> None:
        self._data = data
        self._sink = sink
        self._reductions = (
            reductions if reductions is not None else Reductions.none(data)
        )

    def build(self, blocks: Sequence[ConstraintBlock[Any]]) -> BuiltModel:
        """Check the blocks, then write them in order.

        Raises:
            ModelBuildError: If two blocks have the same name, or if a block requires a family
                that no decision variable and no earlier block provides.
        """
        _check_formulation(blocks)
        tracking = _TrackingSink(self._sink)
        variables = create_decision_variables(tracking, self._data, self._reductions)
        families = Families()
        context = BuildContext(self._data, variables, self._reductions, families)

        rows: dict[str, int] = {}
        for block in blocks:
            rows[block.name] = block.build(context, tracking)
        return BuiltModel(
            variables, self._reductions, rows, families, tuple(tracking.binaries)
        )


def _check_formulation(blocks: Sequence[ConstraintBlock[Any]]) -> None:
    """The consistency of a formulation, checked before anything is written."""
    names = [block.name for block in blocks]
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        raise ModelBuildError(f"blocks with the same name: {repeated}")

    available = set(DECISION_FAMILIES)
    for block in blocks:
        missing = sorted(block.requires - available)
        if missing:
            raise ModelBuildError(
                f"block {block.name!r} requires {missing}, which no earlier block provides"
            )
        available |= block.provides
