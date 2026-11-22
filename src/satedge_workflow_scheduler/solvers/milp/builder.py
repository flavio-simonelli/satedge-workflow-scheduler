"""The builder: it writes the model from the blocks of a formulation, in order.

The builder creates the decision variables, checks that every block finds the families it
requires, and then lets each block write its rows. It does not know the levels, the blocks, or
the document: it receives a list of blocks and runs it. The checks come before any row is
written, so a formulation that is not consistent leaves nothing behind in the model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from satedge_workflow_scheduler.solvers.milp.backends.ports import ModelSink
from satedge_workflow_scheduler.solvers.milp.blocks.base import (
    BuildContext,
    ConstraintBlock,
    Families,
)
from satedge_workflow_scheduler.solvers.milp.errors import ModelBuildError
from satedge_workflow_scheduler.solvers.milp.model_data import ModelData
from satedge_workflow_scheduler.solvers.milp.variables import (
    DECISION_FAMILIES,
    DecisionVariables,
    create_decision_variables,
)


@dataclass(frozen=True)
class BuiltModel:
    """What the builder wrote: the variables and the rows of each block.

    Attributes:
        variables: The decision variables, shared by the blocks.
        rows: Number of rows written by each block, keyed by the name of the block, in the
            order the blocks were built.
        families: The auxiliary variables created by the blocks, by type.
    """

    variables: DecisionVariables
    rows: Mapping[str, int] = field(default_factory=dict)
    families: Families = field(default_factory=Families)


class ModelBuilder:
    """Writes the model of a formulation on a sink.

    Args:
        data: The data of the model.
        sink: Where the variables and the rows go.
    """

    def __init__(self, data: ModelData, sink: ModelSink) -> None:
        self._data = data
        self._sink = sink

    def build(self, blocks: Sequence[ConstraintBlock[Any]]) -> BuiltModel:
        """Check the blocks, then write them in order.

        Raises:
            ModelBuildError: If two blocks have the same name, or if a block requires a family
                that no decision variable and no earlier block provides.
        """
        _check_formulation(blocks)
        variables = create_decision_variables(self._sink, self._data)
        families = Families()
        context = BuildContext(self._data, variables, families)

        rows: dict[str, int] = {}
        for block in blocks:
            rows[block.name] = block.build(context, self._sink)
        return BuiltModel(variables, rows, families)


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
