"""The base of the constraint blocks: the rows of a block, and the way they reach the model.

A block is a class with a name, the equations of the document it writes, and the families of
variables it needs. Its only method to write is :meth:`ConstraintBlock.rows`, which returns the
rows; the template method :meth:`ConstraintBlock.build` writes them to the sink and counts
them, the same way for every block. The sink is the only thing a block can write to.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import ClassVar, final

from satedge_workflow_scheduler.solvers.milp.backends.ports import ModelSink
from satedge_workflow_scheduler.solvers.milp.expr import Row
from satedge_workflow_scheduler.solvers.milp.model_data import ModelData
from satedge_workflow_scheduler.solvers.milp.variables import DecisionVariables


@dataclass(frozen=True)
class BuildContext:
    """What a block reads while it writes its rows.

    Attributes:
        data: The data of the model, in the notation of the document.
        variables: The decision variables, shared by the blocks.
    """

    data: ModelData
    variables: DecisionVariables


class ConstraintBlock(ABC):
    """One family of constraints of the document.

    Subclasses set the class attributes below and implement :meth:`rows`.

    Attributes:
        name: Short name of the block, for the report.
        equations: The equations of the document that the block writes, for example "(2)".
        requires: The families of decision variables the block reads.
    """

    name: ClassVar[str]
    equations: ClassVar[str]
    requires: ClassVar[frozenset[str]]

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        for attribute in ("name", "equations", "requires"):
            if not hasattr(cls, attribute):
                raise TypeError(
                    f"{cls.__name__} must define the attribute {attribute!r}"
                )

    @abstractmethod
    def rows(self, ctx: BuildContext) -> Iterator[Row]:
        """The rows of the block, in order. Each row is named after its key in the model."""

    @final
    def build(self, ctx: BuildContext, sink: ModelSink) -> int:
        """Write the rows of the block to ``sink``, and return how many there are."""
        rows = list(self.rows(ctx))
        sink.add_rows(rows)
        return len(rows)
