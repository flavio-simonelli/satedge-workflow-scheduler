"""The base of the constraint blocks: the rows of a block, and the way they reach the model.

A block is a class with a name, the equations of the document it writes, and the families of
variables it needs and provides. It writes its rows through :meth:`ConstraintBlock.rows`, and it
may create variables first, in :meth:`ConstraintBlock.variables`. The template method
:meth:`ConstraintBlock.build` runs both, in order, and registers the variables in
:class:`Families`, so that the blocks that follow can read them by their type.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import ClassVar, cast, final

from satedge_workflow_scheduler.solvers.milp.backends.ports import ModelSink
from satedge_workflow_scheduler.solvers.milp.errors import ModelBuildError
from satedge_workflow_scheduler.solvers.milp.expr import Row
from satedge_workflow_scheduler.solvers.milp.model_data import ModelData
from satedge_workflow_scheduler.solvers.milp.variables import DecisionVariables


class Families:
    """The auxiliary variables that the blocks created so far, by the type of their container.

    A block that creates variables that other blocks read returns them, in a container, from
    :meth:`ConstraintBlock.variables`. The template method registers the container here, and a
    later block gets it back by its type. Asking for a container that no earlier block created is
    an error, so the order of the formulation is checked when it is built.
    """

    def __init__(self) -> None:
        self._containers: dict[type, object] = {}

    def add(self, container: object) -> None:
        """Register the container of a block, under its type."""
        self._containers[type(container)] = container

    def has(self, kind: type) -> bool:
        """Whether a container of type ``kind`` was created."""
        return kind in self._containers

    def get[T](self, kind: type[T]) -> T:
        """The container of type ``kind``, created by an earlier block.

        Raises:
            ModelBuildError: If no earlier block created it.
        """
        if kind not in self._containers:
            raise ModelBuildError(f"no earlier block created {kind.__name__}")
        return cast(T, self._containers[kind])


@dataclass(frozen=True)
class BuildContext:
    """What a block reads while it writes its rows.

    Attributes:
        data: The data of the model, in the notation of the document.
        variables: The decision variables, shared by the blocks.
        families: The auxiliary variables created by the earlier blocks.
    """

    data: ModelData
    variables: DecisionVariables
    families: Families = field(default_factory=Families)


class OnSubset[K]:
    """A block written on a subset of its keys only, all of them by default.

    The lazy strategy (rule 10 of the document) adds a group of constraints only on the keys
    that a solution violates, for example the satellites whose PLMM is overloaded: the blocks of
    the group receive those keys here.

    Args:
        keys: The keys that the block writes, or ``None`` for all of them.
    """

    def __init__(self, keys: Iterable[K] | None = None) -> None:
        self._only = None if keys is None else frozenset(keys)

    def chosen(self, keys: Iterable[K]) -> tuple[K, ...]:
        """The keys among ``keys`` that the block writes, in the same order."""
        if self._only is None:
            return tuple(keys)
        return tuple(k for k in keys if k in self._only)


class ConstraintBlock[A = None](ABC):
    """One family of constraints of the document.

    Subclasses set the class attributes below and implement :meth:`rows`.

    Attributes:
        name: Short name of the block, for the report.
        equations: The equations of the document that the block writes, for example "(2)".
        requires: The families of variables the block reads: decision variables, or the
            auxiliary ones that an earlier block provides.
        provides: The auxiliary families the block creates, for the blocks that follow it.
    """

    name: ClassVar[str]
    equations: ClassVar[str]
    requires: ClassVar[frozenset[str]]
    provides: ClassVar[frozenset[str]] = frozenset()

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        for attribute in ("name", "equations", "requires"):
            if not hasattr(cls, attribute):
                raise TypeError(
                    f"{cls.__name__} must define the attribute {attribute!r}"
                )

    def variables(self, ctx: BuildContext, sink: ModelSink) -> A:
        """Create the auxiliary variables of the block, before its rows. By default, none."""
        return cast(A, None)

    @abstractmethod
    def rows(self, ctx: BuildContext, aux: A) -> Iterator[Row]:
        """The rows of the block, in order, given the auxiliary variables it created."""

    @final
    def build(self, ctx: BuildContext, sink: ModelSink) -> int:
        """Create the auxiliary variables, register them, write the rows, and return how many."""
        aux = self.variables(ctx, sink)
        if aux is not None:
            ctx.families.add(aux)
        rows = list(self.rows(ctx, aux))
        sink.add_rows(rows)
        return len(rows)
