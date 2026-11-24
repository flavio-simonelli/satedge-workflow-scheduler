"""Tests of the builder: the checks come before any row, and the blocks are written in order."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from sample import sample_instance

from satedge_workflow_scheduler.scheduling.deadline import DeadlineMode
from satedge_workflow_scheduler.solvers.milp.backends.fake_recording import (
    FakeRecordingBackend,
)
from satedge_workflow_scheduler.solvers.milp.blocks.assignment import AssignmentBlock
from satedge_workflow_scheduler.solvers.milp.blocks.base import (
    BuildContext,
    ConstraintBlock,
)
from satedge_workflow_scheduler.solvers.milp.blocks.task_timing import TaskTimingBlock
from satedge_workflow_scheduler.solvers.milp.builder import ModelBuilder
from satedge_workflow_scheduler.solvers.milp.errors import ModelBuildError
from satedge_workflow_scheduler.solvers.milp.expr import Row
from satedge_workflow_scheduler.solvers.milp.model_data import ModelData


def _block(
    name: str,
    requires: frozenset[str] = frozenset(),
    provides: frozenset[str] = frozenset(),
) -> ConstraintBlock:
    """A block that writes nothing: only its declarations matter to the builder."""

    def rows(self: ConstraintBlock, ctx: BuildContext, aux: None) -> Iterator[Row]:
        return iter(())

    cls: type[ConstraintBlock] = type(
        f"Block_{name}",
        (ConstraintBlock,),
        {
            "name": name,
            "equations": "(0)",
            "requires": requires,
            "provides": provides,
            "rows": rows,
        },
    )
    return cls()


def _data() -> ModelData:
    return ModelData(sample_instance(), DeadlineMode.HARD)


def test_builds_the_blocks_in_order_and_counts_their_rows() -> None:
    backend = FakeRecordingBackend()
    built = ModelBuilder(_data(), backend).build([AssignmentBlock(), TaskTimingBlock()])
    assert list(built.rows) == ["assignment", "task timing"]
    assert built.rows["assignment"] == 3  # one per task
    assert built.rows["task timing"] == 1 + 2  # one precedence, two releases
    assert len(backend.rows) == 3 + 1 + 2


def test_the_decision_variables_are_created_before_the_rows() -> None:
    backend = FakeRecordingBackend()
    built = ModelBuilder(_data(), backend).build([AssignmentBlock()])
    assert built.variables is not None
    assert backend.variables  # the variables are in the model
    assert backend.rows[0].name == "assignment[r1.0]"


def test_a_required_family_nobody_provides_is_an_error_before_any_row() -> None:
    backend = FakeRecordingBackend()
    block = _block("needy", requires=frozenset({"ghost"}))
    builder = ModelBuilder(_data(), backend)
    blocks = [AssignmentBlock(), block]
    with pytest.raises(ModelBuildError, match="ghost"):
        builder.build(blocks)
    assert backend.rows == []  # nothing was written


def test_a_family_provided_by_an_earlier_block_satisfies_a_later_one() -> None:
    backend = FakeRecordingBackend()
    producer = _block("producer", provides=frozenset({"bins"}))
    consumer = _block("consumer", requires=frozenset({"bins"}))
    built = ModelBuilder(_data(), backend).build([producer, consumer])
    assert built.rows == {"producer": 0, "consumer": 0}


def test_a_family_provided_by_a_later_block_is_not_enough() -> None:
    backend = FakeRecordingBackend()
    consumer = _block("consumer", requires=frozenset({"bins"}))
    producer = _block("producer", provides=frozenset({"bins"}))
    builder = ModelBuilder(_data(), backend)
    blocks = [consumer, producer]
    with pytest.raises(ModelBuildError, match="bins"):
        builder.build(blocks)


def test_two_blocks_with_the_same_name_are_an_error() -> None:
    backend = FakeRecordingBackend()
    builder = ModelBuilder(_data(), backend)
    blocks = [_block("dup"), _block("dup")]
    with pytest.raises(ModelBuildError, match="same name"):
        builder.build(blocks)
    assert backend.variables == []


def test_the_decision_families_are_available_to_every_block() -> None:
    backend = FakeRecordingBackend()
    built = ModelBuilder(_data(), backend).build(
        [_block("reader", requires=frozenset({"t", "u"}))]
    )
    assert built.rows == {"reader": 0}


def test_the_overlap_block_provides_the_order_and_same_families() -> None:
    from satedge_workflow_scheduler.solvers.milp.blocks.task_overlap import (
        TaskOverlapBlock,
    )

    overlap = TaskOverlapBlock()
    assert {"same", "order"} <= overlap.provides
    reader = _block("reader", requires=frozenset({"order"}))
    backend = FakeRecordingBackend()
    ModelBuilder(_data(), backend).build([overlap, reader])  # no error: provided before
    builder = ModelBuilder(_data(), FakeRecordingBackend())
    with pytest.raises(ModelBuildError, match="reader"):
        builder.build([reader, overlap])
