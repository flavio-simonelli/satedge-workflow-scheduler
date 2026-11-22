"""The formulation of a level: the list of blocks that the model of a configuration contains.

This is the only place that knows which blocks make a level, and it follows the tables of the
document. Each level lists all its blocks, instead of adding the blocks of the previous level,
because some blocks replace others: at level 3 the precedences come with the fixed delays of
the transmissions (``FixedDelayBlock``), and from level 4 they come with the transmissions
themselves (``TaskTimingBlock``).

A block is named here before it is implemented. Asking for a block that has no class yet is an
error, not a silent omission: a model without a block is a different model.
"""

from __future__ import annotations

from typing import Any

from satedge_workflow_scheduler.scheduling.deadline import DeadlineMode
from satedge_workflow_scheduler.solvers.milp.blocks.assignment import AssignmentBlock
from satedge_workflow_scheduler.solvers.milp.blocks.base import ConstraintBlock
from satedge_workflow_scheduler.solvers.milp.blocks.computing_window import (
    ComputingWindowBlock,
)
from satedge_workflow_scheduler.solvers.milp.blocks.delivery import DeliveryBlock
from satedge_workflow_scheduler.solvers.milp.blocks.link_contention import (
    LinkContentionBlock,
)
from satedge_workflow_scheduler.solvers.milp.blocks.link_window import LinkWindowBlock
from satedge_workflow_scheduler.solvers.milp.blocks.path import PathBlock
from satedge_workflow_scheduler.solvers.milp.blocks.store_and_forward import (
    StoreAndForwardBlock,
)
from satedge_workflow_scheduler.solvers.milp.blocks.tardiness import TardinessBlock
from satedge_workflow_scheduler.solvers.milp.blocks.task_overlap import TaskOverlapBlock
from satedge_workflow_scheduler.solvers.milp.blocks.task_timing import TaskTimingBlock
from satedge_workflow_scheduler.solvers.milp.config import Level, MilpConfig
from satedge_workflow_scheduler.solvers.milp.errors import ModelBuildError

_FIXED_DELAY_LEVEL: tuple[str, ...] = (
    "assignment",
    "computing_window",
    "task_overlap",
    "fixed_delay",
)

_ROUTED_LEVEL: tuple[str, ...] = (
    "assignment",
    "computing_window",
    "task_overlap",
    "task_timing",
    "path",
    "store_and_forward",
    "link_window",
    "delivery",
)

_BLOCKS_OF_LEVEL: dict[Level, tuple[str, ...]] = {
    Level.FIXED_DELAY_WINDOWS: _FIXED_DELAY_LEVEL,
    Level.ROUTING: _ROUTED_LEVEL,
    Level.LINK_CONTENTION: (*_ROUTED_LEVEL, "link_contention"),
    Level.PLMM_CAPACITY: (*_ROUTED_LEVEL, "link_contention", "plmm"),
    Level.ONLINE: (*_ROUTED_LEVEL, "link_contention", "plmm", "online"),
}

_STRENGTHENINGS: dict[str, str] = {
    "arrival_without_big_m": "arrival_without_big_m",
    "contact_capacity": "contact_capacity",
}

IMPLEMENTED: dict[str, type[ConstraintBlock[Any]]] = {
    "assignment": AssignmentBlock,
    "computing_window": ComputingWindowBlock,
    "path": PathBlock,
    "store_and_forward": StoreAndForwardBlock,
    "link_window": LinkWindowBlock,
    "link_contention": LinkContentionBlock,
    "delivery": DeliveryBlock,
    "tardiness": TardinessBlock,
    "task_overlap": TaskOverlapBlock,
    "task_timing": TaskTimingBlock,
}
"""The blocks with a class. Each block is registered here when it is written."""


def block_names(config: MilpConfig) -> tuple[str, ...]:
    """The names of the blocks of the model of ``config``, in the order they are built.

    The names are those of the document's levels, and they are returned whether or not the
    block is implemented.
    """
    names = list(_BLOCKS_OF_LEVEL[config.level])
    if config.deadline is DeadlineMode.SOFT:
        names.append("tardiness")
    for flag, name in _STRENGTHENINGS.items():
        if getattr(config.strengthening, flag):
            names.append(name)
    return tuple(names)


def blocks_for(config: MilpConfig) -> tuple[ConstraintBlock, ...]:
    """The blocks of the model of ``config``, instantiated and in the order they are built.

    Raises:
        ModelBuildError: If a block of the level has no class yet.
    """
    missing = [name for name in block_names(config) if name not in IMPLEMENTED]
    if missing:
        raise ModelBuildError(f"blocks not implemented yet: {', '.join(missing)}")
    return tuple(IMPLEMENTED[name]() for name in block_names(config))
