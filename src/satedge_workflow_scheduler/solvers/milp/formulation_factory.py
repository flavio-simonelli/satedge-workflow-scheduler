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

from collections.abc import Callable, Mapping
from typing import Any

from satedge_workflow_scheduler.scheduling.deadline import DeadlineMode
from satedge_workflow_scheduler.solvers.milp.blocks.arrival import ArrivalBlock
from satedge_workflow_scheduler.solvers.milp.blocks.assignment import AssignmentBlock
from satedge_workflow_scheduler.solvers.milp.blocks.base import ConstraintBlock
from satedge_workflow_scheduler.solvers.milp.blocks.computing_window import (
    ComputingWindowBlock,
)
from satedge_workflow_scheduler.solvers.milp.blocks.delivery import DeliveryBlock
from satedge_workflow_scheduler.solvers.milp.blocks.fixed_delay import (
    FixedDelayBlock,
    FixedRouteBlock,
)
from satedge_workflow_scheduler.solvers.milp.blocks.link_contention import (
    LinkContentionBlock,
)
from satedge_workflow_scheduler.solvers.milp.blocks.link_window import LinkWindowBlock
from satedge_workflow_scheduler.solvers.milp.blocks.online import (
    CommittedTasksBlock,
    CommittedTransfersBlock,
)
from satedge_workflow_scheduler.solvers.milp.blocks.path import PathBlock
from satedge_workflow_scheduler.solvers.milp.blocks.plmm_events import PlmmEventsBlock
from satedge_workflow_scheduler.solvers.milp.blocks.plmm_flow import PlmmFlowBlock
from satedge_workflow_scheduler.solvers.milp.blocks.plmm_occupation import (
    PlmmOccupationBlock,
)
from satedge_workflow_scheduler.solvers.milp.blocks.store_and_forward import (
    StoreAndForwardBlock,
)
from satedge_workflow_scheduler.solvers.milp.blocks.strengthening import (
    ArrivalWithoutBigMBlock,
    ContactCapacityBlock,
)
from satedge_workflow_scheduler.solvers.milp.blocks.tardiness import TardinessBlock
from satedge_workflow_scheduler.solvers.milp.blocks.task_overlap import TaskOverlapBlock
from satedge_workflow_scheduler.solvers.milp.blocks.task_timing import TaskTimingBlock
from satedge_workflow_scheduler.solvers.milp.config import (
    LazyGroup,
    Level,
    MilpConfig,
    PlmmFormulation,
)
from satedge_workflow_scheduler.solvers.milp.errors import ModelBuildError

_FIXED_DELAY_LEVEL: tuple[str, ...] = (
    "assignment",
    "computing_window",
    "task_overlap",
    "fixed_route",
    "fixed_delay",
)

_ROUTED_LEVEL: tuple[str, ...] = (
    "assignment",
    "computing_window",
    "task_overlap",
    "task_timing",
    "path",
    "store_and_forward",
    "arrival",
    "link_window",
    "delivery",
)

_BLOCKS_OF_LEVEL: dict[Level, tuple[str, ...]] = {
    Level.FIXED_DELAY_WINDOWS: _FIXED_DELAY_LEVEL,
    Level.ROUTING: _ROUTED_LEVEL,
    Level.LINK_CONTENTION: (*_ROUTED_LEVEL, "link_contention"),
    Level.PLMM_CAPACITY: (*_ROUTED_LEVEL, "link_contention", "plmm"),
}

_PLMM_BLOCKS_OF_FORMULATION: dict[PlmmFormulation, tuple[str, ...]] = {
    PlmmFormulation.FLOW: ("plmm_occupation", "plmm_flow"),
    PlmmFormulation.EVENTS: ("plmm_occupation", "plmm_events"),
}
"""The blocks of the PLMM capacity of each formulation: the occupations, common to both, and the
formulation itself. The level lists the placeholder ``"plmm"``, which this table expands."""

_STRENGTHENINGS: dict[str, str] = {
    "contact_capacity": "contact_capacity",
}
"""The strengthenings that add a block at the end. Rule 7 replaces the arrival instead: see
``block_names``."""

IMPLEMENTED: dict[str, type[ConstraintBlock[Any]]] = {
    "assignment": AssignmentBlock,
    "computing_window": ComputingWindowBlock,
    "path": PathBlock,
    "store_and_forward": StoreAndForwardBlock,
    "arrival": ArrivalBlock,
    "arrival_without_big_m": ArrivalWithoutBigMBlock,
    "contact_capacity": ContactCapacityBlock,
    "link_window": LinkWindowBlock,
    "fixed_route": FixedRouteBlock,
    "fixed_delay": FixedDelayBlock,
    "committed_tasks": CommittedTasksBlock,
    "committed_transfers": CommittedTransfersBlock,
    "link_contention": LinkContentionBlock,
    "plmm_occupation": PlmmOccupationBlock,
    "plmm_flow": PlmmFlowBlock,
    "plmm_events": PlmmEventsBlock,
    "delivery": DeliveryBlock,
    "tardiness": TardinessBlock,
    "task_overlap": TaskOverlapBlock,
    "task_timing": TaskTimingBlock,
}
"""The blocks with a class. Each block is registered here when it is written."""


Restricted = Mapping[LazyGroup, frozenset[Any]]
"""For each lazy group, the keys its blocks are written on: an empty set leaves them out."""

_GROUP_OF_PLACEHOLDER: dict[str, LazyGroup] = {"plmm": LazyGroup.PLMM}
"""The placeholder of each lazy group in the lists of the levels."""

_RESTRICTABLE: dict[
    str, tuple[LazyGroup, Callable[[frozenset[Any] | None], ConstraintBlock[Any]]]
] = {
    "plmm_occupation": (LazyGroup.PLMM, PlmmOccupationBlock),
    "plmm_flow": (LazyGroup.PLMM, PlmmFlowBlock),
    "plmm_events": (LazyGroup.PLMM, PlmmEventsBlock),
}
"""The blocks that can be written on a subset of their keys, with their group: the lazy
strategy writes them only where a solution violates them."""


def block_names(
    config: MilpConfig,
    online: bool = False,
    restricted: Restricted | None = None,
) -> tuple[str, ...]:
    """The names of the blocks of the model of ``config``, in the order they are built.

    The names are those of the document's levels, and they are returned whether or not the
    block is implemented. With ``online``, the instance has an online state, and the blocks
    that fix the committed calendar and impose the plan release are added at the end of the
    level: the tasks at every level, the transmissions only where they are routed. With
    ``restricted``, the blocks of each lazy group in it are written only on its keys, and not at
    all if the set is empty; a group not in it is written on all of its keys.
    """
    names: list[str] = []
    for name in _BLOCKS_OF_LEVEL[config.level]:
        if name == "plmm":
            if _written(restricted, _GROUP_OF_PLACEHOLDER[name]):
                names.extend(_PLMM_BLOCKS_OF_FORMULATION[config.plmm_formulation])
        else:
            names.append(name)
    if config.deadline is DeadlineMode.SOFT:
        names.append("tardiness")
    if online:
        names.append("committed_tasks")
        if config.level.routed:
            names.append("committed_transfers")
    if config.strengthening.arrival_without_big_m:
        names = [
            "arrival_without_big_m" if name == "arrival" else name for name in names
        ]
    for flag, name in _STRENGTHENINGS.items():
        if getattr(config.strengthening, flag):
            names.append(name)
    return tuple(names)


def blocks_for(
    config: MilpConfig,
    online: bool = False,
    restricted: Restricted | None = None,
) -> tuple[ConstraintBlock[Any], ...]:
    """The blocks of the model of ``config``, instantiated and in the order they are built.

    The arguments are those of :func:`block_names`.

    Raises:
        ModelBuildError: If a block of the level has no class yet.
    """
    names = block_names(config, online, restricted)
    missing = [name for name in names if name not in IMPLEMENTED]
    if missing:
        raise ModelBuildError(f"blocks not implemented yet: {', '.join(missing)}")
    blocks: list[ConstraintBlock[Any]] = []
    for name in names:
        if name in _RESTRICTABLE:
            group, make = _RESTRICTABLE[name]
            keys = restricted.get(group) if restricted is not None else None
            blocks.append(make(keys))
        else:
            blocks.append(IMPLEMENTED[name]())
    return tuple(blocks)


def _written(restricted: Restricted | None, group: LazyGroup) -> bool:
    """Whether the blocks of ``group`` are in the model: unless restricted to no key."""
    if restricted is None or group not in restricted:
        return True
    return bool(restricted[group])
