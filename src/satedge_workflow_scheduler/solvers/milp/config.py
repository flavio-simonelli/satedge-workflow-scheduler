"""Configuration of one run of the MILP: which model to build and how to solve it.

:class:`MilpConfig` is immutable pure data. It holds the choices that the factory of the blocks,
the domains, the solve strategy and the report all read, and it rejects the combinations that
are not valid models, so that the error appears when the configuration is written and not after
a long build. It knows nothing about solvers and nothing about instances.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum

from satedge_workflow_scheduler.scheduling.deadline import DeadlineMode
from satedge_workflow_scheduler.solvers.milp.backends.ports import SolveParams
from satedge_workflow_scheduler.solvers.milp.errors import MilpConfigError


class Level(IntEnum):
    """The levels of the document that build the model, each one adding a part.

    Level 2 (packing with fixed delays) is not implemented: the first level is 3. The online
    state (level 7 of the document) is not a level here: it belongs to the instance, which may
    have committed requests and a plan release, and the model adds it at any level.
    """

    FIXED_DELAY_WINDOWS = 3
    """Computing windows of the HPC, with the fixed delays of the transmissions."""
    ROUTING = 4
    """Routing of the data on the network, with visibility windows and store-and-forward."""
    LINK_CONTENTION = 5
    """One transmission at a time on each link."""
    PLMM_CAPACITY = 6
    """The capacity of the PLMM of each satellite."""

    @property
    def routed(self) -> bool:
        """Whether the data follow a path on the network, with visibility windows."""
        return self >= Level.ROUTING

    @property
    def contention(self) -> bool:
        """Whether the links carry one transmission at a time."""
        return self >= Level.LINK_CONTENTION

    @property
    def plmm(self) -> bool:
        """Whether the capacity of the PLMM is modelled."""
        return self >= Level.PLMM_CAPACITY


class LazyGroup(Enum):
    """A group of constraints that the lazy strategy may add only where a solution violates it.

    The model of a lazy group starts without its constraints, and gets them on the keys that a
    solution violates, round after round (section 6, rule 10). A group not in the configuration
    is in the model from the start.
    """

    PLMM = "plmm"
    """The capacity of the PLMM, added on the satellites that a solution overloads."""


class PlmmFormulation(Enum):
    """Which formulation of the capacity of the PLMM the model uses (section 6 of the document)."""

    FLOW = "flow"
    """The flow formulation (FCT): the capacity is a flow between the data that occupy it."""
    EVENTS = "events"
    """The event formulation (OOE): the capacity is checked at each event."""


@dataclass(frozen=True)
class Strengthening:
    """The optional strengthenings of the model (section 6, rules 7 and 8).

    Attributes:
        arrival_without_big_m: Write the arrivals with linear constraints, without big-M, rule 7.
        contact_capacity: Add the capacity cut of each visibility window, rule 8.
    """

    arrival_without_big_m: bool = False
    contact_capacity: bool = False


@dataclass(frozen=True)
class MilpConfig:
    """Everything that defines one run of the MILP.

    Attributes:
        level: The level of the model. The levels add blocks; see :class:`Level`.
        deadline: Hard deadlines (requests may be rejected) or soft ones (all are served).
        reductions: Whether the exact reductions of section 6 (rules 1 to 6) are applied.
        strengthening: The optional strengthenings.
        lazy: The groups of constraints added lazily, only where a solution violates them. Empty
            by default: every constraint is in the model from the start.
        plmm_formulation: Which formulation of the PLMM capacity the model uses. Only meaningful
            from level 6 on.
        polish: Whether the LP is re-solved with the binaries fixed, to remove the residuals
            of the big-M that the integrality tolerance leaves.
        warm_start: Whether the solver starts from the calendar of the priority-EDF/EFT
            heuristic (rule 9). Only for instances without committed requests: the heuristic
            does not reserve them.
        solve: Parameters of the solver.

    Raises:
        MilpConfigError: If the combination of the choices does not define a valid model.
    """

    level: Level
    deadline: DeadlineMode = DeadlineMode.HARD
    reductions: bool = True
    strengthening: Strengthening = field(default_factory=Strengthening)
    lazy: frozenset[LazyGroup] = frozenset()
    plmm_formulation: PlmmFormulation = PlmmFormulation.FLOW
    polish: bool = False
    warm_start: bool = False
    solve: SolveParams = field(default_factory=SolveParams)

    def __post_init__(self) -> None:
        strengthened = (
            self.strengthening.arrival_without_big_m
            or self.strengthening.contact_capacity
        )
        if strengthened and not self.level.routed:
            raise MilpConfigError(
                f"the strengthenings need routed communication, not level {self.level}"
            )
        if LazyGroup.PLMM in self.lazy and not self.level.plmm:
            raise MilpConfigError(
                f"the lazy PLMM needs level 6 or more, not {self.level}"
            )
        if self.solve.time_limit is not None and self.solve.time_limit <= 0:
            raise MilpConfigError("the time limit must be positive")
        if self.solve.mip_gap < 0:
            raise MilpConfigError("the MIP gap must not be negative")
        if self.solve.memory_limit_gb is not None and self.solve.memory_limit_gb <= 0:
            raise MilpConfigError("the memory limit must be positive")
