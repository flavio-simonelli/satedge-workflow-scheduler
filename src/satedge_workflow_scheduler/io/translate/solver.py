"""Translation of a solver document into a solver configuration."""

from __future__ import annotations

from satedge_workflow_scheduler.io.documents import SolverDoc
from satedge_workflow_scheduler.io.errors import DocumentError
from satedge_workflow_scheduler.scheduling.deadline import DeadlineMode
from satedge_workflow_scheduler.solvers.milp.backends.ports import SolveParams
from satedge_workflow_scheduler.solvers.milp.config import (
    LazyGroup,
    Level,
    MilpConfig,
    PlmmFormulation,
    Strengthening,
)
from satedge_workflow_scheduler.solvers.sequential.config import SequentialConfig


def solver_config_from_document(doc: SolverDoc) -> MilpConfig | SequentialConfig:
    """The configuration that ``doc`` describes.

    Raises:
        DocumentError: If the MILP level is not one of the levels, or the document lacks the
            section of its method.
    """
    deadline = DeadlineMode(doc.deadline)
    if doc.method == "sequential":
        return SequentialConfig(
            deadline=deadline,
            max_backtracks=doc.sequential.max_backtracks,
            verify=doc.sequential.verify,
        )
    if doc.milp is None:
        raise DocumentError("method 'milp' needs a 'milp' section")
    milp = doc.milp
    try:
        level = Level(milp.level)
    except ValueError as error:
        raise DocumentError(
            f"milp.level must be one of 3, 4, 5, 6, got {milp.level}"
        ) from error
    return MilpConfig(
        level=level,
        deadline=deadline,
        reductions=milp.reductions,
        strengthening=Strengthening(
            arrival_without_big_m=milp.strengthening.arrival_without_big_m,
            contact_capacity=milp.strengthening.contact_capacity,
        ),
        lazy=frozenset(LazyGroup(group) for group in milp.lazy),
        plmm_formulation=PlmmFormulation(milp.plmm_formulation),
        polish=milp.polish,
        warm_start=milp.warm_start,
        solve=SolveParams(
            time_limit=doc.limits.time_limit_s,
            mip_gap=doc.limits.mip_gap,
            threads=doc.limits.threads,
            memory_limit_gb=doc.limits.memory_limit_gb,
            verbose=doc.verbose,
        ),
    )
