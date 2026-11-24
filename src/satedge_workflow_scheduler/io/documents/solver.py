"""Document of a solver file."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from satedge_workflow_scheduler.io.documents.base import (
    SCHEMA_VERSION,
    Document,
    check_schema_version,
)


class StrengtheningDoc(Document):
    """Optional strengthening cuts of the MILP."""

    arrival_without_big_m: bool = False
    contact_capacity: bool = False


class MilpDoc(Document):
    """Settings of the MILP: the level of the model and its simplifications."""

    level: int
    reductions: bool = True
    strengthening: StrengtheningDoc = Field(default_factory=StrengtheningDoc)
    lazy: list[str] = Field(default_factory=list)
    plmm_formulation: Literal["flow", "events"] = "flow"
    polish: bool = True
    warm_start: bool = False


class SequentialSettingsDoc(Document):
    """Settings of the sequential heuristic."""

    max_backtracks: int = 100
    verify: bool = True


class LimitsDoc(Document):
    """Limits of the MILP solve. A file of the sequential method must not set them."""

    time_limit_s: float | None = None
    mip_gap: float = 1e-6
    threads: int | None = None
    memory_limit_gb: float | None = None


class SolverDoc(Document):
    """The solver file: which method to run, and its settings.

    Attributes:
        schema_version: Version of the format, ``major.minor.patch``.
        method: ``milp`` needs the ``milp`` section; ``sequential`` uses ``sequential``.
        deadline: ``hard`` rejects the requests that miss their deadline; ``soft`` serves them
            all and penalises the lateness.
        milp: Settings of the MILP.
        sequential: Settings of the sequential heuristic.
        limits: Limits of the MILP solve.
        verbose: Whether the solver writes its log.
    """

    schema_version: str = SCHEMA_VERSION
    method: Literal["milp", "sequential"]
    deadline: Literal["hard", "soft"] = "hard"
    milp: MilpDoc | None = None
    sequential: SequentialSettingsDoc = Field(default_factory=SequentialSettingsDoc)
    limits: LimitsDoc = Field(default_factory=LimitsDoc)
    verbose: bool = False

    @field_validator("schema_version")
    @classmethod
    def _same_major(cls, value: str) -> str:
        return check_schema_version(value)

    @model_validator(mode="after")
    def _method_has_its_section(self) -> SolverDoc:
        if self.method == "milp" and self.milp is None:
            raise ValueError("method 'milp' needs a 'milp' section")
        if self.method == "sequential" and "limits" in self.model_fields_set:
            raise ValueError(
                "limits apply to method 'milp' only: the sequential heuristic has no solve "
                "to limit, so remove the 'limits' section"
            )
        return self
