"""Requests: pipelines of tasks with raw input, deadline and priority class."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from satedge_workflow_scheduler.common.errors import InstanceError
from satedge_workflow_scheduler.domain.nodes import NodeId


class Priority(StrEnum):
    """Priority class of a request: urgent or non urgent."""

    URGENT = "urgent"
    NON_URGENT = "non_urgent"


@dataclass(frozen=True)
class Task:
    """A task of a pipeline.

    Attributes:
        duration: Computing time on the HPC, in seconds, container start-up excluded (> 0).
        output_size: Size in Gb of the data produced by the task, which is the input of the
            next task or, for the last task, the final output (> 0).
    """

    duration: float
    output_size: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.duration) and self.duration > 0):
            raise InstanceError(
                f"task duration must be finite and > 0, got {self.duration}"
            )
        if not (math.isfinite(self.output_size) and self.output_size > 0):
            raise InstanceError(
                f"task output size must be finite and > 0, got {self.output_size}"
            )


@dataclass(frozen=True)
class Request:
    """A user request: a pipeline whose final output must reach the ground station.

    Attributes:
        id: Unique identifier.
        priority: Priority class.
        origin: Satellite holding the raw data.
        release: Instant, in seconds, from which the raw data is available (>= 0).
        deadline: Absolute deadline, in seconds, for the delivery of the output to ground (>= 0).
        raw_size: Size in Gb of the raw data (> 0).
        tasks: The pipeline, in execution order (at least one task).
    """

    id: str
    priority: Priority
    origin: NodeId
    release: float
    deadline: float
    raw_size: float
    tasks: tuple[Task, ...]

    def __post_init__(self) -> None:
        if not self.id:
            raise InstanceError("request id must be non-empty")
        if not self.tasks:
            raise InstanceError(
                f"request {self.id}: the pipeline needs at least one task"
            )
        if not (math.isfinite(self.release) and self.release >= 0):
            raise InstanceError(f"request {self.id}: release must be finite and >= 0")
        if not (math.isfinite(self.deadline) and self.deadline >= 0):
            raise InstanceError(f"request {self.id}: deadline must be finite and >= 0")
        if not (math.isfinite(self.raw_size) and self.raw_size > 0):
            raise InstanceError(f"request {self.id}: raw size must be finite and > 0")

    @property
    def output_size(self) -> float:
        """Size in Gb of the final output."""
        return self.tasks[-1].output_size

    @property
    def data_sizes(self) -> tuple[float, ...]:
        """Sizes in Gb of the data of the request: raw input, intermediate data, final output."""
        return (self.raw_size, *(task.output_size for task in self.tasks))

    @property
    def total_duration(self) -> float:
        """Sum of the task durations, in seconds."""
        return sum(task.duration for task in self.tasks)
