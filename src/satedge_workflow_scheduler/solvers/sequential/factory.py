"""The objects that a configuration asks for: one factory per kind of choice.

Each function maps the enum of the configuration to its implementation. Adding a choice means
adding a value to the enum and a case here, and the error of an unknown value says which
implementation is missing, rather than skipping it silently.
"""

from __future__ import annotations

from satedge_workflow_scheduler.solvers.sequential.config import Routing
from satedge_workflow_scheduler.solvers.sequential.routing.ports import Router
from satedge_workflow_scheduler.solvers.sequential.routing.safe_interval import (
    SafeIntervalRouter,
)


def router_for(routing: Routing, eps: float) -> Router:
    """The router of ``routing``, with the tolerance ``eps``."""
    match routing:
        case Routing.SAFE_INTERVAL:
            return SafeIntervalRouter(eps)
    raise NotImplementedError(f"routing {routing!r} has no implementation yet")
