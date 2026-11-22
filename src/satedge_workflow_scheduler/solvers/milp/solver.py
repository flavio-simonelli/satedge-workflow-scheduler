"""The MILP solver: it builds the model of a configuration, solves it, and decodes the solution.

:class:`MilpSolver` is a :class:`~satedge_workflow_scheduler.solvers.base.Solver`, so it can be
used wherever the heuristic is used. It does not know the constraints (the blocks do), nor the
solver library (the backend does): it only puts the parts together, in the order of the pipeline:
data, blocks, builder, objective, solve, decoding.

The backend is given as a factory, so that every solve gets a new model, and so that the tests
can use a backend that needs no licence.
"""

from __future__ import annotations

from collections.abc import Callable

from satedge_workflow_scheduler.domain.instance import ProblemInstance
from satedge_workflow_scheduler.domain.solution import Solution
from satedge_workflow_scheduler.solvers.errors import InfeasibleModelError, SolverError
from satedge_workflow_scheduler.solvers.milp.backends.ports import (
    BackendStatus,
    MilpBackend,
)
from satedge_workflow_scheduler.solvers.milp.blocks.base import BuildContext
from satedge_workflow_scheduler.solvers.milp.builder import ModelBuilder
from satedge_workflow_scheduler.solvers.milp.config import MilpConfig
from satedge_workflow_scheduler.solvers.milp.decoding import decode
from satedge_workflow_scheduler.solvers.milp.formulation_factory import blocks_for
from satedge_workflow_scheduler.solvers.milp.model_data import ModelData
from satedge_workflow_scheduler.solvers.milp.objective import objective


class MilpSolver:
    """Solves an instance with the MILP of a configuration.

    The objective is the lexicographic one of the document. Its constant term is not passed to
    the solver: it does not change the optimal solution, and the value of the solution is
    evaluated on the schedule by the decoder.

    Args:
        config: The level, the deadlines and the solve parameters of the model.
        backend: A factory of the backend that solves the model, called once per solve.
    """

    def __init__(self, config: MilpConfig, backend: Callable[[], MilpBackend]) -> None:
        self._config = config
        self._backend = backend

    def solve(self, instance: ProblemInstance) -> Solution:
        """The solution of ``instance`` found by the MILP.

        Raises:
            InfeasibleModelError: If the solver proves that the model has no solution.
            SolverError: If the solver fails, or stops without a solution.
        """
        data = ModelData(instance, self._config.deadline)
        backend = self._backend()
        blocks = blocks_for(self._config, online=data.is_online)
        model = ModelBuilder(data, backend).build(blocks)
        context = BuildContext(data, model.variables, model.families)
        backend.set_objective(dict(objective(context).terms))

        result = backend.solve(self._config.solve)
        if result.status is BackendStatus.INFEASIBLE:
            raise InfeasibleModelError("the model has no solution")
        if result.status in (BackendStatus.ERROR, BackendStatus.NO_SOLUTION):
            raise SolverError(f"the solver ended with status {result.status}")
        return decode(instance, data, model, result.values, result.status)
