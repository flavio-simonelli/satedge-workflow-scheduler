"""The MILP solver: it builds the model of a configuration, solves it, and decodes the solution.

:class:`MilpSolver` is a :class:`~satedge_workflow_scheduler.solvers.base.Solver`, so it can be
used wherever the heuristic is used. It does not know the constraints (the blocks do), nor the
solver library (the backend does): it reads the data of the instance and hands them to the
strategy of the configuration, which builds, solves and decodes the model (:mod:`.strategy`).

The backend is given as a factory, so that every solve gets a new model, and so that the tests
can use a backend that needs no licence.
"""

from __future__ import annotations

from collections.abc import Callable

from satedge_workflow_scheduler.domain.instance import ProblemInstance
from satedge_workflow_scheduler.domain.solution import Solution
from satedge_workflow_scheduler.solvers.milp.backends.ports import MilpBackend
from satedge_workflow_scheduler.solvers.milp.config import MilpConfig
from satedge_workflow_scheduler.solvers.milp.model_data import ModelData
from satedge_workflow_scheduler.solvers.milp.strategy import strategy_for


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
            SolverError: If the solver fails, or stops without a valid solution.
        """
        data = ModelData(instance, self._config.deadline)
        strategy = strategy_for(self._config)
        return strategy.solve(instance, data, self._config, self._backend)
