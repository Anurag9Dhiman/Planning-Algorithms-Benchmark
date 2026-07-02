from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from benchmark.core.budget import PlanningBudget
from benchmark.core.world_model import State, WorldModel


class Planner(ABC):

    @abstractmethod
    def plan(
        self,
        z: State,
        world_model: WorldModel,
        budget: PlanningBudget,
    ) -> List[int]:
        """
        Return a sequence of actions given the current latent state.
        In receding-horizon mode only action[0] is executed; the rest
        are discarded and replanning happens next step.
        Must not exceed budget — catch BudgetExhausted internally.
        """

    @property
    @abstractmethod
    def name(self) -> str: ...

    # Which world model capabilities this planner uses.
    uses_value: bool = False
    uses_policy: bool = False
