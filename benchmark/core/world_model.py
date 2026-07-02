from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from benchmark.core.budget import PlanningBudget

State = np.ndarray  # latent vector z


class WorldModel(ABC):

    @abstractmethod
    def encode(self, obs: np.ndarray) -> State:
        """Compress raw observation to latent state. Budget-free."""

    @abstractmethod
    def step(
        self, z: State, action: int, budget: PlanningBudget
    ) -> Tuple[State, float, bool]:
        """Simulate one step in latent space. Decrements budget by 1."""

    @abstractmethod
    def action_space(self) -> List[int]:
        """Valid action indices (discrete, same for all states)."""

    def value(self, z: State) -> float:
        """Estimated V(z). Returns 0 if not implemented."""
        return 0.0

    def policy(self, z: State) -> np.ndarray:
        """Action prior P(a|z). Returns uniform if not implemented."""
        n = len(self.action_space())
        return np.ones(n) / n

    def all_states(self) -> list:
        """Enumerate all states. Only feasible for tabular/perfect models."""
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str: ...
