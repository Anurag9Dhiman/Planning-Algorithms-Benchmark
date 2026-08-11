"""
NoisyWorldModel: wraps any WorldModel and injects additive Gaussian noise
into the dynamics step output (z_next).

Used for the noise-robustness experiment: how does each planner degrade
as world model prediction error increases?

sigma is in the same units as the latent space — use eval_wm_quality.py
to report mean ||z|| so sigma values can be interpreted as fractions
of the typical state norm.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from benchmark.core.budget import PlanningBudget
from benchmark.core.world_model import State, WorldModel


class NoisyWorldModel(WorldModel):
    """Wraps a WorldModel, adding N(0, sigma^2 * I) noise to step() z_next."""

    def __init__(self, base: WorldModel, sigma: float = 0.1, seed: int = 0):
        self._base  = base
        self._sigma = sigma
        self._rng   = np.random.default_rng(seed)

    @property
    def name(self) -> str:
        return f"{self._base.name}_noisy"

    @property
    def n_actions(self) -> int:
        return self._base.n_actions

    def encode(self, obs: np.ndarray) -> State:
        return self._base.encode(obs)

    def step(self, z: State, action: int, budget: PlanningBudget) -> Tuple[State, float, bool]:
        z_next, reward, done = self._base.step(z, action, budget)
        if self._sigma > 0:
            z_next = z_next + self._rng.standard_normal(z_next.shape).astype(np.float32) * self._sigma
        return z_next, reward, done

    def value(self, z: State) -> float:
        return self._base.value(z)

    def policy(self, z: State) -> np.ndarray:
        return self._base.policy(z)

    def action_space(self) -> List[int]:
        return self._base.action_space()
