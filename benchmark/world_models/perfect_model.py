"""
Oracle world model: wraps the real environment directly.
Used as the ground-truth baseline for controlled experiments.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from benchmark.core.base_env import BenchmarkEnv
from benchmark.core.budget import PlanningBudget
from benchmark.core.world_model import State, WorldModel


class PerfectModel(WorldModel):
    """
    Uses env.clone_state / restore_state to simulate steps without
    side-effects on the live episode.
    """

    def __init__(self, env: BenchmarkEnv):
        self._env = env

    @property
    def name(self) -> str:
        return "perfect"

    def encode(self, obs: np.ndarray) -> State:
        return obs.copy()

    def step(
        self, z: State, action: int, budget: PlanningBudget
    ) -> Tuple[State, float, bool]:
        budget.consume(1)
        saved = self._env.clone_state()
        self._env.restore_state(self._state_from_latent(z))
        obs, reward, done, _ = self._env.step(action)
        next_z = self.encode(obs)
        self._env.restore_state(saved)
        return next_z, reward, done

    def action_space(self) -> List[int]:
        return list(range(self._env.n_actions))

    def all_states(self) -> List[State]:
        return [self.encode(s) for s in self._env.all_states()]

    # ------------------------------------------------------------------

    def _state_from_latent(self, z: State):
        """
        For the maze env the latent IS the obs (int array).
        Re-construct an internal env state dict from it.
        """
        r, c, gr, gc = int(z[0]), int(z[1]), int(z[2]), int(z[3])
        saved = self._env.clone_state()
        saved["pos"] = (r, c)
        return saved
