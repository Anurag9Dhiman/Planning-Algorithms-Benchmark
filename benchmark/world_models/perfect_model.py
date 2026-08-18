"""
Oracle world model: wraps the real environment directly.
Used as the ground-truth baseline for controlled experiments.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from benchmark.core.base_env import BenchmarkEnv
from benchmark.core.budget import PlanningBudget
from benchmark.core.world_model import State, WorldModel


class PerfectModel(WorldModel):
    """
    Uses env.clone_state / restore_state to simulate steps without
    side-effects on the live episode.

    Latents are integer state IDs backed by an internal registry so
    planners can use them as dict keys / set members without knowing
    the underlying env type.  The registry is cleared on each encode()
    call (once per real env step), keeping memory bounded.
    """

    def __init__(self, env: BenchmarkEnv):
        self._env = env
        self._registry: Dict[int, Any] = {}
        self._counter: int = 0

    @property
    def name(self) -> str:
        return "perfect"

    @property
    def n_actions(self) -> int:
        return self._env.n_actions

    def encode(self, obs: np.ndarray) -> State:
        self._registry.clear()
        self._counter = 0
        return self._store(self._env.clone_state())

    def step(
        self, z: State, action: int, budget: PlanningBudget
    ) -> Tuple[State, float, bool]:
        budget.consume(1)
        saved = self._env.clone_state()
        self._env.restore_state(self._registry[z])
        obs, reward, done, _ = self._env.step(action)
        next_z = self._store(self._env.clone_state())
        self._env.restore_state(saved)
        return next_z, reward, done

    def action_space(self) -> List[int]:
        return list(range(self._env.n_actions))

    def all_states(self) -> List[State]:
        raise NotImplementedError("State enumeration not supported for oracle model.")

    def _store(self, env_state: Any) -> int:
        sid = self._counter
        self._registry[sid] = env_state
        self._counter += 1
        return sid
