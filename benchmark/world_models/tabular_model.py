"""
Tabular world model: learned from rollout data.
Stores (state_hash, action) -> (next_state, reward, done).
Supports controlled model-error ablations via add_corruption().
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from benchmark.core.base_env import BenchmarkEnv
from benchmark.core.budget import PlanningBudget
from benchmark.core.world_model import State, WorldModel


class TabularModel(WorldModel):

    def __init__(self, n_actions: int):
        self._n_actions = n_actions
        self._table: Dict[Tuple, Tuple] = {}   # (hash, action) -> (z', r, done)
        self._value_table: Dict[Tuple, float] = {}

    @property
    def name(self) -> str:
        return "tabular"

    # ------------------------------------------------------------------
    # WorldModel interface
    # ------------------------------------------------------------------

    def encode(self, obs: np.ndarray) -> State:
        return obs.copy()

    def step(
        self, z: State, action: int, budget: PlanningBudget
    ) -> Tuple[State, float, bool]:
        budget.consume(1)
        key = (self._hash(z), action)
        if key not in self._table:
            return z.copy(), 0.0, False   # unknown transition: stay in place
        z_next, r, done = self._table[key]
        return z_next.copy(), r, done

    def value(self, z: State) -> float:
        return self._value_table.get(self._hash(z), 0.0)

    def action_space(self) -> List[int]:
        return list(range(self._n_actions))

    def all_states(self) -> List[State]:
        seen = {}
        for (h, _), (z_next, _, _) in self._table.items():
            if h not in seen:
                seen[h] = z_next
        return list(seen.values())

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def learn_from_env(
        self,
        env: BenchmarkEnv,
        n_episodes: int = 200,
        seed: int = 0,
        rng: Optional[random.Random] = None,
    ) -> None:
        """Populate table from random rollouts."""
        rng = rng or random.Random(seed)
        for ep in range(n_episodes):
            obs = env.reset(seed=seed + ep)
            z = self.encode(obs)
            for _ in range(env.max_steps):
                action = rng.randint(0, env.n_actions - 1)
                obs_next, r, done, _ = env.step(action)
                z_next = self.encode(obs_next)
                key = (self._hash(z), action)
                self._table[key] = (z_next, r, done)
                z = z_next
                if done:
                    break

    def set_value(self, z: State, v: float) -> None:
        self._value_table[self._hash(z)] = v

    def add_corruption(self, p: float, seed: int = 42) -> None:
        """Randomly corrupt a fraction p of stored transitions (ablation)."""
        rng = random.Random(seed)
        keys = list(self._table.keys())
        n_corrupt = int(len(keys) * p)
        for key in rng.sample(keys, min(n_corrupt, len(keys))):
            z_next, r, done = self._table[key]
            noise = np.random.RandomState(seed).randn(*z_next.shape) * 0.5
            self._table[key] = (z_next + noise.astype(z_next.dtype), r, done)

    # ------------------------------------------------------------------

    @staticmethod
    def _hash(z: State) -> tuple:
        return tuple(z.tolist())
