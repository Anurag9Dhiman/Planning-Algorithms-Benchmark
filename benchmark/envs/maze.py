"""
Custom grid-maze environment. Zero external dependencies.

Layout: N×N grid, 4-connected movement (UP/DOWN/LEFT/RIGHT).
Start: top-left (0,0). Goal: bottom-right (N-1, N-1).
Obstacles placed randomly; a path from start to goal is always guaranteed.

Observation: flat int array [row, col, goal_row, goal_col] — hashable, human-readable.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from benchmark.core.base_env import BenchmarkEnv, TaskMeta

# Actions
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
_DELTAS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}


class MazeEnv(BenchmarkEnv):
    """
    Parameters
    ----------
    size        : grid side length (NxN)
    n_obstacles : number of randomly placed walls (BFS ensures reachability)
    slip_prob   : probability of a random action instead of the chosen one
    step_penalty: reward per step (negative → dense cost signal)
    goal_reward : reward on reaching goal
    max_steps   : episode length limit
    """

    def __init__(
        self,
        size: int = 10,
        n_obstacles: int = 15,
        slip_prob: float = 0.0,
        step_penalty: float = -0.01,
        goal_reward: float = 1.0,
        max_steps: int = 200,
    ):
        self.size = size
        self.n_obstacles = n_obstacles
        self.slip_prob = slip_prob
        self.step_penalty = step_penalty
        self.goal_reward = goal_reward
        self._max_steps = max_steps

        self._goal = (size - 1, size - 1)
        self._walls: set = set()
        self._pos: Tuple[int, int] = (0, 0)
        self._steps = 0
        self._rng = random.Random()

    # ------------------------------------------------------------------
    # BenchmarkEnv interface
    # ------------------------------------------------------------------

    def reset(self, seed: int = None) -> np.ndarray:
        self._rng.seed(seed)
        self._walls = self._generate_walls(seed)
        self._pos = (0, 0)
        self._steps = 0
        return self._obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        if self.slip_prob > 0 and self._rng.random() < self.slip_prob:
            action = self._rng.randint(0, 3)

        dr, dc = _DELTAS[action]
        nr, nc = self._pos[0] + dr, self._pos[1] + dc

        if self._valid(nr, nc):
            self._pos = (nr, nc)

        self._steps += 1
        done = self._pos == self._goal
        reward = self.goal_reward if done else self.step_penalty
        return self._obs(), reward, done, {}

    def clone_state(self) -> Any:
        return copy.deepcopy({"pos": self._pos, "steps": self._steps, "walls": self._walls})

    def restore_state(self, state: Any) -> None:
        self._pos = state["pos"]
        self._steps = state["steps"]
        self._walls = state["walls"]

    def all_states(self) -> List[np.ndarray]:
        states = []
        for r in range(self.size):
            for c in range(self.size):
                if (r, c) not in self._walls:
                    states.append(self._make_obs(r, c))
        return states

    @property
    def task_meta(self) -> TaskMeta:
        horizon = "short" if self.size <= 6 else "medium" if self.size <= 12 else "long"
        return TaskMeta(
            tier=1,
            env_name=f"maze_{self.size}x{self.size}",
            horizon=horizon,
            reward_density="sparse" if self.step_penalty == 0 else "dense",
            stochastic=self.slip_prob > 0,
            goal_conditioned=True,
            branching_factor=4,
        )

    @property
    def n_actions(self) -> int:
        return 4

    @property
    def max_steps(self) -> int:
        return self._max_steps

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _obs(self) -> np.ndarray:
        return self._make_obs(*self._pos)

    def _make_obs(self, r: int, c: int) -> np.ndarray:
        return np.array([r, c, self._goal[0], self._goal[1]], dtype=np.int32)

    def _valid(self, r: int, c: int) -> bool:
        return 0 <= r < self.size and 0 <= c < self.size and (r, c) not in self._walls

    def _generate_walls(self, seed: Optional[int]) -> set:
        rng = random.Random(seed)
        walls: set = set()
        forbidden = {(0, 0), self._goal}
        candidates = [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if (r, c) not in forbidden
        ]
        rng.shuffle(candidates)
        for cell in candidates:
            if len(walls) >= self.n_obstacles:
                break
            walls.add(cell)
            if not self._reachable((0, 0), self._goal, walls):
                walls.remove(cell)
        return walls

    def _reachable(self, start: Tuple, goal: Tuple, walls: set) -> bool:
        visited = {start}
        queue = [start]
        while queue:
            r, c = queue.pop()
            if (r, c) == goal:
                return True
            for dr, dc in _DELTAS.values():
                nb = (r + dr, c + dc)
                if (
                    0 <= nb[0] < self.size
                    and 0 <= nb[1] < self.size
                    and nb not in walls
                    and nb not in visited
                ):
                    visited.add(nb)
                    queue.append(nb)
        return False
