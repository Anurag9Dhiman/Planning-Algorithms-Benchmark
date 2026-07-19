"""BFS: optimal in steps (unit cost). Dynamics only."""

from collections import deque
from typing import List

import numpy as np

from benchmark.core.base_planner import Planner
from benchmark.core.budget import BudgetExhausted, PlanningBudget
from benchmark.core.world_model import State, WorldModel


def _hash(z):
    if isinstance(z, np.ndarray):
        return tuple(np.round(z, 4).tolist())
    return z  # int IDs (PerfectModel) or other hashable types


class BFSPlanner(Planner):
    name = "bfs"
    uses_value = False
    uses_policy = False

    def plan(self, z: State, world_model: WorldModel, budget: PlanningBudget) -> List[int]:
        queue = deque()
        queue.append((z, []))
        visited = {_hash(z)}

        try:
            while queue:
                state, path = queue.popleft()
                for action in world_model.action_space():
                    next_z, _, done = world_model.step(state, action, budget)
                    next_path = path + [action]
                    if done:
                        return next_path
                    h = _hash(next_z)
                    if h not in visited:
                        visited.add(h)
                        queue.append((next_z, next_path))
        except BudgetExhausted:
            pass

        return [np.random.randint(0, len(world_model.action_space()))]
