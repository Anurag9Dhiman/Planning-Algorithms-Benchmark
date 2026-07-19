"""Greedy Best-First Search: expands lowest -value(z) first. Track 2."""

import heapq
from typing import List

import numpy as np

from benchmark.core.base_planner import Planner
from benchmark.core.budget import BudgetExhausted, PlanningBudget
from benchmark.core.world_model import State, WorldModel


def _hash(z):
    if isinstance(z, np.ndarray):
        return tuple(np.round(z, 4).tolist())
    return z  # int IDs (PerfectModel) or other hashable types


class GreedyBFSPlanner(Planner):
    name = "greedy_bfs"
    uses_value = True
    uses_policy = False

    def plan(self, z: State, world_model: WorldModel, budget: PlanningBudget) -> List[int]:
        counter = 0
        heap = [(-world_model.value(z), counter, z, [])]
        visited = {_hash(z)}

        try:
            while heap:
                _, _, state, path = heapq.heappop(heap)
                for action in world_model.action_space():
                    next_z, _, done = world_model.step(state, action, budget)
                    next_path = path + [action]
                    if done:
                        return next_path
                    h = _hash(next_z)
                    if h not in visited:
                        visited.add(h)
                        counter += 1
                        heapq.heappush(heap, (-world_model.value(next_z), counter, next_z, next_path))
        except BudgetExhausted:
            pass

        return [np.random.randint(0, len(world_model.action_space()))]
