"""Uniform Cost Search: optimal with variable step costs. Dynamics only."""

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


class UCSPlanner(Planner):
    name = "ucs"
    uses_value = False
    uses_policy = False

    def plan(self, z: State, world_model: WorldModel, budget: PlanningBudget) -> List[int]:
        # heap entries: (neg_cumulative_reward, tie_breaker, state, path)
        counter = 0
        heap = [(0.0, counter, z, [])]
        best_cost = {_hash(z): 0.0}

        try:
            while heap:
                neg_g, _, state, path = heapq.heappop(heap)
                g = -neg_g

                for action in world_model.action_space():
                    next_z, reward, done = world_model.step(state, action, budget)
                    next_path = path + [action]
                    next_g = g + reward
                    if done:
                        return next_path
                    h = _hash(next_z)
                    if h not in best_cost or next_g > best_cost[h]:
                        best_cost[h] = next_g
                        counter += 1
                        heapq.heappush(heap, (-next_g, counter, next_z, next_path))
        except BudgetExhausted:
            pass

        return [np.random.randint(0, len(world_model.action_space()))]
