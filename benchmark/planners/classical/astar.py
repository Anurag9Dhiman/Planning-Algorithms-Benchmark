"""A*: heuristic = -world_model.value(z). Track 2."""

import heapq
from typing import List

import numpy as np

from benchmark.core.base_planner import Planner
from benchmark.core.budget import BudgetExhausted, PlanningBudget
from benchmark.core.world_model import State, WorldModel


def _hash(z: State) -> tuple:
    return tuple(np.round(z, 2).tolist())


class AStarPlanner(Planner):
    name = "astar"
    uses_value = True
    uses_policy = False

    def plan(self, z: State, world_model: WorldModel, budget: PlanningBudget) -> List[int]:
        # f = g + h,  h(z) = -value(z)  (higher value = closer to goal)
        h0 = -world_model.value(z)
        counter = 0
        heap = [(h0, counter, 0.0, z, [])]  # (f, tie, g, state, path)
        best_g: dict = {_hash(z): 0.0}

        try:
            while heap:
                f, _, g, state, path = heapq.heappop(heap)
                for action in world_model.action_space():
                    next_z, reward, done = world_model.step(state, action, budget)
                    next_path = path + [action]
                    if done:
                        return next_path
                    next_g = g + reward
                    h = _hash(next_z)
                    if h not in best_g or next_g > best_g[h]:
                        best_g[h] = next_g
                        heuristic = -world_model.value(next_z)
                        f_new = -(next_g) + heuristic
                        counter += 1
                        heapq.heappush(heap, (f_new, counter, next_g, next_z, next_path))
        except BudgetExhausted:
            pass

        return [np.random.randint(0, len(world_model.action_space()))]
