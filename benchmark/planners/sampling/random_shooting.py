"""Random Shooting: sample N action sequences, return the best. Dynamics only."""

from typing import List

import numpy as np

from benchmark.core.base_planner import Planner
from benchmark.core.budget import BudgetExhausted, PlanningBudget
from benchmark.core.world_model import State, WorldModel


class RandomShootingPlanner(Planner):
    name = "random_shooting"
    uses_value = False
    uses_policy = False

    def __init__(self, n_samples: int = 64, horizon: int = 15):
        self.n_samples = n_samples
        self.horizon = horizon

    def plan(self, z: State, world_model: WorldModel, budget: PlanningBudget) -> List[int]:
        actions = world_model.action_space()
        best_return, best_seq = -np.inf, [np.random.choice(actions)]

        for _ in range(self.n_samples):
            seq = [np.random.choice(actions) for _ in range(self.horizon)]
            total_r = 0.0
            state = z
            try:
                for a in seq:
                    state, r, done = world_model.step(state, a, budget)
                    total_r += r
                    if done:
                        break
            except BudgetExhausted:
                break

            if total_r > best_return:
                best_return, best_seq = total_r, seq

        return best_seq
