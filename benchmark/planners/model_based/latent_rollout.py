"""
Latent Rollout Planner: depth-1 greedy lookahead using value head. Track 2.
At each step picks argmax_a [ r(z,a) + gamma * V(z') ].
"""

from typing import List

import numpy as np

from benchmark.core.base_planner import Planner
from benchmark.core.budget import BudgetExhausted, PlanningBudget
from benchmark.core.world_model import State, WorldModel


class LatentRolloutPlanner(Planner):
    name = "latent_rollout"
    uses_value = True
    uses_policy = False

    def __init__(self, horizon: int = 20, gamma: float = 0.99):
        self.horizon = horizon
        self.gamma = gamma

    def plan(self, z: State, world_model: WorldModel, budget: PlanningBudget) -> List[int]:
        actions = world_model.action_space()
        best_action = np.random.choice(actions)
        best_val = -np.inf

        try:
            for a in actions:
                next_z, r, done = world_model.step(z, a, budget)
                val = r + (0.0 if done else self.gamma * world_model.value(next_z))
                if val > best_val:
                    best_val, best_action = val, a
        except BudgetExhausted:
            pass

        return [best_action]
