"""
Policy Iteration: oracle planner, requires enumerable state space.
Alternates policy evaluation and greedy improvement until convergence.
"""

from typing import Dict, List

import numpy as np

from benchmark.core.base_planner import Planner
from benchmark.core.budget import BudgetConfig, BudgetExhausted, PlanningBudget
from benchmark.core.world_model import State, WorldModel


def _hash(z: State) -> tuple:
    return tuple(z.tolist())


class PolicyIterationPlanner(Planner):
    name = "policy_iteration"
    uses_value = False
    uses_policy = False

    def __init__(self, gamma: float = 0.99, eval_iters: int = 100):
        self.gamma = gamma
        self.eval_iters = eval_iters
        self._policy: Dict[tuple, int] = {}
        self._fitted = False

    def fit(self, world_model: WorldModel) -> None:
        states = world_model.all_states()
        actions = world_model.action_space()
        budget = PlanningBudget(BudgetConfig(max_model_calls=10_000_000, max_wall_time_s=3600))

        # Init random policy
        pi: Dict[tuple, int] = {_hash(s): np.random.choice(actions) for s in states}
        V: Dict[tuple, float] = {_hash(s): 0.0 for s in states}

        while True:
            # --- Policy Evaluation ---
            for _ in range(self.eval_iters):
                for s in states:
                    h = _hash(s)
                    a = pi[h]
                    try:
                        s_next, r, done = world_model.step(s, a, budget)
                    except BudgetExhausted:
                        break
                    v_next = 0.0 if done else V.get(_hash(s_next), 0.0)
                    V[h] = r + self.gamma * v_next

            # --- Policy Improvement ---
            stable = True
            for s in states:
                h = _hash(s)
                old_a = pi[h]
                best_a, best_q = old_a, -np.inf
                for a in actions:
                    try:
                        s_next, r, done = world_model.step(s, a, budget)
                    except BudgetExhausted:
                        break
                    v_next = 0.0 if done else V.get(_hash(s_next), 0.0)
                    q = r + self.gamma * v_next
                    if q > best_q:
                        best_q, best_a = q, a
                pi[h] = best_a
                if best_a != old_a:
                    stable = False

            if stable:
                break

        self._policy = pi
        self._fitted = True

    def plan(self, z: State, world_model: WorldModel, budget: PlanningBudget) -> List[int]:
        if not self._fitted:
            self.fit(world_model)
        action = self._policy.get(_hash(z), np.random.choice(world_model.action_space()))
        return [action]
