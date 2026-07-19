"""
Value Iteration: oracle planner, requires enumerable state space.
Computes globally optimal V* then acts greedily. Only works with
PerfectModel or TabularModel on small envs.
"""

from typing import Dict, List, Tuple

import numpy as np

from benchmark.core.base_planner import Planner
from benchmark.core.budget import BudgetExhausted, PlanningBudget
from benchmark.core.world_model import State, WorldModel


def _hash(z):
    if isinstance(z, np.ndarray):
        return tuple(z.tolist())
    return z  # int IDs (PerfectModel) or other hashable types


class ValueIterationPlanner(Planner):
    name = "value_iteration"
    uses_value = False
    uses_policy = False

    def __init__(self, gamma: float = 0.99, eps: float = 1e-4, max_iters: int = 1000):
        self.gamma = gamma
        self.eps = eps
        self.max_iters = max_iters
        self._policy: Dict[tuple, int] = {}
        self._fitted = False

    def fit(self, world_model: WorldModel) -> None:
        """Pre-compute policy via VI. Call once before planning."""
        states = world_model.all_states()
        actions = world_model.action_space()
        V: Dict[tuple, float] = {_hash(s): 0.0 for s in states}

        # Use a dummy unlimited budget for VI (it's an offline oracle)
        from benchmark.core.budget import BudgetConfig
        budget = PlanningBudget(BudgetConfig(max_model_calls=10_000_000, max_wall_time_s=3600))

        for _ in range(self.max_iters):
            delta = 0.0
            new_V = {}
            for s in states:
                h = _hash(s)
                best = -np.inf
                for a in actions:
                    try:
                        s_next, r, done = world_model.step(s, a, budget)
                    except BudgetExhausted:
                        break
                    v_next = 0.0 if done else V.get(_hash(s_next), 0.0)
                    q = r + self.gamma * v_next
                    best = max(best, q)
                new_V[h] = best if best > -np.inf else 0.0
                delta = max(delta, abs(new_V[h] - V.get(h, 0.0)))
            V.update(new_V)
            if delta < self.eps:
                break

        # Extract greedy policy
        for s in states:
            best_a, best_q = None, -np.inf
            for a in actions:
                try:
                    s_next, r, done = world_model.step(s, a, budget)
                except BudgetExhausted:
                    break
                v_next = 0.0 if done else V.get(_hash(s_next), 0.0)
                q = r + self.gamma * v_next
                if q > best_q:
                    best_q, best_a = q, a
            if best_a is not None:
                self._policy[_hash(s)] = best_a

        self._fitted = True

    def plan(self, z: State, world_model: WorldModel, budget: PlanningBudget) -> List[int]:
        if not self._fitted:
            self.fit(world_model)
        action = self._policy.get(_hash(z), np.random.choice(world_model.action_space()))
        return [action]
