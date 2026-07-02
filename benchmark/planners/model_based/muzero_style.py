"""
MuZero-style planner: MCTS with learned policy prior + value bootstrap. Track 2.
Uses PUCT selection: Q(z,a) + c * P(a|z) * sqrt(N(z)) / (1 + N(z,a)).
"""

import math
import random
from typing import Dict, List, Optional

import numpy as np

from benchmark.core.base_planner import Planner
from benchmark.core.budget import BudgetExhausted, PlanningBudget
from benchmark.core.world_model import State, WorldModel


class _PUCTNode:
    __slots__ = ("z", "parent", "action_taken", "children",
                 "visit_count", "total_value", "prior", "untried_actions")

    def __init__(self, z: State, prior: np.ndarray, actions: List[int],
                 parent=None, action_taken=None):
        self.z = z
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        self.children: Dict[int, "_PUCTNode"] = {}
        self.visit_count = 0
        self.total_value = 0.0
        self.untried_actions = list(actions)
        random.shuffle(self.untried_actions)

    def puct_score(self, c: float) -> float:
        if self.visit_count == 0:
            return float("inf")
        q = self.total_value / self.visit_count
        a = self.action_taken if self.action_taken is not None else 0
        p = self.prior[a] if self.parent is not None else 1.0
        n_parent = self.parent.visit_count if self.parent else 1
        return q + c * p * math.sqrt(n_parent) / (1 + self.visit_count)


class MuZeroStylePlanner(Planner):
    name = "muzero_style"
    uses_value = True
    uses_policy = True

    def __init__(self, n_simulations: int = 200, c_puct: float = 1.25):
        self.n_simulations = n_simulations
        self.c_puct = c_puct

    def plan(self, z: State, world_model: WorldModel, budget: PlanningBudget) -> List[int]:
        actions = world_model.action_space()
        prior = world_model.policy(z)
        root = _PUCTNode(z, prior, actions)

        try:
            for _ in range(self.n_simulations):
                node = self._select(root, world_model, budget)
                value = world_model.value(node.z)
                self._backprop(node, value)
        except BudgetExhausted:
            pass

        if not root.children:
            return [random.choice(actions)]
        best = max(root.children.values(), key=lambda n: n.visit_count)
        return [best.action_taken]

    def _select(self, node: _PUCTNode, wm: WorldModel, budget: PlanningBudget) -> _PUCTNode:
        while True:
            if node.untried_actions:
                return self._expand(node, wm, budget)
            if not node.children:
                return node
            node = max(node.children.values(), key=lambda n: n.puct_score(self.c_puct))

    def _expand(self, node: _PUCTNode, wm: WorldModel, budget: PlanningBudget) -> _PUCTNode:
        action = node.untried_actions.pop()
        next_z, _, _ = wm.step(node.z, action, budget)
        prior = wm.policy(next_z)
        child = _PUCTNode(next_z, prior, wm.action_space(), parent=node, action_taken=action)
        node.children[action] = child
        return child

    @staticmethod
    def _backprop(node: _PUCTNode, value: float) -> None:
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            node = node.parent
