"""
MCTS: UCB1 tree search with random rollouts. Two variants share this file:
  - MCTSPlanner        (Track 1): value bootstrap disabled (pure rollout)
  - MCTSValuePlanner   (Track 2): uses world_model.value() at leaf nodes
"""

import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from benchmark.core.base_planner import Planner
from benchmark.core.budget import BudgetExhausted, PlanningBudget
from benchmark.core.world_model import State, WorldModel


def _hash(z: State) -> tuple:
    return tuple(np.round(z, 2).tolist())


class _Node:
    __slots__ = ("z", "parent", "action_taken", "children", "visit_count", "total_value", "untried_actions")

    def __init__(self, z: State, actions: List[int], parent=None, action_taken=None):
        self.z = z
        self.parent = parent
        self.action_taken = action_taken
        self.children: Dict[int, "_Node"] = {}
        self.visit_count = 0
        self.total_value = 0.0
        self.untried_actions = list(actions)
        random.shuffle(self.untried_actions)

    @property
    def q_value(self) -> float:
        return self.total_value / self.visit_count if self.visit_count > 0 else 0.0

    def ucb1(self, c: float) -> float:
        if self.visit_count == 0:
            return float("inf")
        return self.q_value + c * math.sqrt(math.log(self.parent.visit_count) / self.visit_count)


class _MCTSBase(Planner):
    def __init__(self, n_simulations: int = 200, c_puct: float = 1.41, rollout_depth: int = 10):
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.rollout_depth = rollout_depth

    def plan(self, z: State, world_model: WorldModel, budget: PlanningBudget) -> List[int]:
        actions = world_model.action_space()
        root = _Node(z, actions)

        try:
            for _ in range(self.n_simulations):
                node, cum_reward = self._select(root, world_model, budget)
                value = self._evaluate(node, world_model, budget) + cum_reward
                self._backprop(node, value)
        except BudgetExhausted:
            pass

        if not root.children:
            return [random.choice(actions)]
        best = max(root.children.values(), key=lambda n: n.visit_count)
        return [best.action_taken]

    def _select(self, node: _Node, wm: WorldModel, budget: PlanningBudget) -> Tuple[_Node, float]:
        cum_reward = 0.0
        while True:
            if node.untried_actions:
                return self._expand(node, wm, budget, cum_reward)
            if not node.children:
                return node, cum_reward
            node = max(node.children.values(), key=lambda n: n.ucb1(self.c_puct))
            # re-simulate the edge to get reward
            _, r, done = wm.step(node.parent.z if node.parent else node.z,
                                 node.action_taken, budget)
            cum_reward += r
            if done:
                return node, cum_reward

    def _expand(self, node: _Node, wm: WorldModel, budget: PlanningBudget, cum_reward: float) -> Tuple[_Node, float]:
        action = node.untried_actions.pop()
        next_z, r, done = wm.step(node.z, action, budget)
        child = _Node(next_z, wm.action_space(), parent=node, action_taken=action)
        node.children[action] = child
        return child, cum_reward + r

    def _evaluate(self, node: _Node, wm: WorldModel, budget: PlanningBudget) -> float:
        raise NotImplementedError

    @staticmethod
    def _backprop(node: _Node, value: float) -> None:
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            node = node.parent


class MCTSPlanner(_MCTSBase):
    """Track 1: random rollout, no value head."""
    name = "mcts"
    uses_value = False
    uses_policy = False

    def _evaluate(self, node: _Node, wm: WorldModel, budget: PlanningBudget) -> float:
        z = node.z
        total = 0.0
        for _ in range(self.rollout_depth):
            a = random.choice(wm.action_space())
            z, r, done = wm.step(z, a, budget)
            total += r
            if done:
                break
        return total


class MCTSValuePlanner(_MCTSBase):
    """Track 2: bootstraps leaf value with world_model.value(z)."""
    name = "mcts_value"
    uses_value = True
    uses_policy = False

    def _evaluate(self, node: _Node, wm: WorldModel, budget: PlanningBudget) -> float:
        return wm.value(node.z)
