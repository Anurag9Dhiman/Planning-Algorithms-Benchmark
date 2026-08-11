"""
LeWM Paper CEM Planner — matches the original LeWorldModel planning protocol.

Key differences from generic CEM:
  1. Terminal cost only: ||z_H - z_goal||  (not cumulative step rewards)
     The paper evaluates action sequences by their FINAL predicted state,
     not accumulated rewards along the way.

  2. Single canonical goal latent: world_model._single_goal_latent
     Uses the mean of the goal bank rather than min-distance to 202 points,
     giving a cleaner monotonic potential field.

  3. Budget-aware iteration: fills the full budget per planning round.

Paper reference: LeWorldModel (arXiv 2603.19312)
  - CEM: 300 samples, 30 iterations, horizon=5 (continuous-action envs)
  - We adapt to discrete actions and a 500-call-per-round budget.
"""

from __future__ import annotations

import random
from typing import List

import numpy as np

from benchmark.core.base_planner import Planner
from benchmark.core.budget import BudgetExhausted, PlanningBudget
from benchmark.core.world_model import State, WorldModel


class LeWMCEMPlanner(Planner):
    """
    CEM planner designed for LeWM's latent space.

    Uses terminal-cost-only evaluation: plans find the action sequence whose
    predicted FINAL state is closest to the goal, rather than maximising
    cumulative per-step reward. This matches JEPA.criterion() from the paper.

    Requires world_model._single_goal_latent (set by load_goal_latents()).
    Falls back to random action if no goal latent is available.
    """

    name = "lewm_cem"
    uses_value = False
    uses_policy = False

    def __init__(
        self,
        n_samples: int = 100,
        elite_frac: float = 0.1,
        n_iters: int = 3,
        horizon: int = 5,
    ):
        self.n_samples = n_samples
        self.elite_frac = elite_frac
        self.n_iters = n_iters
        self.horizon = horizon

    def plan(
        self,
        z: State,
        world_model: WorldModel,
        budget: PlanningBudget,
    ) -> List[int]:
        goal = getattr(world_model, "_single_goal_latent", None)
        if goal is None:
            return [random.choice(world_model.action_space())]

        actions = world_model.action_space()
        n_actions = len(actions)
        n_elite = max(1, int(self.n_samples * self.elite_frac))

        # Uniform categorical distribution over each planning step
        logits = np.zeros((self.horizon, n_actions), dtype=np.float32)
        best_action = random.choice(actions)
        best_cost = float("inf")

        try:
            for _ in range(self.n_iters):
                probs = _softmax(logits)  # (horizon, n_actions)
                costs: List[float] = []
                seqs: List[List[int]] = []

                for _ in range(self.n_samples):
                    seq = [
                        int(np.random.choice(n_actions, p=probs[t]))
                        for t in range(self.horizon)
                    ]
                    # Roll out horizon steps through the predictor
                    state = z
                    for a_idx in seq:
                        state, _, _ = world_model.step(state, actions[a_idx], budget)

                    # Terminal cost: L2 distance from predicted final state to goal
                    cost = float(np.linalg.norm(state - goal))
                    costs.append(cost)
                    seqs.append(seq)

                # Elite update
                elite_idx = np.argsort(costs)[:n_elite]
                if costs[elite_idx[0]] < best_cost:
                    best_cost = costs[elite_idx[0]]
                    best_action = actions[seqs[elite_idx[0]][0]]

                # Refit distribution from elite sequences
                new_logits = np.zeros_like(logits)
                for i in elite_idx:
                    for t, a_idx in enumerate(seqs[i]):
                        new_logits[t, a_idx] += 1.0
                logits = np.log(new_logits / n_elite + 1e-8)

        except BudgetExhausted:
            pass

        return [best_action]


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)
