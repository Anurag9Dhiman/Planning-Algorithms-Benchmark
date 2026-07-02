"""Cross-Entropy Method: iteratively refine action distribution. Track 2."""

from typing import List

import numpy as np

from benchmark.core.base_planner import Planner
from benchmark.core.budget import BudgetExhausted, PlanningBudget
from benchmark.core.world_model import State, WorldModel


class CEMPlanner(Planner):
    name = "cem"
    uses_value = True
    uses_policy = False

    def __init__(
        self,
        n_samples: int = 64,
        elite_frac: float = 0.1,
        n_iters: int = 5,
        horizon: int = 15,
        gamma: float = 0.99,
    ):
        self.n_samples = n_samples
        self.elite_frac = elite_frac
        self.n_iters = n_iters
        self.horizon = horizon
        self.gamma = gamma

    def plan(self, z: State, world_model: WorldModel, budget: PlanningBudget) -> List[int]:
        actions = world_model.action_space()
        n_actions = len(actions)
        n_elite = max(1, int(self.n_samples * self.elite_frac))

        # Distribution over action indices: one categorical per step
        # Represented as logits (H, n_actions), initialised uniform
        logits = np.zeros((self.horizon, n_actions))
        best_seq = [np.random.choice(actions)] * self.horizon

        try:
            for _ in range(self.n_iters):
                probs = self._softmax(logits)   # (H, n_actions)
                returns, seqs = [], []

                for _ in range(self.n_samples):
                    seq = [
                        np.random.choice(n_actions, p=probs[t])
                        for t in range(self.horizon)
                    ]
                    total_r, state = 0.0, z
                    for t, a in enumerate(seq):
                        state, r, done = world_model.step(state, actions[a], budget)
                        total_r += (self.gamma ** t) * r
                        if done:
                            break
                    total_r += world_model.value(state)
                    returns.append(total_r)
                    seqs.append(seq)

                elite_idx = np.argsort(returns)[-n_elite:]
                elite_seqs = [seqs[i] for i in elite_idx]
                # MLE update: count elite action frequencies per step
                new_logits = np.zeros_like(logits)
                for seq in elite_seqs:
                    for t, a in enumerate(seq):
                        new_logits[t, a] += 1.0
                logits = np.log(new_logits / n_elite + 1e-8)
                best_seq = [actions[a] for a in elite_seqs[-1]]

        except BudgetExhausted:
            pass

        return best_seq

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)
