"""Beam Search planners: dynamics-only and value-guided variants."""

from typing import List, Tuple

import numpy as np

from benchmark.core.base_planner import Planner
from benchmark.core.budget import BudgetExhausted, PlanningBudget
from benchmark.core.world_model import State, WorldModel


class BeamSearchPlanner(Planner):
    name = "beam_search"
    uses_value = False
    uses_policy = False

    def __init__(self, beam_width: int = 10, horizon: int = 20):
        self.beam_width = beam_width
        self.horizon = horizon

    def plan(self, z: State, world_model: WorldModel, budget: PlanningBudget) -> List[int]:
        # beam: list of (cumulative_reward, state, action_sequence)
        beam: List[Tuple[float, State, List[int]]] = [(0.0, z, [])]
        actions = world_model.action_space()
        best_seq = [np.random.choice(actions)]

        try:
            for _ in range(self.horizon):
                candidates = []
                for cum_r, state, seq in beam:
                    for a in actions:
                        next_z, r, done = world_model.step(state, a, budget)
                        next_seq = seq + [a]
                        next_r = cum_r + r
                        if done:
                            return next_seq
                        candidates.append((next_r, next_z, next_seq))

                # keep top beam_width by cumulative reward
                candidates.sort(key=lambda x: x[0], reverse=True)
                beam = candidates[: self.beam_width]
                if beam:
                    best_seq = beam[0][2]
        except BudgetExhausted:
            pass

        return best_seq if best_seq else [np.random.choice(actions)]


class ValueBeamSearchPlanner(Planner):
    """
    Beam search scored by cumulative_reward + gamma^depth * V(z_next).

    Identical expansion to BeamSearchPlanner but uses the world model's value
    function to bootstrap from the beam frontier, turning it into a Track-2
    planner. Equivalent to a width-K version of greedy latent rollout.
    """

    name = "value_beam_search"
    uses_value = True
    uses_policy = False

    def __init__(self, beam_width: int = 10, horizon: int = 20, gamma: float = 0.99):
        self.beam_width = beam_width
        self.horizon    = horizon
        self.gamma      = gamma

    def plan(self, z: State, world_model: WorldModel, budget: PlanningBudget) -> List[int]:
        # beam: (score, cum_reward, state, action_seq, depth)
        beam: List[Tuple[float, float, State, List[int], int]] = [
            (0.0, 0.0, z, [], 0)
        ]
        actions  = world_model.action_space()
        best_seq = [np.random.choice(actions)]

        try:
            for _ in range(self.horizon):
                candidates = []
                for _, cum_r, state, seq, depth in beam:
                    for a in actions:
                        next_z, r, done = world_model.step(state, a, budget)
                        next_seq  = seq + [a]
                        next_cum  = cum_r + r
                        if done:
                            return next_seq
                        v     = world_model.value(next_z)
                        score = next_cum + self.gamma ** (depth + 1) * v
                        candidates.append((score, next_cum, next_z, next_seq, depth + 1))

                candidates.sort(key=lambda x: x[0], reverse=True)
                beam = candidates[: self.beam_width]
                if beam:
                    best_seq = beam[0][3]
        except BudgetExhausted:
            pass

        return best_seq if best_seq else [np.random.choice(actions)]
