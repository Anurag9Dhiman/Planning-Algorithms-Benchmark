"""Beam Search: keep top-K candidates at each depth step. Dynamics only."""

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
