"""
BenchmarkRunner: executes one (planner, env, world_model) combination
for N episodes and returns EpisodeResult objects.
"""

from __future__ import annotations

import random
import time
from typing import List, Optional

import numpy as np

from benchmark.core.base_env import BenchmarkEnv
from benchmark.core.base_planner import Planner
from benchmark.core.budget import BudgetConfig, BudgetExhausted, PlanningBudget
from benchmark.core.world_model import WorldModel
from benchmark.metrics.evaluator import EpisodeResult


class BenchmarkRunner:
    def __init__(
        self,
        budget_config: BudgetConfig,
        receding_horizon: bool = True,
        verbose: bool = False,
    ):
        self.budget_config = budget_config
        self.receding_horizon = receding_horizon
        self.verbose = verbose

    def run(
        self,
        planner: Planner,
        env: BenchmarkEnv,
        world_model: WorldModel,
        n_episodes: int,
        seeds: Optional[List[int]] = None,
    ) -> List[EpisodeResult]:

        if seeds is None:
            seeds = list(range(n_episodes))
        seeds = seeds[:n_episodes]

        results = []
        for seed in seeds:
            result = self._run_episode(planner, env, world_model, seed)
            results.append(result)
            if self.verbose:
                print(
                    f"  [{planner.name}|{world_model.name}|{env.task_meta.env_name}]"
                    f"  seed={seed}"
                    f"  return={result.episode_return:.3f}"
                    f"  success={result.success}"
                    f"  calls={result.model_calls_used}"
                )
        return results

    def _run_episode(
        self,
        planner: Planner,
        env: BenchmarkEnv,
        world_model: WorldModel,
        seed: int,
    ) -> EpisodeResult:

        obs = env.reset(seed=seed)
        z = world_model.encode(obs)
        total_reward = 0.0
        total_model_calls = 0
        all_actions: List[int] = []
        all_rewards: List[float] = []
        budget_exhausted = False
        t0 = time.time()

        for step in range(env.max_steps):
            budget = PlanningBudget(self.budget_config)
            try:
                action_seq = planner.plan(z, world_model, budget)
            except BudgetExhausted:
                action_seq = [random.randint(0, env.n_actions - 1)]
                budget_exhausted = True

            total_model_calls += budget.model_calls_used
            to_execute = [action_seq[0]] if self.receding_horizon else action_seq

            done = False
            for action in to_execute:
                obs, reward, done, _ = env.step(action)
                z = world_model.encode(obs)
                total_reward += reward
                all_actions.append(action)
                all_rewards.append(reward)
                if done or len(all_actions) >= env.max_steps:
                    break

            if budget.exhausted and not budget_exhausted:
                budget_exhausted = True

            if done:
                break

        return EpisodeResult(
            planner=planner.name,
            world_model=world_model.name,
            env_name=env.task_meta.env_name,
            task_meta=env.task_meta,
            seed=seed,
            episode_return=total_reward,
            steps_taken=len(all_actions),
            success=done and total_reward > 0,
            model_calls_used=total_model_calls,
            wall_time_s=time.time() - t0,
            budget_exhausted=budget_exhausted,
            actions_taken=all_actions,
            rewards=all_rewards,
        )
