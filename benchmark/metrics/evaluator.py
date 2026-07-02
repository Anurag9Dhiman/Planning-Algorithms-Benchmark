from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from benchmark.core.base_env import TaskMeta


@dataclass
class EpisodeResult:
    planner: str
    world_model: str
    env_name: str
    task_meta: TaskMeta
    seed: int
    episode_return: float
    steps_taken: int
    success: bool
    model_calls_used: int
    wall_time_s: float
    budget_exhausted: bool
    actions_taken: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)

    @property
    def budget_efficiency(self) -> float:
        calls = max(self.model_calls_used, 1)
        return self.episode_return / calls


@dataclass
class AggregateResult:
    planner: str
    world_model: str
    env_name: str
    n_episodes: int
    mean_return: float
    std_return: float
    success_rate: float
    mean_model_calls: float
    mean_wall_time_s: float
    budget_efficiency: float
    budget_exhaustion_rate: float
    mean_steps_success: Optional[float]   # None if no successes


def aggregate(results: List[EpisodeResult]) -> AggregateResult:
    returns = [r.episode_return for r in results]
    successes = [r.success for r in results]
    calls = [r.model_calls_used for r in results]
    times = [r.wall_time_s for r in results]
    efficiencies = [r.budget_efficiency for r in results]
    exhausted = [r.budget_exhausted for r in results]
    success_steps = [r.steps_taken for r in results if r.success]

    return AggregateResult(
        planner=results[0].planner,
        world_model=results[0].world_model,
        env_name=results[0].env_name,
        n_episodes=len(results),
        mean_return=float(np.mean(returns)),
        std_return=float(np.std(returns)),
        success_rate=float(np.mean(successes)),
        mean_model_calls=float(np.mean(calls)),
        mean_wall_time_s=float(np.mean(times)),
        budget_efficiency=float(np.mean(efficiencies)),
        budget_exhaustion_rate=float(np.mean(exhausted)),
        mean_steps_success=float(np.mean(success_steps)) if success_steps else None,
    )
