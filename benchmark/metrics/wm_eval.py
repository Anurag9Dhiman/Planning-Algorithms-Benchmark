"""
World model prediction quality evaluation.

Measures how accurately a world model predicts next latent, reward, and done
signal against the real environment. Uses random-action trajectories so results
are dynamics-only and not confounded by planner quality.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np

from benchmark.core.base_env import BenchmarkEnv
from benchmark.core.budget import BudgetConfig, PlanningBudget
from benchmark.core.world_model import WorldModel

_UNLIMITED = BudgetConfig(max_model_calls=10_000_000, max_wall_time_s=86_400)

DEFAULT_HORIZONS = [1, 2, 5, 10, 20]


@dataclass
class ModelEvalResult:
    world_model: str
    env_name: str
    n_episodes: int
    n_steps: int
    # Latent prediction error (L2); None for integer-ID latents (PerfectModel)
    latent_error_1step: Optional[float]
    # Mean L2 error at each open-loop rollout horizon
    rollout_error: Dict[int, float] = field(default_factory=dict)
    # Reward prediction
    reward_mae: float = 0.0
    reward_sign_accuracy: float = 0.0  # fraction where sign(r̂) == sign(r)
    # Done signal binary accuracy
    done_accuracy: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        # JSON needs string keys
        d["rollout_error"] = {str(k): v for k, v in self.rollout_error.items()}
        return d

    def pretty(self) -> str:
        lines = [
            f"  world_model : {self.world_model}",
            f"  env         : {self.env_name}",
            f"  episodes    : {self.n_episodes}  steps: {self.n_steps}",
        ]
        if self.latent_error_1step is not None:
            lines.append(f"  latent L2 (1-step) : {self.latent_error_1step:.4f}")
            for h, e in sorted(self.rollout_error.items()):
                lines.append(f"  latent L2 ({h:2d}-step) : {e:.4f}")
        else:
            lines.append("  latent error       : N/A (integer-ID latents)")
        lines += [
            f"  reward MAE         : {self.reward_mae:.4f}",
            f"  reward sign acc    : {self.reward_sign_accuracy:.3f}",
            f"  done accuracy      : {self.done_accuracy:.3f}",
        ]
        return "\n".join(lines)


def evaluate(
    world_model: WorldModel,
    env: BenchmarkEnv,
    n_episodes: int = 10,
    max_steps_per_ep: int = 100,
    horizons: List[int] = None,
    seed_offset: int = 9000,
) -> ModelEvalResult:
    """
    Roll out random-action episodes in the real env, then compare
    world_model predictions against ground-truth next observations.

    Two types of error are measured:
    - One-step: predict z_{t+1} from z_t; compare to encode(o_{t+1})
    - Multi-step (open-loop): chain predictions from z_t for H steps;
      compare final z_{t+H} to encode(o_{t+H})
    """
    if horizons is None:
        horizons = DEFAULT_HORIZONS
    max_horizon = max(horizons)

    latent_errors_1step: List[float] = []
    rollout_buf: Dict[int, List[float]] = {h: [] for h in horizons}
    reward_abs_errors: List[float] = []
    reward_sign_hits: List[float] = []
    done_hits: List[float] = []
    total_steps = 0

    for ep in range(n_episodes):
        obs = env.reset(seed=seed_offset + ep)

        # Collect a real trajectory (observations + actions)
        traj_obs = [obs]
        traj_actions: List[int] = []
        traj_rewards: List[float] = []
        traj_dones: List[bool] = []

        for _ in range(max_steps_per_ep + max_horizon):
            action = random.randint(0, env.n_actions - 1)
            obs, reward, done, _ = env.step(action)
            traj_obs.append(obs)
            traj_actions.append(action)
            traj_rewards.append(reward)
            traj_dones.append(done)
            if done:
                break

        T = len(traj_actions)
        # Encode all real observations into latents
        real_z = [world_model.encode(o) for o in traj_obs]
        is_vector = isinstance(real_z[0], np.ndarray)

        eval_steps = min(T, max_steps_per_ep)

        # ── One-step prediction ──────────────────────────────────────────────
        for t in range(eval_steps):
            budget = PlanningBudget(_UNLIMITED)
            try:
                z_pred, r_pred, done_pred = world_model.step(
                    real_z[t], traj_actions[t], budget
                )
            except Exception:
                continue

            total_steps += 1

            if is_vector:
                latent_errors_1step.append(
                    float(np.linalg.norm(z_pred - real_z[t + 1]))
                )

            reward_abs_errors.append(abs(r_pred - traj_rewards[t]))
            reward_sign_hits.append(
                float(np.sign(r_pred) == np.sign(traj_rewards[t]))
            )
            done_hits.append(float(bool(done_pred) == traj_dones[t]))

        # ── Multi-step open-loop rollout ─────────────────────────────────────
        if is_vector:
            for t in range(eval_steps):
                z_roll = real_z[t]
                for h in range(1, max_horizon + 1):
                    if t + h >= len(real_z):
                        break
                    budget = PlanningBudget(_UNLIMITED)
                    try:
                        z_roll, _, _ = world_model.step(
                            z_roll, traj_actions[t + h - 1], budget
                        )
                    except Exception:
                        break
                    if h in rollout_buf:
                        rollout_buf[h].append(
                            float(np.linalg.norm(z_roll - real_z[t + h]))
                        )

    return ModelEvalResult(
        world_model=world_model.name,
        env_name=env.task_meta.env_name,
        n_episodes=n_episodes,
        n_steps=total_steps,
        latent_error_1step=(
            float(np.mean(latent_errors_1step)) if latent_errors_1step else None
        ),
        rollout_error={
            h: float(np.mean(v)) for h, v in rollout_buf.items() if v
        },
        reward_mae=float(np.mean(reward_abs_errors)) if reward_abs_errors else 0.0,
        reward_sign_accuracy=(
            float(np.mean(reward_sign_hits)) if reward_sign_hits else 0.0
        ),
        done_accuracy=float(np.mean(done_hits)) if done_hits else 0.0,
    )
