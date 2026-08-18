"""
MiniGrid adapter for the benchmark.

Wraps a Gymnasium MiniGrid environment and exposes:
  - RGB pixel observations (full grid render, not partial view)
  - clone_state / restore_state for PerfectModel compatibility
  - task_meta reflecting the chosen env
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Tuple

import numpy as np

from benchmark.core.base_env import BenchmarkEnv, TaskMeta

# Action subset used by MiniGrid (we keep all 7 but mask unused ones via max_steps)
# 0:left, 1:right, 2:forward, 3:pickup, 4:drop, 5:toggle, 6:done
_MINIGRID_N_ACTIONS = 7

_TASK_META = {
    "MiniGrid-Empty-5x5-v0":       TaskMeta(1, "minigrid_empty_5x5",    "short",  "sparse", False, True,  _MINIGRID_N_ACTIONS),
    "MiniGrid-Empty-8x8-v0":       TaskMeta(1, "minigrid_empty_8x8",    "short",  "sparse", False, True,  _MINIGRID_N_ACTIONS),
    "MiniGrid-FourRooms-v0":       TaskMeta(1, "minigrid_fourrooms",     "medium", "sparse", False, True,  _MINIGRID_N_ACTIONS),
    "MiniGrid-DoorKey-8x8-v0":     TaskMeta(1, "minigrid_doorkey_8x8",  "medium", "sparse", False, True,  _MINIGRID_N_ACTIONS),
    "MiniGrid-MultiRoom-N4-S5-v0": TaskMeta(1, "minigrid_multiroom_n4", "long",   "sparse", False, True,  _MINIGRID_N_ACTIONS),
    "MiniGrid-LavaGap-S7-v0":      TaskMeta(1, "minigrid_lavagap_s7",   "short",  "sparse", False, False, _MINIGRID_N_ACTIONS),
}

# Stochastic suffix appended to env_name when slip_prob > 0
_STOCHASTIC_SUFFIX = "_stochastic"


class MiniGridWrapper(BenchmarkEnv):
    """
    Parameters
    ----------
    env_id      : Gymnasium MiniGrid environment ID
    render_size : pixel size of the full-grid RGB render (H=W)
    """

    def __init__(self, env_id: str = "MiniGrid-Empty-8x8-v0", render_size: int = 224, slip_prob: float = 0.0):
        try:
            import gymnasium as gym
            import minigrid  # registers envs
        except ImportError as e:
            raise ImportError("Install minigrid: pip install minigrid gymnasium") from e

        self._env_id = env_id
        self._render_size = render_size
        self._slip_prob = slip_prob
        self._env = gym.make(env_id, render_mode="rgb_array")
        self._last_obs: np.ndarray = None
        self._steps = 0

    # ── BenchmarkEnv interface ────────────────────────────────────────

    def reset(self, seed: int = None) -> np.ndarray:
        obs, _ = self._env.reset(seed=seed)
        self._steps = 0
        self._last_obs = self._render()
        return self._last_obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        if self._slip_prob > 0 and np.random.random() < self._slip_prob:
            action = np.random.randint(self.n_actions)
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._steps += 1
        done = terminated or truncated
        self._last_obs = self._render()
        return self._last_obs, float(reward), done, info

    def clone_state(self) -> Any:
        return copy.deepcopy(self._env)

    def restore_state(self, state: Any) -> None:
        self._env = copy.deepcopy(state)

    def all_states(self) -> List[Any]:
        raise NotImplementedError("MiniGrid state space is too large to enumerate.")

    @property
    def task_meta(self) -> TaskMeta:
        meta = _TASK_META.get(
            self._env_id,
            TaskMeta(1, self._env_id, "medium", "sparse", False, True, _MINIGRID_N_ACTIONS),
        )
        if self._slip_prob > 0:
            return TaskMeta(
                meta.tier,
                meta.env_name + _STOCHASTIC_SUFFIX,
                meta.horizon,
                meta.reward_density,
                True,
                meta.goal_conditioned,
                meta.branching_factor,
            )
        return meta

    @property
    def n_actions(self) -> int:
        return _MINIGRID_N_ACTIONS

    @property
    def max_steps(self) -> int:
        # MiniGrid sets max_steps internally; mirror it
        return getattr(self._env, "max_steps", 500)

    # ── Helpers ───────────────────────────────────────────────────────

    def _render(self) -> np.ndarray:
        """Returns (render_size, render_size, 3) uint8 RGB array."""
        frame = self._env.render()   # (H, W, 3) uint8
        if frame.shape[0] != self._render_size or frame.shape[1] != self._render_size:
            from PIL import Image
            frame = np.array(
                Image.fromarray(frame).resize(
                    (self._render_size, self._render_size), Image.BILINEAR
                )
            )
        return frame
