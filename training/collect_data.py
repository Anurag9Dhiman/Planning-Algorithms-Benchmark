"""
Collect rollout data from MiniGrid environments for world model training.

Saves a dataset of (obs_t, action_t, reward_t, obs_{t+1}, done_t) tuples
as a compressed numpy archive.

Policy mix (--epsilon-greedy):
  Each episode is either fully random (with prob 1-eps) or BFS-guided to goal
  (with prob eps). BFS-guided episodes guarantee at least one positive-reward
  trajectory, which is essential for sparse-reward envs like DoorKey where a
  pure random policy almost never solves the task.

Usage:
    # simple random collection (good for Empty-8x8, FourRooms)
    python training/collect_data.py \
        --env MiniGrid-Empty-8x8-v0 \
        --episodes 1000 \
        --output data/minigrid_empty_8x8.npz

    # mixed collection (required for DoorKey)
    python training/collect_data.py \
        --env MiniGrid-DoorKey-8x8-v0 \
        --episodes 5000 \
        --epsilon-greedy 0.3 \
        --output data/minigrid_doorkey.npz
"""

import argparse
import random
import sys
from collections import deque
from pathlib import Path
from typing import List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.envs.minigrid_wrap import MiniGridWrapper


# ── BFS oracle (uses perfect model via env clone) ────────────────────────────

def _bfs_actions(env: MiniGridWrapper, max_steps: int) -> Optional[List[int]]:
    """Return a BFS-optimal action sequence, or None if goal not found.

    Works directly on gym env copies to avoid MiniGridWrapper overhead and
    rendering (PIL resize) during search.  Key includes door open/closed
    state so DoorKey and MultiRoom are handled correctly.
    """
    import copy as _copy

    start = env.clone_state()  # copy of env._env (gymnasium env object)

    def _key(gym_env):
        u = gym_env.unwrapped
        ag = tuple(int(x) for x in u.agent_pos)
        dr = int(u.agent_dir)
        cr = getattr(u, "carrying", None)
        cr_type = type(cr).__name__ if cr else None
        grid = u.grid
        door_states = tuple(
            (x, y, bool(grid.get(x, y).is_open))
            for x in range(grid.width)
            for y in range(grid.height)
            if grid.get(x, y) is not None and hasattr(grid.get(x, y), 'is_open')
        )
        return (ag, dr, cr_type, door_states)

    queue = deque([(start, [])])
    visited_keys = {_key(start)}

    while queue:
        state, actions = queue.popleft()
        if len(actions) >= max_steps:
            continue
        for a in range(env.n_actions):
            e2 = _copy.deepcopy(state)
            _, reward, terminated, truncated, _ = e2.step(a)
            done = terminated or truncated
            if done and reward > 0:
                return actions + [a]
            k = _key(e2)
            if k not in visited_keys:
                visited_keys.add(k)
                queue.append((e2, actions + [a]))

    return None


def _collect_episode(env, rng, use_bfs: bool, seed: int):
    """Collect one episode; returns list of (obs, action, reward, next_obs, done)."""
    obs = env.reset(seed=seed)
    transitions = []

    if use_bfs:
        plan = _bfs_actions(env, max_steps=env._bfs_depth if hasattr(env, "_bfs_depth") else min(env.max_steps, 200))
        obs = env.reset(seed=seed)  # reset again after BFS exploration
        if plan is not None:
            for a in plan:
                next_obs, reward, done, _ = env.step(a)
                transitions.append((obs, a, reward, next_obs, done))
                obs = next_obs
                if done:
                    return transitions
            # BFS plan exhausted without done — fall through to random
        # BFS failed or plan ran out: finish episode randomly

    for _ in range(env.max_steps - len(transitions)):
        a = rng.randint(0, env.n_actions - 1)
        next_obs, reward, done, _ = env.step(a)
        transitions.append((obs, a, reward, next_obs, done))
        obs = next_obs
        if done:
            break

    return transitions


# ── Main collection loop ──────────────────────────────────────────────────────

def collect(
    env_id: str,
    n_episodes: int,
    seed: int,
    render_size: int,
    epsilon_greedy: float = 0.0,
    max_steps_per_ep: int = None,
    bfs_depth: int = None,
) -> dict:
    env = MiniGridWrapper(env_id, render_size=render_size)
    if bfs_depth is not None:
        env._bfs_depth = bfs_depth
    rng = random.Random(seed)

    ep_max_steps = max_steps_per_ep or env.max_steps
    max_total = n_episodes * ep_max_steps

    # Pre-allocate arrays to avoid Python list overhead (avoids 2x peak RAM)
    obs_buf      = np.empty((max_total, render_size, render_size, 3), dtype=np.uint8)
    next_obs_buf = np.empty_like(obs_buf)
    act_buf      = np.empty(max_total, dtype=np.int64)
    rew_buf      = np.empty(max_total, dtype=np.float32)
    done_buf     = np.empty(max_total, dtype=bool)

    idx = 0
    successes = 0

    for ep in range(n_episodes):
        use_bfs = epsilon_greedy > 0 and rng.random() < epsilon_greedy
        transitions = _collect_episode(env, rng, use_bfs=use_bfs, seed=seed + ep)

        # Cap to max_steps_per_ep if set
        if max_steps_per_ep is not None:
            transitions = transitions[:max_steps_per_ep]

        for obs, a, r, next_obs, done in transitions:
            obs_buf[idx]      = obs
            next_obs_buf[idx] = next_obs
            act_buf[idx]      = a
            rew_buf[idx]      = r
            done_buf[idx]     = done
            idx += 1

        if transitions and transitions[-1][2] > 0:
            successes += 1

        if (ep + 1) % 200 == 0:
            print(
                f"  ep {ep+1}/{n_episodes}"
                f"  transitions={idx}"
                f"  success_rate={successes/(ep+1)*100:.1f}%"
            )

    return {
        "obs":      obs_buf[:idx],
        "action":   act_buf[:idx],
        "reward":   rew_buf[:idx],
        "next_obs": next_obs_buf[:idx],
        "done":     done_buf[:idx],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",            default="MiniGrid-Empty-8x8-v0")
    parser.add_argument("--episodes",       type=int,   default=2000)
    parser.add_argument("--seed",           type=int,   default=0)
    parser.add_argument("--render-size",    type=int,   default=224)
    parser.add_argument("--epsilon-greedy", type=float, default=0.0,
                        help="Fraction of episodes guided by BFS to goal (0=pure random).")
    parser.add_argument("--max-steps",      type=int,   default=None,
                        help="Cap each episode at this many steps (default: env max_steps).")
    parser.add_argument("--bfs-depth",      type=int,   default=None,
                        help="BFS search depth for guided episodes (default: same as max-steps).")
    parser.add_argument("--output",         default="data/minigrid_empty_8x8.npz")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    print(f"Collecting {args.episodes} episodes from {args.env}  "
          f"(epsilon_greedy={args.epsilon_greedy}"
          f"{f', max_steps={args.max_steps}' if args.max_steps else ''}"
          f"{f', bfs_depth={args.bfs_depth}' if args.bfs_depth else ''}) ...")
    data = collect(args.env, args.episodes, args.seed, args.render_size,
                   args.epsilon_greedy, args.max_steps, args.bfs_depth)
    np.savez_compressed(args.output, **data)

    n = len(data["obs"])
    print(f"\nSaved {n} transitions -> {args.output}")
    print(f"  obs shape    : {data['obs'].shape}")
    print(f"  reward range : [{data['reward'].min():.3f}, {data['reward'].max():.3f}]")
    print(f"  done rate    : {data['done'].mean()*100:.1f}%")
    print(f"  success rate : {(data['reward'] > 0).mean()*100:.2f}% of transitions have r>0")


if __name__ == "__main__":
    main()
