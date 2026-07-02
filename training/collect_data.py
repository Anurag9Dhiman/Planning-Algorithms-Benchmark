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
    """Return a BFS-optimal action sequence, or None if goal not found."""
    import copy
    start = env.clone_state()
    queue = deque([(start, [])])
    visited = {id(start)}  # approximate; reset each call

    # Use a hash of (agent_pos, agent_dir, carrying) as visited key
    def _key(s):
        e2 = copy.deepcopy(env)
        e2.restore_state(s)
        ag = e2._env.unwrapped.agent_pos
        dr = e2._env.unwrapped.agent_dir
        cr = getattr(e2._env.unwrapped, "carrying", None)
        cr_type = type(cr).__name__ if cr else None
        return (ag, dr, cr_type)

    visited_keys = {_key(start)}

    while queue:
        state, actions = queue.popleft()
        if len(actions) >= max_steps:
            continue
        for a in range(env.n_actions):
            e2 = MiniGridWrapper.__new__(MiniGridWrapper)
            e2.__dict__.update(env.__dict__)
            e2._env = __import__("copy").deepcopy(env._env)
            e2.restore_state(state)
            _, reward, done, _ = e2.step(a)
            nstate = e2.clone_state()
            k = _key(nstate)
            if done and reward > 0:
                return actions + [a]
            if k not in visited_keys:
                visited_keys.add(k)
                queue.append((nstate, actions + [a]))
    return None


def _collect_episode(env, rng, use_bfs: bool, seed: int):
    """Collect one episode; returns list of (obs, action, reward, next_obs, done)."""
    obs = env.reset(seed=seed)
    transitions = []

    if use_bfs:
        plan = _bfs_actions(env, max_steps=min(env.max_steps, 200))
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
) -> dict:
    env = MiniGridWrapper(env_id, render_size=render_size)
    rng = random.Random(seed)

    obs_list, act_list, rew_list, next_obs_list, done_list = [], [], [], [], []
    successes = 0

    for ep in range(n_episodes):
        use_bfs = epsilon_greedy > 0 and rng.random() < epsilon_greedy
        transitions = _collect_episode(env, rng, use_bfs=use_bfs, seed=seed + ep)

        for obs, a, r, next_obs, done in transitions:
            obs_list.append(obs)
            act_list.append(a)
            rew_list.append(r)
            next_obs_list.append(next_obs)
            done_list.append(done)

        if transitions and transitions[-1][2] > 0:
            successes += 1

        if (ep + 1) % 200 == 0:
            print(
                f"  ep {ep+1}/{n_episodes}"
                f"  transitions={len(obs_list)}"
                f"  success_rate={successes/(ep+1)*100:.1f}%"
            )

    return {
        "obs":      np.array(obs_list,      dtype=np.uint8),
        "action":   np.array(act_list,      dtype=np.int64),
        "reward":   np.array(rew_list,      dtype=np.float32),
        "next_obs": np.array(next_obs_list, dtype=np.uint8),
        "done":     np.array(done_list,     dtype=bool),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",            default="MiniGrid-Empty-8x8-v0")
    parser.add_argument("--episodes",       type=int,   default=2000)
    parser.add_argument("--seed",           type=int,   default=0)
    parser.add_argument("--render-size",    type=int,   default=224)
    parser.add_argument("--epsilon-greedy", type=float, default=0.0,
                        help="Fraction of episodes guided by BFS to goal (0=pure random).")
    parser.add_argument("--output",         default="data/minigrid_empty_8x8.npz")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    print(f"Collecting {args.episodes} episodes from {args.env}  "
          f"(epsilon_greedy={args.epsilon_greedy}) ...")
    data = collect(args.env, args.episodes, args.seed, args.render_size, args.epsilon_greedy)
    np.savez_compressed(args.output, **data)

    n = len(data["obs"])
    print(f"\nSaved {n} transitions -> {args.output}")
    print(f"  obs shape    : {data['obs'].shape}")
    print(f"  reward range : [{data['reward'].min():.3f}, {data['reward'].max():.3f}]")
    print(f"  done rate    : {data['done'].mean()*100:.1f}%")
    print(f"  success rate : {(data['reward'] > 0).mean()*100:.2f}% of transitions have r>0")


if __name__ == "__main__":
    main()
