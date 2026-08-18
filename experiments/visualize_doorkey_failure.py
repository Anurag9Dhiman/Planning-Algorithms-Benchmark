"""
Visualize WHY flat planners fail on DoorKey-8x8.

Produces two plots saved to --output-dir:
  1. value_heatmap.png  — LeWM goalft value V(s)=-L2(z,z_goal) as a heatmap
                          over every (x,y) cell; shows the planner's "gradient"
                          and why it is misleading (goal is behind locked door).
  2. episode_frames.png — Frame grid of a real failed UCS episode: what the
                          agent actually did step by step.

Usage:
    python experiments/visualize_doorkey_failure.py \
        --checkpoint /scratch/.../checkpoints/base_lewm_minigrid_doorkey_8x8_goalft.pt \
        --goal-data  /scratch/.../data/minigrid_doorkey_8x8_goalft_data.npz \
        --output-dir /scratch/.../viz/doorkey_failure \
        --seed 3
"""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.envs.minigrid_wrap import MiniGridWrapper
from benchmark.world_models.base_lewm import BaseLeWM
from benchmark.core.budget import BudgetConfig, PlanningBudget


# ── helpers ───────────────────────────────────────────────────────────────────

def encode_obs(wm, obs_np):
    """Encode a single (H,W,3) uint8 obs → latent vector."""
    return wm.encode(obs_np)   # BaseLeWM.encode() accepts (H,W,3) uint8 numpy


def value_for_obs(wm, obs_np):
    """Scalar planning value = -min L2(z, z_goal_k) over goal bank."""
    z = encode_obs(wm, obs_np)
    return wm.value(z)         # BaseLeWM.value() returns -min_k L2(z, z_goal_k)


def make_budget(n=10000):
    return PlanningBudget(BudgetConfig(max_model_calls=n, max_wall_time_s=3600.0))


# ── value heatmap ─────────────────────────────────────────────────────────────

def make_value_heatmap(wm, env, seed, out_path):
    """
    Sweep every (x, y) cell in the DoorKey grid, place the agent there
    (facing right), render, encode, compute value. Overlay on the base frame.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("matplotlib not available — skipping heatmap")
        return

    env.reset(seed=seed)
    base_frame = env._render()          # reference frame (actual start obs)
    unwrapped = env._env.unwrapped

    grid = unwrapped.grid
    W, H = grid.width, grid.height

    values = np.full((H, W), np.nan)

    for y in range(H):
        for x in range(W):
            cell = grid.get(x, y)
            # skip walls, doors, objects — agent can only stand on floor/goal
            if cell is not None and cell.type not in ("floor", "goal"):
                continue
            # clone env, place agent
            env2 = MiniGridWrapper.__new__(MiniGridWrapper)
            env2.__dict__.update(env.__dict__)
            env2._env = copy.deepcopy(env._env)
            env2._env.unwrapped.agent_pos = np.array([x, y])
            env2._env.unwrapped.agent_dir = 0  # facing right
            obs = env2._render()
            values[y, x] = value_for_obs(wm, obs)

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "LeWM (goal-ft) Value Heatmap — DoorKey-8x8\n"
        "V(s) = −L2(encode(s), z_goal)  |  brighter = higher value (closer to goal)",
        fontsize=11
    )

    # Left: base render
    axes[0].imshow(base_frame)
    axes[0].set_title("Environment (start state)")
    axes[0].axis("off")

    # Right: value heatmap
    vmin = np.nanmin(values)
    vmax = np.nanmax(values)
    im = axes[1].imshow(
        values, cmap="RdYlGn", vmin=vmin, vmax=vmax,
        interpolation="nearest", aspect="auto"
    )
    axes[1].set_title(
        "Value heatmap (agent pos → value)\n"
        "Red = low (far from goal) | Green = high (near goal)\n"
        "KEY: must be collected BEFORE goal is reachable"
    )
    axes[1].set_xlabel("x (grid column)")
    axes[1].set_ylabel("y (grid row)")
    plt.colorbar(im, ax=axes[1], label="V(s)")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap → {out_path}")


# ── episode frame grid ────────────────────────────────────────────────────────

def run_and_save_episode(wm, env, seed, out_path, max_steps=30, budget=200):
    """
    Run one episode using a greedy flat planner (best-action by value),
    record frames + annotations, save a grid image.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not available — skipping episode frames")
        return

    obs = env.reset(seed=seed)
    frames, annotations = [], []
    total_calls = 0
    n_actions = env.n_actions

    for step in range(max_steps):
        frames.append(obs.copy())

        # greedy one-step lookahead (flat planner logic)
        best_a, best_v = 0, -1e9
        env2 = MiniGridWrapper.__new__(MiniGridWrapper)
        env2.__dict__.update(env.__dict__)
        env2._env = copy.deepcopy(env._env)

        bgt = make_budget(n_actions + 1)
        for a in range(n_actions):
            env3 = MiniGridWrapper.__new__(MiniGridWrapper)
            env3.__dict__.update(env2.__dict__)
            env3._env = copy.deepcopy(env2._env)
            next_obs, reward, done, _ = env3.step(a)
            total_calls += 1
            if done and reward > 0:
                best_a, best_v = a, 1e9
                break
            v = value_for_obs(wm, next_obs)
            if v > best_v:
                best_v, best_a = v, a

        action_names = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]
        annotations.append({
            "step": step,
            "action": action_names[best_a],
            "value": best_v,
            "carrying": str(getattr(env._env.unwrapped, "carrying", None)),
        })

        obs, reward, done, _ = env.step(best_a)
        if done:
            frames.append(obs.copy())
            annotations.append({
                "step": step + 1,
                "action": "—",
                "value": value_for_obs(wm, obs),
                "carrying": "DONE",
            })
            break

    # ── layout ────────────────────────────────────────────────────────────────
    n = len(frames)
    cols = min(6, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.8, rows * 3.2))
    fig.suptitle(
        "DoorKey-8x8 — Greedy Flat Planner (LeWM goal-ft)\n"
        "Action chosen = max V(next_state) = min L2 to goal latent",
        fontsize=11
    )

    axes = np.array(axes).reshape(-1)
    for i, (frame, ann) in enumerate(zip(frames, annotations)):
        ax = axes[i]
        ax.imshow(frame)
        carrying = ann["carrying"]
        color = "green" if "Key" in carrying else "red"
        ax.set_title(
            f"Step {ann['step']}: {ann['action']}\n"
            f"V={ann['value']:.2f}  carry={carrying}",
            fontsize=7, color=color
        )
        ax.axis("off")

    for j in range(len(frames), len(axes)):
        axes[j].axis("off")

    # legend
    red_p   = mpatches.Patch(color="red",   label="Not carrying key")
    green_p = mpatches.Patch(color="green", label="Carrying key / done")
    fig.legend(handles=[red_p, green_p], loc="lower center", ncol=2, fontsize=9)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved episode frames → {out_path}")
    print(f"Total model calls: {total_calls}  |  Steps taken: {len(frames)}")
    for ann in annotations:
        print(f"  step={ann['step']:>2}  action={ann['action']:<8}  "
              f"V={ann['value']:>7.3f}  carrying={ann['carrying']}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--goal-data",  required=True)
    p.add_argument("--output-dir", default="viz/doorkey_failure")
    p.add_argument("--seed",       type=int, default=3)
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    out_dir = Path(args.output_dir)

    print(f"Loading LeWM goalft from {args.checkpoint} ...")
    wm = BaseLeWM(n_actions=7, checkpoint_path=args.checkpoint, device=args.device)
    wm.load_goal_latents(args.goal_data)
    wm.calibrate_done_threshold(args.goal_data)
    print(f"  goal latents: {wm._goal_latents.shape}  "
          f"threshold={wm.done_distance_threshold:.4f}")

    env = MiniGridWrapper("MiniGrid-DoorKey-8x8-v0", render_size=224)

    print(f"\n[1/2] Generating value heatmap (seed={args.seed}) ...")
    make_value_heatmap(wm, env, args.seed, out_dir / "value_heatmap.png")

    print(f"\n[2/2] Running greedy episode (seed={args.seed}) ...")
    run_and_save_episode(wm, env, args.seed, out_dir / "episode_frames.png")

    print(f"\nAll outputs in {out_dir}/")


if __name__ == "__main__":
    main()
