"""
Multi-Step Rollout Quality Evaluation
======================================
Measures how quickly each world model's imagination drifts from reality.

Experiment:
  1. Pick a starting observation from the held-out dataset
  2. Encode it -> z_0
  3. Roll out k steps using the model's dynamics with the RECORDED actions
     (open-loop: we replay what actually happened, not the model's own policy)
  4. At each step k, compare imagined z_k to actual encode(obs_{t+k})
  5. Report mean MSE at each rollout depth

This isolates the dynamics quality of the base model (encoder + dynamics only).
No value head, no done head, no planner involved.

Usage:
    python training/eval_rollout_quality.py \
        --model   dino_wm \
        --ckpt    checkpoints/dino_wm_minigrid_empty_8x8.pt \
        --data    data/minigrid_empty_8x8.npz \
        --env     minigrid_empty_8x8 \
        --max-steps 20 \
        --n-seqs    200 \
        --device    cpu
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.core.budget import BudgetConfig, PlanningBudget
from benchmark.registry import make_env, make_world_model


# ── Rollout evaluation ────────────────────────────────────────────────────────

def find_valid_starts(done_arr: np.ndarray, max_steps: int) -> list:
    """
    Return indices where a rollout of max_steps can run without crossing a
    done=True boundary. Falls back to shorter sequences if none found.
    """
    N = len(done_arr)
    # Strict: no done anywhere in the window
    strict = [i for i in range(N - max_steps)
              if not np.any(done_arr[i: i + max_steps])]
    if strict:
        return strict
    # Relaxed: just don't start on a done transition
    return [i for i in range(N - max_steps) if not done_arr[i]]


def evaluate_rollout(
    model,
    obs_arr: np.ndarray,
    action_arr: np.ndarray,
    done_arr: np.ndarray,
    max_steps: int = 20,
    n_seqs: int = 200,
    seed: int = 42,
) -> dict:
    """
    Returns: {step: (mean_mse, std_mse)} for step in 1..max_steps
    """
    rng = np.random.default_rng(seed)
    N   = len(obs_arr)

    valid_starts = find_valid_starts(done_arr, max_steps)
    if not valid_starts:
        raise RuntimeError("No valid starting positions found in dataset.")

    n_seqs  = min(n_seqs, len(valid_starts))
    starts  = rng.choice(valid_starts, size=n_seqs, replace=False)

    # Large budget — evaluation only, not planning; wall_time set high
    budget  = PlanningBudget(BudgetConfig(
        max_model_calls=n_seqs * max_steps + 10,
        max_wall_time_s=3600.0,
    ))
    mse_by_step = {k: [] for k in range(1, max_steps + 1)}

    for idx, start in enumerate(starts):
        z = model.encode(obs_arr[start])   # initial encoding

        for k in range(1, max_steps + 1):
            step_idx = start + k - 1
            if step_idx >= N:
                break

            # One imagined step using recorded action
            z, _, _ = model.step(z, int(action_arr[step_idx]), budget)

            # Actual encoded observation at this timestep
            if start + k >= N:
                break
            z_real = model.encode(obs_arr[start + k])

            mse = float(np.mean((z - z_real) ** 2))
            mse_by_step[k].append(mse)

        if (idx + 1) % 50 == 0:
            print(f"  sequences: {idx + 1}/{n_seqs}")

    return {
        k: (float(np.mean(v)), float(np.std(v)))
        for k, v in mse_by_step.items()
        if v
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Multi-step rollout quality evaluation")
    p.add_argument("--model",     required=True, help="World model name (dino_wm, lewm, cnn_wm)")
    p.add_argument("--ckpt",      required=True, help="Checkpoint path (.pt)")
    p.add_argument("--data",      required=True, help="Dataset path (.npz)")
    p.add_argument("--env",       required=True, help="Environment name")
    p.add_argument("--max-steps", type=int, default=20,  help="Max rollout length")
    p.add_argument("--n-seqs",    type=int, default=200, help="Number of sequences")
    p.add_argument("--device",    default="cpu")
    p.add_argument("--output-dir", default="results/rollout_quality")
    args = p.parse_args()

    print(f"\n{'='*60}")
    print(f"Multi-step Rollout Quality")
    print(f"Model   : {args.model}")
    print(f"Env     : {args.env}")
    print(f"Ckpt    : {args.ckpt}")
    print(f"{'='*60}")

    env   = make_env(args.env)
    model = make_world_model(args.model, env, checkpoint=args.ckpt, device=args.device)

    data       = np.load(args.data)
    obs_arr    = data["obs"]       # (N, H, W, 3) uint8
    action_arr = data["action"]    # (N,)
    done_arr   = data["done"]      # (N,) bool

    # Held-out evaluation set: last 20%
    N         = len(obs_arr)
    start_idx = int(0.8 * N)
    obs_arr    = obs_arr[start_idx:]
    action_arr = action_arr[start_idx:]
    done_arr   = done_arr[start_idx:]

    print(f"\nEval set  : {len(obs_arr)} transitions (last 20%)")
    print(f"Sequences : {args.n_seqs}  ·  Max rollout steps : {args.max_steps}")
    print(f"\nRunning rollout evaluation ...")

    results = evaluate_rollout(
        model, obs_arr, action_arr, done_arr,
        max_steps=args.max_steps,
        n_seqs=args.n_seqs,
    )

    # ── Print table ───────────────────────────────────────────────────────────
    print(f"\n{'─'*52}")
    print(f"  ROLLOUT MSE  ·  {args.model}  ·  {args.env}")
    print(f"{'─'*52}")
    print(f"  {'Step':>4}  {'MSE (mean)':>12}  {'Std':>10}  {'vs step-1':>10}")
    print(f"{'─'*52}")

    step1_mean = results[1][0] if 1 in results else 1.0
    for k in sorted(results.keys()):
        mean, std = results[k]
        ratio = mean / step1_mean if step1_mean > 0 else float("nan")
        print(f"  {k:>4}  {mean:>12.4f}  {std:>10.4f}  {ratio:>9.2f}x")
    print(f"{'─'*52}\n")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model}_{args.env}.csv"

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "mse_mean", "mse_std"])
        for k in sorted(results.keys()):
            mean, std = results[k]
            writer.writerow([k, round(mean, 6), round(std, 6)])

    print(f"Saved -> {out_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    steps = sorted(results.keys())
    means = [results[k][0] for k in steps]
    stds  = [results[k][1] for k in steps]
    means = np.array(means)
    stds  = np.array(stds)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, means, color="#2855D8", linewidth=2, marker="o",
            markersize=4, label=args.model)
    ax.fill_between(steps, means - 0.5 * stds, means + 0.5 * stds,
                    color="#2855D8", alpha=0.15)
    ax.set_xlabel("Rollout step", fontsize=11)
    ax.set_ylabel("MSE (imagined vs real)", fontsize=11)
    ax.set_title(f"Multi-step Rollout Error — {args.model} · {args.env}",
                 fontsize=12)
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(fontsize=10)
    fig.tight_layout()

    plot_path = out_dir / f"{args.model}_{args.env}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Plot   -> {plot_path}")


if __name__ == "__main__":
    main()
