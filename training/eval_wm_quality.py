"""
World Model Quality Evaluation
================================
Computes four metrics on held-out transitions to quantify model quality
independently of planning success:

  - Dynamics MSE   : mean ||encode(next_obs) - step_z_next||^2
  - Done F1        : F1 score of binary done prediction vs ground truth
  - Reward F1      : F1 score of (reward_pred > 0) vs (reward_actual > 0)
  - Value Pearson  : Pearson-r between V(encode(obs)) and MC return

Uses the last 20% of the npz as a held-out evaluation set (transitions
the model was trained on but a clean proxy for calibration quality).

Usage:
    python training/eval_wm_quality.py \
        --model   cnn_wm \
        --ckpt    /scratch/.../checkpoints/cnn_wm_fixed_minigrid_empty_8x8.pt \
        --data    /scratch/.../data/minigrid_empty_8x8.npz \
        --env     minigrid_empty_8x8 \
        --device  cuda
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.core.budget import BudgetConfig, PlanningBudget
from benchmark.registry import make_env, make_world_model


def mc_returns(rewards: np.ndarray, dones: np.ndarray, gamma: float = 0.99) -> np.ndarray:
    n = len(rewards)
    mc = np.zeros(n, dtype=np.float32)
    R = 0.0
    for i in range(n - 1, -1, -1):
        R = rewards[i] + (0.0 if dones[i] else gamma * R)
        mc[i] = R
    return mc


def f1(preds: np.ndarray, labels: np.ndarray) -> dict:
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1s  = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1s,
            "tp": tp, "fp": fp, "fn": fn,
            "n_pos_pred": int((preds == 1).sum()),
            "n_pos_true": int((labels == 1).sum())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",   required=True, help="World model name (cnn_wm, dino_wm, lewm, ...)")
    p.add_argument("--ckpt",    required=True, help="Checkpoint path")
    p.add_argument("--data",    required=True, help="Training .npz dataset")
    p.add_argument("--env",     required=True, help="Environment name (minigrid_empty_8x8, ...)")
    p.add_argument("--device",  default="cuda")
    p.add_argument("--n-eval",  type=int, default=3000, help="Transitions to evaluate on")
    p.add_argument("--gamma",   type=float, default=0.99)
    args = p.parse_args()

    print(f"\n{'='*60}")
    print(f"Model  : {args.model}  ({args.env})")
    print(f"Ckpt   : {args.ckpt}")
    print(f"{'='*60}")

    env   = make_env(args.env)
    model = make_world_model(args.model, env, checkpoint=args.ckpt, device=args.device)

    data       = np.load(args.data)
    obs_arr    = data["obs"]        # (N, H, W, 3) uint8
    action_arr = data["action"]     # (N,) int
    reward_arr = data["reward"]     # (N,) float
    next_arr   = data["next_obs"]   # (N, H, W, 3) uint8
    done_arr   = data["done"]       # (N,) bool

    N = len(obs_arr)
    # Use the last 20% as held-out evaluation set
    start    = int(0.8 * N)
    indices  = np.arange(start, N)
    rng      = np.random.default_rng(42)
    if len(indices) > args.n_eval:
        indices = rng.choice(indices, size=args.n_eval, replace=False)
    indices.sort()

    n = len(indices)
    print(f"Evaluating on {n} transitions (indices {indices[0]}–{indices[-1]})\n")

    # Precompute MC returns over the full dataset (needed for value Pearson)
    mc = mc_returns(reward_arr, done_arr, gamma=args.gamma)

    # ── Encode all evaluation observations ──────────────────────────────────
    print("Encoding observations ...")
    z_obs      = np.zeros((n, ), dtype=object)  # will store numpy arrays
    z_next_obs = np.zeros((n, ), dtype=object)
    z_list     = []
    z_next_list = []
    for j, i in enumerate(indices):
        z_list.append(model.encode(obs_arr[i]))
        z_next_list.append(model.encode(next_arr[i]))
        if (j + 1) % 500 == 0:
            print(f"  {j+1}/{n}")

    z_arr      = np.stack(z_list)       # (n, D)
    z_next_arr = np.stack(z_next_list)  # (n, D)

    # Report latent norm for noise calibration
    norms = np.linalg.norm(z_arr, axis=1)
    print(f"\nLatent norm: mean={norms.mean():.3f}  std={norms.std():.3f}  "
          f"min={norms.min():.3f}  max={norms.max():.3f}")

    # ── Dynamics MSE ─────────────────────────────────────────────────────────
    print("\nComputing dynamics MSE ...")
    dummy = PlanningBudget(BudgetConfig(max_model_calls=n + 1))
    z_pred_list = []
    for j, i in enumerate(indices):
        z_pred, _, _ = model.step(z_list[j], int(action_arr[i]), dummy)
        z_pred_list.append(z_pred)
    z_pred_arr = np.stack(z_pred_list)  # (n, D)

    sq_err  = ((z_pred_arr - z_next_arr) ** 2).sum(axis=1)  # (n,)
    dyn_mse = float(sq_err.mean())
    dyn_std = float(sq_err.std())
    print(f"  Dynamics MSE : {dyn_mse:.4f}  ± {dyn_std:.4f}")

    # ── Done F1 ──────────────────────────────────────────────────────────────
    print("\nComputing done F1 ...")
    # Re-run step to get done predictions (uses model's internal threshold)
    dummy2 = PlanningBudget(BudgetConfig(max_model_calls=n + 1))
    done_pred = np.zeros(n, dtype=np.int32)
    for j, i in enumerate(indices):
        _, _, d = model.step(z_list[j], int(action_arr[i]), dummy2)
        done_pred[j] = int(d)
    done_true = done_arr[indices].astype(np.int32)

    done_metrics = f1(done_pred, done_true)
    print(f"  Done precision : {done_metrics['precision']:.3f}")
    print(f"  Done recall    : {done_metrics['recall']:.3f}")
    print(f"  Done F1        : {done_metrics['f1']:.3f}")
    print(f"  Positives pred/true : {done_metrics['n_pos_pred']} / {done_metrics['n_pos_true']}")

    # ── Reward F1 ─────────────────────────────────────────────────────────────
    print("\nComputing reward F1 ...")
    dummy3 = PlanningBudget(BudgetConfig(max_model_calls=n + 1))
    rew_pred = np.zeros(n, dtype=np.int32)
    for j, i in enumerate(indices):
        _, r, _ = model.step(z_list[j], int(action_arr[i]), dummy3)
        rew_pred[j] = int(r > 0.5)
    rew_true = (reward_arr[indices] > 0).astype(np.int32)

    rew_metrics = f1(rew_pred, rew_true)
    print(f"  Reward precision : {rew_metrics['precision']:.3f}")
    print(f"  Reward recall    : {rew_metrics['recall']:.3f}")
    print(f"  Reward F1        : {rew_metrics['f1']:.3f}")
    print(f"  Positives pred/true : {rew_metrics['n_pos_pred']} / {rew_metrics['n_pos_true']}")

    # ── Value Pearson ──────────────────────────────────────────────────────────
    print("\nComputing value Pearson correlation ...")
    v_pred = np.array([model.value(z_list[j]) for j in range(n)], dtype=np.float32)
    v_true = mc[indices].astype(np.float32)

    # Only compute correlation on non-trivial transitions (mc > 0 exists?)
    if v_true.std() < 1e-6:
        print("  WARNING: MC returns have zero variance — correlation undefined.")
        val_r = float("nan")
    else:
        val_r, val_p = pearsonr(v_pred, v_true)
        print(f"  Value Pearson r : {val_r:.4f}  (p={val_p:.3e})")
        print(f"  V(z) range      : [{v_pred.min():.3f}, {v_pred.max():.3f}]")
        print(f"  MC return range : [{v_true.min():.3f}, {v_true.max():.3f}]")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"SUMMARY  {args.model}  ·  {args.env}")
    print(f"{'─'*60}")
    print(f"  Dynamics MSE  : {dyn_mse:.4f}")
    print(f"  Done F1       : {done_metrics['f1']:.3f}")
    print(f"  Reward F1     : {rew_metrics['f1']:.3f}")
    print(f"  Value Pearson : {val_r:.4f}" if not (isinstance(val_r, float) and val_r != val_r) else "  Value Pearson : nan")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
