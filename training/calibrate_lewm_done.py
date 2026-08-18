"""
Calibrate the geometric done-detection threshold for a LeWM checkpoint.

Takes a checkpoint that already has goal_latents baked in (from bake_goal_latents.py),
encodes all goal transitions from training data, computes the min-distance to stored
goal latents, and sets done_distance_threshold at the `--recall` percentile so that
that fraction of real goal arrivals are correctly detected.

Usage:
    python training/calibrate_lewm_done.py \
        --checkpoint /scratch/.../checkpoints/lewm_goal_heuristic_minigrid_empty_8x8.pt \
        --data       /scratch/.../data/minigrid_empty_8x8.npz \
        --output     /scratch/.../checkpoints/lewm_geodone_minigrid_empty_8x8.pt \
        --recall     0.95
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.world_models.lewm import LeWM


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="LeWM checkpoint with goal_latents baked in")
    p.add_argument("--data",       required=True, help="Training .npz dataset")
    p.add_argument("--output",     required=True, help="Output checkpoint path")
    p.add_argument("--recall",     type=float, default=0.95,
                   help="Fraction of goal transitions to capture (default 0.95)")
    p.add_argument("--n-actions",  type=int,   default=7)
    p.add_argument("--device",     default="cuda")
    args = p.parse_args()

    print(f"Loading: {args.checkpoint}")
    model = LeWM(n_actions=args.n_actions, checkpoint_path=args.checkpoint, device=args.device)

    assert model._goal_latents is not None, \
        "Checkpoint has no goal_latents. Run bake_goal_latents.py first."

    threshold = model.calibrate_done_distance_threshold(args.data, recall=args.recall)

    # FPR estimate: run predictor on a sample of non-goal transitions
    from benchmark.core.budget import BudgetConfig, PlanningBudget

    data       = np.load(args.data)
    done_mask  = data["done"] & (data["reward"] > 0)
    obs_arr    = data["obs"]
    action_arr = data["action"]
    non_goal_idxs = np.where(~done_mask)[0]

    rng    = np.random.default_rng(42)
    sample = rng.choice(non_goal_idxs, size=min(500, len(non_goal_idxs)), replace=False)
    dummy  = PlanningBudget(BudgetConfig(max_model_calls=len(sample) + 1))

    fp_count = 0
    for i in sample:
        z_curr  = model.encode(obs_arr[i])
        z_next, _, _ = model.step(z_curr, int(action_arr[i]), dummy)
        dist    = float(np.linalg.norm(model._goal_latents - z_next, axis=1).min())
        if dist < threshold:
            fp_count += 1
    print(f"[LeWM] Estimated false positive rate on {len(sample)} non-goal transitions: "
          f"{fp_count/len(sample):.2%}")

    print(f"Saving to: {args.output}")
    model.save(args.output)
    print("Done.")


if __name__ == "__main__":
    main()
