"""
Option 1 — Bake goal latents into an existing LeWM checkpoint.

No retraining. Encodes all successful terminal observations from the dataset,
stores them inside the checkpoint, and saves a new file. At inference, the
value() method uses -min_distance to goal latents as a heuristic instead of
the trained value head.

Usage:
    python training/bake_goal_latents.py \
        --checkpoint /scratch/.../lewm_fixed_minigrid_empty_8x8.pt \
        --data       /scratch/.../data/minigrid_empty_8x8.npz \
        --output     /scratch/.../checkpoints/lewm_goal_heuristic_minigrid_empty_8x8.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.world_models.lewm import LeWM


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Existing LeWM checkpoint")
    p.add_argument("--data",       required=True, help="Training .npz dataset")
    p.add_argument("--output",     required=True, help="Output checkpoint path")
    p.add_argument("--n-actions",  type=int, default=7)
    p.add_argument("--device",     default="cuda")
    args = p.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    model = LeWM(n_actions=args.n_actions, checkpoint_path=args.checkpoint, device=args.device)

    print(f"Encoding goal observations from: {args.data}")
    model.load_goal_latents(args.data)

    print(f"Saving to: {args.output}")
    model.save(args.output)
    print(f"Done — {len(model._goal_latents)} goal latents baked in.")


if __name__ == "__main__":
    main()
