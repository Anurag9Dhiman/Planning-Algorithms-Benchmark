"""
Evaluate world model prediction quality for all (model, env) pairs.

Usage:
    python experiments/eval_world_models.py \
        --envs minigrid_empty_8x8 minigrid_fourrooms \
        --world-models dino_wm lewm \
        --checkpoints /path/to/dino.pt /path/to/lewm.pt \
        --episodes 20 --output results/wm_eval
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.metrics.wm_eval import evaluate
from benchmark.registry import make_env, make_world_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", nargs="+", required=True)
    parser.add_argument("--world-models", nargs="+", required=True)
    parser.add_argument("--checkpoints", nargs="+", default=[])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--output", default="results/wm_eval")
    args = parser.parse_args()

    checkpoints = args.checkpoints or ([None] * len(args.world_models))
    if len(checkpoints) < len(args.world_models):
        checkpoints += [None] * (len(args.world_models) - len(checkpoints))
    wm_ckpt = dict(zip(args.world_models, checkpoints))

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    all_results = []

    for env_name in args.envs:
        env = make_env(env_name)
        for wm_name in args.world_models:
            print(f"\n── Evaluating {wm_name.upper()} on {env_name} ──")
            wm = make_world_model(
                wm_name, env, checkpoint=wm_ckpt.get(wm_name), device=args.device
            )
            result = evaluate(
                wm, env,
                n_episodes=args.episodes,
                max_steps_per_ep=args.max_steps,
            )
            print(result.pretty())
            all_results.append(result.to_dict())

    out_path = out_dir / f"wm_eval_{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
