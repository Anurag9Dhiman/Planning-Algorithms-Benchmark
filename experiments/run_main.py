"""
Full benchmark sweep: all planners × all envs × specified world models.
Usage:
    python experiments/run_main.py --config configs/default.yaml
    python experiments/run_main.py --planners track1 track2 --envs maze_10x10 maze_20x20
"""

import argparse
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.core.budget import BudgetConfig
from benchmark.metrics.evaluator import aggregate
from benchmark.metrics.logger import ResultLogger
from benchmark.registry import (
    TRACK1_PLANNERS, TRACK2_PLANNERS,
    make_env, make_planner, make_world_model,
)
from benchmark.runner import BenchmarkRunner


def resolve_planners(spec) -> list:
    if spec == "all":
        return TRACK1_PLANNERS + TRACK2_PLANNERS
    if spec == "track1":
        return TRACK1_PLANNERS
    if spec == "track2":
        return TRACK2_PLANNERS
    if isinstance(spec, list):
        out = []
        for s in spec:
            if s == "all":
                out += TRACK1_PLANNERS + TRACK2_PLANNERS
            elif s == "track1":
                out += TRACK1_PLANNERS
            elif s == "track2":
                out += TRACK2_PLANNERS
            else:
                out.append(s)
        return out
    return [spec]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--planners", nargs="+")
    parser.add_argument("--envs", nargs="+")
    parser.add_argument("--world-models", nargs="+")
    parser.add_argument("--checkpoints", nargs="+",
                        help="Checkpoint paths, one per --world-models entry (for dino_wm / lewm)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--wall-time", type=float,
                        help="Max wall time per planning round (seconds). Overrides config.")
    parser.add_argument("--output")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    planner_names = resolve_planners(args.planners or cfg.get("planners", "track1"))
    env_names = args.envs or cfg["envs"]
    wm_names = args.world_models or cfg.get("world_models", ["perfect"])
    n_episodes = args.episodes or cfg["execution"]["n_episodes"]
    seeds = cfg["execution"].get("seeds", list(range(n_episodes)))
    budget_cfg = BudgetConfig(
        max_model_calls=args.budget or cfg["budget"]["max_model_calls"],
        max_wall_time_s=args.wall_time or cfg["budget"]["max_wall_time_s"],
    )
    results_dir = args.output or cfg["results_dir"]
    run_id = time.strftime("%Y%m%d_%H%M%S")
    logger = ResultLogger(f"{results_dir}/{run_id}")
    runner = BenchmarkRunner(budget_cfg, receding_horizon=cfg["execution"]["receding_horizon"], verbose=True)

    print(f"Planners    : {planner_names}")
    print(f"Envs        : {env_names}")
    print(f"World models: {wm_names}")
    print(f"Episodes    : {n_episodes}  Budget: {budget_cfg.max_model_calls}\n")

    # Map world model name -> checkpoint path (None for perfect/tabular)
    checkpoints = args.checkpoints or ([None] * len(wm_names))
    if len(checkpoints) < len(wm_names):
        checkpoints += [None] * (len(wm_names) - len(checkpoints))
    wm_checkpoint_map = dict(zip(wm_names, checkpoints))

    for env_name in env_names:
        env = make_env(env_name)
        for wm_name in wm_names:
            ckpt = wm_checkpoint_map.get(wm_name)
            world_model = make_world_model(wm_name, env, checkpoint=ckpt, device=args.device)
            print(f"\n── {wm_name.upper()} on {env_name} ──")
            for planner_name in planner_names:
                planner_kw = cfg.get("planner_kwargs", {}).get(planner_name, {})
                planner = make_planner(planner_name, **planner_kw)

                if hasattr(planner, "fit"):
                    print(f"[{planner_name}] fitting oracle policy on {env_name} ...")
                    planner.fit(world_model)

                results = runner.run(planner, env, world_model, n_episodes, seeds)
                agg = aggregate(results)
                for r in results:
                    logger.log_episode(r)
                logger.log_aggregate(agg)

    print(f"\nDone. Results: {logger.run_dir}")


if __name__ == "__main__":
    main()
