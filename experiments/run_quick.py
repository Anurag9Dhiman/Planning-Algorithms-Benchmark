"""
Smoke test: runs a small subset of planners on maze_5x5 with PerfectModel.
Usage:
    python experiments/run_quick.py
    python experiments/run_quick.py --episodes 3 --budget 200
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.core.budget import BudgetConfig
from benchmark.metrics.evaluator import aggregate
from benchmark.metrics.logger import ResultLogger
from benchmark.registry import make_env, make_planner, TRACK1_PLANNERS, TRACK2_PLANNERS
from benchmark.runner import BenchmarkRunner
from benchmark.world_models.perfect_model import PerfectModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument("--env", default="maze_5x5")
    parser.add_argument("--planners", nargs="+",
                        default=["bfs", "ucs", "random_shooting", "mcts", "astar", "mcts_value"])
    parser.add_argument("--output", default="results/quick_run")
    args = parser.parse_args()

    env = make_env(args.env)
    world_model = PerfectModel(env)
    budget_config = BudgetConfig(max_model_calls=args.budget, max_wall_time_s=60.0)
    runner = BenchmarkRunner(budget_config, receding_horizon=True, verbose=True)
    logger = ResultLogger(args.output)

    print(f"\nEnvironment : {args.env}")
    print(f"World model : {world_model.name}")
    print(f"Budget      : {args.budget} model calls")
    print(f"Episodes    : {args.episodes}\n")

    for planner_name in args.planners:
        planner = make_planner(planner_name)

        # Oracle planners need to be fitted first
        if hasattr(planner, "fit"):
            print(f"[{planner_name}] fitting oracle policy ...")
            planner.fit(world_model)

        results = runner.run(planner, env, world_model, n_episodes=args.episodes)
        agg = aggregate(results)

        for r in results:
            logger.log_episode(r)
        logger.log_aggregate(agg)

        print(
            f"\n  {planner_name:20s}  "
            f"return={agg.mean_return:+.3f}±{agg.std_return:.3f}  "
            f"success={agg.success_rate*100:.0f}%  "
            f"calls={agg.mean_model_calls:.0f}  "
            f"efficiency={agg.budget_efficiency:.4f}"
        )

    print(f"\nResults saved to: {logger.run_dir}")


if __name__ == "__main__":
    main()
