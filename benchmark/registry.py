"""
Central registry of planners and environments.
Each entry maps a name -> (class, default_kwargs).
"""

from benchmark.planners.classical.bfs import BFSPlanner
from benchmark.planners.classical.ucs import UCSPlanner
from benchmark.planners.classical.astar import AStarPlanner
from benchmark.planners.classical.greedy_bfs import GreedyBFSPlanner
from benchmark.planners.sampling.random_shooting import RandomShootingPlanner
from benchmark.planners.sampling.beam_search import BeamSearchPlanner, ValueBeamSearchPlanner
from benchmark.planners.sampling.cem import CEMPlanner
from benchmark.planners.sampling.lewm_cem import LeWMCEMPlanner
from benchmark.planners.sampling.mcts import MCTSPlanner, MCTSValuePlanner
from benchmark.planners.model_based.latent_rollout import LatentRolloutPlanner
from benchmark.planners.model_based.value_guided_astar import ValueGuidedAStarPlanner
from benchmark.planners.model_based.muzero_style import MuZeroStylePlanner
from benchmark.planners.dp.value_iteration import ValueIterationPlanner
from benchmark.planners.dp.policy_iteration import PolicyIterationPlanner
from benchmark.envs.maze import MazeEnv
from benchmark.envs.minigrid_wrap import MiniGridWrapper

# ── Planners ──────────────────────────────────────────────────────────────────
# Track 1: dynamics only
# Track 2: dynamics + value (and optionally policy)
# Oracle:  requires state enumeration — PerfectModel / TabularModel only

PLANNER_REGISTRY = {
    # Track 1
    "bfs":              (BFSPlanner,            {}),
    "ucs":              (UCSPlanner,            {}),
    "random_shooting":  (RandomShootingPlanner, {"n_samples": 64,  "horizon": 15}),
    "beam_search":        (BeamSearchPlanner,      {"beam_width": 10, "horizon": 20}),
    "value_beam_search":  (ValueBeamSearchPlanner, {"beam_width": 10, "horizon": 20, "gamma": 0.99}),
    "mcts":             (MCTSPlanner,           {"n_simulations": 200, "c_puct": 1.41, "rollout_depth": 10}),
    # Track 2
    "astar":            (AStarPlanner,          {}),
    "greedy_bfs":       (GreedyBFSPlanner,      {}),
    "cem":              (CEMPlanner,            {"n_samples": 64, "elite_frac": 0.1, "n_iters": 5, "horizon": 15}),
    "lewm_cem":         (LeWMCEMPlanner,        {"n_samples": 100, "elite_frac": 0.1, "n_iters": 3, "horizon": 5}),
    "mcts_value":       (MCTSValuePlanner,      {"n_simulations": 200, "c_puct": 1.41}),
    "latent_rollout":   (LatentRolloutPlanner,  {"horizon": 20, "gamma": 0.99}),
    "value_guided_astar": (ValueGuidedAStarPlanner, {}),
    "muzero_style":     (MuZeroStylePlanner,    {"n_simulations": 200, "c_puct": 1.25}),
    # Oracle (offline — fit() must be called before plan())
    "value_iteration":  (ValueIterationPlanner, {"gamma": 0.99, "eps": 1e-4}),
    "policy_iteration": (PolicyIterationPlanner,{"gamma": 0.99, "eval_iters": 100}),
}

ENV_REGISTRY = {
    # Maze (no extra deps, works with PerfectModel / TabularModel)
    "maze_5x5":              (MazeEnv, {"size": 5,  "n_obstacles": 3,  "max_steps": 50}),
    "maze_10x10":            (MazeEnv, {"size": 10, "n_obstacles": 15, "max_steps": 150}),
    "maze_20x20":            (MazeEnv, {"size": 20, "n_obstacles": 60, "max_steps": 400}),
    "maze_10x10_stochastic": (MazeEnv, {"size": 10, "n_obstacles": 15, "slip_prob": 0.1, "max_steps": 150}),
    # MiniGrid — requires `pip install minigrid gymnasium`; use with DINOWorldModel / LeWM
    "minigrid_empty_5x5":             (MiniGridWrapper, {"env_id": "MiniGrid-Empty-5x5-v0"}),
    "minigrid_empty_8x8":             (MiniGridWrapper, {"env_id": "MiniGrid-Empty-8x8-v0"}),
    "minigrid_fourrooms":             (MiniGridWrapper, {"env_id": "MiniGrid-FourRooms-v0"}),
    "minigrid_doorkey_8x8":           (MiniGridWrapper, {"env_id": "MiniGrid-DoorKey-8x8-v0"}),
    "minigrid_multiroom_n4":          (MiniGridWrapper, {"env_id": "MiniGrid-MultiRoom-N4-S5-v0"}),
    # Stochastic: same env, slip_prob=0.1; world model trained on deterministic data → mismatch
    "minigrid_empty_8x8_stochastic":  (MiniGridWrapper, {"env_id": "MiniGrid-Empty-8x8-v0", "slip_prob": 0.1}),
}

TRACK1_PLANNERS = ["bfs", "ucs", "random_shooting", "beam_search", "mcts"]
TRACK2_PLANNERS = ["astar", "greedy_bfs", "cem", "mcts_value", "latent_rollout", "value_guided_astar", "muzero_style", "value_beam_search"]
ORACLE_PLANNERS = ["value_iteration", "policy_iteration"]
# LeWM-specific planner: paper-protocol CEM with terminal cost
LEWM_PLANNERS   = ["lewm_cem"]


def make_planner(name: str, **overrides):
    cls, defaults = PLANNER_REGISTRY[name]
    return cls(**{**defaults, **overrides})


def make_env(name: str, **overrides):
    cls, defaults = ENV_REGISTRY[name]
    return cls(**{**defaults, **overrides})


def make_world_model(name: str, env, checkpoint: str = None, device: str = "cpu"):
    """
    name : "perfect" | "tabular" | "dino_wm" | "lewm"
    env  : BenchmarkEnv instance (needed by perfect/tabular; n_actions by neural)
    checkpoint : path to .pt file (required for dino_wm / lewm)
    """
    if name == "perfect":
        from benchmark.world_models.perfect_model import PerfectModel
        return PerfectModel(env)
    if name == "tabular":
        from benchmark.world_models.tabular_model import TabularModel
        model = TabularModel(env.n_actions)
        model.learn_from_env(env, n_episodes=300)
        return model
    if name == "dino_wm":
        from benchmark.world_models.dino_wm import DINOWorldModel
        return DINOWorldModel(
            n_actions=env.n_actions,
            checkpoint_path=checkpoint,
            device=device,
        )
    if name == "lewm":
        from benchmark.world_models.lewm import LeWM
        return LeWM(
            n_actions=env.n_actions,
            checkpoint_path=checkpoint,
            device=device,
        )
    if name == "cnn_wm":
        from benchmark.world_models.cnn_wm import CNNWorldModel
        return CNNWorldModel(
            n_actions=env.n_actions,
            checkpoint_path=checkpoint,
            device=device,
        )
    if name in ("dino_wm_ft", "dino_wm_ft2"):
        from benchmark.world_models.dino_wm import DINOWorldModel
        return DINOWorldModel(
            n_actions=env.n_actions,
            checkpoint_path=checkpoint,
            device=device,
        )
    if name == "lewm_geodone":
        # Goal-latent heuristic + calibrated geometric done detection
        from benchmark.world_models.lewm import LeWM
        return LeWM(
            n_actions=env.n_actions,
            checkpoint_path=checkpoint,
            device=device,
        )
    if name == "lewm_goal_heuristic":
        # Option 1: goal-latent heuristic baked into checkpoint (no retraining)
        from benchmark.world_models.lewm import LeWM
        return LeWM(
            n_actions=env.n_actions,
            checkpoint_path=checkpoint,
            device=device,
        )
    if name == "lewm_valueft":
        # Option 2: frozen-JEPA head fine-tune with w_val=10
        from benchmark.world_models.lewm import LeWM
        return LeWM(
            n_actions=env.n_actions,
            checkpoint_path=checkpoint,
            device=device,
        )
    if name == "lewm_her":
        # Option 3: HER-augmented value targets
        from benchmark.world_models.lewm import LeWM
        return LeWM(
            n_actions=env.n_actions,
            checkpoint_path=checkpoint,
            device=device,
        )
    if name == "lewm_goal_cond":
        from benchmark.world_models.lewm import GoalCondLeWM
        return GoalCondLeWM(
            n_actions=env.n_actions,
            checkpoint_path=checkpoint,
            device=device,
        )
    # cnn_wm_noisy_<sigma> — wraps cnn_wm_fixed with additive Gaussian noise on z_next
    if name.startswith("cnn_wm_noisy_"):
        sigma = float(name[len("cnn_wm_noisy_"):])
        from benchmark.world_models.cnn_wm import CNNWorldModel
        from benchmark.world_models.noisy_wrapper import NoisyWorldModel
        base = CNNWorldModel(n_actions=env.n_actions, checkpoint_path=checkpoint, device=device)
        return NoisyWorldModel(base, sigma=sigma)
    if name == "base_dino_wm":
        from benchmark.world_models.base_dino_wm import BaseDINOModel
        return BaseDINOModel(
            n_actions=env.n_actions,
            checkpoint_path=checkpoint,
            device=device,
        )
    if name == "base_lewm":
        from benchmark.world_models.base_lewm import BaseLeWM
        return BaseLeWM(
            n_actions=env.n_actions,
            checkpoint_path=checkpoint,
            device=device,
        )
    raise ValueError(f"Unknown world model: {name}")
