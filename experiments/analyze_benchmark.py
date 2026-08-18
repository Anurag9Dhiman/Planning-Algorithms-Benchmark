"""
Benchmark analysis: environment taxonomy, FLOP-normalized SR, per-group guidelines.

Usage:
    python experiments/analyze_benchmark.py
"""

import json
import glob
import os
from collections import defaultdict

# ── FLOP table (GFLOPs per call) ──────────────────────────────────────────────
# encode(): called once per episode (initial obs)
# step():   called mean_model_calls times per episode
FLOPS = {
    "dino_wm": {
        "encode_gflops": 21.7,    # DINOv2 ViT-S/8 at 224x224 (standard)
        "step_gflops":   0.0002,  # small dynamics MLP
    },
    "lewm_goalft": {
        "encode_gflops": 1.28,    # ViT-tiny at 224x224 (standard)
        "step_gflops":   0.005,   # ARPredictor (small causal transformer)
    },
}

# ── Environment taxonomy ───────────────────────────────────────────────────────
ENV_GROUPS = {
    # Simple Navigation
    "minigrid_empty_8x8":              "Simple Navigation",
    "minigrid_empty_5x5":              "Simple Navigation",
    "maze_5x5":                        "Simple Navigation",
    # Complex Navigation (long horizon, multi-room)
    "minigrid_fourrooms":              "Complex Navigation",
    "maze_10x10":                      "Complex Navigation",
    "maze_20x20":                      "Complex Navigation",
    # Stochastic (slip probability — world model assumes deterministic)
    "maze_10x10_stochastic":           "Stochastic",
    "minigrid_empty_8x8_stochastic":   "Stochastic",
    # Compositional (multi-step subgoals: pick up key → toggle door → reach goal)
    "minigrid_doorkey_8x8":            "Compositional",
    # Visual Complexity (multi-room layout, partial observability, longer horizon)
    "minigrid_multiroom_n4":           "Visual Complexity",
}

ENV_PROPS = {
    "minigrid_empty_8x8":            {"horizon": "short",  "rooms": 1,  "stochastic": False, "compositional": False},
    "minigrid_fourrooms":            {"horizon": "long",   "rooms": 4,  "stochastic": False, "compositional": False},
    "maze_10x10_stochastic":         {"horizon": "medium", "rooms": 1,  "stochastic": True,  "compositional": False},
    "minigrid_empty_8x8_stochastic": {"horizon": "short",  "rooms": 1,  "stochastic": True,  "compositional": False},
    "minigrid_doorkey_8x8":          {"horizon": "medium", "rooms": 2,  "stochastic": False, "compositional": True},
    "minigrid_multiroom_n4":         {"horizon": "long",   "rooms": 4,  "stochastic": False, "compositional": False},
}

# ── Result sources (model -> list of result dirs to search) ───────────────────
RESULTS_BASE = os.environ.get("RESULTS_BASE", "results")
SOURCES = {
    "dino_wm":    [f"{RESULTS_BASE}/base_benchmark",
                   f"{RESULTS_BASE}/doorkey_bench",
                   f"{RESULTS_BASE}/multiroom_bench",
                   f"{RESULTS_BASE}/stochastic_bench"],
    "lewm_goalft":[f"{RESULTS_BASE}/lewm_goalft_bench",
                   f"{RESULTS_BASE}/stochastic_bench",
                   f"{RESULTS_BASE}/doorkey_goalft_bench",   # job 36d
                   f"{RESULTS_BASE}/multiroom_goalft_bench"], # job 36d
    # NOTE: base_lewm excluded — goal metric (L2 distance) is uninformative
    # without goal-conditioned fine-tuning; results are near-0% on all envs.
    # DoorKey and MultiRoom goalft added via jobs 36a-36d (fixed BFS).
}

# world_model field stored in results -> our model key
WM_NAME_MAP = {
    "base_dino_wm": "dino_wm",
    "dino_wm":      "dino_wm",
}

MODEL_LABELS = {
    "dino_wm":     "DINO-WM",
    "lewm_goalft": "LeWM (goal-ft)",
}


# ── Load results ──────────────────────────────────────────────────────────────

def load_results():
    """Returns dict: (model, env, planner) -> AggregateResult dict."""
    records = {}
    for model_key, dirs in SOURCES.items():
        for d in dirs:
            for summary_path in glob.glob(f"{d}/*/*/summary.json"):
                try:
                    data = json.load(open(summary_path))
                except Exception:
                    continue
                for r in data:
                    stored_wm = r.get("world_model", "")
                    if "goalft" in summary_path:
                        # any result whose full path contains "goalft" is a goalft run
                        # (covers stochastic_bench/base_lewm_goalft_* subdirs too)
                        resolved = "lewm_goalft"
                    elif stored_wm in WM_NAME_MAP:
                        resolved = WM_NAME_MAP[stored_wm]
                    else:
                        # skip results whose world_model doesn't belong to this source
                        # (e.g. base_lewm entries sitting in doorkey_bench/multiroom_bench)
                        continue
                    key = (resolved, r["env_name"], r["planner"])
                    if key not in records or r["success_rate"] > records[key]["success_rate"]:
                        records[key] = r
    return records


def flop_normalized_sr(record, model):
    """Successes per TFLOP (1 TFLOP = 1000 GFLOPs)."""
    f = FLOPS.get(model)
    if f is None:
        return None
    ep_gflops = f["encode_gflops"] + record["mean_model_calls"] * f["step_gflops"]
    ep_tflops = ep_gflops / 1000.0
    if ep_tflops == 0:
        return None
    return record["success_rate"] / ep_tflops


# ── Print analysis ────────────────────────────────────────────────────────────

def main():
    records = load_results()

    # Group by (env_group, env, model)
    by_group = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))  # group -> env -> model -> {planner -> record}
    for (model, env, planner), r in records.items():
        group = ENV_GROUPS.get(env, "Other")
        by_group[group][env][model][planner] = r

    # ── Per-group summary ─────────────────────────────────────────────────────
    print("=" * 70)
    print("BENCHMARK ANALYSIS — Environment Groups")
    print("=" * 70)

    all_flop_rows = []

    for group in ["Simple Navigation", "Complex Navigation", "Stochastic", "Compositional", "Visual Complexity", "Other"]:
        if group not in by_group:
            continue
        print(f"\n{'─'*70}")
        print(f"GROUP: {group}")
        print(f"{'─'*70}")

        for env, model_data in by_group[group].items():
            props = ENV_PROPS.get(env, {})
            prop_str = "  |  ".join([
                f"horizon={props.get('horizon','?')}",
                f"rooms={props.get('rooms','?')}",
                f"stochastic={props.get('stochastic','?')}",
                f"compositional={props.get('compositional','?')}",
            ])
            print(f"\n  Env: {env}")
            print(f"  Properties: {prop_str}")
            print()
            print(f"  {'Model':<22} {'Planner':<25} {'SR':>5}  {'Calls':>8}  {'FLOP-SR':>10}")
            print(f"  {'─'*75}")

            # collect best per model
            best_per_model = {}
            for model, planners in model_data.items():
                for planner, r in planners.items():
                    fsr = flop_normalized_sr(r, model)
                    row = {
                        "model": model, "env": env, "planner": planner,
                        "sr": r["success_rate"], "calls": r["mean_model_calls"],
                        "fsr": fsr,
                    }
                    all_flop_rows.append(row)
                    if r["success_rate"] > 0:
                        print(f"  {MODEL_LABELS[model]:<22} {planner:<25} "
                              f"{r['success_rate']*100:>4.0f}%  "
                              f"{r['mean_model_calls']:>8.0f}  "
                              f"{fsr*1000:>9.3f}‰" if fsr else
                              f"  {MODEL_LABELS[model]:<22} {planner:<25} "
                              f"{r['success_rate']*100:>4.0f}%  "
                              f"{r['mean_model_calls']:>8.0f}  {'N/A':>10}")
                    if model not in best_per_model or r["success_rate"] > best_per_model[model]["sr"]:
                        best_per_model[model] = {"planner": planner, "sr": r["success_rate"], "fsr": fsr}

            # Summary row per model
            print()
            print(f"  Best per model:")
            for model, b in best_per_model.items():
                fsr_str = f"{b['fsr']*1000:.3f}‰" if b["fsr"] else "N/A"
                print(f"    {MODEL_LABELS[model]:<22}  best={b['planner']:<25}  "
                      f"SR={b['sr']*100:.0f}%  FLOP-SR={fsr_str}")

    # ── FLOP-normalized SR table ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("FLOP-NORMALIZED SUCCESS RATE (successes per TFLOP)")
    print(f"  encode FLOPs: DINO-WM={FLOPS['dino_wm']['encode_gflops']} GFLOPs  "
          f"LeWM={FLOPS['lewm_goalft']['encode_gflops']} GFLOPs")
    print(f"  step FLOPs:   DINO-WM={FLOPS['dino_wm']['step_gflops']} GFLOPs  "
          f"LeWM={FLOPS['lewm_goalft']['step_gflops']} GFLOPs")
    print(f"{'='*70}")

    # Top 10 by FLOP-SR across all conditions
    positive = [r for r in all_flop_rows if r["sr"] > 0 and r["fsr"] is not None]
    positive.sort(key=lambda x: x["fsr"], reverse=True)
    print(f"\n  Top results by FLOP efficiency:")
    print(f"  {'Model':<22} {'Env':<22} {'Planner':<25} {'SR':>5}  {'Calls':>8}  {'FLOP-SR':>10}")
    print(f"  {'─'*90}")
    for r in positive[:15]:
        fsr_str = f"{r['fsr']*1000:.4f}‰"
        print(f"  {MODEL_LABELS[r['model']]:<22} {r['env']:<22} {r['planner']:<25} "
              f"{r['sr']*100:>4.0f}%  {r['calls']:>8.0f}  {fsr_str:>10}")

    # ── Per-group guidelines ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("GUIDELINES PER ENVIRONMENT GROUP")
    print(f"{'='*70}")

    guidelines = {
        "Simple Navigation": {
            "description": "Small grid, single room, short horizon (~5-15 steps), deterministic.",
            "findings": [
                "DINO-WM dominates: pretrained DINOv2 features give smooth latent gradients toward goal.",
                "Best planner: mcts_value (55%) — value head + MCTS tree search most effective.",
                "LeWM goal-ft competitive: bfs (35%) — goal-calibrated latent space enables L2 planning.",
            ],
            "recommend": {
                "DINO-WM":        "mcts_value or value_guided_astar",
                "LeWM (goal-ft)": "bfs or astar",
            },
            "avoid": "random_shooting, beam_search — too random for structured envs.",
        },
        "Complex Navigation": {
            "description": "Multi-room, long horizon (~50-100 steps), deterministic. Planning horizon bottleneck.",
            "findings": [
                "Both models struggle — max SR is 10% (DINO-WM value_guided_astar).",
                "Bottleneck is predictor rollout accuracy, not encoder metric quality.",
                "Prediction error compounds over long horizons — imagined states diverge from reality.",
                "LeWM goal-ft mcts reaches 10% — tree search more robust to rollout drift.",
            ],
            "recommend": {
                "DINO-WM":        "value_guided_astar (10%) — only planner with consistent signal",
                "LeWM (goal-ft)": "mcts (10%) — slight improvement from better metric",
            },
            "avoid": "BFS/UCS — exhausts budget exploring dead-end latent paths at long horizon.",
        },
        "Stochastic": {
            "description": "Slip probability on transitions — world model trained on deterministic data.",
            "findings": [
                "Both models reach 35% SR with UCS — receding-horizon re-planning compensates for slippage.",
                "DINO-WM CEM also 35% — sampling-based planners robust to stochastic transitions.",
                "Deterministic planners (beam_search, latent_rollout) collapse to 0% — commit to a fixed path.",
                "LeWM goalft FLOP-SR much lower than DINO-WM — ViT-tiny encoder is cheaper but less accurate.",
            ],
            "recommend": {
                "DINO-WM":        "cem or mcts (sampling-based, re-plans every step)",
                "LeWM (goal-ft)": "lewm_cem — terminal cost robust to mid-path stochasticity",
            },
            "avoid": "BFS, UCS, A* — deterministic path planning breaks under stochastic transitions.",
        },
        "Compositional": {
            "description": "Multi-step subgoals (pick up key → toggle door → reach goal). DoorKey-8x8.",
            "findings": [
                "Both models nearly 0% — flat search cannot decompose the key→door→goal subgoal chain.",
                "DINO-WM: 0% on all planners; best accidental SR via lewm_cem (5% on 1 seed only).",
                "LeWM goal-ft: ucs 5%, mcts 5% — goalft calibration helps marginally but budget exhausted.",
                "Requires hierarchical planning or option-based methods beyond current planner suite.",
            ],
            "recommend": {
                "DINO-WM":        "mcts_value — deepest search, best chance of accidental subgoal discovery",
                "LeWM (goal-ft)": "lewm_cem — goal-calibrated terminal cost may reward partial progress",
            },
            "avoid": "BFS, UCS — exhausts budget on flat search in exponentially large state space.",
        },
        "Visual Complexity": {
            "description": "Multiple rooms, varied layouts, longer horizon. MultiRoom-N4-S5.",
            "findings": [
                "DINO-WM: pretrained DINOv2 features handle diverse visual textures across rooms.",
                "LeWM goal-ft: pending (job 36d) — fixed BFS (door states + no-render) enables collection.",
                "Horizon bottleneck compounds with visual diversity — both models expected to struggle.",
            ],
            "recommend": {
                "DINO-WM":        "value_guided_astar — consistent signal despite long horizon",
                "LeWM (goal-ft)": "mcts — tree search more robust to long-horizon rollout drift",
            },
            "avoid": "BFS, UCS — exponential state explosion across many rooms.",
        },
    }

    for group, g in guidelines.items():
        print(f"\n  [{group}]")
        print(f"  {g['description']}")
        print(f"\n  Key findings:")
        for f in g["findings"]:
            print(f"    • {f}")
        print(f"\n  Recommended planner:")
        for model, rec in g["recommend"].items():
            print(f"    {model:<22}: {rec}")
        print(f"\n  Avoid: {g['avoid']}")


if __name__ == "__main__":
    main()
