"""
Rollout Quality Comparison Plot
=================================
Reads CSV outputs from eval_rollout_quality.py and produces a
side-by-side comparison figure: one subplot per environment,
one line per world model.

Y-axis uses log scale so DINO-WM and LeWM (very different magnitudes)
are both visible on the same plot.

Usage:
    python training/plot_rollout_comparison.py \
        --input-dir  results/rollout_quality \
        --output-dir results/rollout_quality
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── Style ─────────────────────────────────────────────────────────────────────

MODEL_STYLE = {
    "base_dino_wm": {"color": "#2855D8", "label": "DINO-WM (base)", "marker": "o"},
    "base_lewm":    {"color": "#D84228", "label": "LeWM (base)",    "marker": "s"},
    "dino_wm":      {"color": "#2855D8", "label": "DINO-WM",        "marker": "o"},
    "lewm":         {"color": "#D84228", "label": "LeWM",           "marker": "s"},
    "cnn_wm":       {"color": "#28A838", "label": "CNN-WM",         "marker": "^"},
}

ENV_TITLE = {
    "minigrid_empty_8x8":  "Empty 8×8",
    "minigrid_fourrooms":  "FourRooms",
}


# ── Load CSVs ─────────────────────────────────────────────────────────────────

def load_csv(path: Path):
    """Returns {step: (mean, std)}"""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {int(r["step"]): (float(r["mse_mean"]), float(r["mse_std"]))
            for r in rows}


def find_csvs(input_dir: Path):
    """Returns {(model, env): {step: (mean, std)}}"""
    data = {}
    for f in sorted(input_dir.glob("*.csv")):
        parts = f.stem.split("_", 1)
        # Filename format: {model}_{env}.csv
        # model can be: dino_wm, lewm, cnn_wm
        # env can be: minigrid_empty_8x8, minigrid_fourrooms
        for model_key in MODEL_STYLE:
            if f.stem.startswith(model_key):
                env_key = f.stem[len(model_key) + 1:]
                data[(model_key, env_key)] = load_csv(f)
                break
    return data


# ── Plot ──────────────────────────────────────────────────────────────────────

def make_comparison_plot(data: dict, output_dir: Path):
    envs   = sorted({env for (_, env) in data.keys()})
    models = sorted({m   for (m, _)  in data.keys()})

    n_envs = len(envs)
    fig, axes = plt.subplots(1, n_envs, figsize=(6 * n_envs, 4.5), sharey=False)
    if n_envs == 1:
        axes = [axes]

    for ax, env in zip(axes, envs):
        for model in models:
            if (model, env) not in data:
                continue
            results = data[(model, env)]
            style   = MODEL_STYLE.get(model, {"color": "gray",
                                               "label": model,
                                               "marker": "x"})
            steps = sorted(results.keys())
            means = np.array([results[k][0] for k in steps])
            stds  = np.array([results[k][1] for k in steps])

            ax.plot(steps, means,
                    color=style["color"],
                    linewidth=2,
                    marker=style["marker"],
                    markersize=5,
                    label=style["label"])
            ax.fill_between(steps,
                            np.maximum(means - 0.5 * stds, 1e-6),
                            means + 0.5 * stds,
                            color=style["color"],
                            alpha=0.12)

        ax.set_yscale("log")
        ax.set_xlabel("Rollout step", fontsize=11)
        ax.set_ylabel("MSE (imagined vs real)", fontsize=11)
        ax.set_title(ENV_TITLE.get(env, env), fontsize=13, fontweight="bold")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)
        ax.legend(fontsize=10, framealpha=0.9)
        ax.set_xlim(left=1)

    fig.suptitle("Multi-Step Rollout Error: How Quickly Does Imagination Drift?",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    out_path = output_dir / "rollout_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison plot saved -> {out_path}")
    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir",  default="results/rollout_quality")
    p.add_argument("--output-dir", default="results/rollout_quality")
    args = p.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = find_csvs(input_dir)
    if not data:
        print(f"No CSV files found in {input_dir}")
        return

    print(f"Found results for: {list(data.keys())}")
    make_comparison_plot(data, output_dir)


if __name__ == "__main__":
    main()
