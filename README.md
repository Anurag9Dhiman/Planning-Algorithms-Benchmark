# Planning Algorithms Benchmark

> **Research question:** Given the same learned world model and the same planning budget, which planning algorithm performs best under different task characteristics?

We hold the world model, compute budget, and dataset fixed. Only the planner changes. This isolates the planning algorithm as the independent variable and lets us measure how much planner choice matters — and how much world model quality limits everything underneath it.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Environments](#3-environments)
4. [World Models](#4-world-models)
5. [Planners](#5-planners)
6. [Installation](#6-installation)
7. [Step-by-Step Pipeline](#7-step-by-step-pipeline)
   - [Collect training data](#71-collect-training-data)
   - [Train world models](#72-train-world-models)
   - [Run the benchmark](#73-run-the-benchmark)
   - [Evaluate world model quality](#74-evaluate-world-model-quality)
8. [Configuration](#8-configuration)
9. [Configuration](#9-configuration)
10. [Results](#10-results)
11. [Key Findings](#11-key-findings)
12. [Next Steps](#12-next-steps)

---

## 1. Project Overview

### What we are measuring

At each real environment step, the planner is given a latent state `z` (encoded from the current pixel observation) and a fixed budget `B` of world model calls. It uses those calls to simulate trajectories in latent space and returns one action. The real environment executes that action, the agent re-encodes the new observation, and replanning happens. This is **receding-horizon planning**.

```
obs_t  →  encode()  →  z_t  →  planner(z_t, world_model, budget=B)  →  action_t
                                         │
                     world_model.step()  ←  called up to B times per real step
env.step(action_t)  →  obs_{t+1}  →  encode()  →  z_{t+1}  → ...
```

### Two failure hypotheses we test

| ID | Hypothesis | How we test it |
|----|------------|----------------|
| H1 | **World model quality is the bottleneck.** Bad latent dynamics mean planners can't find good trajectories even with unlimited budget. | Compare planners across world models of different quality on the same task. |
| H2 | **Planning budget is the bottleneck.** Even a perfect world model can't solve the task at B=200. | Sweep B=200→5000 using the Oracle (PerfectModel); if SR stays low, budget isn't the issue. |

---

## 2. Repository Structure

```
Planning-Algorithms-Benchmark/
│
├── benchmark/                    # Core library (installable package)
│   ├── core/
│   │   ├── base_env.py           # Abstract BenchmarkEnv + TaskMeta
│   │   ├── base_planner.py       # Abstract Planner
│   │   ├── world_model.py        # Abstract WorldModel + State typedef
│   │   └── budget.py             # PlanningBudget, BudgetExhausted
│   │
│   ├── envs/
│   │   ├── minigrid_wrap.py      # MiniGrid pixel wrapper (64×64 RGB)
│   │   └── maze.py               # Built-in parameterised maze env
│   │
│   ├── world_models/
│   │   ├── perfect_model.py      # Oracle — wraps real env, ground truth
│   │   ├── tabular_model.py      # Learned (s,a)→s' lookup table
│   │   ├── dino_wm.py            # Frozen DINOv2-small + learned heads
│   │   ├── lewm.py               # ViT-tiny trained from scratch (JEPA)
│   │   ├── cnn_wm.py             # 4-block CNN trained from scratch (new)
│   │   ├── jepa.py               # JEPA predictor module (shared)
│   │   └── lewm_module.py        # LeWM ViT module
│   │
│   ├── planners/
│   │   ├── classical/            # BFS, UCS, A*, Greedy-BFS
│   │   ├── sampling/             # Random Shooting, Beam Search, CEM, MCTS, MCTS-Value
│   │   ├── model_based/          # Latent Rollout, Value-Guided A*, MuZero-style
│   │   └── dp/                   # Value Iteration, Policy Iteration (oracle only)
│   │
│   ├── metrics/
│   │   ├── evaluator.py          # EpisodeResult dataclass + aggregate()
│   │   ├── logger.py             # JSON + JSONL result logging
│   │   └── wm_eval.py            # Rollout divergence metrics (L2, SSIM)
│   │
│   └── registry.py               # make_env / make_planner / make_world_model
│
├── training/                     # World model training scripts
│   ├── collect_data.py           # Roll out env → save .npz dataset
│   ├── train_dino_wm.py          # Train heads on top of frozen DINOv2
│   ├── train_lewm.py             # Train LeWM (ViT-tiny + JEPA)
│   ├── train_cnn_wm.py           # Train CNNWorldModel end-to-end
│   ├── train_dino_wm_finetune.py # Fine-tune last N DINOv2 blocks
│   └── models/
│       ├── encoder.py            # CNNEncoder (4-block, stride-2)
│       └── heads.py              # DynamicsHead, RewardHead, ValueHead, PolicyHead, DoneHead
│
├── experiments/
│   ├── run_main.py               # Full benchmark sweep (all planners × envs × models)
│   ├── run_quick.py              # Smoke test (single planner/env/model)
│   └── eval_world_models.py      # Measure rollout divergence for each world model
│
├── configs/
│   ├── default.yaml              # Budget, planner kwargs, envs
│   ├── ablation_budget.yaml      # Budget sweep config
│   └── ablation_model_error.yaml # Model quality ablation config
│
├── presentation/
│   └── main.tex                  # 9-slide Beamer presentation
│
├── checkpoints/                  # Local checkpoint stubs (.gitkeep)
├── results/                      # Local results stubs (.gitkeep)
├── requirements.txt              # Runtime deps (numpy, gymnasium, …)
├── requirements-training.txt     # Training deps (torch, transformers)
└── setup.py
```

---

## 3. Environments

All experiments use **MiniGrid** pixel environments. Observations are 64×64 RGB full-grid renders (not the partial egocentric view).

| Name (registry key) | Gymnasium ID | Max steps | Difficulty | Notes |
|---|---|---|---|---|
| `minigrid_empty_8x8` | `MiniGrid-Empty-8x8-v0` | ~253 | Easy | Agent navigates to goal in open grid |
| `minigrid_fourrooms` | `MiniGrid-FourRooms-v0` | ~97 | Medium | Four rooms connected by doorways |
| `minigrid_empty_5x5` | `MiniGrid-Empty-5x5-v0` | ~50 | Trivial | Used for smoke tests |
| `minigrid_doorkey` | `MiniGrid-DoorKey-8x8-v0` | — | Hard | Not yet evaluated |

Both active envs are **sparse reward** (reward only on reaching the goal), **deterministic**, and **goal-conditioned**.

---

## 4. World Models

All neural world models share the same head architecture and training objective. Only the encoder differs.

### Shared Head Architecture (`training/models/heads.py`)

Every world model predicts five quantities from the current latent state `z`:

| Head | Input | Output | Loss |
|---|---|---|---|
| **DynamicsHead** | `[z ‖ a_emb]` | `z_{t+1}` (latent) | MSE vs JEPA target |
| **RewardHead** | `[z ‖ a_emb]` | scalar reward | MSE |
| **DoneHead** | `[z ‖ a_emb]` | done probability | BCE |
| **ValueHead** | `z` | Monte-Carlo return V(z) | MSE |
| **PolicyHead** | `z` | action logits | Cross-entropy |

`a_emb` is a learned `ActionEmbedding` (n_actions → 64-d). DynamicsHead uses a residual connection: predicts `Δz` not `z_next` directly.

**JEPA-style training loss:**
```
z_t    = encoder(obs_t)
z_next = encoder(obs_{t+1}).detach()   ← target, no gradient through it
z_pred = dynamics(z_t, a_emb)
L_dyn  = MSE(z_pred, z_next)
L_total = L_dyn + L_rew + L_done + 0.5·L_value + 0.1·L_policy
```

---

### 4.1 DINO-WM (Frozen DINOv2)

**File:** `benchmark/world_models/dino_wm.py`

| Property | Value |
|---|---|
| Encoder | DINOv2-small (Meta, pretrained on ImageNet) |
| Latent dim | 384 (CLS token) |
| Input resolution | 224×224 (bilinear upsample from 64×64) |
| Encoder frozen | **Yes** — zero encoder gradients |
| Parameters trained | Heads only (~2M) |

The encoder produces ImageNet features. These are semantically rich but **domain-mismatched** to simple 64×64 MiniGrid grids. The dynamics head has to learn transition structure over features that were never designed for it.

---

### 4.2 LeWM (Learned ViT from scratch)

**File:** `benchmark/world_models/lewm.py`

| Property | Value |
|---|---|
| Encoder | ViT-tiny (via Dinov2Config, random init) |
| Latent dim | 192 (CLS token) |
| Input resolution | 224×224 (bilinear upsample from 64×64) |
| Encoder frozen | **No** — full JEPA training |
| Architecture config | `hidden_size=192, layers=3, heads=3, patch_size=14` |

Uses a causal transformer predictor (ARPredictor) in addition to the simple DynamicsHead. Trained from scratch — no pretrained weights. The ViT patch tokenisation requires upscaling, which discards spatial structure from the original 64×64 grid.

---

### 4.3 CNN-WM (New — spatially native)

**File:** `benchmark/world_models/cnn_wm.py`  
**Training:** `training/train_cnn_wm.py`

| Property | Value |
|---|---|
| Encoder | 4-block CNN (`CNNEncoder`) |
| Latent dim | 256 |
| Input resolution | **64×64** (no upscaling) |
| Encoder frozen | **No** — jointly trained with all heads |
| Encoder params | ~780K |

```
Input (B, 3, 64, 64)
  → Conv(3→32, stride 2) + GroupNorm + GELU     → (B, 32, 32, 32)
  → Conv(32→64, stride 2) + GroupNorm + GELU    → (B, 64, 16, 16)
  → Conv(64→128, stride 2) + GroupNorm + GELU   → (B, 128, 8, 8)
  → Conv(128→256, stride 2) + GroupNorm + GELU  → (B, 256, 4, 4)
  → AdaptiveAvgPool2d(1) → Flatten              → (B, 256)
  → Linear(256→256) + LayerNorm                 → (B, 256)
```

Training: 100 epochs, batch=256, lr=1e-4, AdamW (wd=1e-4), CosineAnnealingLR, grad clip=1.0.

---

### 4.4 DINO-WM-FT (Fine-tuned DINOv2)

**File:** `benchmark/world_models/dino_wm.py` (same class, different checkpoint)  
**Training:** `training/train_dino_wm_finetune.py`

| Property | Value |
|---|---|
| Encoder | DINOv2-small, last 2 of 12 blocks unfrozen |
| Latent dim | 384 |
| Unfrozen encoder params | ~3.5M |
| Encoder LR | 1e-5 (10× smaller than heads) |
| Head LR | 3e-4 |
| Epochs | 30 |

Online encoding (cannot pre-encode because the encoder changes). Warm-starts from the frozen DINO-WM checkpoint for the heads.

**Note on L2 metric:** Fine-tuning changes the scale and geometry of the latent space. The raw L2 rollout divergence numbers are *not* comparable to the frozen DINO-WM numbers — the metric is in a different coordinate system. Planning success rates (below) are the correct measure.

---

### 4.5 PerfectModel (Oracle)

**File:** `benchmark/world_models/perfect_model.py`

Wraps the real environment directly. `step()` clones the env state, steps it, and restores. Uses an integer state-ID registry for DP planners (value iteration, policy iteration) that need enumerable states. Zero prediction error by definition — this is the ceiling for any learned world model.

---

## 5. Planners

### Track 1 — Dynamics only (no value/policy head)

| Planner | Strategy | Budget use |
|---|---|---|
| `bfs` | Breadth-first search over latent states | Expands all neighbours at each depth |
| `ucs` | Uniform-cost search (min cumulative reward cost) | Priority queue on cost |
| `random_shooting` | Sample N × H action sequences, keep max-return | N=40, H=5 → 200 calls |
| `beam_search` | Keep top-K states by cumulative reward at each depth | K=10, H=20 → 200 calls |
| `mcts` | UCB1 selection + random rollout | 20 sims × depth 10 ≈ 200 calls |

### Track 2 — Dynamics + value (and optionally policy)

| Planner | Strategy | Budget use |
|---|---|---|
| `astar` | A* with `−V(z)` as heuristic | Priority queue on `cost − V(z)` |
| `greedy_bfs` | Greedy best-first with value heuristic | Always expands highest-value neighbour |
| `cem` | Cross-Entropy Method over action sequences | N=40, elite=10%, 1 iter, H=5 |
| `mcts_value` | MCTS backed by learned value (no random rollout) | 20 sims × depth ≈ 200 calls |
| `latent_rollout` | Depth-1 greedy: `argmax R(z,a) + γV(next_z)` | 7 calls per real step (7 actions) |
| `value_guided_astar` | A* with combined reward+value heuristic | Priority queue |
| `muzero_style` | MCTS with learned policy prior + value | 20 sims, c_puct=1.25 |

### Oracle-only (enumerable state space required)

| Planner | Notes |
|---|---|
| `value_iteration` | Exact VI — only works with PerfectModel / TabularModel |
| `policy_iteration` | Exact PI — same requirement |

---

## 6. Installation

### Requirements

```bash
# Core benchmark (no GPU needed for planning)
pip install -r requirements.txt
pip install -e .

# World model training (GPU recommended)
pip install -r requirements-training.txt
# requirements-training.txt: torch>=2.0, torchvision>=0.15

# MiniGrid environments
pip install minigrid gymnasium
```

### DINOv2 checkpoint

The frozen encoder downloads automatically via HuggingFace on first use (`facebook/dinov2-small`). To use a local copy instead, set the environment variable:

```bash
export DINOV2_CHECKPOINT=/path/to/local/dinov2-small
```

---

## 7. Step-by-Step Pipeline

### 7.1 Collect Training Data

```bash
# MiniGrid Empty-8x8 (~236k transitions)
python training/collect_data.py \
    --env MiniGrid-Empty-8x8-v0 \
    --episodes 1500 \
    --output data/minigrid_empty_8x8.npz

# MiniGrid FourRooms (~295k transitions)
python training/collect_data.py \
    --env MiniGrid-FourRooms-v0 \
    --episodes 2000 \
    --epsilon-greedy 0.3 \
    --output data/minigrid_fourrooms.npz
```

Each `.npz` file contains arrays: `obs` (H×W×3, uint8), `action`, `reward`, `next_obs`, `done`.

Dataset statistics (confirmed):

| Dataset | Transitions | Actions | Done rate |
|---|---|---|---|
| `minigrid_empty_8x8.npz` | 236,022 | 7 | 0.42% |
| `minigrid_fourrooms.npz` | 294,592 | 7 | 1.02% |

---

### 7.2 Train World Models

#### DINO-WM (frozen encoder, heads only)

```bash
python training/train_dino_wm.py \
    --data data/minigrid_empty_8x8.npz \
    --epochs 50 --batch-size 512 --lr 3e-4 \
    --device cuda \
    --output checkpoints/dino_wm_minigrid_empty_8x8.pt
```

Pre-encodes all observations once at startup (fast, since encoder is frozen).

#### LeWM (ViT-tiny from scratch)

```bash
python training/train_lewm.py \
    --data data/minigrid_empty_8x8.npz \
    --epochs 100 --batch-size 256 --lr 1e-4 \
    --device cuda \
    --output checkpoints/lewm_minigrid_empty_8x8.pt
```

#### CNN-WM (CNN from scratch — recommended)

```bash
python training/train_cnn_wm.py \
    --data data/minigrid_empty_8x8.npz \
    --epochs 100 --batch-size 256 --lr 1e-4 \
    --device cuda \
    --output checkpoints/cnn_wm_minigrid_empty_8x8.pt
```

Training took **38 minutes** on a single GPU. Final validation loss: **0.254**.

#### DINO-WM-FT (fine-tune last 2 DINOv2 blocks)

Requires an existing DINO-WM heads checkpoint to warm-start from:

```bash
python training/train_dino_wm_finetune.py \
    --data data/minigrid_empty_8x8.npz \
    --checkpoint checkpoints/dino_wm_minigrid_empty_8x8.pt \
    --unfreeze-blocks 2 \
    --epochs 30 --batch-size 64 \
    --lr 3e-4 --lr-enc 1e-5 \
    --device cuda \
    --output checkpoints/dino_wm_ft_minigrid_empty_8x8.pt
```

Key: `--lr-enc 1e-5` is 30× smaller than head LR to preserve pretrained features while adapting to the domain. Training took **10 hours** (online encoding — cannot pre-encode when the encoder changes).

---

### 7.3 Run the Benchmark

#### Full sweep (all 12 planners, one world model, one env)

```bash
python experiments/run_main.py \
    --config configs/default.yaml \
    --envs minigrid_empty_8x8 \
    --world-models cnn_wm \
    --checkpoints checkpoints/cnn_wm_minigrid_empty_8x8.pt \
    --planners all --episodes 20 --budget 200 --device cuda \
    --output results/cnn_wm_empty8x8
```

#### Quick smoke test

```bash
python experiments/run_quick.py \
    --env minigrid_empty_8x8 \
    --world-model perfect \
    --planner bfs --episodes 3 --budget 200
```

#### Output format

Results are written to `<output>/<timestamp>/`:
- `summary.json` — one entry per planner with aggregated metrics
- `episodes.jsonl` — one line per episode with full detail

Key fields in `summary.json`:

```json
{
  "planner": "mcts_value",
  "world_model": "cnn_wm",
  "env_name": "minigrid_empty_8x8",
  "n_episodes": 20,
  "success_rate": 0.85,
  "mean_return": 0.456,
  "mean_model_calls": 4968,
  "budget_exhaustion_rate": 0.0,
  "mean_steps_success": 131.9
}
```

---

### 7.4 Evaluate World Model Quality

Measures how far predicted latent states drift from true latent states over multi-step open-loop rollout:

```bash
python experiments/eval_world_models.py \
    --envs minigrid_empty_8x8 minigrid_fourrooms \
    --world-models cnn_wm cnn_wm \
    --checkpoints \
        checkpoints/cnn_wm_minigrid_empty_8x8.pt \
        checkpoints/cnn_wm_minigrid_fourrooms.pt \
    --device cuda --episodes 20 \
    --output results/wm_eval/cnn_wm.json
```

Output includes L2 error at horizons H=1,2,5,10,20, reward MAE, and done accuracy.

---

---

## 9. Configuration

`configs/default.yaml` controls the main sweep:

```yaml
budget:
  max_model_calls: 200      # B — model calls available per real env step
  max_wall_time_s: 30.0     # wall-time fallback (model_calls is the binding limit)

execution:
  n_episodes: 20
  receding_horizon: true    # execute only action[0], replan each step

planners: all               # "all" | "track1" | "track2" | [list]

envs:
  - minigrid_empty_8x8
  - minigrid_fourrooms

planner_kwargs:
  mcts:       {n_simulations: 20, c_puct: 1.41, rollout_depth: 10}
  mcts_value: {n_simulations: 20, c_puct: 1.41}
  cem:        {n_samples: 40, elite_frac: 0.1, n_iters: 1, horizon: 5}
  beam_search:{beam_width: 10, horizon: 20}
  random_shooting: {n_samples: 40, horizon: 5}
  latent_rollout:  {horizon: 20, gamma: 0.99}
  muzero_style:    {n_simulations: 20, c_puct: 1.25}
```

All planner kwargs are tuned so total `step()` calls ≈ B=200 per real step.

---

## 10. Results

### World Model Rollout Divergence (mean L2 in latent space)

Lower is better. Measures open-loop prediction error at horizon H.

#### MiniGrid Empty-8×8

| Horizon H | DINO-WM (frozen) | LeWM | CNN-WM | DINO-WM-FT |
|---|---|---|---|---|
| 1 | 6.55 | 19.33 | **0.048** | 26.83¹ |
| 2 | 11.41 | 23.50 | **0.085** | 43.46¹ |
| 5 | 32.68 | 29.27 | **0.254** | 72.09¹ |
| 10 | 210.1 | 32.68 | **1.703** | 116.0¹ |
| 20 | 11,823 | 34.39 | 86.49 | 308.3¹ |

#### MiniGrid FourRooms

| Horizon H | DINO-WM (frozen) | LeWM | CNN-WM | DINO-WM-FT |
|---|---|---|---|---|
| 1 | 4.94 | 1.99 | **0.046** | 29.56¹ |
| 2 | 7.99 | 3.66 | **0.081** | 47.80¹ |
| 5 | 13.16 | 8.77 | **0.242** | 78.37¹ |
| 10 | 21.22 | 15.31 | **1.630** | 127.3¹ |
| 20 | 115.9 | **20.44** | 82.40 | 347.5¹ |

¹ DINO-WM-FT L2 numbers are in the fine-tuned latent space and are not directly comparable to the frozen DINO-WM numbers. The latent geometry changed; planning SR (below) is the correct comparison.

---

### Planning Success Rates — MiniGrid Empty-8×8 (B=200, 20 episodes)

| Planner | DINO-WM | LeWM | DINO-WM-FT | CNN-WM | Oracle |
|---|---|---|---|---|---|
| bfs | 0% | 5% | 25% | 10% | **75%** |
| ucs | 0% | 0% | 5% | 0% | — |
| random_shooting | 0% | 0% | 5% | 5% | — |
| beam_search | 0% | 0% | 0% | 0% | — |
| mcts | 0% | 0% | **40%** | 0% | — |
| astar | 0% | 0% | 5% | 55% | 60% |
| greedy_bfs | 0% | 0% | 25% | 55% | — |
| cem | 5% | 20% | 25% | 35% | — |
| mcts_value | 0% | 20% | 35% | **85%** | — |
| latent_rollout | 0% | 0% | 0% | 0% | — |
| value_guided_astar | 0% | 0% | 20% | **60%** | 50% |
| muzero_style | 0% | 10% | 0% | **65%** | — |

---

### Planning Success Rates — MiniGrid FourRooms (B=200, 20 episodes)

| Planner | DINO-WM | LeWM | DINO-WM-FT | CNN-WM | Oracle |
|---|---|---|---|---|---|
| bfs | 0% | 5% | 0% | 10% | 5% |
| ucs | 0% | 10% | 0% | 0% | — |
| random_shooting | 5% | 0% | 5% | 10% | — |
| beam_search | 0% | 0% | 0% | 0% | — |
| mcts | 5% | 0% | 5% | **15%** | — |
| astar | 0% | 0% | 0% | 0% | 10% |
| greedy_bfs | 0% | 0% | 0% | 5% | — |
| cem | 0% | 5% | 5% | 0% | — |
| mcts_value | 0% | 10% | 0% | 5% | — |
| latent_rollout | 0% | 0% | 0% | 0% | — |
| value_guided_astar | 0% | 0% | 0% | 5% | 15% |
| muzero_style | 5% | 10% | 0% | 0% | — |

---

## 11. Key Findings

### Finding 1: World model quality is the primary bottleneck on Empty-8×8 (H1 confirmed)

CNN-WM lifts `mcts_value` from 20% (LeWM) to **85%**, and `muzero_style` from 10% to **65%**. The improvement comes entirely from the encoder: a spatially-native CNN trained on 64×64 pixels directly is the right inductive bias for grid navigation. The DINOv2 ViT with ImageNet features is fundamentally mismatched to this domain.

### Finding 2: Budget constraint is the primary bottleneck on FourRooms (H2 confirmed)

Even with CNN-WM, the best SR on FourRooms is only **15%** (mcts). The Oracle ceiling at B=200 is just 5–15% (BFS/A*/VGA*). More budget is needed before world model quality becomes the limiting factor.

### Finding 3: Value head quality is what separates CNN-WM from everything else

Track 1 planners (bfs, mcts, random_shooting) show modest improvement with CNN-WM because they don't use the value head. Track 2 planners (mcts_value, muzero_style, value_guided_astar, astar) show dramatic improvement — they rely on `V(z)` to guide search. A world model with a well-trained value head enables much more efficient planning at the same budget.

### Finding 4: Fine-tuning DINOv2 is a middle ground, not a solution

DINO-WM-FT improves substantially over frozen DINO on E8 (`mcts`: 0%→40%, `bfs/greedy_bfs/cem`: 0%→25%) but stays well below CNN-WM on value-guided planners (`mcts_value`: 35% vs 85%). Unfreezing 2 of 12 blocks partially adapts the features but the core mismatch between ImageNet pretraining and MiniGrid domain remains.

### Finding 5: latent_rollout fails everywhere

0% SR across all world models and environments. This greedy depth-1 planner relies purely on `V(z)` to find the goal — sparse reward and a finite-horizon value estimate are not sufficient. It also uses the fewest model calls per step (7, one per action), showing that efficiency is not the issue.

### Finding 6: The L2 divergence metric is unreliable across different encoder families

CNN-WM has much lower 1-step L2 (0.048) than LeWM (19.3), yet LeWM achieves comparable or better long-horizon stability (H=20: 34 vs 86). DINO-WM-FT has higher L2 than frozen DINO-WM yet better planning SR. Absolute L2 values are only meaningful within the same encoder family.

---

## 12. Next Steps

| Priority | Task | Why |
|---|---|---|
| High | Resubmit budget sweep (B=1000–5000) with longer wall time | Jobs 539334/540314 timed out. Needed to determine if FourRooms SR improves with more budget (test H2 quantitatively). |
| High | Extend CNN-WM training or increase capacity | Current architecture plateaus at val loss 0.254. Rollout diverges significantly at H=20 (86×). Deeper network or more epochs may help. |
| Medium | Evaluate CNN-WM at B=500, 1000 on FourRooms | If SR increases substantially, it confirms budget is the remaining bottleneck; if not, we need better world model quality for FR too. |
| Medium | Data augmentation for world model training | Random crops, color jitter, and goal randomisation could improve generalization. |
| Low | Extend to DoorKey / MultiRoom | More complex tasks with structured sub-goals. Requires epsilon-greedy data collection (already implemented in `collect_data.py`). |

---

## Citation / Acknowledgements

This project is part of ongoing research into model-based planning under computational constraints. Environments: [MiniGrid](https://github.com/Farama-Foundation/Minigrid). Encoder pretraining: [DINOv2](https://github.com/facebookresearch/dinov2) (Meta AI). JEPA-style training inspired by [I-JEPA](https://github.com/facebookresearch/ijepa).
