# Planning Algorithms Benchmark

Benchmarking planning algorithms with learned world models in discrete action spaces (MiniGrid environments).

## Installation

```bash
pip install -r requirements.txt
pip install -r requirements-training.txt
```

## Pipeline

**1. Collect training data**
```bash
python training/collect_data.py \
    --env MiniGrid-Empty-8x8-v0 \
    --episodes 1500 \
    --output data/minigrid_empty_8x8.npz
```

**2. Train a world model**
```bash
python training/train_dino_wm.py \
    --data data/minigrid_empty_8x8.npz \
    --output checkpoints/dino_wm_empty8x8.pt \
    --device cuda
```

**3. Run the benchmark**
```bash
python experiments/run_main.py \
    --world-models dino_wm \
    --checkpoints checkpoints/dino_wm_empty8x8.pt \
    --envs minigrid_empty_8x8 \
    --planners all \
    --episodes 20 \
    --device cuda
```

## World Models

- `dino_wm` — frozen DINOv2-small encoder + learned MLP heads
- `lewm` — ViT-tiny encoder trained from scratch with JEPA + SIGReg

> DINOv2 weights download automatically via HuggingFace on first use. To use a local copy: `export DINOV2_CHECKPOINT=/path/to/dinov2-small`

## Planners

**Track 1** (encode + step only): `bfs`, `ucs`, `mcts`, `beam_search`, `random_shooting`

**Track 2** (+ value function): `astar`, `cem`, `mcts_value`, `muzero_style`, `value_guided_astar`, `greedy_bfs`, `latent_rollout`, `value_beam_search`

## Configuration

Edit `configs/default.yaml` to change budget, episodes, and planner hyperparameters.
