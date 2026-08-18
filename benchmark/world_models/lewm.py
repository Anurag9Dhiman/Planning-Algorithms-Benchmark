"""
LeWM — benchmark adapter wrapping the official JEPA implementation.

Source: https://github.com/lucas-maes/le-wm  (jepa.py / lewm_module.py)

Architecture
------------
  Encoder   : ViT-tiny trained from scratch, no pretraining
              (H,W,3) -> CLS token -> projector -> z ∈ R^EMBED_DIM
  Predictor : ARPredictor (causal transformer over embedding + action history)
              f(z_t, a_t) -> z_{t+1}
  Training  : next-embedding prediction loss + SIGReg (Gaussian latent regulariser)

Discrete-action adaptation
--------------------------
  - Actions are one-hot encoded and fed through the Embedder action encoder.
  - history_size=1 so each step() is self-contained (planners branch freely).
  - RewardHeadZ and DoneHeadZ are lightweight z-only heads trained on collected
    (z_next, r, done) tuples; they do NOT need an action embedding because the
    reward/done signal already conditions on the predicted next state.
  - ValueHead (from training/models/heads.py) predicts V(z) for Track-2 planners.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# jepa.py and lewm_module.py live alongside this file
sys.path.insert(0, str(Path(__file__).parent))
from jepa import JEPA
from lewm_module import ARPredictor, Embedder, MLP, SIGReg  # noqa: F401 (SIGReg used in train script)

from benchmark.core.budget import PlanningBudget
from benchmark.core.world_model import State, WorldModel
from training.models.heads import ValueHead

EMBED_DIM = 192  # matches config/train/model/lewm.yaml


# ── Lightweight z-only heads ──────────────────────────────────────────────────

class _ZHead(nn.Module):
    """Shared MLP backbone for z-only heads."""

    def __init__(self, embed_dim: int, hidden: int, out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, out),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


class RewardHeadZ(_ZHead):
    """Predicts scalar reward from predicted next-state embedding."""

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__(embed_dim, 256, 1)


class DoneHeadZ(_ZHead):
    """Predicts episode-termination logit from predicted next-state embedding."""

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__(embed_dim, 128, 1)


# ── JEPA construction ────────────────────────────────────────────────────────

def _build_jepa(n_actions: int, image_size: int = 224) -> JEPA:
    """
    Build JEPA with ViT-tiny encoder (trained from scratch) + ARPredictor.
    Mirrors config/train/model/lewm.yaml but using transformers.Dinov2Model
    as the ViT backend (no stable_pretraining dependency).
    """
    from transformers import Dinov2Config, Dinov2Model

    vit_cfg = Dinov2Config(
        hidden_size=EMBED_DIM,
        num_hidden_layers=3,
        num_attention_heads=3,
        mlp_ratio=4,
        patch_size=14,
        image_size=image_size,
        initializer_range=0.02,
    )
    encoder = Dinov2Model(vit_cfg)

    predictor = ARPredictor(
        num_frames=1,
        input_dim=EMBED_DIM,
        hidden_dim=EMBED_DIM,
        output_dim=EMBED_DIM,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dim_head=64,
        dropout=0.1,
    )

    action_encoder = Embedder(
        input_dim=n_actions,
        smoothed_dim=n_actions,
        emb_dim=EMBED_DIM,
        mlp_scale=4,
    )

    projector = MLP(input_dim=EMBED_DIM, hidden_dim=2048, output_dim=EMBED_DIM,
                    norm_fn=nn.BatchNorm1d)
    pred_proj = MLP(input_dim=EMBED_DIM, hidden_dim=2048, output_dim=EMBED_DIM,
                    norm_fn=nn.BatchNorm1d)

    return JEPA(encoder=encoder, predictor=predictor, action_encoder=action_encoder,
                projector=projector, pred_proj=pred_proj)


# ── World model ───────────────────────────────────────────────────────────────

class LeWM(WorldModel):
    """
    LeWM world model for the planning benchmark.

    Parameters
    ----------
    n_actions       : number of discrete actions in the environment
    checkpoint_path : .pt checkpoint written by training/train_lewm.py
                      (None = random init, useful for sanity-checking shapes)
    device          : "cpu" | "cuda"
    image_size      : spatial resolution fed to the ViT encoder
    done_threshold  : sigmoid(logit) >= threshold -> done=True
    """

    name = "lewm"

    def __init__(
        self,
        n_actions: int,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
        image_size: int = 224,
        done_threshold: float = 0.03,
        done_distance_threshold: Optional[float] = None,
    ):
        self.n_actions               = n_actions
        self.done_threshold          = done_threshold
        self.done_distance_threshold = done_distance_threshold  # geometric done
        self.image_size              = image_size
        self.device                  = torch.device(device)

        self._jepa   = _build_jepa(n_actions, image_size).to(self.device)
        self._reward = RewardHeadZ(EMBED_DIM).to(self.device)
        self._done   = DoneHeadZ(EMBED_DIM).to(self.device)
        self._value  = ValueHead(EMBED_DIM).to(self.device)
        # Goal-latent heuristic — set via load_goal_latents() or from checkpoint
        self._goal_latents: Optional[np.ndarray] = None

        if checkpoint_path is not None:
            self.load(checkpoint_path)

        self._eval_mode()

    # ── WorldModel interface ──────────────────────────────────────────────────

    def encode(self, obs: np.ndarray) -> State:
        """(H, W, 3) uint8 -> (EMBED_DIM,) float32 numpy array."""
        x = self._to_tensor(obs)
        with torch.no_grad():
            out = self._jepa.encoder(pixel_values=x, interpolate_pos_encoding=True)
            cls = out.last_hidden_state[:, 0]       # (1, EMBED_DIM)
            z   = self._jepa.projector(cls)         # (1, EMBED_DIM)
        return z.squeeze(0).cpu().numpy()

    def step(
        self, z: State, action: int, budget: PlanningBudget
    ) -> Tuple[State, float, bool]:
        budget.consume(1)

        # Accept GPU tensors to avoid repeated CPU↔GPU round trips when
        # called in a tight planning loop (planners may pass z_next back directly)
        if isinstance(z, torch.Tensor):
            z_t = z.to(self.device).unsqueeze(0).unsqueeze(0)
        else:
            z_t = torch.from_numpy(z).to(self.device).unsqueeze(0).unsqueeze(0)

        a_oh = self._one_hot(action)                            # (1, 1, n_actions)

        with torch.no_grad():
            a_emb  = self._jepa.action_encoder(a_oh)           # (1, 1, D)
            pred   = self._jepa.predictor(z_t, a_emb)          # (1, 1, D)
            z_flat = pred[:, -1]                                # (1, D)
            z_next = self._jepa.pred_proj(z_flat)              # (1, D)

            reward = float(self._reward(z_next).squeeze().cpu())

        z_next_np = z_next.squeeze(0).cpu().numpy()

        # Geometric done: use min-distance to goal latents when calibrated
        if self._goal_latents is not None and self.done_distance_threshold is not None:
            dist = float(np.linalg.norm(self._goal_latents - z_next_np, axis=1).min())
            done = dist < self.done_distance_threshold
        else:
            with torch.no_grad():
                done_prob = float(torch.sigmoid(self._done(z_next)).squeeze().cpu())
            done = done_prob >= self.done_threshold

        return z_next_np, reward, done

    def value(self, z: State) -> float:
        if self._goal_latents is not None:
            dists = np.linalg.norm(self._goal_latents - z, axis=1)
            return -float(dists.min())
        z_t = torch.from_numpy(z).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self._value(z_t).item()

    def calibrate_done_distance_threshold(
        self, npz_path: str, recall: float = 0.95
    ) -> float:
        """
        Compute done_distance_threshold using predictor-path distances.

        For each goal transition (done=True, reward>0), we encode obs, run it
        through the JEPA predictor with the recorded action, and compute the
        min-distance from the predicted z_next to the stored goal latents.
        This mirrors exactly what happens at inference time, so the threshold
        is calibrated to the predictor's output distribution rather than the
        encoder's (which would give trivially-zero distances).

        Sets done_distance_threshold at the `recall`-th percentile.
        """
        from benchmark.core.budget import BudgetConfig, PlanningBudget

        assert self._goal_latents is not None, "Call load_goal_latents() first"

        data      = np.load(npz_path)
        done_mask = data["done"] & (data["reward"] > 0)
        obs_arr   = data["obs"]
        action_arr = data["action"]
        idxs      = np.where(done_mask)[0]

        # Use a large dummy budget — we only consume 1 call per transition
        dummy_budget = PlanningBudget(BudgetConfig(max_model_calls=len(idxs) + 1))

        goal_dists = []
        for i in idxs:
            z_curr = self.encode(obs_arr[i])
            z_next, _, _ = self.step(z_curr, int(action_arr[i]), dummy_budget)
            dist = float(np.linalg.norm(self._goal_latents - z_next, axis=1).min())
            goal_dists.append(dist)

        threshold = float(np.percentile(goal_dists, recall * 100))
        self.done_distance_threshold = threshold
        print(f"[LeWM] done_distance_threshold={threshold:.4f}  "
              f"(recall={recall:.0%} over {len(goal_dists)} goal transitions, "
              f"p50={np.percentile(goal_dists,50):.4f} p95={np.percentile(goal_dists,95):.4f})")
        return threshold

    def load_goal_latents(self, npz_path: str) -> None:
        """Encode all successful terminal observations and cache as goal-latent heuristic."""
        data = np.load(npz_path)
        done_mask  = data["done"]
        reward_arr = data["reward"]
        next_obs   = data["next_obs"]
        goal_obs   = next_obs[done_mask & (reward_arr > 0)]
        latents = []
        for o in goal_obs:
            latents.append(self.encode(o))
        self._goal_latents = np.stack(latents).astype(np.float32)
        print(f"[LeWM] Loaded {len(latents)} goal latents from {npz_path}")

    def policy(self, z: State) -> np.ndarray:
        return np.ones(self.n_actions, dtype=np.float32) / self.n_actions

    def action_space(self) -> List[int]:
        return list(range(self.n_actions))

    # ── Checkpoint I/O ───────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        state = {
            "jepa":      self._jepa.state_dict(),
            "reward":    self._reward.state_dict(),
            "done":      self._done.state_dict(),
            "value":     self._value.state_dict(),
            "n_actions": self.n_actions,
        }
        if self._goal_latents is not None:
            state["goal_latents"] = torch.from_numpy(self._goal_latents)
        if self.done_distance_threshold is not None:
            state["done_distance_threshold"] = self.done_distance_threshold
        torch.save(state, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self._jepa.load_state_dict(ckpt["jepa"])
        self._reward.load_state_dict(ckpt["reward"])
        self._done.load_state_dict(ckpt["done"])
        self._value.load_state_dict(ckpt["value"])
        if "goal_latents" in ckpt and ckpt["goal_latents"] is not None:
            self._goal_latents = ckpt["goal_latents"].cpu().numpy()
        if "done_distance_threshold" in ckpt:
            self.done_distance_threshold = float(ckpt["done_distance_threshold"])
        self._eval_mode()

    # ── Parameter groups (used by training script) ───────────────────────────

    def jepa_parameters(self):
        return list(self._jepa.parameters())

    def head_parameters(self):
        return (
            list(self._reward.parameters())
            + list(self._done.parameters())
            + list(self._value.parameters())
        )

    def set_train(self) -> None:
        for m in (self._jepa, self._reward, self._done, self._value):
            m.train()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _eval_mode(self) -> None:
        for m in (self._jepa, self._reward, self._done, self._value):
            m.eval()

    def _to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        """(H, W, 3) uint8 -> (1, 3, image_size, image_size) float32 on device."""
        x = torch.from_numpy(obs).permute(2, 0, 1).float().div_(255.0)
        x = x.unsqueeze(0).to(self.device)
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size),
                              mode="bilinear", align_corners=False)
        return x

    def _one_hot(self, action: int) -> torch.Tensor:
        """int -> (1, 1, n_actions) float tensor."""
        return F.one_hot(
            torch.tensor([[action]], device=self.device),
            num_classes=self.n_actions,
        ).float()


# ── Option 4: Goal-conditioned done head ─────────────────────────────────────

class DoneHeadGoalCond(_ZHead):
    """done(z_next, z_goal) — concatenates predicted next state with goal latent."""

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__(embed_dim * 2, 256, 1)

    def forward(self, z_next: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_next, z_goal], dim=-1)).squeeze(-1)


class GoalCondLeWM(LeWM):
    """
    LeWM with a goal-conditioned done head: done(z_next, z_goal).
    At inference, z_goal is the mean of goal latents stored in the checkpoint.
    At training time, z_goal is the actual terminal latent of the episode.
    """

    name = "lewm_goal_cond"

    def __init__(
        self,
        n_actions: int,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
        image_size: int = 224,
        done_threshold: float = 0.5,
    ):
        # Init parent WITHOUT loading checkpoint (we have a different done head shape)
        super().__init__(n_actions, checkpoint_path=None, device=device,
                         image_size=image_size, done_threshold=done_threshold)
        # Replace z-only done head with goal-conditioned version
        self._done = DoneHeadGoalCond(EMBED_DIM).to(self.device)
        self._mean_goal_latent: Optional[torch.Tensor] = None

        if checkpoint_path is not None:
            self._load_goal_cond(checkpoint_path)
        self._eval_mode()

    def step(
        self, z: State, action: int, budget: PlanningBudget
    ) -> Tuple[State, float, bool]:
        budget.consume(1)
        if isinstance(z, torch.Tensor):
            z_t = z.to(self.device).unsqueeze(0).unsqueeze(0)
        else:
            z_t = torch.from_numpy(z).to(self.device).unsqueeze(0).unsqueeze(0)

        a_oh = self._one_hot(action)
        with torch.no_grad():
            a_emb  = self._jepa.action_encoder(a_oh)
            pred   = self._jepa.predictor(z_t, a_emb)
            z_next = self._jepa.pred_proj(pred[:, -1])       # (1, D)

            reward = self._reward(z_next).squeeze().cpu().item()

            if self._mean_goal_latent is not None:
                g = self._mean_goal_latent.unsqueeze(0)       # (1, D)
                done_prob = torch.sigmoid(self._done(z_next, g)).cpu().item()
            else:
                # Fall back to z-only done if no goal latent loaded
                done_prob = 0.0

        done = bool(done_prob >= self.done_threshold)
        return z_next.squeeze(0).cpu().numpy(), reward, done

    def save_goal_cond(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        state = {
            "jepa":           self._jepa.state_dict(),
            "reward":         self._reward.state_dict(),
            "done_goal_cond": self._done.state_dict(),
            "value":          self._value.state_dict(),
            "n_actions":      self.n_actions,
        }
        if self._mean_goal_latent is not None:
            state["mean_goal_latent"] = self._mean_goal_latent.cpu()
        torch.save(state, path)

    def _load_goal_cond(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self._jepa.load_state_dict(ckpt["jepa"])
        self._reward.load_state_dict(ckpt["reward"])
        self._value.load_state_dict(ckpt["value"])
        if "done_goal_cond" in ckpt:
            # Checkpoint saved by save_goal_cond() — load goal-cond done head
            self._done.load_state_dict(ckpt["done_goal_cond"])
        # else: warm-starting from a base LeWM checkpoint; done head stays random init
        if "mean_goal_latent" in ckpt:
            self._mean_goal_latent = ckpt["mean_goal_latent"].to(self.device)
        self._eval_mode()

    def done_head_parameters(self):
        return list(self._done.parameters())
