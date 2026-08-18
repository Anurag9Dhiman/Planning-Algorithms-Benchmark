"""
Base DINO-WM — encoder + dynamics only.

No value head, no done head, no reward head.
This is the original DINO-WM concept: frozen DINOv2 encoder
with a learned dynamics MLP trained purely on next-state prediction.

Used for:
  - Multi-step rollout quality evaluation (base dynamics experiment)
  - Establishing what the model can do before any planning components are added
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Dinov2Model

from benchmark.core.budget import PlanningBudget
from benchmark.core.world_model import State, WorldModel
from training.models.heads import ActionEmbedding, DynamicsHead

DINO_CHECKPOINT  = os.environ.get("DINOV2_CHECKPOINT", "facebook/dinov2-small")
LATENT_DIM       = 384
ACTION_EMB_DIM   = 64

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _preprocess(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(obs).permute(2, 0, 1).float() / 255.0
    x = x.unsqueeze(0)
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    return (x.to(device) - _MEAN.to(device)) / _STD.to(device)


class BaseDINOModel(WorldModel):
    """
    DINO-WM as described in the paper: frozen DINOv2 encoder + dynamics MLP.
    Goal-directed planning uses goal-latent distance (no learned heads):
      done   = min_k ||z_next - z_goal_k|| < done_distance_threshold
      reward = -min_k ||z_next - z_goal_k||
    Call load_goal_latents(npz_path) then calibrate_done_threshold(npz_path)
    before running planners.
    """

    name = "base_dino_wm"

    def __init__(
        self,
        n_actions: int,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
    ):
        self.n_actions              = n_actions
        self.device                 = torch.device(device)
        self._goal_latents          = None   # (K, LATENT_DIM) float32
        self.done_distance_threshold = None  # set by calibrate_done_threshold()

        # Frozen DINOv2 encoder
        self._encoder = Dinov2Model.from_pretrained(DINO_CHECKPOINT)
        self._encoder.eval()
        for p in self._encoder.parameters():
            p.requires_grad = False
        self._encoder.to(self.device)

        # Dynamics only — no value/done/reward heads
        self._action_emb = ActionEmbedding(n_actions, ACTION_EMB_DIM).to(self.device)
        self._dynamics   = DynamicsHead(LATENT_DIM, ACTION_EMB_DIM).to(self.device)

        if checkpoint_path is not None:
            self.load(checkpoint_path)

        self._action_emb.eval()
        self._dynamics.eval()

    def encode(self, obs: np.ndarray) -> State:
        x = _preprocess(obs, self.device)
        with torch.no_grad():
            out = self._encoder(pixel_values=x)
        return out.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()

    def step(
        self, z: State, action: int, budget: PlanningBudget
    ) -> Tuple[State, float, bool]:
        budget.consume(1)
        z_t   = torch.from_numpy(z).unsqueeze(0).to(self.device)
        a_t   = torch.tensor([action], dtype=torch.long, device=self.device)
        a_emb = self._action_emb(a_t)
        with torch.no_grad():
            z_next = self._dynamics(z_t, a_emb)
        z_next_np = z_next.squeeze(0).cpu().numpy()

        if self._goal_latents is not None:
            dist   = float(np.linalg.norm(self._goal_latents - z_next_np, axis=1).min())
            reward = -dist
            done   = (dist < self.done_distance_threshold) if self.done_distance_threshold is not None else False
        else:
            reward, done = 0.0, False

        return z_next_np, reward, done

    def value(self, z: State) -> float:
        if self._goal_latents is not None:
            return -float(np.linalg.norm(self._goal_latents - z, axis=1).min())
        return 0.0

    def load_goal_latents(self, npz_path: str) -> None:
        """Encode all successful terminal observations as goal latents."""
        data      = np.load(npz_path)
        done_mask = data["done"] & (data["reward"] > 0)
        goal_obs  = data["next_obs"][done_mask]
        latents   = [self.encode(o) for o in goal_obs]
        self._goal_latents = np.stack(latents).astype(np.float32)
        print(f"[BaseDINO] Loaded {len(latents)} goal latents from {npz_path}")

    def calibrate_done_threshold(self, npz_path: str, recall: float = 0.95) -> float:
        """Set done_distance_threshold from dataset goal transitions."""
        from benchmark.core.budget import BudgetConfig, PlanningBudget
        assert self._goal_latents is not None, "Call load_goal_latents() first"
        data       = np.load(npz_path)
        done_mask  = data["done"] & (data["reward"] > 0)
        obs_arr    = data["obs"]
        action_arr = data["action"]
        idxs       = np.where(done_mask)[0]
        dummy      = PlanningBudget(BudgetConfig(max_model_calls=len(idxs) + 1,
                                                  max_wall_time_s=3600.0))
        dists = []
        for i in idxs:
            z      = self.encode(obs_arr[i])
            z_next, _, _ = self.step(z, int(action_arr[i]), dummy)
            dists.append(float(np.linalg.norm(self._goal_latents - z_next, axis=1).min()))
        self.done_distance_threshold = float(np.percentile(dists, recall * 100))
        print(f"[BaseDINO] done_distance_threshold={self.done_distance_threshold:.4f} "
              f"(recall={recall:.0%} over {len(dists)} goal transitions)")
        return self.done_distance_threshold

    def action_space(self) -> List[int]:
        return list(range(self.n_actions))

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self._action_emb.load_state_dict(state["action_emb"])
        self._dynamics.load_state_dict(state["dynamics"])

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "action_emb": self._action_emb.state_dict(),
            "dynamics":   self._dynamics.state_dict(),
        }, path)

    def trainable_parameters(self):
        return (list(self._action_emb.parameters())
                + list(self._dynamics.parameters()))
