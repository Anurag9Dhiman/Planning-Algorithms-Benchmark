"""
Base LeWM — JEPA encoder + predictor only.

No value head, no done head, no reward head.
This is the original LeWM architecture: ViT-tiny encoder trained
from scratch using the JEPA next-embedding prediction objective.

Used for:
  - Multi-step rollout quality evaluation (base dynamics experiment)
  - Establishing what the JEPA architecture can do before planning
    components are added
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from lewm import _build_jepa, EMBED_DIM

from benchmark.core.budget import PlanningBudget
from benchmark.core.world_model import State, WorldModel


class BaseLeWM(WorldModel):
    """
    LeWM as described in the paper: JEPA encoder + predictor.
    Goal-directed planning uses goal-latent distance (no learned heads):
      done   = min_k ||z_next - z_goal_k|| < done_distance_threshold
      reward = -min_k ||z_next - z_goal_k||
    Call load_goal_latents(npz_path) then calibrate_done_threshold(npz_path)
    before running planners.
    """

    name = "base_lewm"

    def __init__(
        self,
        n_actions: int,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
    ):
        self.n_actions               = n_actions
        self.device                  = torch.device(device)
        self._goal_latents           = None  # (K, EMBED_DIM) float32
        self._single_goal_latent     = None  # (EMBED_DIM,) mean of goal bank — used by LeWMCEMPlanner
        self.done_distance_threshold = None  # set by calibrate_done_threshold()

        # Full JEPA graph (encoder + projector + predictor + action_encoder)
        self._jepa = _build_jepa(n_actions).to(self.device)

        if checkpoint_path is not None:
            self.load(checkpoint_path)

        self._jepa.eval()

    def encode(self, obs: np.ndarray) -> State:
        x = torch.from_numpy(obs).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        x = F.interpolate(x.to(self.device), size=(224, 224),
                          mode="bilinear", align_corners=False)
        with torch.no_grad():
            out = self._jepa.encoder(pixel_values=x, interpolate_pos_encoding=True)
            cls = out.last_hidden_state[:, 0]
            z   = self._jepa.projector(cls)
        return z.squeeze(0).cpu().numpy()

    def step(
        self, z: State, action: int, budget: PlanningBudget
    ) -> Tuple[State, float, bool]:
        budget.consume(1)
        z_t  = torch.from_numpy(z).to(self.device).unsqueeze(0).unsqueeze(0)  # (1,1,D)
        a_oh = F.one_hot(
            torch.tensor([action], device=self.device), num_classes=self.n_actions
        ).float().unsqueeze(1)                                                  # (1,1,n)
        with torch.no_grad():
            a_emb  = self._jepa.action_encoder(a_oh)
            pred   = self._jepa.predictor(z_t, a_emb)
            z_next = self._jepa.pred_proj(pred[:, -1])
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
        # Single canonical goal latent — mean of bank, used by LeWMCEMPlanner
        # (cleaner than min-distance to 202 points; for fixed-goal envs like
        # Empty-8x8 the mean is a reliable representative of the goal region)
        self._single_goal_latent = self._goal_latents.mean(axis=0)
        print(f"[BaseLeWM] Loaded {len(latents)} goal latents from {npz_path}")

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
        print(f"[BaseLeWM] done_distance_threshold={self.done_distance_threshold:.4f} "
              f"(recall={recall:.0%} over {len(dists)} goal transitions)")
        return self.done_distance_threshold

    def action_space(self) -> List[int]:
        return list(range(self.n_actions))

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self._jepa.load_state_dict(state["jepa"])

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"jepa": self._jepa.state_dict()}, path)

    def trainable_parameters(self):
        return list(self._jepa.parameters())
