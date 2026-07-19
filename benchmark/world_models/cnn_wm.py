"""
CNN World Model — encoder trained from scratch jointly with all heads.

Architecture
------------
  Encoder  : CNNEncoder  (64×64 RGB → 256-d latent)
             4 conv blocks with stride-2 downsampling + global avg pool
             No pretrained weights; trained end-to-end on MiniGrid data.
  Dynamics : DynamicsHead  f(z, a) -> z'   (residual MLP)
  Reward   : RewardHead    r(z, a) -> scalar
  Value    : ValueHead     V(z)    -> scalar
  Policy   : PolicyHead    π(z)    -> logits
  Done     : DoneHead      d(z, a) -> logit

Why CNN over ViT for MiniGrid
------------------------------
  MiniGrid observations are 64×64 flat-colour grids.  A ViT with patch_size=14
  requires upscaling to 224×224 and sees 256 patches on a near-constant image.
  A CNN respects the spatial locality of the task (agent pos, goal pos) and
  trains 10× faster with far fewer parameters.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmark.core.budget import PlanningBudget
from benchmark.core.world_model import State, WorldModel
from training.models.encoder import CNNEncoder
from training.models.heads import (
    ActionEmbedding, DoneHead, DynamicsHead,
    PolicyHead, RewardHead, ValueHead,
)

LATENT_DIM    = 256
ACTION_EMB_DIM = 64


class CNNWorldModel(WorldModel):
    """
    Parameters
    ----------
    n_actions       : number of discrete actions
    checkpoint_path : saved .pt file (None = random init)
    device          : "cpu" | "cuda"
    done_threshold  : sigmoid prob above which step() returns done=True
    """

    name = "cnn_wm"

    def __init__(
        self,
        n_actions: int,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
        done_threshold: float = 0.5,
    ):
        self.n_actions     = n_actions
        self.done_threshold = done_threshold
        self.device        = torch.device(device)

        self._encoder  = CNNEncoder(latent_dim=LATENT_DIM).to(self.device)
        self._action   = ActionEmbedding(n_actions, ACTION_EMB_DIM).to(self.device)
        self._dynamics = DynamicsHead(LATENT_DIM, ACTION_EMB_DIM).to(self.device)
        self._reward   = RewardHead(LATENT_DIM, ACTION_EMB_DIM).to(self.device)
        self._value    = ValueHead(LATENT_DIM).to(self.device)
        self._policy   = PolicyHead(LATENT_DIM, n_actions).to(self.device)
        self._done     = DoneHead(LATENT_DIM, ACTION_EMB_DIM).to(self.device)

        if checkpoint_path is not None:
            self.load(checkpoint_path)
        self._set_eval()

    # ── WorldModel interface ──────────────────────────────────────────────────

    def encode(self, obs: np.ndarray) -> State:
        """(H, W, 3) uint8 -> (LATENT_DIM,) float32 numpy array."""
        x = self._to_tensor(obs)
        with torch.no_grad():
            z = self._encoder(x)
        return z.squeeze(0).cpu().numpy()

    def step(
        self, z: State, action: int, budget: PlanningBudget
    ) -> Tuple[State, float, bool]:
        budget.consume(1)
        z_t   = torch.from_numpy(z).unsqueeze(0).to(self.device)
        a_t   = torch.tensor([action], dtype=torch.long, device=self.device)
        a_emb = self._action(a_t)

        with torch.no_grad():
            z_next = self._dynamics(z_t, a_emb)
            aux = torch.stack([
                self._reward(z_t, a_emb).squeeze(),
                torch.sigmoid(self._done(z_t, a_emb)).squeeze(),
            ]).cpu()

        return z_next.squeeze(0).cpu().numpy(), float(aux[0]), bool(aux[1] >= self.done_threshold)

    def value(self, z: State) -> float:
        z_t = torch.from_numpy(z).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self._value(z_t).item()

    def policy(self, z: State) -> np.ndarray:
        z_t = torch.from_numpy(z).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return F.softmax(self._policy(z_t), dim=-1).squeeze(0).cpu().numpy()

    def action_space(self) -> List[int]:
        return list(range(self.n_actions))

    # ── Checkpoint I/O ────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            "encoder":  self._encoder.state_dict(),
            "action":   self._action.state_dict(),
            "dynamics": self._dynamics.state_dict(),
            "reward":   self._reward.state_dict(),
            "value":    self._value.state_dict(),
            "policy":   self._policy.state_dict(),
            "done":     self._done.state_dict(),
            "n_actions": self.n_actions,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self._encoder.load_state_dict(ckpt["encoder"])
        self._action.load_state_dict(ckpt["action"])
        self._dynamics.load_state_dict(ckpt["dynamics"])
        self._reward.load_state_dict(ckpt["reward"])
        self._value.load_state_dict(ckpt["value"])
        self._policy.load_state_dict(ckpt["policy"])
        self._done.load_state_dict(ckpt["done"])

    def trainable_parameters(self):
        params = []
        for m in (self._encoder, self._action, self._dynamics,
                  self._reward, self._value, self._policy, self._done):
            params += list(m.parameters())
        return params

    def set_train(self) -> None:
        for m in (self._encoder, self._action, self._dynamics,
                  self._reward, self._value, self._policy, self._done):
            m.train()

    def _set_eval(self) -> None:
        for m in (self._encoder, self._action, self._dynamics,
                  self._reward, self._value, self._policy, self._done):
            m.eval()

    def _to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        """(H, W, 3) uint8 -> (1, 3, H, W) float32 on device in [0, 1]."""
        return torch.from_numpy(obs).permute(2, 0, 1).float().div_(255.0).unsqueeze(0).to(self.device)
