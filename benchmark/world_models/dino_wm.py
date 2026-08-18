"""
DINO-WM World Model.

Architecture:
  Encoder  : frozen DINOv2-small (facebook/dinov2-small via HuggingFace)
             RGB (H,W,3) -> CLS token z ∈ R^384
  Dynamics : learned DynamicsHead  f(z, a) -> z'
  Reward   : learned RewardHead    r(z, a) -> scalar
  Value    : learned ValueHead     V(z)    -> scalar
  Policy   : learned PolicyHead    π(z)    -> logits (optional)
  Done     : learned DoneHead      d(z, a) -> logit  (optional)

The encoder is always frozen. Only the heads are trained (see training/train_dino_wm.py).
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
from training.models.heads import (
    ActionEmbedding,
    DoneHead,
    DynamicsHead,
    PolicyHead,
    RewardHead,
    ValueHead,
)

DINO_CHECKPOINT = os.environ.get("DINOV2_CHECKPOINT", "facebook/dinov2-small")
LATENT_DIM = 384
ACTION_EMB_DIM = 64

# ImageNet normalisation (DINOv2 standard)
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _preprocess(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    obs: (H, W, 3) uint8 numpy array
    Returns: (1, 3, 224, 224) float tensor, ImageNet-normalised
    """
    x = torch.from_numpy(obs).permute(2, 0, 1).float() / 255.0   # (3, H, W)
    x = x.unsqueeze(0)                                             # (1, 3, H, W)
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    mean = _MEAN.to(device)
    std  = _STD.to(device)
    return (x.to(device) - mean) / std


class DINOWorldModel(WorldModel):
    """
    Parameters
    ----------
    n_actions       : number of discrete actions in the environment
    checkpoint_path : path to a saved heads checkpoint (.pt file)
                      If None, heads are randomly initialised (useful for
                      architecture smoke-tests; values will be meaningless).
    device          : "cpu" | "cuda"
    done_threshold  : sigmoid probability above which step() returns done=True.
                      Set to 0.9 (not 0.5) to suppress false positives: the done head
                      fires on ~50% of out-of-distribution (dynamics-predicted) latents
                      at the default 0.5 threshold, causing planners to return spurious
                      terminal paths. On real encoder outputs done_accuracy=99.94%.
    """

    name = "dino_wm"

    def __init__(
        self,
        n_actions: int,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
        done_threshold: float = 0.9,
    ):
        self.n_actions = n_actions
        self.done_threshold = done_threshold
        self.device = torch.device(device)

        # ── Frozen encoder ────────────────────────────────────────────
        self._encoder = Dinov2Model.from_pretrained(DINO_CHECKPOINT)
        self._encoder.eval()
        for p in self._encoder.parameters():
            p.requires_grad = False
        self._encoder.to(self.device)

        # ── Learnable heads ───────────────────────────────────────────
        self._action_emb = ActionEmbedding(n_actions, ACTION_EMB_DIM).to(self.device)
        self._dynamics   = DynamicsHead(LATENT_DIM, ACTION_EMB_DIM).to(self.device)
        self._reward     = RewardHead(LATENT_DIM, ACTION_EMB_DIM).to(self.device)
        self._value      = ValueHead(LATENT_DIM).to(self.device)
        self._policy     = PolicyHead(LATENT_DIM, n_actions).to(self.device)
        self._done       = DoneHead(LATENT_DIM, ACTION_EMB_DIM).to(self.device)

        if checkpoint_path is not None:
            self.load(checkpoint_path)

        self._set_eval()

    # ── WorldModel interface ──────────────────────────────────────────

    def encode(self, obs: np.ndarray) -> State:
        """obs: (H, W, 3) uint8 -> z: (384,) float32 numpy array."""
        x = _preprocess(obs, self.device)
        with torch.no_grad():
            out = self._encoder(pixel_values=x)
        cls = out.last_hidden_state[:, 0, :]   # (1, 384) CLS token
        return cls.squeeze(0).cpu().numpy()

    def step(
        self, z: State, action: int, budget: PlanningBudget
    ) -> Tuple[State, float, bool]:
        budget.consume(1)
        z_t   = torch.from_numpy(z).unsqueeze(0).to(self.device)      # (1, 384)
        a_t   = torch.tensor([action], dtype=torch.long, device=self.device)
        a_emb = self._action_emb(a_t)                                  # (1, 64)

        with torch.no_grad():
            z_next = self._dynamics(z_t, a_emb)                        # (1, 384)
            # Single CPU-GPU sync instead of two .item() calls
            aux = torch.stack([
                self._reward(z_t, a_emb).squeeze(),
                torch.sigmoid(self._done(z_t, a_emb)).squeeze(),
            ]).cpu()

        reward = float(aux[0])
        done   = bool(aux[1] >= self.done_threshold)
        return z_next.squeeze(0).cpu().numpy(), reward, done

    def value(self, z: State) -> float:
        z_t = torch.from_numpy(z).unsqueeze(0).to(self.device)
        with torch.no_grad():
            v = self._value(z_t).item()
        return v

    def policy(self, z: State) -> np.ndarray:
        z_t = torch.from_numpy(z).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self._policy(z_t)
            probs  = F.softmax(logits, dim=-1)
        return probs.squeeze(0).cpu().numpy()

    def action_space(self) -> List[int]:
        return list(range(self.n_actions))

    # ── Checkpoint I/O ───────────────────────────────────────────────

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self._action_emb.load_state_dict(state["action_emb"])
        self._dynamics.load_state_dict(state["dynamics"])
        self._reward.load_state_dict(state["reward"])
        self._value.load_state_dict(state["value"])
        self._policy.load_state_dict(state["policy"])
        self._done.load_state_dict(state["done"])
        self._set_eval()

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "action_emb": self._action_emb.state_dict(),
                "dynamics":   self._dynamics.state_dict(),
                "reward":     self._reward.state_dict(),
                "value":      self._value.state_dict(),
                "policy":     self._policy.state_dict(),
                "done":       self._done.state_dict(),
            },
            path,
        )

    def trainable_parameters(self):
        """Return parameters that should be optimised (encoder excluded)."""
        return (
            list(self._action_emb.parameters())
            + list(self._dynamics.parameters())
            + list(self._reward.parameters())
            + list(self._value.parameters())
            + list(self._policy.parameters())
            + list(self._done.parameters())
        )

    def set_train(self) -> None:
        self._action_emb.train()
        self._dynamics.train()
        self._reward.train()
        self._value.train()
        self._policy.train()
        self._done.train()

    def _set_eval(self) -> None:
        self._encoder.eval()
        self._action_emb.eval()
        self._dynamics.eval()
        self._reward.eval()
        self._value.eval()
        self._policy.eval()
        self._done.eval()
