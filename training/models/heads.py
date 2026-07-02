"""
Learnable heads that sit on top of the frozen DINOv2 encoder.

All heads take a latent z ∈ R^latent_dim and optionally an action embedding.
Action embedding: learned lookup table (n_actions, action_emb_dim).
"""

import torch
import torch.nn as nn


class ActionEmbedding(nn.Module):
    def __init__(self, n_actions: int, emb_dim: int = 64):
        super().__init__()
        self.emb = nn.Embedding(n_actions, emb_dim)

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        return self.emb(action)


class DynamicsHead(nn.Module):
    """
    Predicts next latent: f(z_t, a_t) -> z_{t+1}
    Input:  [z || a_emb]  (latent_dim + action_emb_dim)
    Output: z_{t+1}       (latent_dim)
    """

    def __init__(self, latent_dim: int = 384, action_emb_dim: int = 64):
        super().__init__()
        in_dim = latent_dim + action_emb_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, latent_dim),
        )
        # Residual: predict delta instead of full next state
        self._residual = nn.Linear(in_dim, latent_dim, bias=False)

    def forward(self, z: torch.Tensor, a_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, a_emb], dim=-1)
        return z + self.net(x) + self._residual(x)


class RewardHead(nn.Module):
    """
    Predicts reward: r(z_t, a_t) -> scalar
    """

    def __init__(self, latent_dim: int = 384, action_emb_dim: int = 64):
        super().__init__()
        in_dim = latent_dim + action_emb_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, z: torch.Tensor, a_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, a_emb], dim=-1)
        return self.net(x).squeeze(-1)


class ValueHead(nn.Module):
    """
    Estimates state value: V(z_t) -> scalar
    """

    def __init__(self, latent_dim: int = 384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.GELU(),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


class PolicyHead(nn.Module):
    """
    Action prior: P(a|z_t) -> logits over n_actions
    Used by MuZero-style planner.
    """

    def __init__(self, latent_dim: int = 384, n_actions: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.GELU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)   # raw logits; caller applies softmax


class DoneHead(nn.Module):
    """
    Predicts episode termination: done(z_t, a_t) -> logit
    """

    def __init__(self, latent_dim: int = 384, action_emb_dim: int = 64):
        super().__init__()
        in_dim = latent_dim + action_emb_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, z: torch.Tensor, a_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, a_emb], dim=-1)
        return self.net(x).squeeze(-1)   # logit; apply sigmoid for probability
