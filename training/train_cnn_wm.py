"""
Train CNNWorldModel end-to-end on MiniGrid transition data.

Loss terms
----------
  L_dynamics : MSE(z_pred, z_next.detach())   JEPA-style: target is encoded next obs
  L_reward   : MSE(r_head(z_t, a), r)
  L_done     : BCE(done_head(z_t, a), done)
  L_value    : MSE(V(z_t), MC_return)
  L_policy   : CE(π(z_t), a)                  behaviour cloning prior

The encoder is NOT frozen — all parameters train jointly.
The JEPA-style target (z_next = encode(next_obs).detach()) prevents mode collapse
without requiring a separate EMA target network.

Usage
-----
    python training/train_cnn_wm.py \\
        --data data/minigrid_empty_8x8.npz \\
        --epochs 100 \\
        --batch-size 256 \\
        --output checkpoints/cnn_wm_empty8x8.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.world_models.cnn_wm import CNNWorldModel
from training.train_dino_wm import TransitionDataset


# ── Dataset ───────────────────────────────────────────────────────────────────

class PixelTransitionDataset(Dataset):
    """Raw pixel pairs — encoder trains jointly so we cannot pre-encode."""

    def __init__(self, npz_path: str, gamma: float = 0.99):
        base = TransitionDataset(npz_path, gamma)
        self.obs       = base.obs          # (N, H, W, 3) uint8
        self.action    = base.action       # (N,) int64
        self.reward    = base.reward       # (N,) float32
        self.next_obs  = base.next_obs     # (N, H, W, 3) uint8
        self.done      = base.done         # (N,) bool
        self.mc_return = base.mc_return    # (N,) float32
        self.n_actions = int(self.action.max()) + 1

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        obs      = torch.from_numpy(self.obs[idx]).permute(2, 0, 1).float().div_(255.0)
        next_obs = torch.from_numpy(self.next_obs[idx]).permute(2, 0, 1).float().div_(255.0)
        return (
            obs,
            int(self.action[idx]),
            float(self.reward[idx]),
            next_obs,
            float(self.done[idx]),
            float(self.mc_return[idx]),
        )


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device(args.device)

    ds    = PixelTransitionDataset(args.data, gamma=args.gamma)
    n_val = max(1, int(len(ds) * 0.1))
    train_ds, val_ds = random_split(
        ds, [len(ds) - n_val, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    kw = dict(batch_size=args.batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kw)

    model = CNNWorldModel(n_actions=ds.n_actions, device=str(device))

    optimiser = torch.optim.AdamW(
        model.trainable_parameters(), lr=args.lr, weight_decay=args.wd,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    # ── Class-imbalance weights ───────────────────────────────────────
    done_pos_rate   = float(ds.done.astype(float).mean())
    done_pw         = (args.done_pos_weight if args.done_pos_weight is not None
                       else (1.0 - done_pos_rate) / max(done_pos_rate, 1e-6))
    reward_pos_rate = float((ds.reward > 0).mean())
    reward_pw       = (args.reward_pos_weight if args.reward_pos_weight is not None
                       else (1.0 - reward_pos_rate) / max(reward_pos_rate, 1e-6))
    args.done_pw   = done_pw
    args.reward_pw = reward_pw

    n_params = sum(p.numel() for p in model.trainable_parameters())
    print(f"\nTraining CNNWorldModel  |  params={n_params:,}")
    print(f"  n_actions={ds.n_actions}  latent_dim=256")
    print(f"  train={len(train_ds)}  val={len(val_ds)}  batch={args.batch_size}  device={device}")
    print(f"  done_pos_weight={done_pw:.1f}  reward_pos_weight={reward_pw:.1f}\n")

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        # ── Train ──────────────────────────────────────────────────────
        model.set_train()
        t_loss = 0.0
        for batch in train_loader:
            loss = _forward(model, batch, args, device)
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
            optimiser.step()
            t_loss += loss.item()
        t_loss /= len(train_loader)
        scheduler.step()

        # ── Validate ───────────────────────────────────────────────────
        model._set_eval()
        v_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                v_loss += _forward(model, batch, args, device).item()
        v_loss /= len(val_loader)

        print(f"Epoch {epoch:3d}/{args.epochs}  train={t_loss:.4f}  val={v_loss:.4f}")

        if v_loss < best_val:
            best_val = v_loss
            model.save(args.output)
            print(f"  saved -> {args.output}")

    print(f"\nDone. Best val: {best_val:.4f}  Checkpoint: {args.output}")


def _forward(model, batch, args, device):
    obs, action, reward, next_obs, done, mc = batch
    obs      = obs.to(device)
    next_obs = next_obs.to(device)
    action   = torch.as_tensor(action, dtype=torch.long, device=device)
    reward   = torch.as_tensor(reward, dtype=torch.float32, device=device)
    done     = torch.as_tensor(done,   dtype=torch.float32, device=device)
    mc       = torch.as_tensor(mc,     dtype=torch.float32, device=device)

    z_t    = model._encoder(obs)                         # (B, 256)
    z_next = model._encoder(next_obs).detach()           # target: stop gradient

    a_emb  = model._action(action)                       # (B, 64)
    z_pred = model._dynamics(z_t, a_emb)                 # (B, 256)

    done_pw_t = torch.tensor([args.done_pw], device=device)
    rew_pred  = model._reward(z_t, a_emb)
    rew_w     = torch.where(reward > 0, reward.new_full(reward.shape, args.reward_pw), torch.ones_like(reward))
    l_dyn    = F.mse_loss(z_pred, z_next)
    l_rew    = (rew_w * (rew_pred - reward) ** 2).mean()
    l_done   = F.binary_cross_entropy_with_logits(model._done(z_t, a_emb), done, pos_weight=done_pw_t)
    l_value  = F.mse_loss(model._value(z_t), mc)
    l_policy = F.cross_entropy(model._policy(z_t), action)

    return (
        args.w_dyn    * l_dyn
        + args.w_rew  * l_rew
        + args.w_done * l_done
        + args.w_val  * l_value
        + args.w_pol  * l_policy
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       required=True)
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch-size", type=int,   default=256)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--wd",         type=float, default=1e-4)
    parser.add_argument("--gamma",      type=float, default=0.99)
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output",     default="checkpoints/cnn_wm.pt")
    parser.add_argument("--w-dyn",  type=float, default=1.0)
    parser.add_argument("--w-rew",  type=float, default=1.0)
    parser.add_argument("--w-done", type=float, default=1.0)
    parser.add_argument("--w-val",  type=float, default=0.5)
    parser.add_argument("--w-pol",  type=float, default=0.1)
    # Class-imbalance corrections (auto-computed from data when not set)
    parser.add_argument("--done-pos-weight",   type=float, default=None,
                        help="pos_weight for done BCE; auto-computed from dataset if omitted")
    parser.add_argument("--reward-pos-weight", type=float, default=None,
                        help="weight for positive-reward transitions in weighted MSE; auto-computed if omitted")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
