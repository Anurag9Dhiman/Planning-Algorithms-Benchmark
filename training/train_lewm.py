"""
Train LeWM end-to-end on a collected MiniGrid/maze dataset.

Loss terms
----------
  L_jepa   : MSE(z_pred, z_next)   — next-embedding prediction (JEPA objective)
  L_sigreg : SIGReg(latents)        — Gaussian latent regulariser
  L_reward : MSE(r_head(z_pred), r) — reward prediction from predicted next state
  L_done   : BCE(done_head(z_pred), done)
  L_value  : MSE(V(z_t), MC_return)

No policy loss — LeWM has no policy head; a uniform prior is used at inference.
All parameters (encoder + predictor + heads) are trained jointly.

Usage:
    python training/train_lewm.py \
        --data data/minigrid_empty_8x8.npz \
        --epochs 50 \
        --batch-size 128 \
        --output checkpoints/lewm_empty8x8.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.world_models.lewm import LeWM, SIGReg, EMBED_DIM
from training.train_dino_wm import TransitionDataset  # reuse npz loading + MC returns


# ── Dataset ───────────────────────────────────────────────────────────────────

class PixelDataset(Dataset):
    """Raw pixel pairs — encoder is trained jointly so we cannot pre-encode."""

    def __init__(self, npz_path: str, gamma: float = 0.99):
        base = TransitionDataset(npz_path, gamma)
        self.obs       = base.obs          # (N, H, W, 3) uint8
        self.action    = base.action       # (N,) int
        self.reward    = base.reward       # (N,) float32
        self.next_obs  = base.next_obs     # (N, H, W, 3) uint8
        self.done      = base.done         # (N,) float32
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

    ds      = PixelDataset(args.data, gamma=args.gamma)
    n_val   = max(1, int(len(ds) * 0.1))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    kw = dict(batch_size=args.batch_size, num_workers=2, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kw)

    model  = LeWM(n_actions=ds.n_actions, device=str(device))
    sigreg = SIGReg(knots=17, num_proj=512).to(device)

    optimiser = torch.optim.AdamW(
        model.jepa_parameters() + model.head_parameters()
        + list(sigreg.parameters()),
        lr=args.lr, weight_decay=args.wd,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    print(f"\nTraining LeWM for {args.epochs} epochs")
    print(f"  n_actions={ds.n_actions}  embed_dim={EMBED_DIM}")
    print(f"  train={n_train}  val={n_val}  batch={args.batch_size}  device={device}\n")

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        # ── Train ──────────────────────────────────────────────────────
        model.set_train()
        sigreg.train()
        t_loss = 0.0
        for obs, action, reward, next_obs, done, mc in train_loader:
            loss = _forward(model, sigreg, obs, action, reward, next_obs, done, mc, args, device)
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.jepa_parameters() + model.head_parameters(), 1.0
            )
            optimiser.step()
            t_loss += loss.item()
        t_loss /= len(train_loader)
        scheduler.step()

        # ── Validate ───────────────────────────────────────────────────
        model._eval_mode()
        sigreg.eval()
        v_loss = 0.0
        with torch.no_grad():
            for obs, action, reward, next_obs, done, mc in val_loader:
                v_loss += _forward(
                    model, sigreg, obs, action, reward, next_obs, done, mc, args, device
                ).item()
        v_loss /= len(val_loader)

        print(f"Epoch {epoch:3d}/{args.epochs}  train={t_loss:.4f}  val={v_loss:.4f}")

        if v_loss < best_val:
            best_val = v_loss
            model.save(args.output)
            print(f"  saved -> {args.output}")

    print(f"\nDone. Best val: {best_val:.4f}  Checkpoint: {args.output}")


def _forward(model, sigreg, obs, action, reward, next_obs, done, mc, args, device):
    """Single batch forward; returns combined scalar loss."""
    obs      = obs.to(device)
    next_obs = next_obs.to(device)
    action   = torch.as_tensor(action, device=device)
    reward   = torch.as_tensor(reward, dtype=torch.float32, device=device)
    done     = torch.as_tensor(done,   dtype=torch.float32, device=device)
    mc       = torch.as_tensor(mc,     dtype=torch.float32, device=device)
    B        = obs.size(0)

    # ── Encode obs and next_obs through JEPA encoder + projector ──────
    def _encode_proj(x):
        out = model._jepa.encoder(pixel_values=x, interpolate_pos_encoding=True)
        cls = out.last_hidden_state[:, 0]      # (B, D)
        return model._jepa.projector(cls)      # (B, D)

    z_t    = _encode_proj(obs)       # (B, D)
    z_next = _encode_proj(next_obs)  # (B, D)  — prediction target

    # ── Action encoding: one-hot -> Embedder ──────────────────────────
    a_oh  = F.one_hot(action, num_classes=model.n_actions).float()  # (B, n_actions)
    a_in  = a_oh.unsqueeze(1)                                        # (B, 1, n_actions)
    a_emb = model._jepa.action_encoder(a_in)                        # (B, 1, D)

    # ── Predict next latent ────────────────────────────────────────────
    z_in   = z_t.unsqueeze(1)                                 # (B, 1, D)
    pred   = model._jepa.predictor(z_in, a_emb)              # (B, 1, D)
    z_pred = model._jepa.pred_proj(pred[:, -1])              # (B, D)

    # ── JEPA prediction loss ───────────────────────────────────────────
    l_jepa = F.mse_loss(z_pred, z_next.detach())

    # ── SIGReg on projected latents ────────────────────────────────────
    # SIGReg expects (T, B, D) — stack both z_t and z_pred as timesteps
    latents = torch.stack([z_t, z_pred], dim=0)           # (2, B, D)
    l_sig   = sigreg(latents)

    # ── Auxiliary heads (on predicted next state) ──────────────────────
    l_rew   = F.mse_loss(model._reward(z_pred), reward)
    l_done  = F.binary_cross_entropy_with_logits(model._done(z_pred), done)
    l_value = F.mse_loss(model._value(z_t), mc)

    return (
        args.w_jepa   * l_jepa
        + args.w_sig  * l_sig
        + args.w_rew  * l_rew
        + args.w_done * l_done
        + args.w_val  * l_value
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       required=True,   help="Path to .npz dataset")
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch-size", type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=5e-5)
    parser.add_argument("--wd",         type=float, default=1e-3)
    parser.add_argument("--gamma",      type=float, default=0.99)
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output",     default="checkpoints/lewm.pt")
    # Loss weights
    parser.add_argument("--w-jepa", type=float, default=1.0,  help="JEPA prediction loss weight")
    parser.add_argument("--w-sig",  type=float, default=0.09, help="SIGReg weight (from paper)")
    parser.add_argument("--w-rew",  type=float, default=1.0)
    parser.add_argument("--w-done", type=float, default=1.0)
    parser.add_argument("--w-val",  type=float, default=0.5)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
