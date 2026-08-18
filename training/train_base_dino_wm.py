"""
Train Base DINO-WM — dynamics only, no value/done/reward heads.

Loss:
  L = MSE(z_pred, z_next)   ← next-state prediction only

This is the original DINO-WM concept before any planning components are added.

Usage:
    python training/train_base_dino_wm.py \
        --data    data/minigrid_empty_8x8.npz \
        --epochs  50 \
        --output  checkpoints/base_dino_wm_minigrid_empty_8x8.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import Dinov2Model
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.world_models.base_dino_wm import (
    BaseDINOModel, DINO_CHECKPOINT, LATENT_DIM, ACTION_EMB_DIM, _preprocess
)
from training.models.heads import ActionEmbedding, DynamicsHead


# ── Dataset ───────────────────────────────────────────────────────────────────

class LatentTransitionDataset(Dataset):
    """Pre-encodes all observations with frozen DINOv2 to speed up training."""

    def __init__(self, npz_path: str, encoder, device, batch_size: int = 512):
        data           = np.load(npz_path)
        obs_raw        = data["obs"]        # (N, H, W, 3)
        next_obs_raw   = data["next_obs"]   # (N, H, W, 3)
        self.actions   = torch.from_numpy(data["action"].astype(np.int64))

        print(f"Pre-encoding {len(obs_raw)} observations with DINOv2 ...")
        self.z      = self._encode_all(obs_raw,      encoder, device, batch_size)
        self.z_next = self._encode_all(next_obs_raw, encoder, device, batch_size)
        print("Done.")

    @staticmethod
    def _encode_all(obs_arr, encoder, device, batch_size):
        N      = len(obs_arr)
        latents = []
        encoder.eval()
        with torch.no_grad():
            for i in range(0, N, batch_size):
                batch = obs_arr[i: i + batch_size]
                imgs  = torch.stack([
                    _preprocess(b, device).squeeze(0) for b in batch
                ]).to(device)
                out = encoder(pixel_values=imgs)
                latents.append(out.last_hidden_state[:, 0, :].cpu())
                if (i // batch_size + 1) % 20 == 0:
                    print(f"  encoded {min(i + batch_size, N)}/{N}")
        return torch.cat(latents, dim=0)   # (N, 384)

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return self.z[idx], self.actions[idx], self.z_next[idx]


# ── Training ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",       required=True)
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--batch-size", type=int,   default=256)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--device",     default="cuda")
    p.add_argument("--output",     required=True)
    p.add_argument("--n-actions",  type=int,   default=7)
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"\nTraining Base DINO-WM (dynamics only)")
    print(f"Data   : {args.data}")
    print(f"Output : {args.output}")
    print(f"Device : {device}\n")

    # Load frozen encoder for pre-encoding
    encoder = Dinov2Model.from_pretrained(DINO_CHECKPOINT).to(device)
    encoder.eval()
    for p_ in encoder.parameters():
        p_.requires_grad = False

    dataset = LatentTransitionDataset(args.data, encoder, device)

    n_val   = max(1, int(0.1 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    # Model — dynamics only
    action_emb = ActionEmbedding(args.n_actions, ACTION_EMB_DIM).to(device)
    dynamics   = DynamicsHead(LATENT_DIM, ACTION_EMB_DIM).to(device)

    params    = list(action_emb.parameters()) + list(dynamics.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # ── Train ────────────────────────────────────────────────────────────
        action_emb.train(); dynamics.train()
        train_loss = 0.0
        for z, a, z_next in train_loader:
            z, a, z_next = z.to(device), a.to(device), z_next.to(device)
            a_emb  = action_emb(a)
            z_pred = dynamics(z, a_emb)
            loss   = criterion(z_pred, z_next)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(z)
        train_loss /= n_train

        # ── Validate ─────────────────────────────────────────────────────────
        action_emb.eval(); dynamics.eval()
        val_loss = 0.0
        with torch.no_grad():
            for z, a, z_next in val_loader:
                z, a, z_next = z.to(device), a.to(device), z_next.to(device)
                a_emb  = action_emb(a)
                z_pred = dynamics(z, a_emb)
                val_loss += criterion(z_pred, z_next).item() * len(z)
        val_loss /= n_val

        print(f"Epoch {epoch:>3}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "action_emb": action_emb.state_dict(),
                "dynamics":   dynamics.state_dict(),
            }, args.output)
            print(f"  ✓ saved (val_loss={val_loss:.4f})")

    print(f"\nBest val loss : {best_val_loss:.4f}")
    print(f"Checkpoint    : {args.output}")


if __name__ == "__main__":
    main()
