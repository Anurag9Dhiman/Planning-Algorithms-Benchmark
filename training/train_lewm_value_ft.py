"""
Option 2 — Fine-tune LeWM value/done/reward heads with the JEPA encoder frozen.

Loads an existing LeWM checkpoint, freezes the JEPA backbone, pre-encodes all
observations once, then trains only the auxiliary heads with:
  - High value loss weight (w_val=10.0) to force goal-discriminative V(z)
  - Class-balanced BCE for done (pos_weight auto-computed from dataset)
  - Class-balanced weighted MSE for reward

Usage:
    python training/train_lewm_value_ft.py \
        --checkpoint lewm_fixed_minigrid_empty_8x8.pt \
        --data       data/minigrid_empty_8x8.npz \
        --output     checkpoints/lewm_valueft_minigrid_empty_8x8.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.world_models.lewm import LeWM, EMBED_DIM
from training.train_dino_wm import TransitionDataset


def _batch_encode(model: LeWM, images: np.ndarray, batch_size: int = 128) -> np.ndarray:
    """Encode (N, H, W, 3) uint8 -> (N, EMBED_DIM) float32 in batches."""
    N   = len(images)
    out = np.zeros((N, EMBED_DIM), dtype=np.float32)
    dev = model.device
    img_size = model.image_size

    for start in range(0, N, batch_size):
        end   = min(start + batch_size, N)
        batch = torch.stack([
            torch.from_numpy(images[i]).permute(2, 0, 1).float().div_(255.0)
            for i in range(start, end)
        ]).to(dev)
        if batch.shape[-1] != img_size or batch.shape[-2] != img_size:
            batch = F.interpolate(batch, size=(img_size, img_size),
                                  mode="bilinear", align_corners=False)
        with torch.no_grad():
            cls    = model._jepa.encoder(pixel_values=batch,
                                         interpolate_pos_encoding=True).last_hidden_state[:, 0]
            z_proj = model._jepa.projector(cls)
        out[start:end] = z_proj.cpu().numpy()
        if start % (batch_size * 20) == 0:
            print(f"  encoded {end}/{N}")
    return out


def train(args):
    device = torch.device(args.device)

    model = LeWM(n_actions=args.n_actions, checkpoint_path=args.checkpoint, device=str(device))

    # Freeze JEPA
    for p in model._jepa.parameters():
        p.requires_grad = False
    model._jepa.eval()

    base = TransitionDataset(args.data, gamma=args.gamma)
    N    = len(base)

    print(f"\nPre-encoding {N} transitions (frozen JEPA) ...")
    z     = _batch_encode(model, base.obs,      args.enc_batch)
    z_nxt = _batch_encode(model, base.next_obs, args.enc_batch)

    reward    = base.reward.astype(np.float32)
    done      = base.done.astype(np.float32)
    mc_return = base.mc_return.astype(np.float32)

    # Class-imbalance weights
    done_pw   = (1.0 - done.mean())   / max(done.mean(),               1e-6)
    reward_pw = (1.0 - (reward > 0).mean()) / max((reward > 0).mean(), 1e-6)
    done_pw_t = torch.tensor([done_pw], device=device)
    print(f"done_pos_weight={done_pw:.1f}  reward_pos_weight={reward_pw:.1f}")
    print(f"w_val={args.w_val}  epochs={args.epochs}")

    ds      = TensorDataset(
        torch.from_numpy(z),
        torch.from_numpy(z_nxt),
        torch.from_numpy(reward),
        torch.from_numpy(done),
        torch.from_numpy(mc_return),
    )
    n_val  = max(1, int(N * 0.1))
    tr_ds, va_ds = random_split(ds, [N - n_val, n_val],
                                generator=torch.Generator().manual_seed(42))
    kw = dict(batch_size=args.batch_size, num_workers=2, pin_memory=True)
    tr_loader = DataLoader(tr_ds, shuffle=True,  **kw)
    va_loader = DataLoader(va_ds, shuffle=False, **kw)

    head_params = (
        list(model._value.parameters())
        + list(model._done.parameters())
        + list(model._reward.parameters())
    )
    opt  = torch.optim.AdamW(head_params, lr=args.lr, weight_decay=1e-4)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val = float("inf")

    def _batch_loss(z_, z_nxt_, rew_, done_, mc_):
        z_    = z_.to(device);  z_nxt_  = z_nxt_.to(device)
        rew_  = rew_.to(device); done_  = done_.to(device); mc_ = mc_.to(device)
        rew_w = torch.where(rew_ > 0, rew_.new_full(rew_.shape, reward_pw), torch.ones_like(rew_))
        l_rew   = (rew_w * (model._reward(z_nxt_) - rew_) ** 2).mean()
        l_done  = F.binary_cross_entropy_with_logits(model._done(z_nxt_), done_, pos_weight=done_pw_t)
        l_value = F.mse_loss(model._value(z_), mc_)
        return args.w_rew * l_rew + args.w_done * l_done + args.w_val * l_value

    print()
    for epoch in range(1, args.epochs + 1):
        model._value.train(); model._done.train(); model._reward.train()
        t_loss = 0.0
        for batch in tr_loader:
            loss = _batch_loss(*batch)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head_params, 1.0)
            opt.step()
            t_loss += loss.item()
        t_loss /= len(tr_loader)
        sch.step()

        model._value.eval(); model._done.eval(); model._reward.eval()
        v_loss = 0.0
        with torch.no_grad():
            for batch in va_loader:
                v_loss += _batch_loss(*batch).item()
        v_loss /= len(va_loader)

        print(f"Epoch {epoch:3d}/{args.epochs}  train={t_loss:.4f}  val={v_loss:.4f}")
        if v_loss < best_val:
            best_val = v_loss
            model.save(args.output)
            print(f"  saved -> {args.output}")

    print(f"\nDone. Best val: {best_val:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data",       required=True)
    p.add_argument("--output",     required=True)
    p.add_argument("--n-actions",  type=int,   default=7)
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--batch-size", type=int,   default=1024)
    p.add_argument("--enc-batch",  type=int,   default=128,  help="Batch size for pre-encoding")
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--gamma",      type=float, default=0.99)
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--w-val",  type=float, default=10.0)
    p.add_argument("--w-done", type=float, default=1.0)
    p.add_argument("--w-rew",  type=float, default=1.0)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
