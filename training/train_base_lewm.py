"""
Train Base LeWM — JEPA encoder + predictor only, no extra heads.

Loss:
  L = L_JEPA + λ * L_SIGReg
  L_JEPA  = MSE(predictor(z_t, a_t), stop_grad(target_encoder(o_{t+1})))
  L_SIGReg = Gaussian latent regulariser (prevents collapse)

This is the original LeWM training before value/done heads are added.

Usage:
    python training/train_base_lewm.py \
        --data    data/minigrid_empty_8x8.npz \
        --epochs  100 \
        --output  checkpoints/base_lewm_minigrid_empty_8x8.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "benchmark" / "world_models"))

from lewm import _build_jepa, EMBED_DIM
from lewm_module import SIGReg

SIGREG_LAMBDA = 0.1


# ── Dataset ───────────────────────────────────────────────────────────────────

class TransitionDataset(Dataset):
    def __init__(self, npz_path: str):
        data          = np.load(npz_path)
        self.obs      = data["obs"]        # (N, H, W, 3) uint8
        self.actions  = data["action"]     # (N,)
        self.next_obs = data["next_obs"]   # (N, H, W, 3) uint8

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        obs      = torch.from_numpy(self.obs[idx]).permute(2, 0, 1).float() / 255.0
        next_obs = torch.from_numpy(self.next_obs[idx]).permute(2, 0, 1).float() / 255.0
        action   = int(self.actions[idx])
        return obs, action, next_obs

    @staticmethod
    def collate(batch):
        obs, actions, next_obs = zip(*batch)
        obs      = F.interpolate(torch.stack(obs),      size=(224, 224),
                                 mode="bilinear", align_corners=False)
        next_obs = F.interpolate(torch.stack(next_obs), size=(224, 224),
                                 mode="bilinear", align_corners=False)
        actions  = torch.tensor(actions, dtype=torch.long)
        return obs, actions, next_obs


# ── Training ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",       required=True)
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--batch-size", type=int,   default=128)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--ema-tau",    type=float, default=0.99,
                   help="EMA decay for target encoder")
    p.add_argument("--device",     default="cuda")
    p.add_argument("--output",     required=True)
    p.add_argument("--n-actions",  type=int, default=7)
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"\nTraining Base LeWM (JEPA only — no value/done heads)")
    print(f"Data   : {args.data}")
    print(f"Output : {args.output}\n")

    dataset = TransitionDataset(args.data)
    n_val   = max(1, int(0.1 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True,
                              collate_fn=TransitionDataset.collate)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=2, pin_memory=True,
                              collate_fn=TransitionDataset.collate)

    # Online JEPA (encoder + projector + predictor + action_encoder)
    jepa   = _build_jepa(args.n_actions).to(device)
    sigreg = SIGReg(EMBED_DIM).to(device)

    # Target JEPA — EMA copy, no gradient
    import copy
    target_jepa = copy.deepcopy(jepa).to(device)
    for p_ in target_jepa.parameters():
        p_.requires_grad = False

    params    = list(jepa.parameters()) + list(sigreg.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)

    best_val_loss = float("inf")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    def _encode(model, obs_t):
        """Encode batch of images (B,C,H,W) -> (B, 1, D) for predictor input."""
        out = model.encoder(pixel_values=obs_t, interpolate_pos_encoding=True)
        cls = out.last_hidden_state[:, 0]          # (B, D)
        z   = model.projector(cls)                 # (B, D)
        return z.unsqueeze(1)                      # (B, 1, D)

    def _predict(model, z_t, actions_t):
        """z_t: (B,1,D), actions_t: (B,) long -> z_next: (B,D)"""
        a_oh  = F.one_hot(actions_t, num_classes=args.n_actions).float().unsqueeze(1)
        a_emb = model.action_encoder(a_oh)         # (B, 1, D)
        pred  = model.predictor(z_t, a_emb)        # (B, 1, D)
        return model.pred_proj(pred[:, -1])         # (B, D)

    for epoch in range(1, args.epochs + 1):
        # ── Train ────────────────────────────────────────────────────────────
        jepa.train()
        train_loss = 0.0

        for obs, actions, next_obs in train_loader:
            obs, actions, next_obs = (obs.to(device), actions.to(device),
                                      next_obs.to(device))
            z_t   = _encode(jepa, obs)             # (B, 1, D)
            z_hat = _predict(jepa, z_t, actions)   # (B, D)

            with torch.no_grad():
                z_tgt = _encode(target_jepa, next_obs)[:, 0]   # (B, D)

            loss_jepa   = F.mse_loss(z_hat, z_tgt)
            loss_sigreg = sigreg(z_t[:, 0])
            loss        = loss_jepa + SIGREG_LAMBDA * loss_sigreg

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            # EMA update
            with torch.no_grad():
                tau = args.ema_tau
                for p_on, p_tg in zip(jepa.parameters(), target_jepa.parameters()):
                    p_tg.data = tau * p_tg.data + (1 - tau) * p_on.data

            train_loss += loss_jepa.item() * len(obs)

        train_loss /= n_train

        # ── Validate ─────────────────────────────────────────────────────────
        jepa.eval()
        val_loss = 0.0
        with torch.no_grad():
            for obs, actions, next_obs in val_loader:
                obs, actions, next_obs = (obs.to(device), actions.to(device),
                                          next_obs.to(device))
                z_t   = _encode(jepa, obs)
                z_hat = _predict(jepa, z_t, actions)
                z_tgt = _encode(target_jepa, next_obs)[:, 0]
                val_loss += F.mse_loss(z_hat, z_tgt).item() * len(obs)
        val_loss /= n_val

        print(f"Epoch {epoch:>3}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save to /tmp first then move to avoid Lustre mid-write failures
            import tempfile, shutil
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
                tmp_path = tmp.name
            torch.save({"jepa": jepa.state_dict()}, tmp_path)
            shutil.move(tmp_path, args.output)
            print(f"  ✓ saved (val_loss={val_loss:.4f})")

    print(f"\nBest val loss : {best_val_loss:.4f}")
    print(f"Checkpoint    : {args.output}")


if __name__ == "__main__":
    main()
