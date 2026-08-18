"""
Fine-tune DINO-WM: unfreeze last N transformer blocks of DINOv2.

Key differences from train_dino_wm.py
--------------------------------------
  1. Encoder is NOT frozen — last `--unfreeze-blocks` blocks get gradients.
  2. Cannot pre-encode latents (encoder changes each step) → online encoding.
  3. Differential LR: encoder blocks at lr_enc, heads at lr.
  4. Starts from an existing checkpoint if --checkpoint is given, otherwise
     random head init with pretrained encoder (cold start).

Usage
-----
    python training/train_dino_wm_finetune.py \\
        --data       /scratch/.../data/minigrid_empty_8x8.npz \\
        --checkpoint /home/.../checkpoints/dino_wm_minigrid_empty_8x8.pt \\
        --unfreeze-blocks 2 \\
        --epochs 30 \\
        --batch-size 64 \\
        --output /scratch/.../checkpoints/dino_wm_ft_empty8x8.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import Dinov2Model

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.world_models.dino_wm import DINOWorldModel, DINO_CHECKPOINT, _preprocess, LATENT_DIM, ACTION_EMB_DIM
from training.train_dino_wm import TransitionDataset
from training.models.heads import ActionEmbedding, DynamicsHead, RewardHead, ValueHead, PolicyHead, DoneHead


# ── Pixel dataset (cannot pre-encode when encoder is unfrozen) ────────────────

class PixelTransitionDataset(Dataset):
    def __init__(self, npz_path: str, gamma: float = 0.99):
        base = TransitionDataset(npz_path, gamma)
        self.obs       = base.obs
        self.action    = base.action
        self.reward    = base.reward
        self.next_obs  = base.next_obs
        self.done      = base.done.astype(np.float32)
        self.mc_return = base.mc_return
        self.n_actions = int(self.action.max()) + 1

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return (
            self.obs[idx],       # (H, W, 3) uint8 — preprocess in collate
            int(self.action[idx]),
            float(self.reward[idx]),
            self.next_obs[idx],
            float(self.done[idx]),
            float(self.mc_return[idx]),
        )


def _preprocess_batch(imgs: np.ndarray, device: torch.device) -> torch.Tensor:
    """(B, H, W, 3) uint8 -> (B, 3, 224, 224) ImageNet-normalised."""
    MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    STD  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    x = torch.from_numpy(imgs).permute(0, 3, 1, 2).float().div_(255.0).to(device)
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    return (x - MEAN) / STD


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

    n_actions = ds.n_actions

    # ── Load encoder ──────────────────────────────────────────────────
    encoder = Dinov2Model.from_pretrained(DINO_CHECKPOINT).to(device)

    # Freeze everything first, then selectively unfreeze last N blocks
    for p in encoder.parameters():
        p.requires_grad = False

    # DINOv2 blocks are in encoder.encoder.layer (list of TransformerBlocks)
    n_blocks = len(encoder.encoder.layer)
    for block in encoder.encoder.layer[n_blocks - args.unfreeze_blocks:]:
        for p in block.parameters():
            p.requires_grad = True
    # Also unfreeze layernorm after the last block
    for p in encoder.layernorm.parameters():
        p.requires_grad = True

    n_enc_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"Unfreezing last {args.unfreeze_blocks}/{n_blocks} DINOv2 blocks: {n_enc_params:,} params")

    # ── Build heads (or load from checkpoint) ─────────────────────────
    action_emb = ActionEmbedding(n_actions, ACTION_EMB_DIM).to(device)
    dynamics   = DynamicsHead(LATENT_DIM, ACTION_EMB_DIM).to(device)
    reward_h   = RewardHead(LATENT_DIM, ACTION_EMB_DIM).to(device)
    value_h    = ValueHead(LATENT_DIM).to(device)
    policy_h   = PolicyHead(LATENT_DIM, n_actions).to(device)
    done_h     = DoneHead(LATENT_DIM, ACTION_EMB_DIM).to(device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        action_emb.load_state_dict(ckpt["action_emb"])
        dynamics.load_state_dict(ckpt["dynamics"])
        reward_h.load_state_dict(ckpt["reward"])
        value_h.load_state_dict(ckpt["value"])
        policy_h.load_state_dict(ckpt["policy"])
        done_h.load_state_dict(ckpt["done"])
        print(f"Loaded heads from {args.checkpoint}")

    head_params = (
        list(action_emb.parameters()) + list(dynamics.parameters())
        + list(reward_h.parameters()) + list(value_h.parameters())
        + list(policy_h.parameters()) + list(done_h.parameters())
    )
    enc_params = [p for p in encoder.parameters() if p.requires_grad]

    optimiser = torch.optim.AdamW([
        {"params": enc_params,  "lr": args.lr_enc},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    # ── Class-imbalance weights ───────────────────────────────────────
    done_pos_rate   = float(ds.done.astype(float).mean())
    done_pw         = (args.done_pos_weight if args.done_pos_weight is not None
                       else (1.0 - done_pos_rate) / max(done_pos_rate, 1e-6))
    reward_pos_rate = float((ds.reward > 0).mean())
    reward_pw       = (args.reward_pos_weight if args.reward_pos_weight is not None
                       else (1.0 - reward_pos_rate) / max(reward_pos_rate, 1e-6))
    done_pw_t = torch.tensor([done_pw], device=device)

    print(f"\nFine-tuning DINO-WM | enc_lr={args.lr_enc}  head_lr={args.lr}")
    print(f"  train={len(train_ds)}  val={len(val_ds)}  batch={args.batch_size}  device={device}")
    print(f"  done_pos_weight={done_pw:.1f}  reward_pos_weight={reward_pw:.1f}\n")

    best_val = float("inf")
    heads = [action_emb, dynamics, reward_h, value_h, policy_h, done_h]

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        for h in heads: h.train()
        t_loss = 0.0

        for batch in train_loader:
            obs_np, action, reward, next_obs_np, done, mc = batch
            obs_t      = _preprocess_batch(obs_np.numpy(),      device)
            next_obs_t = _preprocess_batch(next_obs_np.numpy(), device)
            action = torch.as_tensor(action, dtype=torch.long,    device=device)
            reward = torch.as_tensor(reward, dtype=torch.float32, device=device)
            done   = torch.as_tensor(done,   dtype=torch.float32, device=device)
            mc     = torch.as_tensor(mc,     dtype=torch.float32, device=device)

            z_t    = encoder(pixel_values=obs_t).last_hidden_state[:, 0]       # (B, 384)
            z_next = encoder(pixel_values=next_obs_t).last_hidden_state[:, 0].detach()

            a_emb     = action_emb(action)
            z_pred    = dynamics(z_t, a_emb)
            r_pred_ft = reward_h(z_t, a_emb)
            rew_w_ft  = torch.where(reward > 0, reward.new_full(reward.shape, reward_pw), torch.ones_like(reward))
            l_dyn     = F.mse_loss(z_pred, z_next)
            l_rew     = (rew_w_ft * (r_pred_ft - reward) ** 2).mean()
            l_done    = F.binary_cross_entropy_with_logits(done_h(z_t, a_emb), done, pos_weight=done_pw_t)
            l_value   = F.mse_loss(value_h(z_t), mc)
            l_policy  = F.cross_entropy(policy_h(z_t), action)

            loss = (args.w_dyn * l_dyn + args.w_rew * l_rew
                    + args.w_done * l_done + args.w_val * l_value
                    + args.w_pol * l_policy)

            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(enc_params + head_params, 1.0)
            optimiser.step()
            t_loss += loss.item()

        t_loss /= len(train_loader)
        scheduler.step()

        # ── Validate ───────────────────────────────────────────────────
        encoder.eval()
        for h in heads: h.eval()
        v_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                obs_np, action, reward, next_obs_np, done, mc = batch
                obs_t      = _preprocess_batch(obs_np.numpy(),      device)
                next_obs_t = _preprocess_batch(next_obs_np.numpy(), device)
                action = torch.as_tensor(action, dtype=torch.long,    device=device)
                reward = torch.as_tensor(reward, dtype=torch.float32, device=device)
                done   = torch.as_tensor(done,   dtype=torch.float32, device=device)
                mc     = torch.as_tensor(mc,     dtype=torch.float32, device=device)

                z_t    = encoder(pixel_values=obs_t).last_hidden_state[:, 0]
                z_next = encoder(pixel_values=next_obs_t).last_hidden_state[:, 0]
                a_emb  = action_emb(action)

                r_pred_v = reward_h(z_t, a_emb)
                rew_w_v  = torch.where(reward > 0, reward.new_full(reward.shape, reward_pw), torch.ones_like(reward))
                v_loss += (
                    F.mse_loss(dynamics(z_t, a_emb), z_next)
                    + (rew_w_v * (r_pred_v - reward) ** 2).mean()
                    + F.binary_cross_entropy_with_logits(done_h(z_t, a_emb), done, pos_weight=done_pw_t)
                    + F.mse_loss(value_h(z_t), mc)
                ).item()

        v_loss /= len(val_loader)
        print(f"Epoch {epoch:3d}/{args.epochs}  train={t_loss:.4f}  val={v_loss:.4f}")

        if v_loss < best_val:
            best_val = v_loss
            _save(args.output, encoder, action_emb, dynamics, reward_h, value_h, policy_h, done_h, n_actions)
            print(f"  saved -> {args.output}")

    print(f"\nDone. Best val: {best_val:.4f}  Checkpoint: {args.output}")


def _save(path, encoder, action_emb, dynamics, reward_h, value_h, policy_h, done_h, n_actions):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "encoder":    encoder.state_dict(),
        "action_emb": action_emb.state_dict(),
        "dynamics":   dynamics.state_dict(),
        "reward":     reward_h.state_dict(),
        "value":      value_h.state_dict(),
        "policy":     policy_h.state_dict(),
        "done":       done_h.state_dict(),
        "n_actions":  n_actions,
    }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",             required=True)
    parser.add_argument("--checkpoint",       default=None, help="Existing DINO-WM heads checkpoint to warm-start from")
    parser.add_argument("--unfreeze-blocks",  type=int,   default=2,    help="Number of DINOv2 blocks to unfreeze (from the end)")
    parser.add_argument("--epochs",           type=int,   default=30)
    parser.add_argument("--batch-size",       type=int,   default=64,   help="Keep small — online encoding is expensive")
    parser.add_argument("--lr",               type=float, default=3e-4, help="Head learning rate")
    parser.add_argument("--lr-enc",           type=float, default=1e-5, help="Encoder fine-tune learning rate (much smaller)")
    parser.add_argument("--gamma",            type=float, default=0.99)
    parser.add_argument("--device",           default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output",           default="checkpoints/dino_wm_ft.pt")
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
