"""
Train DINO-WM heads on a collected MiniGrid dataset.

The DINOv2 encoder is kept frozen. Only the dynamics, reward, value,
policy, and done heads are trained.

Loss terms:
  L_dynamics  = MSE(z_pred, z_next)           -- latent prediction
  L_reward    = MSE(r_pred, r_true)           -- reward prediction
  L_done      = BCE(done_logit, done_true)    -- termination prediction
  L_value     = MSE(V(z_t), MC_return_t)     -- value regression
  L_policy    = CE(π(z_t), best_action)      -- behaviour cloning (optional)

Usage:
    python training/train_dino_wm.py \
        --data data/minigrid_empty_8x8.npz \
        --env  MiniGrid-Empty-8x8-v0 \
        --epochs 50 \
        --batch-size 256 \
        --output checkpoints/dino_wm_empty8x8.pt
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

from benchmark.world_models.dino_wm import DINOWorldModel, DINO_CHECKPOINT, _preprocess
from training.models.heads import (
    ActionEmbedding, DynamicsHead, RewardHead, ValueHead, PolicyHead, DoneHead
)

LATENT_DIM    = 384
ACTION_EMB_DIM = 64


# ── Dataset ───────────────────────────────────────────────────────────────────

class TransitionDataset(Dataset):
    """
    Each item: (obs_t, action_t, reward_t, obs_{t+1}, done_t, mc_return_t)
    MC returns are pre-computed with γ=0.99 over the episode.
    """

    def __init__(self, npz_path: str, gamma: float = 0.99):
        data = np.load(npz_path)
        self.obs      = data["obs"]        # (N, H, W, 3)  uint8
        self.action   = data["action"]     # (N,)           int64
        self.reward   = data["reward"]     # (N,)           float32
        self.next_obs = data["next_obs"]   # (N, H, W, 3)  uint8
        self.done     = data["done"]       # (N,)           bool
        self.mc_return = self._compute_mc_returns(self.reward, self.done, gamma)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return (
            self.obs[idx],
            int(self.action[idx]),
            float(self.reward[idx]),
            self.next_obs[idx],
            bool(self.done[idx]),
            float(self.mc_return[idx]),
        )

    @staticmethod
    def _compute_mc_returns(rewards: np.ndarray, dones: np.ndarray, gamma: float) -> np.ndarray:
        mc = np.zeros_like(rewards)
        running = 0.0
        for i in reversed(range(len(rewards))):
            running = rewards[i] + gamma * running * (1.0 - float(dones[i]))
            mc[i] = running
        return mc


# ── Pre-encode latents ────────────────────────────────────────────────────────

@torch.no_grad()
def encode_dataset(dataset: TransitionDataset, encoder: Dinov2Model, device: torch.device, batch_size: int = 128):
    """Pre-compute all latents to avoid running DINO on every training step."""
    N = len(dataset)
    latents     = np.zeros((N, LATENT_DIM), dtype=np.float32)
    next_latents = np.zeros((N, LATENT_DIM), dtype=np.float32)

    print("Pre-encoding observations with DINOv2 ...")
    for start in range(0, N, batch_size):
        end  = min(start + batch_size, N)
        obs_batch      = dataset.obs[start:end]         # (B, H, W, 3)
        next_obs_batch = dataset.next_obs[start:end]

        def batch_encode(imgs):
            tensors = []
            for img in imgs:
                tensors.append(_preprocess(img, device))
            x = torch.cat(tensors, dim=0)               # (B, 3, 224, 224)
            return encoder(pixel_values=x).last_hidden_state[:, 0, :].cpu().numpy()

        latents[start:end]      = batch_encode(obs_batch)
        next_latents[start:end] = batch_encode(next_obs_batch)

        if (start // batch_size) % 10 == 0:
            print(f"  encoded {end}/{N}")

    return latents, next_latents


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    device = torch.device(args.device)

    dataset = TransitionDataset(args.data, gamma=args.gamma)
    n_val   = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    # Infer n_actions from data
    n_actions = int(dataset.action.max()) + 1

    # Load frozen encoder for pre-encoding
    encoder = Dinov2Model.from_pretrained(DINO_CHECKPOINT).to(device).eval()

    # Pre-encode all latents (avoids redundant DINO forward passes)
    all_latents, all_next_latents = encode_dataset(dataset, encoder, device, batch_size=64)
    del encoder   # free VRAM; encoder not needed after this

    # Build heads
    action_emb = ActionEmbedding(n_actions, ACTION_EMB_DIM).to(device)
    dynamics   = DynamicsHead(LATENT_DIM, ACTION_EMB_DIM).to(device)
    reward_h   = RewardHead(LATENT_DIM, ACTION_EMB_DIM).to(device)
    value_h    = ValueHead(LATENT_DIM).to(device)
    policy_h   = PolicyHead(LATENT_DIM, n_actions).to(device)
    done_h     = DoneHead(LATENT_DIM, ACTION_EMB_DIM).to(device)

    params = (
        list(action_emb.parameters())
        + list(dynamics.parameters())
        + list(reward_h.parameters())
        + list(value_h.parameters())
        + list(policy_h.parameters())
        + list(done_h.parameters())
    )
    optimiser = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    train_indices = train_ds.indices
    val_indices   = val_ds.indices

    def get_batch(indices, shuffle=True):
        if shuffle:
            idx = np.random.choice(indices, size=min(args.batch_size, len(indices)), replace=False)
        else:
            idx = np.array(indices[:args.batch_size])
        z      = torch.from_numpy(all_latents[idx]).to(device)
        z_next = torch.from_numpy(all_next_latents[idx]).to(device)
        a      = torch.from_numpy(dataset.action[idx]).long().to(device)
        r      = torch.from_numpy(dataset.reward[idx]).float().to(device)
        d      = torch.from_numpy(dataset.done[idx].astype(np.float32)).to(device)
        mc     = torch.from_numpy(dataset.mc_return[idx]).float().to(device)
        return z, a, r, z_next, d, mc

    # ── Class-imbalance weights (computed from full dataset) ──────────
    done_pos_rate   = float(dataset.done.astype(float).mean())
    done_pw         = (args.done_pos_weight if args.done_pos_weight is not None
                       else (1.0 - done_pos_rate) / max(done_pos_rate, 1e-6))
    reward_pos_rate = float((dataset.reward > 0).mean())
    reward_pw       = (args.reward_pos_weight if args.reward_pos_weight is not None
                       else (1.0 - reward_pos_rate) / max(reward_pos_rate, 1e-6))
    done_pw_t = torch.tensor([done_pw], device=device)

    best_val_loss = float("inf")
    print(f"\nTraining DINO-WM heads for {args.epochs} epochs ...")
    print(f"  train={n_train}  val={n_val}  batch={args.batch_size}  device={device}")
    print(f"  done_pos_weight={done_pw:.1f}  reward_pos_weight={reward_pw:.1f}\n")

    for epoch in range(1, args.epochs + 1):
        # ── Train ──────────────────────────────────────────────────
        for mod in [action_emb, dynamics, reward_h, value_h, policy_h, done_h]:
            mod.train()

        n_batches = max(1, n_train // args.batch_size)
        train_loss = 0.0
        for _ in range(n_batches):
            z, a, r, z_next, d, mc = get_batch(train_indices)
            a_emb = action_emb(a)

            z_pred    = dynamics(z, a_emb)
            r_pred    = reward_h(z, a_emb)
            done_pred = done_h(z, a_emb)
            v_pred    = value_h(z)
            pi_logits = policy_h(z)

            l_dyn    = F.mse_loss(z_pred, z_next)
            rew_w    = torch.where(r > 0, r.new_full(r.shape, reward_pw), torch.ones_like(r))
            l_rew    = (rew_w * (r_pred - r) ** 2).mean()
            l_done   = F.binary_cross_entropy_with_logits(done_pred, d, pos_weight=done_pw_t)
            l_value  = F.mse_loss(v_pred, mc)
            l_policy = F.cross_entropy(pi_logits, a)   # behaviour cloning

            loss = (
                args.w_dyn   * l_dyn
                + args.w_rew   * l_rew
                + args.w_done  * l_done
                + args.w_value * l_value
                + args.w_policy * l_policy
            )

            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimiser.step()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= n_batches

        # ── Validate ───────────────────────────────────────────────
        for mod in [action_emb, dynamics, reward_h, value_h, policy_h, done_h]:
            mod.eval()

        with torch.no_grad():
            z, a, r, z_next, d, mc = get_batch(val_indices, shuffle=False)
            a_emb  = action_emb(a)
            r_pred = reward_h(z, a_emb)
            rew_w  = torch.where(r > 0, r.new_full(r.shape, reward_pw), torch.ones_like(r))
            l_dyn  = F.mse_loss(dynamics(z, a_emb), z_next).item()
            l_rew  = (rew_w * (r_pred - r) ** 2).mean().item()
            l_done = F.binary_cross_entropy_with_logits(done_h(z, a_emb), d, pos_weight=done_pw_t).item()
            l_val  = F.mse_loss(value_h(z), mc).item()
            val_loss = l_dyn + l_rew + l_done + l_val

        print(
            f"Epoch {epoch:3d}/{args.epochs}"
            f"  train={train_loss:.4f}"
            f"  val={val_loss:.4f}"
            f"  dyn={l_dyn:.4f}  rew={l_rew:.4f}  done={l_done:.4f}  val_v={l_val:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(args.output, action_emb, dynamics, reward_h, value_h, policy_h, done_h, n_actions)
            print(f"  ✓ checkpoint saved ({args.output})")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {args.output}")


def _save_checkpoint(path, action_emb, dynamics, reward_h, value_h, policy_h, done_h, n_actions):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "action_emb": action_emb.state_dict(),
            "dynamics":   dynamics.state_dict(),
            "reward":     reward_h.state_dict(),
            "value":      value_h.state_dict(),
            "policy":     policy_h.state_dict(),
            "done":       done_h.state_dict(),
            "n_actions":  n_actions,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",      required=True,    help="Path to .npz dataset")
    parser.add_argument("--env",       default="MiniGrid-Empty-8x8-v0")
    parser.add_argument("--epochs",    type=int,   default=50)
    parser.add_argument("--batch-size",type=int,   default=256)
    parser.add_argument("--lr",        type=float, default=3e-4)
    parser.add_argument("--gamma",     type=float, default=0.99)
    parser.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output",    default="checkpoints/dino_wm.pt")
    # Loss weights
    parser.add_argument("--w-dyn",    type=float, default=1.0)
    parser.add_argument("--w-rew",    type=float, default=1.0)
    parser.add_argument("--w-done",   type=float, default=1.0)
    parser.add_argument("--w-value",  type=float, default=0.5)
    parser.add_argument("--w-policy", type=float, default=0.1)
    # Class-imbalance corrections (auto-computed from data when not set)
    parser.add_argument("--done-pos-weight",   type=float, default=None,
                        help="pos_weight for done BCE; auto-computed from dataset if omitted")
    parser.add_argument("--reward-pos-weight", type=float, default=None,
                        help="weight for positive-reward transitions in weighted MSE; auto-computed if omitted")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
