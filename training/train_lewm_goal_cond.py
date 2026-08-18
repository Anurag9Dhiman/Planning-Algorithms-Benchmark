"""
Option 4 — Train GoalCondLeWM: done(z_next, z_goal) concatenation head.

Identical backbone training (JEPA + SIGReg + reward/value heads) but with the
goal-conditioned done head:
  - Successful episodes: z_goal = encoded terminal next_obs of the episode
  - Failed episodes:     z_goal = encoded terminal next_obs of a random
                                  successful episode (hindsight relabeling)

The done head receives cat(z_next, z_goal) and should learn to fire when
z_next ≈ z_goal in latent space.

Saves in GoalCondLeWM format (key "done_goal_cond" instead of "done"), with
the mean goal latent baked in for zero-configuration inference.

Usage:
    python training/train_lewm_goal_cond.py \
        --checkpoint /scratch/.../lewm_fixed_minigrid_empty_8x8.pt \
        --data       /scratch/.../data/minigrid_empty_8x8.npz \
        --output     /scratch/.../checkpoints/lewm_goal_cond_minigrid_empty_8x8.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.world_models.lewm import GoalCondLeWM, SIGReg, EMBED_DIM
from training.train_dino_wm import TransitionDataset


# ── Dataset ───────────────────────────────────────────────────────────────────

def _assign_goal_indices(reward: np.ndarray, done: np.ndarray,
                         rng: np.random.Generator) -> np.ndarray:
    """
    Per-transition index into next_obs that is the goal observation.

    Successful episodes → their own terminal next_obs index.
    Failed episodes     → a random terminal next_obs from a successful episode.
    """
    ends         = np.where(done)[0]
    starts       = np.concatenate([[0], ends[:-1] + 1])
    success_ends = ends[reward[ends] > 0]
    if len(success_ends) == 0:
        success_ends = ends  # fallback

    goal_idx = np.zeros(len(done), dtype=np.int64)
    for ep_start, ep_end in zip(starts, ends):
        g = ep_end if reward[ep_end] > 0 else int(rng.choice(success_ends))
        goal_idx[ep_start:ep_end + 1] = g
    return goal_idx


class GoalCondDataset(Dataset):
    def __init__(self, npz_path: str, gamma: float = 0.99, seed: int = 42):
        base           = TransitionDataset(npz_path, gamma)
        self.obs       = base.obs
        self.action    = base.action
        self.reward    = base.reward.astype(np.float32)
        self.next_obs  = base.next_obs
        self.done      = base.done.astype(np.float32)
        self.mc_return = base.mc_return.astype(np.float32)
        self.n_actions = int(self.action.max()) + 1

        rng           = np.random.default_rng(seed)
        self.goal_idx = _assign_goal_indices(self.reward, self.done, rng)

        n_ep   = int(self.done.sum())
        n_succ = int((self.reward[self.done.astype(bool)] > 0).sum())
        print(f"GoalCondDataset: {len(self.obs)} transitions | {n_ep} eps | {n_succ} successful")

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        def _t(arr):
            return torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0)
        return (
            _t(self.obs[idx]),
            int(self.action[idx]),
            float(self.reward[idx]),
            _t(self.next_obs[idx]),
            float(self.done[idx]),
            float(self.mc_return[idx]),
            _t(self.next_obs[self.goal_idx[idx]]),
        )


# ── Forward pass ──────────────────────────────────────────────────────────────

def _forward(model, sigreg, obs, action, reward, next_obs, done, mc, goal_obs, args, device):
    obs      = obs.to(device);      next_obs  = next_obs.to(device)
    goal_obs = goal_obs.to(device)
    action   = torch.as_tensor(action, device=device)
    reward   = torch.as_tensor(reward, dtype=torch.float32, device=device)
    done     = torch.as_tensor(done,   dtype=torch.float32, device=device)
    mc       = torch.as_tensor(mc,     dtype=torch.float32, device=device)

    def _enc(x):
        out = model._jepa.encoder(pixel_values=x, interpolate_pos_encoding=True)
        return model._jepa.projector(out.last_hidden_state[:, 0])

    z_t    = _enc(obs)
    z_next = _enc(next_obs)
    z_goal = _enc(goal_obs)

    a_oh   = F.one_hot(action, num_classes=model.n_actions).float().unsqueeze(1)
    a_emb  = model._jepa.action_encoder(a_oh)
    pred   = model._jepa.predictor(z_t.unsqueeze(1), a_emb)
    z_pred = model._jepa.pred_proj(pred[:, -1])

    l_jepa = F.mse_loss(z_pred, z_next.detach())
    l_sig  = sigreg(torch.stack([z_t, z_pred], dim=0))

    done_pw_t = torch.tensor([args.done_pw], device=device)
    rew_w   = torch.where(reward > 0, reward.new_full(reward.shape, args.reward_pw),
                          torch.ones_like(reward))
    l_rew   = (rew_w * (model._reward(z_pred) - reward) ** 2).mean()
    l_done  = F.binary_cross_entropy_with_logits(
        model._done(z_pred, z_goal), done, pos_weight=done_pw_t
    )
    l_value = F.mse_loss(model._value(z_t), mc)

    return (
        args.w_jepa * l_jepa
        + args.w_sig  * l_sig
        + args.w_rew  * l_rew
        + args.w_done * l_done
        + args.w_val  * l_value
    )


# ── Goal-latent baking ────────────────────────────────────────────────────────

def _bake_mean_goal_latent(model: GoalCondLeWM, ds: GoalCondDataset,
                            device: torch.device) -> None:
    """Encode all successful terminal observations and store their mean."""
    done_bool = ds.done.astype(bool)
    goal_obs  = ds.next_obs[done_bool & (ds.reward > 0)]
    if len(goal_obs) == 0:
        return

    latents = []
    for i in range(0, len(goal_obs), 32):
        batch = torch.stack([
            torch.from_numpy(goal_obs[j]).permute(2, 0, 1).float().div_(255.0)
            for j in range(i, min(i + 32, len(goal_obs)))
        ]).to(device)
        if batch.shape[-1] != model.image_size:
            batch = F.interpolate(batch, size=(model.image_size, model.image_size),
                                  mode="bilinear", align_corners=False)
        with torch.no_grad():
            out = model._jepa.encoder(pixel_values=batch, interpolate_pos_encoding=True)
            z   = model._jepa.projector(out.last_hidden_state[:, 0])
        latents.append(z)

    model._mean_goal_latent = torch.cat(latents, dim=0).mean(dim=0).to(device)
    print(f"  Baked mean goal latent from {len(goal_obs)} successful terminals.")


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    device = torch.device(args.device)

    ds    = GoalCondDataset(args.data, gamma=args.gamma)
    n_val = max(1, int(len(ds) * 0.1))
    tr_ds, va_ds = random_split(ds, [len(ds) - n_val, n_val],
                                generator=torch.Generator().manual_seed(42))
    kw = dict(batch_size=args.batch_size, num_workers=2, pin_memory=True)
    tr_loader = DataLoader(tr_ds, shuffle=True,  **kw)
    va_loader = DataLoader(va_ds, shuffle=False, **kw)

    model  = GoalCondLeWM(n_actions=ds.n_actions,
                          checkpoint_path=args.checkpoint,
                          device=str(device))
    sigreg = SIGReg(knots=17, num_proj=512).to(device)

    done_pos_rate   = float(ds.done.mean())
    args.done_pw    = (1.0 - done_pos_rate)   / max(done_pos_rate,  1e-6)
    reward_pos_rate = float((ds.reward > 0).mean())
    args.reward_pw  = (1.0 - reward_pos_rate) / max(reward_pos_rate, 1e-6)

    print(f"\nGoalCondLeWM | epochs={args.epochs} | device={device}")
    print(f"  done_pw={args.done_pw:.1f}  reward_pw={args.reward_pw:.1f}")
    print(f"  train={len(tr_ds)}  val={len(va_ds)}\n")

    all_params = (
        model.jepa_parameters()
        + model.head_parameters()
        + model.done_head_parameters()
        + list(sigreg.parameters())
    )
    opt = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=args.wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.set_train(); sigreg.train()
        t_loss = 0.0
        for obs, action, reward, next_obs, done, mc, goal_obs in tr_loader:
            loss = _forward(model, sigreg, obs, action, reward, next_obs,
                            done, mc, goal_obs, args, device)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.jepa_parameters() + model.head_parameters()
                + model.done_head_parameters(), 1.0)
            opt.step()
            t_loss += loss.item()
        t_loss /= len(tr_loader)
        sch.step()

        model._eval_mode(); sigreg.eval()
        v_loss = 0.0
        with torch.no_grad():
            for obs, action, reward, next_obs, done, mc, goal_obs in va_loader:
                v_loss += _forward(model, sigreg, obs, action, reward, next_obs,
                                   done, mc, goal_obs, args, device).item()
        v_loss /= len(va_loader)

        print(f"Epoch {epoch:3d}/{args.epochs}  train={t_loss:.4f}  val={v_loss:.4f}")
        if v_loss < best_val:
            best_val = v_loss
            _bake_mean_goal_latent(model, ds, device)
            model.save_goal_cond(args.output)
            print(f"  saved -> {args.output}")

    print(f"\nDone. Best val: {best_val:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None, help="Warm-start JEPA+heads from existing checkpoint")
    p.add_argument("--data",       required=True)
    p.add_argument("--output",     required=True)
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--batch-size", type=int,   default=64)
    p.add_argument("--lr",         type=float, default=5e-5)
    p.add_argument("--wd",         type=float, default=1e-3)
    p.add_argument("--gamma",      type=float, default=0.99)
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--w-jepa",  type=float, default=1.0)
    p.add_argument("--w-sig",   type=float, default=0.09)
    p.add_argument("--w-rew",   type=float, default=1.0)
    p.add_argument("--w-done",  type=float, default=1.0)
    p.add_argument("--w-val",   type=float, default=0.5)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
