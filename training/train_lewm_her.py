"""
Option 3 — Train LeWM with Hindsight Experience Replay (HER) value targets.

Identical to train_lewm.py (full end-to-end JEPA + heads) except:
  - Failed episodes get augmented value targets:
      V_HER(t) = gamma^(steps_to_ep_end - t) * her_val
      V_target(t) = max(MC_return(t), V_HER(t))
  - Successful episodes: V_target(t) = MC_return(t) (unchanged)
  - Higher value loss weight (w_val=5.0 default) to leverage the denser signal

This gives the value head gradient signal in failed episodes, nudging it to
predict higher values for states closer to episode end (which are also likely
closer to the goal in goal-conditioned tasks).

Usage:
    python training/train_lewm_her.py \
        --checkpoint lewm_fixed_minigrid_empty_8x8.pt \
        --data       data/minigrid_empty_8x8.npz \
        --output     checkpoints/lewm_her_minigrid_empty_8x8.pt
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
from training.train_dino_wm import TransitionDataset


def _compute_her_returns(reward: np.ndarray, done: np.ndarray,
                         gamma: float = 0.99, her_val: float = 0.1) -> np.ndarray:
    """
    Augmented value targets:
      - Successful episodes: MC return (unchanged)
      - Failed episodes: max(MC_return, gamma^(T-t) * her_val)
    """
    # Standard MC returns
    mc = np.zeros_like(reward)
    running = 0.0
    for i in reversed(range(len(reward))):
        running = reward[i] + gamma * running * (1.0 - float(done[i]))
        mc[i]   = running

    # Episode boundaries
    ends   = np.where(done)[0]
    starts = np.concatenate([[0], ends[:-1] + 1])

    her = mc.copy()
    for ep_start, ep_end in zip(starts, ends):
        # Only augment failed episodes (no positive reward in episode)
        if reward[ep_start:ep_end + 1].max() <= 0:
            ep_len = ep_end - ep_start + 1
            for t in range(ep_len):
                steps_remaining = ep_len - 1 - t
                her_target = gamma ** steps_remaining * her_val
                her[ep_start + t] = max(mc[ep_start + t], her_target)

    return her.astype(np.float32)


class HERDataset(Dataset):
    def __init__(self, npz_path: str, gamma: float = 0.99, her_val: float = 0.1):
        base = TransitionDataset(npz_path, gamma)
        self.obs       = base.obs
        self.action    = base.action
        self.reward    = base.reward
        self.next_obs  = base.next_obs
        self.done      = base.done
        self.n_actions = int(self.action.max()) + 1

        print("Computing HER value targets ...")
        self.mc_return = _compute_her_returns(self.reward, self.done, gamma, her_val)
        n_success = int(self.done[self.reward > 0].sum())
        orig_mc   = TransitionDataset.__new__(TransitionDataset)
        orig_mc.mc_return = base.mc_return
        delta = np.mean(self.mc_return - base.mc_return)
        print(f"  {n_success} successful episodes; mean V_target lift = {delta:.4f}")

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


def _forward(model, sigreg, obs, action, reward, next_obs, done, mc, args, device):
    obs      = obs.to(device)
    next_obs = next_obs.to(device)
    action   = torch.as_tensor(action, device=device)
    reward   = torch.as_tensor(reward, dtype=torch.float32, device=device)
    done     = torch.as_tensor(done,   dtype=torch.float32, device=device)
    mc       = torch.as_tensor(mc,     dtype=torch.float32, device=device)

    def _enc(x):
        out = model._jepa.encoder(pixel_values=x, interpolate_pos_encoding=True)
        return model._jepa.projector(out.last_hidden_state[:, 0])

    z_t    = _enc(obs)
    z_next = _enc(next_obs)

    a_oh   = F.one_hot(action, num_classes=model.n_actions).float().unsqueeze(1)
    a_emb  = model._jepa.action_encoder(a_oh)
    pred   = model._jepa.predictor(z_t.unsqueeze(1), a_emb)
    z_pred = model._jepa.pred_proj(pred[:, -1])

    l_jepa = F.mse_loss(z_pred, z_next.detach())
    l_sig  = sigreg(torch.stack([z_t, z_pred], dim=0))

    done_pw_t = torch.tensor([args.done_pw], device=device)
    rew_w   = torch.where(reward > 0, reward.new_full(reward.shape, args.reward_pw), torch.ones_like(reward))
    l_rew   = (rew_w * (model._reward(z_pred) - reward) ** 2).mean()
    l_done  = F.binary_cross_entropy_with_logits(model._done(z_pred), done, pos_weight=done_pw_t)
    l_value = F.mse_loss(model._value(z_t), mc)

    return (
        args.w_jepa * l_jepa
        + args.w_sig  * l_sig
        + args.w_rew  * l_rew
        + args.w_done * l_done
        + args.w_val  * l_value
    )


def train(args):
    device = torch.device(args.device)

    ds     = HERDataset(args.data, gamma=args.gamma, her_val=args.her_val)
    n_val  = max(1, int(len(ds) * 0.1))
    tr_ds, va_ds = random_split(ds, [len(ds) - n_val, n_val],
                                generator=torch.Generator().manual_seed(42))
    kw = dict(batch_size=args.batch_size, num_workers=2, pin_memory=True)
    tr_loader = DataLoader(tr_ds, shuffle=True,  **kw)
    va_loader = DataLoader(va_ds, shuffle=False, **kw)

    model  = LeWM(n_actions=ds.n_actions,
                  checkpoint_path=args.checkpoint,
                  device=str(device))
    sigreg = SIGReg(knots=17, num_proj=512).to(device)

    # Class-imbalance weights
    done_pos_rate   = float(ds.done.astype(float).mean())
    args.done_pw    = (1.0 - done_pos_rate)   / max(done_pos_rate, 1e-6)
    reward_pos_rate = float((ds.reward > 0).mean())
    args.reward_pw  = (1.0 - reward_pos_rate) / max(reward_pos_rate, 1e-6)

    print(f"\nHER LeWM fine-tune | epochs={args.epochs} | device={device}")
    print(f"  done_pw={args.done_pw:.1f}  reward_pw={args.reward_pw:.1f}")
    print(f"  w_val={args.w_val}  train={len(tr_ds)}  val={len(va_ds)}\n")

    opt = torch.optim.AdamW(
        model.jepa_parameters() + model.head_parameters() + list(sigreg.parameters()),
        lr=args.lr, weight_decay=args.wd,
    )
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.set_train(); sigreg.train()
        t_loss = 0.0
        for batch in tr_loader:
            loss = _forward(model, sigreg, *batch, args, device)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.jepa_parameters() + model.head_parameters(), 1.0)
            opt.step()
            t_loss += loss.item()
        t_loss /= len(tr_loader)
        sch.step()

        model._eval_mode(); sigreg.eval()
        v_loss = 0.0
        with torch.no_grad():
            for batch in va_loader:
                v_loss += _forward(model, sigreg, *batch, args, device).item()
        v_loss /= len(va_loader)

        print(f"Epoch {epoch:3d}/{args.epochs}  train={t_loss:.4f}  val={v_loss:.4f}")
        if v_loss < best_val:
            best_val = v_loss
            model.save(args.output)
            print(f"  saved -> {args.output}")

    print(f"\nDone. Best val: {best_val:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None, help="Start from existing checkpoint (fine-tune)")
    p.add_argument("--data",       required=True)
    p.add_argument("--output",     required=True)
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--batch-size", type=int,   default=128)
    p.add_argument("--lr",         type=float, default=5e-5)
    p.add_argument("--wd",         type=float, default=1e-3)
    p.add_argument("--gamma",      type=float, default=0.99)
    p.add_argument("--her-val",    type=float, default=0.1, help="Synthetic HER reward magnitude")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--w-jepa",  type=float, default=1.0)
    p.add_argument("--w-sig",   type=float, default=0.09)
    p.add_argument("--w-rew",   type=float, default=1.0)
    p.add_argument("--w-done",  type=float, default=1.0)
    p.add_argument("--w-val",   type=float, default=5.0, help="Higher than default to use denser HER signal")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
