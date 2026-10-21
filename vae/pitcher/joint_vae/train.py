"""Train the joint VAE on the stage-1 set (human train presets + synthetic rows) on the Mac (CPU).

Modality dropout per row: both experts 50%, parameters only 25%, sound only 25%; both decoders are always
scored. beta warms up from 50% of its target (Le Vaillant 2021: from 0 hurt validation) over --warmup
epochs. Each epoch, on the human val presets (data/inverter/eval, split "val") and the held-out synthetic
rows, with z = posterior mean:
  invert   sound only -> parameters: parameter loss (the inverter role)
  hear     parameters only -> sound code: sound loss (is z sound-aware?)
  both     both experts: parameter + sound loss
Kept model: lowest human-val (invert params + hear sound); early stop after --patience epochs.
--weights screening: sound-weighted parameter loss (weights.py: per-feature audibility from the stage-0
screening, mean 1 per kind); the human-val score then uses the same weights.

Writes  runs/joint_vae/<run>/{model.pt, config.json, metrics.csv}
Run:    .venv/bin/python -m pitcher.joint_vae.train --run j48 [--epochs 150 --z 32 --beta 0.25]
        .venv/bin/python -m pitcher.joint_vae.train --run j48w --weights screening [--w-power 1]
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from pitcher.core.surge_io import PITCHER_ROOT
from pitcher.inverter.train import to_t
from pitcher.joint_vae.data import N_CLS, JointSet, JSpec, load_eval
from pitcher.joint_vae import weights as W
from pitcher.joint_vae.model import JointVAE, loss_fn, recon

RUNS = PITCHER_ROOT / "runs" / "joint_vae"
MODES = {"invert": (0.0, 1.0), "hear": (1.0, 0.0), "both": (1.0, 1.0)}


def batch(t, a, y, idx):
    return {k: to_t(v[idx]) for k, v in t.items()}, to_t(a[idx]), to_t(y[idx])


@torch.no_grad()
def evaluate(model, t, a, y, idx, bs=512, w=None):
    model.eval()
    res = {}
    for mode, (p, s) in MODES.items():
        P, S = [], []
        for i in range(0, len(idx), bs):
            b, aa, yy = batch(t, a, y, idx[i:i + bs])
            n = len(yy)
            params, sound, _, _ = model(b, aa, yy, torch.full((n,), p), torch.full((n,), s), sample=False)
            mse, bce, ce = recon(params, b, w)
            P += (mse + bce + ce).tolist()
            S += (0.5 * ((sound - aa) ** 2).sum(-1)).tolist()
        res[f"{mode}_params"], res[f"{mode}_sound"] = float(np.mean(P)), float(np.mean(S))
    return res


def build(cfg):
    return JointVAE(cfg["n_num"], cfg["n_bool"], cfg["cat_sizes"], k_sound=cfg["k"], n_cls=N_CLS,
                    z_dim=cfg["z"], hidden=cfg["hidden"], dropout=cfg["dropout"])


def load_model(run):
    cfg = json.loads((RUNS / run / "config.json").read_text())
    m = build(cfg)
    m.load_state_dict(torch.load(RUNS / run / "model.pt"))
    return m.eval(), cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="j48")
    ap.add_argument("--data", default="pilot")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--z", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--beta", type=float, default=0.25)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--warmup-start", type=float, default=0.5, help="beta starts at this share of its target")
    ap.add_argument("--free-bits", type=float, default=0.05)
    ap.add_argument("--w-sound", type=float, default=1.0)
    ap.add_argument("--p-both", type=float, default=0.5)
    ap.add_argument("--weights", default="none", choices=["none", "screening"], help="sound-weighted parameter loss")
    ap.add_argument("--w-power", type=float, default=1.0, help="weight = (dB / median dB) ^ power")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    out = RUNS / args.run
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    js = JSpec()
    ds = JointSet(args.data, js)
    ev_meta, _, _, ev_t, ev_y, ev_a = load_eval(js)
    val_h = np.flatnonzero((ev_meta.split == "val").to_numpy())
    tr, va = np.flatnonzero(~ds.is_val), np.flatnonzero(ds.is_val)
    print(f"data: {len(tr)} train rows ({(ds.meta.source.iloc[tr] == 'human').sum()} human), {len(va)} synthetic val, "
          f"{len(val_h)} human val; {len(js.num)} num, {len(js.bool)} bool, {len(js.cat)} cat ({js.n_ord} stepped knobs); "
          f"loaded in {time.time() - t0:.0f}s", flush=True)

    cfg = dict(vars(args), n_num=len(js.num), n_bool=len(js.bool), cat_sizes=js.cat_sizes, k=ds.a.shape[1], n_train=len(tr))
    w = W.tensors(W.fit(js, args.w_power)) if args.weights == "screening" else None
    model = build(cfg)
    cfg["n_params"] = sum(p.numel() for p in model.parameters())
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    rng = np.random.default_rng(args.seed)
    best, best_ep, rows = float("inf"), 0, []
    p_one = (1 - args.p_both) / 2
    for ep in range(1, args.epochs + 1):
        te = time.time()
        beta = args.beta * min(1.0, args.warmup_start + (1 - args.warmup_start) * (ep - 1) / max(1, args.warmup))
        model.train()
        order = rng.permutation(tr)
        acc = []
        for i in range(0, len(order) - args.batch + 1, args.batch):
            b, a, y = batch(ds.t, ds.a, ds.y, order[i:i + args.batch])
            r = torch.rand(len(y))
            use_p = torch.where(r < p_one, 0.0, 1.0)          # r < p_one: sound only
            use_a = torch.where(r >= 1 - p_one, 0.0, 1.0)     # r >= 1 - p_one: parameters only
            params, sound, mu, lv = model(b, a, y, use_p, use_a)
            loss, parts = loss_fn(params, sound, mu, lv, b, a, beta, args.w_sound, args.free_bits, w)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            acc.append(parts)
        sched.step()
        tr_m = {k: float(np.mean([p[k] for p in acc])) for k in acc[0]}
        hv = evaluate(model, ev_t, ev_a, ev_y, val_h, w=w)
        sv = evaluate(model, ds.t, ds.a, ds.y, va, w=w)
        score = hv["invert_params"] + args.w_sound * hv["hear_sound"]
        row = dict(epoch=ep, beta=beta, **{f"train_{k}": v for k, v in tr_m.items()}, **{f"human_{k}": v for k, v in hv.items()},
                   **{f"syn_{k}": v for k, v in sv.items()}, score=score, s=time.time() - te)
        rows.append(row)
        pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
        mark = ""
        if score < best:
            best, best_ep, mark = score, ep, " *"
            torch.save(model.state_dict(), out / "model.pt")
        print(f"epoch {ep:3d} | train params {tr_m['params']:.1f} sound {tr_m['sound']:.1f} kl {tr_m['kl']:.1f} | "
              f"human: invert {hv['invert_params']:.1f} hear {hv['hear_sound']:.1f} both {hv['both_params']:.1f}/{hv['both_sound']:.1f} | "
              f"syn: invert {sv['invert_params']:.1f} hear {sv['hear_sound']:.1f} | {time.time() - te:.0f}s{mark}", flush=True)
        if ep - best_ep >= args.patience:
            print(f"early stop: no improvement for {args.patience} epochs")
            break
    print(f"best epoch {best_ep}, human-val score {best:.2f}; model: {out / 'model.pt'}")


if __name__ == "__main__":
    main()
