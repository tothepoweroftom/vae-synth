"""Step 5: train the CVAE (PIPELINE_VAE_SURGE.md, Step 5).

AdamW (weight decay optional), category-balanced batches (each label drawn equally often), KL warm-up from 0 to beta_max,
optional free bits, early stopping on val reconstruction. Logged per epoch: train loss, val
reconstruction (MSE / BCE / CE), KL, active latent units, val accuracy of key switches.

Writes  runs/vae/<run>/model.pt, config.json, metrics.csv
Run:    .venv/bin/python -m pitcher.step4_vae.train --run base [--z 12 --beta 0.5 --warmup 100 --epochs 600]
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from pitcher.step4_vae.model import CVAE, loss_fn, recon_terms
from pitcher.core.surge_io import PITCHER_ROOT
from pitcher.step4_vae.data import LABELS, load

RUNS = PITCHER_ROOT / "runs" / "vae"
KEY_SWITCHES = ["scene/osc/0/type", "scene/filterunit/0/model", "scene/polymode", "scene/filterblock_configuration"]


def tensors(d):
    """numpy arrays -> torch tensors via lists (no numpy bridge in torch 2.2.2 + numpy 2)."""
    out = {}
    for k, v in d.items():
        if k == "ids":
            continue
        dtype = torch.long if v.dtype.kind == "i" else torch.float32
        out[k] = torch.tensor(v.tolist(), dtype=dtype)
    out["y1h"] = F.one_hot(out["y"], len(LABELS)).float()
    return out


def batch(t, idx):
    return {k: v[idx] for k, v in t.items()}


@torch.no_grad()
def evaluate(model, t, spec):
    model.eval()
    mu, logvar = model.encode(t["x_in"], t["y1h"])
    out = model.decode(mu, t["y1h"])
    mse, bce, ce = recon_terms(out, t)
    kl_dim = (0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)).mean(0)
    active = int((mu.var(0) > 0.01).sum())
    acc = {}
    for name in KEY_SWITCHES:
        if name in spec.cat:
            j = spec.cat.index(name)
            m = t["m_cat"][:, j] > 0
            acc[name] = float((out[2][j].argmax(-1)[m] == t["x_cat"][m, j]).float().mean()) if m.any() else float("nan")
    return dict(val_rec=float((mse + bce + ce).mean()), val_mse=float(mse.mean()), val_bce=float(bce.mean()),
                val_ce=float(ce.mean()), val_kl=float(kl_dim.sum()), active_units=active,
                kl_per_dim=[round(float(k), 3) for k in kl_dim]), acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="base")
    ap.add_argument("--z", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--free_bits", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--patience", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    spec, data = load()
    tr, va = tensors(data["train"]), tensors(data["val"])
    model = CVAE(spec.n_in, len(spec.cont), len(spec.bool), spec.cat_sizes, len(LABELS), args.z, args.hidden, args.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    run_dir = RUNS / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(vars(args), n_in=spec.n_in, n_cont=len(spec.cont), n_bool=len(spec.bool), n_cat=len(spec.cat),
               cat_sizes=spec.cat_sizes, n_train=len(data["train"]["y"]), n_val=len(data["val"]["y"]),
               n_params=sum(p.numel() for p in model.parameters()))
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=1))
    print(f"train {cfg['n_train']}, val {cfg['n_val']}; input {spec.n_in}, outputs cont {cfg['n_cont']} / bool "
          f"{cfg['n_bool']} / cat {cfg['n_cat']} ({sum(spec.cat_sizes)} logits); {cfg['n_params']:,} parameters")

    y = data["train"]["y"]
    weights = 1.0 / np.bincount(y)[y]
    weights = weights / weights.sum()
    n = len(y)
    best, best_epoch, rows, t0 = float("inf"), 0, [], time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        beta = args.beta * min(1.0, epoch / max(1, args.warmup))
        idx_all = rng.choice(n, size=n, replace=True, p=weights)   # category-balanced epoch
        losses = []
        for s in range(0, n, args.batch):
            b = batch(tr, torch.tensor(idx_all[s:s + args.batch].tolist()))
            out, mu, logvar = model(b["x_in"], b["y1h"])
            loss, parts, _ = loss_fn(out, mu, logvar, b, beta, args.free_bits)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(parts)
        ev, acc = evaluate(model, va, spec)
        row = dict(epoch=epoch, beta=beta, train_loss=float(np.mean([p["rec"] + beta * p["kl"] for p in losses])),
                   train_rec=float(np.mean([p["rec"] for p in losses])), train_kl=float(np.mean([p["kl"] for p in losses])),
                   **{k: v for k, v in ev.items() if k != "kl_per_dim"}, **{f"acc:{k}": v for k, v in acc.items()})
        rows.append(row)
        if epoch > args.warmup and ev["val_rec"] < best:  # select only once beta has reached its target
            best, best_epoch = ev["val_rec"], epoch
            torch.save(model.state_dict(), run_dir / "model.pt")
            (run_dir / "best.json").write_text(json.dumps(dict(epoch=epoch, **ev, acc=acc), indent=1))
        if epoch % 25 == 0 or epoch == 1:
            print(f"  epoch {epoch:4d} beta {beta:.3f} | train rec {row['train_rec']:.2f} kl {row['train_kl']:.2f} | "
                  f"val rec {ev['val_rec']:.2f} (mse {ev['val_mse']:.2f} bce {ev['val_bce']:.2f} ce {ev['val_ce']:.2f}) "
                  f"kl {ev['val_kl']:.2f} active {ev['active_units']}/{args.z} | {time.time() - t0:.0f}s", flush=True)
        if epoch > args.warmup and epoch - best_epoch > args.patience:
            print(f"  early stop at epoch {epoch} (best {best_epoch})")
            break
    pd.DataFrame(rows).to_csv(run_dir / "metrics.csv", index=False)
    b = json.loads((run_dir / "best.json").read_text())
    print(f"\nbest epoch {b['epoch']}: val rec {b['val_rec']:.2f} (mse {b['val_mse']:.2f}, bce {b['val_bce']:.2f}, "
          f"ce {b['val_ce']:.2f}), kl {b['val_kl']:.2f}, active units {b['active_units']}/{args.z}")
    print("KL per latent dim:", b["kl_per_dim"])
    print("val accuracy of key switches:", {k.split("/")[-2] + "/" + k.split("/")[-1]: round(v, 3) for k, v in b["acc"].items()})


if __name__ == "__main__":
    main()
