"""Train inverter v0 (TEXT2PRESET_PLAN.md §4) on a stage-1 set.

Batches mix human and synthetic rows as they come (the human train presets are ~3% of the pilot set);
AdamW, cosine learning rate, mel augmentation. Each epoch: loss on the held-out synthetic rows (2%) and on
the human val presets (data/inverter/eval, split "val"), plus the accuracy of key switches on those; the
model with the lowest human-val loss is kept (the inverter has to work on real presets, not only on the
sampler's own distribution). The sound score (render -> invert -> render) is pitcher.inverter.evaluate.

Writes  runs/inverter/<run>/{model.pt, config.json, metrics.csv}
Run:    .venv/bin/python -m pitcher.inverter.train --run v0 --data pilot [--epochs 20 --batch 128]
        (first time: .venv/bin/python -m pitcher.inverter.data   builds the human val / test renders)
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from pitcher.core.surge_io import PITCHER_ROOT
from pitcher.inverter.data import MelSet, load_eval
from pitcher.inverter.model import Inverter, augment
from pitcher.step4_vae.data import Spec
from pitcher.step4_vae.model import recon_terms

RUNS = PITCHER_ROOT / "runs" / "inverter"
KEY_SWITCHES = ["scene/osc/0/type", "scene/filterunit/0/model", "scene/filterblock_configuration", "scene/wsunit/type"]


def to_t(a):
    """numpy -> torch without the numpy C API (torch 2.2.2 + numpy 2): through a byte buffer."""
    a = np.ascontiguousarray(a)
    dt = {np.dtype(np.float16): torch.float16, np.dtype(np.float32): torch.float32, np.dtype(np.int64): torch.int64}[a.dtype]
    return torch.frombuffer(bytearray(a.tobytes()), dtype=dt).reshape(a.shape)


def batch(mel, t, idx):
    b = {k: to_t(v[idx]) for k, v in t.items()}
    return to_t(mel[idx]).float(), b


@torch.no_grad()
def evaluate(model, mel, t, spec, idx, bs=256):
    model.eval()
    parts, acc = [], {k: [] for k in KEY_SWITCHES if k in spec.cat}
    for i in range(0, len(idx), bs):
        x, b = batch(mel, t, idx[i:i + bs])
        out, _ = model(x)
        mse, bce, ce = recon_terms(out, b)
        parts.append(torch.stack([mse, bce, ce], 1))
        for k in acc:
            j = spec.cat.index(k)
            m = b["m_cat"][:, j] > 0
            acc[k] += (out[2][j].argmax(-1)[m] == b["x_cat"][m, j]).tolist()
    p = torch.cat(parts).mean(0).tolist()
    return dict(loss=sum(p), mse=p[0], bce=p[1], ce=p[2]), {k.split("/")[-2] + "/" + k.split("/")[-1]: float(np.mean(v)) for k, v in acc.items() if v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v0")
    ap.add_argument("--data", default="pilot")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=0, help="subsample the training rows (smoke tests)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    out = RUNS / args.run
    out.mkdir(parents=True, exist_ok=True)

    spec = Spec()
    t0 = time.time()
    ds = MelSet(args.data, spec)
    ev_meta, ev_mel, _, ev_t = load_eval(spec)
    rng0 = np.random.default_rng(args.seed + 1)
    val_h = np.flatnonzero((ev_meta.split == "val").to_numpy())
    tr = np.flatnonzero(~ds.is_val)
    if args.max_rows:
        tr = np.sort(rng0.choice(tr, min(args.max_rows, len(tr)), replace=False))
    va = np.flatnonzero(ds.is_val)
    print(f"data: {len(tr)} train rows ({(ds.meta.source.iloc[tr] == 'human').sum()} human), {len(va)} synthetic val, "
          f"{len(val_h)} human val presets; loaded in {time.time() - t0:.0f}s", flush=True)

    model = Inverter(len(spec.cont), len(spec.bool), spec.cat_sizes, dropout=args.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * (len(tr) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    cfg = dict(vars(args), n_cont=len(spec.cont), n_bool=len(spec.bool), cat_sizes=spec.cat_sizes,
               n_params=sum(p.numel() for p in model.parameters()), n_train=len(tr))
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    rng = np.random.default_rng(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    best, rows = float("inf"), []
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = rng.permutation(tr)
        te, losses = time.time(), []
        for i in range(0, len(perm) - args.batch + 1, args.batch):
            x, b = batch(ds.mel, ds.t, np.sort(perm[i:i + args.batch]))
            out_, _ = model(augment(x, gen))
            mse, bce, ce = recon_terms(out_, b)
            loss = (mse + bce + ce).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
        sv, _ = evaluate(model, ds.mel, ds.t, spec, va)
        hv, acc = evaluate(model, ev_mel, ev_t, spec, val_h)
        row = dict(epoch=ep, train=float(np.mean(losses)), syn_val=sv["loss"], human_val=hv["loss"],
                   human_mse=hv["mse"], human_bce=hv["bce"], human_ce=hv["ce"], **{f"acc_{k}": v for k, v in acc.items()},
                   seconds=time.time() - te)
        rows.append(row)
        pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
        if hv["loss"] < best:
            best = hv["loss"]
            torch.save(model.state_dict(), out / "model.pt")
        print(f"epoch {ep:3d} | train {row['train']:.2f} | synthetic val {sv['loss']:.2f} | human val {hv['loss']:.2f} "
              f"(mse {hv['mse']:.2f} bce {hv['bce']:.2f} ce {hv['ce']:.2f}) | "
              + " ".join(f"{k} {v:.2f}" for k, v in acc.items()) + f" | {row['seconds']:.0f}s"
              + (" *" if hv["loss"] == best else ""), flush=True)
    print(f"best human-val loss {best:.2f}; model: {out / 'model.pt'}")


if __name__ == "__main__":
    main()
