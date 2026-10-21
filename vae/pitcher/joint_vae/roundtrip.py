"""Round-trip check of a VAE decoder in sound: preset -> encoder -> z -> decoder -> preset -> render.

Per human val / test preset (never seen in training): encode, decode, render the decoded preset with
ANOTHER seed (id + 1007) and compare with the stored render of the original (seed id + 7; level-matched mel
distance, stage1.exact). Same character = <= 6.7 dB (Surge vs Surge, Round 14). Tells whether a point in z
decodes into a preset that sounds like it: the ceiling of text -> z -> decoder generation.
  joint, params -> mean         z = posterior mean from the parameters only, most likely values
  joint, both -> mean           z from parameters + sound code
  joint, params best of 8       the mean + 7 draws from the posterior, the closest render counts
                                (propose -> render -> select with 8 proposals)
  old VAE (h256_d01) -> mean    the parameter-only CVAE (pitcher/step4_vae), same presets
  oracle                        the true preset with the other seed (runs/inverter/v0/eval)

Writes  runs/joint_vae/<run>/roundtrip/{per_preset.csv, summary.json}
Run:    .venv/bin/python -m pitcher.joint_vae.roundtrip --run j48 [--old h256_d01 --workers 4]
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch

from pitcher.core.surge_io import PITCHER_ROOT
from pitcher.inverter.evaluate import _task
from pitcher.inverter.train import to_t
from pitcher.joint_vae.data import JSpec, load_eval
from pitcher.joint_vae.evaluate import INV, posterior
from pitcher.joint_vae.train import RUNS, load_model
from pitcher.synth_data.common import CHARACTER_DB, run_pool

SEED_OFFSET = 1000
N_DRAWS = 7


@torch.no_grad()
def decode_rows(model, js, z, y1):
    (num, bl, cats), _ = model.decode(z, y1)
    return [js.to_row(num[i].tolist(), (bl[i] > 0).float().tolist(), [int(c[i].argmax()) for c in cats]) for i in range(len(z))]


@torch.no_grad()
def old_vae_rows(run, ids):
    """Decoded presets of the parameter-only CVAE for these preset ids (val / test)."""
    from pitcher.step4_vae.data import Spec, load
    from pitcher.step4_vae.eval import rows_from
    from pitcher.step4_vae.train import tensors
    from pitcher.step6_sample.latent import load_model as load_old
    model, _ = load_old(run)
    spec, data = load(Spec())
    rows = {}
    for split in ("val", "test"):
        t = tensors(data[split])
        mu, _ = model.encode(t["x_in"], t["y1h"])
        cont, bl, cats = model.decode(mu, t["y1h"])
        rows.update(zip(data[split]["ids"].tolist(), rows_from(spec, cont, bl, cats)))
    return [rows.get(int(i)) for i in ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="j48")
    ap.add_argument("--old", default="h256_d01", help="parameter-only VAE run to compare ('' to skip)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    out = RUNS / args.run / "roundtrip"
    out.mkdir(parents=True, exist_ok=True)
    js = JSpec()
    model, _ = load_model(args.run)
    meta, mel, codes, t, y, a = load_eval(js)
    if args.limit:
        meta, mel, a, y = meta.iloc[:args.limit], mel[:args.limit], a[:args.limit], y[:args.limit]
        t = {k: v[:args.limit] for k, v in t.items()}
    ids = meta.id.to_numpy()
    seeds = ids + 7 + SEED_OFFSET

    variants = {}
    mu_p, lv_p, y1 = posterior(model, b=t, y=y)
    variants["joint_params"] = decode_rows(model, js, mu_p, y1)
    mu_b, _, _ = posterior(model, a=a, b=t, y=y)
    variants["joint_both"] = decode_rows(model, js, mu_b, y1)
    g = torch.Generator().manual_seed(0)
    for k in range(N_DRAWS):
        z = mu_p + torch.randn(mu_p.shape, generator=g) * (0.5 * lv_p).exp()
        variants[f"draw{k}"] = decode_rows(model, js, z, y1)
    if args.old:
        variants["old_vae"] = old_vae_rows(args.old, ids)

    tasks = [[((q, v), rows[q], int(seeds[q]), mel[q]) for v, rows in variants.items() if rows[q] is not None]
             for q in range(len(ids))]
    D = {}
    for res in run_pool(_task, tasks, args.workers, "round trip"):
        D.update(dict(res))
    inv = pd.read_csv(INV).set_index("preset_id")
    rows = []
    for q in range(len(ids)):
        draws = [D[(q, f"draw{k}")] for k in range(N_DRAWS)]
        rows.append(dict(preset_id=int(ids[q]), split=meta.split.iloc[q], label=meta.label.iloc[q],
                         oracle=inv.loc[int(ids[q]), "oracle"], joint_params=D[(q, "joint_params")], joint_both=D[(q, "joint_both")],
                         joint_params_best8=np.nanmin([D[(q, "joint_params")]] + draws),
                         old_vae=D.get((q, "old_vae"), np.nan)))
    P = pd.DataFrame(rows)
    P.to_csv(out / "per_preset.csv", index=False)
    cols = ["oracle", "old_vae", "joint_params", "joint_both", "joint_params_best8"]
    summ = dict(run=args.run, old=args.old, n=len(P), threshold_db=CHARACTER_DB)
    for c in cols:
        summ[c] = dict(same_character=float((P[c] <= CHARACTER_DB).mean()), median_db=float(P[c].median()),
                       p90_db=float(P[c].quantile(0.9)), silent=int(P[c].isna().sum()))
    summ["by_label_same_character"] = P.groupby("label")[cols].apply(lambda S: (S <= CHARACTER_DB).mean()).round(3).to_dict(orient="index")
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(f"\n{len(P)} val + test presets; same character = <= {CHARACTER_DB} dB (decoded preset, other seed, vs the original)")
    for c in cols:
        s = summ[c]
        print(f"  {c:20s} same character {s['same_character']:.0%}  median {s['median_db']:.1f} dB  90th {s['p90_db']:.1f} dB  silent {s['silent']}")
    pd.set_option("display.width", 200)
    print(pd.DataFrame(summ["by_label_same_character"]).T.to_string())
    print(f"written: {out}")


if __name__ == "__main__":
    main()
