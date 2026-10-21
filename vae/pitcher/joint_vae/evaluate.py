"""Evaluate a joint VAE on the human val / test presets (never seen in training).

1. Inverter role (same protocol as pitcher.inverter.evaluate): sound code only -> posterior -> 16 candidate
   presets (0: posterior mean, most likely values; 1-7: z drawn from the posterior; 8-15: posterior mean
   with sampled switches) -> rendered with ANOTHER seed (id + 1007) -> level-matched mel distance to the
   target; the closest counts. Oracle, retrieval and inverter v0 from runs/inverter/v0/eval/per_preset.csv
   (same presets, same seeds). Same character = <= 6.7 dB (Surge vs Surge, Round 14).
2. z probe (as step4_vae.zprobe, same measures from runs/vae/h256_d01/zprobe/measures.parquet): ridge and
   boosted probes fit on the 1,375 train presets, R^2 on the 344 val + test presets, from z of the
   parameters only, the sound only, or both; reference: the parameter-only VAE's z + label.

Writes  runs/joint_vae/<run>/eval/{per_preset.csv, zprobe.csv, summary.json, examples/*.flac}
Run:    .venv/bin/python -m pitcher.joint_vae.evaluate --run j48 [--workers 4 --limit 0]
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from pitcher.core.surge_io import PITCHER_ROOT
from pitcher.inverter.evaluate import _example, _task, snap_routes
from pitcher.inverter.train import KEY_SWITCHES, to_t
from pitcher.joint_vae.data import JointSet, JSpec, load_eval
from pitcher.joint_vae.train import RUNS, load_model
from pitcher.step4_vae.data import LABELS
from pitcher.step4_vae.zprobe import MEASURES, r2
from pitcher.synth_data.common import CHARACTER_DB, run_pool

SEED_OFFSET = 1000
INV = PITCHER_ROOT / "runs" / "inverter" / "v0" / "eval" / "per_preset.csv"
ZREF = PITCHER_ROOT / "runs" / "vae" / "h256_d01" / "zprobe"


@torch.no_grad()
def posterior(model, a=None, b=None, y=None):
    n = len(y)
    y1 = F.one_hot(to_t(y), model.n_cls).float()
    bb = {k: to_t(v) for k, v in b.items()} if b is not None else None
    use_p = torch.full((n,), 1.0 if b is not None else 0.0)
    use_a = torch.full((n,), 1.0 if a is not None else 0.0)
    mu, lv = model.posterior(bb, to_t(a) if a is not None else None, y1, use_p, use_a)
    return mu, lv, y1


@torch.no_grad()
def candidates(model, js, a, y, n=16, seed=0):
    """sound codes (B, k) + labels (B,) -> [[full codes row] * n] * B."""
    g = torch.Generator().manual_seed(seed)
    mu, lv, y1 = posterior(model, a=a, y=y)
    n_post = min(7, n - 1)
    out = [[None] * n for _ in range(len(y))]
    for k in range(n):
        z = mu + torch.randn(mu.shape, generator=g) * (0.5 * lv).exp() if 1 <= k <= n_post else mu
        (num, bl, cats), _ = model.decode(z, y1)
        sample = k > n_post
        for i in range(len(y)):
            if sample:
                bo = (torch.rand(bl[i].shape, generator=g) < torch.sigmoid(bl[i])).float()
                ci = [int(torch.multinomial(torch.softmax(c[i], -1), 1, generator=g)) for c in cats]
            else:
                bo = (bl[i] > 0).float()
                ci = [int(c[i].argmax()) for c in cats]
            out[i][k] = snap_routes(js.spec, js.to_row(num[i].tolist(), bo.tolist(), ci))
    return out


def zprobe(model, js, ds, ev_meta, ev_t, ev_y, ev_a):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression, RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    df = pd.read_parquet(ZREF / "measures.parquet")
    orig = df[df.kind == "orig"].assign(decay=lambda d: d.decay.clip(-60, 0)).set_index("preset_id")
    hum = np.flatnonzero((ds.meta.source == "human").to_numpy())
    tr_ids, te_ids = ds.meta.base.iloc[hum].to_numpy().astype(int), ev_meta.id.to_numpy().astype(int)
    tr = dict(t={k: v[hum] for k, v in ds.t.items()}, a=ds.a[hum], y=ds.y[hum])
    te = dict(t=ev_t, a=ev_a, y=ev_y)
    zs = {}
    for name, use in (("z from params", "p"), ("z from sound", "a"), ("z from both", "pa")):
        f = lambda d: posterior(model, a=d["a"] if "a" in use else None, b=d["t"] if "p" in use else None, y=d["y"])[0].tolist()
        zs[name] = (np.array(f(tr), np.float32), np.array(f(te), np.float32))
    onehot = lambda y: np.eye(len(LABELS) + 1)[y]
    sets = {"label": (onehot(tr["y"]), onehot(te["y"])), **zs,
            "z from params + label": (np.hstack([zs["z from params"][0], onehot(tr["y"])]), np.hstack([zs["z from params"][1], onehot(te["y"])]))}
    rows = []
    for name, (A, B) in sets.items():
        for target in MEASURES:
            ya, yb = orig.loc[tr_ids, target].to_numpy(), orig.loc[te_ids, target].to_numpy()
            ka, kb = np.isfinite(ya), np.isfinite(yb)
            for mname, make in (("linear", lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13)))),
                                ("boosted", lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=.05, max_leaf_nodes=15,
                                                                                  l2_regularization=1.0, random_state=0))):
                m = make().fit(A[ka], ya[ka])
                rows.append(dict(features=name, target=target, model=mname, score=r2(yb[kb], m.predict(B[kb]))))
        if name.startswith("z") and "label" not in name:
            m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)).fit(A, tr["y"])
            rows.append(dict(features=name, target="category", model="linear", score=float((m.predict(B) == te["y"]).mean())))
    P = pd.DataFrame(rows)
    best = P.groupby(["features", "target"]).score.max().unstack()
    ref = pd.read_csv(ZREF / "probe.csv")
    best.loc["old VAE: z + label"] = ref[ref.features == "z + label"].groupby("target").score.max()
    best.loc["old VAE: params"] = ref[ref.features == "params"].groupby("target").score.max()
    return P, best[MEASURES + ["category"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="j48")
    ap.add_argument("--data", default="pilot")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--examples", type=int, default=12)
    ap.add_argument("--no-render", action="store_true", help="z probe only")
    args = ap.parse_args()
    out = RUNS / args.run / "eval"
    (out / "examples").mkdir(parents=True, exist_ok=True)
    js = JSpec()
    model, cfg = load_model(args.run)
    meta, mel, codes, t, y, a = load_eval(js)
    ds = JointSet(args.data, js)
    summ = dict(run=args.run, n_candidates=args.n)

    P_z, best = zprobe(model, js, ds, meta, t, y, a)
    P_z.to_csv(out / "zprobe.csv", index=False)
    summ["zprobe_best_r2"] = best.round(3).to_dict(orient="index")
    pd.set_option("display.width", 200)
    print("z probe (best of linear / boosted; R^2, category = accuracy), val + test presets:")
    print(best.round(2).to_string())

    # switch accuracy, sound only -> most likely value
    mu, lv, y1 = posterior(model, a=a, y=y)
    with torch.no_grad():
        (_, _, cats), _ = model.decode(mu, y1)
    acc = {}
    for k in KEY_SWITCHES:
        j = js.cat.index(k)
        m = t["m_cat"][:, j] > 0
        acc[k.split("/")[-2] + "/" + k.split("/")[-1]] = float((np.array(cats[j].argmax(-1).tolist())[m] == t["x_cat"][m, j]).mean())
    summ["switch_accuracy"] = acc
    print("key switches from the sound (most likely value):", {k: round(v, 2) for k, v in acc.items()})
    if args.no_render:
        (out / "summary.json").write_text(json.dumps(summ, indent=1))
        return

    if args.limit:
        meta, mel, a, y = meta.iloc[:args.limit], mel[:args.limit], a[:args.limit], y[:args.limit]
    ids = meta.id.to_numpy()
    seeds = ids + 7 + SEED_OFFSET
    cands = []
    for i in range(0, len(ids), 64):
        cands += candidates(model, js, a[i:i + 64], y[i:i + 64], args.n, seed=i)
    tasks = [[((q, k), r, int(seeds[q]), mel[q]) for k, r in enumerate(cands[q])] for q in range(len(ids))]
    D = {}
    for res in run_pool(_task, tasks, args.workers, "eval"):
        D.update(dict(res))
    inv = pd.read_csv(INV).set_index("preset_id")
    rows = []
    for q in range(len(ids)):
        d = np.array([D[(q, k)] for k in range(args.n)], dtype=float)
        bk = int(np.nanargmin(d)) if np.isfinite(d).any() else 0
        r = inv.loc[int(ids[q])]
        rows.append(dict(preset_id=int(ids[q]), split=meta.split.iloc[q], label=meta.label.iloc[q], oracle=r.oracle,
                         retrieval=r.retrieval, inverter_v0=r.inverter_best, joint_top=d[0], joint_post=np.nanmin(d[1:8]),
                         joint_best=np.nanmin(d), best_candidate=bk, joint_silent=int(np.isnan(d).sum())))
    P = pd.DataFrame(rows)
    P["joint_or_retrieval"] = P[["joint_best", "retrieval"]].min(axis=1)
    P.to_csv(out / "per_preset.csv", index=False)
    cols = ["oracle", "inverter_v0", "joint_top", "joint_post", "joint_best", "retrieval", "joint_or_retrieval"]
    for split, S in [("val", P[P.split == "val"]), ("test", P[P.split == "test"]), ("all", P)]:
        if len(S):
            summ[split] = {c: dict(same_character=float((S[c] <= CHARACTER_DB).mean()), median_db=float(S[c].median())) for c in cols}
    summ["by_label_test_same_character"] = P[P.split == "test"].groupby("label")[cols].apply(
        lambda S: (S <= CHARACTER_DB).mean()).round(3).to_dict(orient="index")
    summ["best_candidate_counts"] = {"mean (0)": int((P.best_candidate == 0).sum()),
                                     "posterior draws (1-7)": int(P.best_candidate.between(1, 7).sum()),
                                     "sampled switches (8-15)": int((P.best_candidate >= 8).sum())}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(f"\n{len(P)} presets, {args.n} candidates; same character = <= {CHARACTER_DB} dB (other seed for all renders)")
    for split in ("val", "test", "all"):
        if split in summ:
            print(f"  {split:5s} " + "  ".join(f"{c} {v['same_character']:.0%} ({v['median_db']:.1f})" for c, v in summ[split].items()))
    print("  best candidate:", summ["best_candidate_counts"])
    print("  test, same character by category:")
    print(pd.DataFrame(summ["by_label_test_same_character"]).T.to_string())

    ex = P[P.split == "test"].groupby("label").head(max(1, args.examples // 8)).head(args.examples)
    items = []
    for _, r in ex.iterrows():
        q = int(np.flatnonzero(ids == r.preset_id)[0])
        tag = f"{r.label}_{r.preset_id}"
        items.append((str(out / "examples" / f"{tag}_joint_{r.joint_best:.1f}dB.flac"), cands[q][int(r.best_candidate)], int(seeds[q])))
    for _ in run_pool(_example, [items[i:i + 3] for i in range(0, len(items), 3)], args.workers, "examples"):
        pass
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
