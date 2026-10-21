"""Sound score of an inverter (TEXT2PRESET_PLAN.md §4) on the human val / test presets.

Per preset: its stored mel (template rebuild at C3, seed id + 7) -> inverter -> N candidate presets
(candidate 0 = most likely values, the others with sampled switches, model.py) -> each rendered with
ANOTHER seed (id + 1007: the inverter can't know Surge's random phase / drift) -> level-matched mel
distance to the target (stage1.exact, the mel_distance_db rule) -> the closest is kept.
References, rendered the same way:
  oracle     the true preset with the other seed (the best any inversion can do: seed noise only)
  retrieval  the preset of the nearest training sound (PCA candidates + exact distance over the stage-1
             set), the no-training baseline
  combined   best of the inverter's candidates and the retrieved preset
Score: "same character" rate (distance <= 6.7 dB, listening test Round 14) and median distance.
Also the accuracy of key switches (most likely value vs the true one). A few examples are written as
FLAC (target, inverter best, retrieval) for listening.

Writes  runs/inverter/<run>/eval/{per_preset.csv, summary.json, examples/*.flac}
Run:    .venv/bin/python -m pitcher.inverter.evaluate --run v0 [--n 16 --workers 4 --limit 0]
"""

import argparse
import json

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from pitcher.core.surge_io import SR
from pitcher.inverter.data import MelSet, full_row, load_eval
from pitcher.inverter.model import Inverter
from pitcher.inverter.train import RUNS, KEY_SWITCHES, to_t
from pitcher.step4_vae.data import Spec
from pitcher.synth_data.common import CHARACTER_DB, run_pool
from pitcher.synth_data.stage1 import exact, init_worker, render_mel

ROUTE_SNAP = 0.02   # |depth| < 0.04 = no routing (the heads never output exactly 0.5)
SEED_OFFSET = 1000


def load_model(run):
    cfg = json.loads((RUNS / run / "config.json").read_text())
    m = Inverter(cfg["n_cont"], cfg["n_bool"], cfg["cat_sizes"], dropout=cfg.get("dropout", 0.2))
    m.load_state_dict(torch.load(RUNS / run / "model.pt"))
    return m.eval(), cfg


def snap_routes(spec, row):
    for f in spec.cont:
        if f.startswith("route/") and abs(row[f] - 0.5) < ROUTE_SNAP:
            row[f] = 0.5
    return row


@torch.no_grad()
def candidates(model, spec, mel, n, seed=0):
    """mel (B, T, F) numpy -> [[full codes row] * n] * B."""
    out, _ = model(to_t(mel).float())
    cont, bl, cats = out
    g = torch.Generator().manual_seed(seed)
    res = []
    for b in range(len(mel)):
        rows = []
        for k in range(n):
            if k == 0:
                bk, ck = bl[b], [c[b] for c in cats]
            else:
                bk = torch.where(torch.rand(bl[b].shape, generator=g) < torch.sigmoid(bl[b]), 1.0, -1.0)
                ck = [torch.nn.functional.one_hot(torch.multinomial(torch.softmax(c[b], -1), 1, generator=g)[0], len(c[b])).float()
                      for c in cats]
            r = spec.to_codes(cont[b].tolist(), bk.tolist(), [c.tolist() for c in ck])
            rows.append(snap_routes(spec, r))
        res.append(rows)
    return res


def _task(items):
    out = []
    for key, row, seed, target in items:
        M, _, _, info = render_mel(row, seed)
        d = float(exact(target.astype(np.float32), M.astype(np.float32)[None])[0]) if not info["silent"] else np.nan
        out.append((key, d))
    return out


def _example(items):
    out = []
    for path, row, seed in items:
        _, audio, gain, _ = render_mel(row, seed)
        y = audio * 10 ** (gain / 20)
        sf.write(path, (y / max(1.0, float(np.abs(y).max()) / 0.98)).T, SR, format="FLAC", subtype="PCM_24")
        out.append(path)
    return out


def retrieve(ds, query, k=20, seed=0):
    """index into ds of the nearest stored training sound per query mel (exact distance among k PCA candidates)."""
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    T = (query.shape[1] // 4) * 4
    pool = lambda M: M[:, :T].astype(np.float32).reshape(len(M), T // 4, 4, -1).mean(2).reshape(len(M), -1)
    tr = np.flatnonzero(~ds.is_val)
    rng = np.random.default_rng(seed)
    pca = PCA(64, svd_solver="randomized", random_state=seed).fit(pool(ds.mel[rng.choice(tr, 4000, replace=False)]))
    E = np.vstack([pca.transform(pool(ds.mel[tr[i:i + 2000]])) for i in range(0, len(tr), 2000)])
    _, J = NearestNeighbors(n_neighbors=k).fit(E).kneighbors(pca.transform(pool(query)))
    best = []
    for q, js in enumerate(J):
        d = exact(query[q].astype(np.float32), ds.mel[tr[js]].astype(np.float32))
        best.append(int(tr[js[d.argmin()]]))
    return np.array(best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v0")
    ap.add_argument("--data", default="pilot")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="only the first N presets (smoke test)")
    ap.add_argument("--examples", type=int, default=12)
    args = ap.parse_args()
    out = RUNS / args.run / "eval"
    (out / "examples").mkdir(parents=True, exist_ok=True)
    spec = Spec()
    model, cfg = load_model(args.run)
    meta, mel, codes, t = load_eval(spec)
    if args.limit:
        meta, mel, codes = meta.iloc[:args.limit], mel[:args.limit], codes.iloc[:args.limit]
        t = {k: v[:args.limit] for k, v in t.items()}
    ids = meta.id.to_numpy()
    seeds = ids + 7 + SEED_OFFSET

    cands = []
    for i in range(0, len(mel), 64):
        cands += candidates(model, spec, mel[i:i + 64], args.n, seed=i)
    ds = MelSet(args.data, spec)
    nn = retrieve(ds, mel)
    kept = list(codes.columns.drop("id"))
    tasks = []
    for q in range(len(ids)):
        truth = full_row(spec, {f: float(codes.iloc[q][f]) for f in kept})
        ret = full_row(spec, {f: float(ds.codes.iloc[nn[q]][f]) for f in kept})
        items = [((q, "oracle"), truth, int(seeds[q]), mel[q]), ((q, "retrieval"), ret, int(seeds[q]), mel[q])]
        items += [((q, k), r, int(seeds[q]), mel[q]) for k, r in enumerate(cands[q])]
        tasks.append(items)
    D = {}
    for res in run_pool(_task, tasks, args.workers, "eval"):
        D.update(dict(res))

    rows = []
    for q in range(len(ids)):
        inv = np.array([D[(q, k)] for k in range(args.n)], dtype=float)
        best_k = int(np.nanargmin(inv)) if np.isfinite(inv).any() else 0
        rows.append(dict(preset_id=int(ids[q]), split=meta.split.iloc[q], label=meta.label.iloc[q],
                         oracle=D[(q, "oracle")], retrieval=D[(q, "retrieval")], retrieved_row=int(nn[q]),
                         retrieved_source=ds.meta.source.iloc[nn[q]], inverter_top=inv[0], inverter_best=np.nanmin(inv),
                         best_candidate=best_k, inverter_silent=int(np.isnan(inv).sum())))
    P = pd.DataFrame(rows)
    P["combined"] = P[["inverter_best", "retrieval"]].min(axis=1)
    P.to_csv(out / "per_preset.csv", index=False)

    with torch.no_grad():
        o, _ = model(to_t(mel).float())
    acc = {}
    for k in KEY_SWITCHES:
        j = spec.cat.index(k)
        m = t["m_cat"][:, j] > 0
        acc[k] = float((np.array(o[2][j].argmax(-1).tolist())[m] == t["x_cat"][m, j]).mean())

    cols = ["oracle", "inverter_top", "inverter_best", "retrieval", "combined"]
    summ = dict(run=args.run, n_candidates=args.n, n_presets=len(P), switch_accuracy=acc)
    for split, S in [("val", P[P.split == "val"]), ("test", P[P.split == "test"]), ("all", P)]:
        if len(S):
            summ[split] = {c: dict(same_character=float((S[c] <= CHARACTER_DB).mean()), median_db=float(S[c].median()))
                           for c in cols}
    summ["by_label_test_same_character"] = P[P.split == "test"].groupby("label")[cols].apply(
        lambda S: (S <= CHARACTER_DB).mean()).round(3).to_dict(orient="index")
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    pd.set_option("display.width", 200)
    print(f"\n{len(P)} presets, {args.n} candidates each; same character = <= {CHARACTER_DB} dB (other seed for all renders)")
    for split in ("val", "test"):
        if split in summ:
            print(f"  {split:5s} " + "  ".join(f"{c} {v['same_character']:.0%} ({v['median_db']:.1f} dB)" for c, v in summ[split].items()))
    print("  key switches (most likely value):", {k.split('/')[-2] + '/' + k.split('/')[-1]: round(v, 2) for k, v in acc.items()})
    print("  test, same character by category:")
    print(pd.DataFrame(summ["by_label_test_same_character"]).T.to_string())

    # listening examples: test presets spread over the categories
    ex = P[P.split == "test"].groupby("label").head(max(1, args.examples // 8)).head(args.examples)
    items = []
    for _, r in ex.iterrows():
        q = int(np.flatnonzero(ids == r.preset_id)[0])
        tag = f"{r.label}_{r.preset_id}"
        items += [(str(out / "examples" / f"{tag}_a_target.flac"), full_row(spec, {f: float(codes.iloc[q][f]) for f in kept}), int(ids[q] + 7)),
                  (str(out / "examples" / f"{tag}_b_inverter_{r.inverter_best:.1f}dB.flac"), cands[q][int(r.best_candidate)], int(seeds[q])),
                  (str(out / "examples" / f"{tag}_c_retrieval_{r.retrieval:.1f}dB.flac"),
                   full_row(spec, {f: float(ds.codes.iloc[int(r.retrieved_row)][f]) for f in kept}), int(seeds[q]))]
    for _ in run_pool(_example, [items[i:i + 3] for i in range(0, len(items), 3)], args.workers, "examples"):
        pass
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
