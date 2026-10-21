"""Stage 1 of the synthetic Surge dataset (TEXT2PRESET_PLAN.md §3.2): render the sampled rows, store
(parameters, mel spectrogram) pairs for the inverter, and measure coverage in sound.

Rows come from pitcher.synth_data.sampler (perturb / layered / vae) and are built inside the render
workers. Chunk 0 is the cleaned human TRAIN presets (source "human": inverter training data too, and the
starting pool for coverage); the other chunks hold the synthetic rows with FIXED shares per source,
shuffled, so the sources can be compared fairly on the coverage curve. Per row:
  - deterministic render at C3 with the row's own seed (= its id), SurgeIO.render_plan (Round 29)
  - rendered for 6 s at most (2 s hold + release capped at 4 s: only the first 6 s are stored)
  - validity: silent (loudest 50 ms window below -90 dBFS) or NaN -> invalid; flags kept: hot (samples
    above 0 dBFS in the raw render), tail capped (still sounding at 6 s, i.e. cut off by the window)
  - level-normalised (loudest 50 ms window to -20 dBFS, the post-processing rule), first 6 s
    (zero-padded), 64-band mel in dB (core.audio_metrics, the metric's own: 2048 / 512 at 48 kHz =
    559 frames), floored 80 dB below its peak, float16 (72 KB)
  - parameters: the kept features (NaN = inactive = masked for the inverter)
  - ~1% of the normalised audio saved as FLAC for listening
No CLAP here (7 per second on the Mac CPU): computed later on a GPU from re-renders (rows are deterministic).

Coverage (--analyse-only, or after rendering): in generation order, the level-matched mel distance of each
valid render to its nearest earlier one (the 6.7 dB "same character" line of Round 14 decides "new").
Nearest-neighbour candidates come from a PCA of the time-pooled mels (k nearest per render, blocks of 2,000, plus
earlier rows of the same block); the distance itself is exact on the stored mels (mel_distance_db's rule:
cells within 80 dB of the louder peak). A recall check compares k (default 40) with 200 candidates.

Writes  data/synth/stage1/<name>/chunk_XXX/{meta.parquet, codes.parquet, mel.npy, audio/*.flac}
        data/synth/stage1/<name>/coverage.parquet;  runs/synth/stage1_<name>/{report.json, coverage.png}
Run:    .venv/bin/python -m pitcher.synth_data.stage1 --name pilot --n 50000      (~40 min on 4 workers)
        .venv/bin/python -m pitcher.synth_data.stage1 --name pilot --analyse-only
        .venv/bin/python -m pitcher.synth_data.stage1 --name pilot --listen   (listening folder: data/synth/stage1/<name>/listen/)
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")   # 4 render workers on 4 cores: no extra maths threads per worker

import argparse
import json
import math
import time

import numpy as np
import pandas as pd
import soundfile as sf

from pitcher.core.audio_metrics import FLOOR_DB, HOP, N_FFT, mel_db
from pitcher.core.surge_io import SR
from pitcher.step4_vae.data import LABELS
from pitcher.synth_data import common
from pitcher.synth_data.common import CHARACTER_DB, OUT, RUNS, _loudest_window_db, run_pool
from pitcher.synth_data.sampler import STRENGTH_P, Space

NOTE, DUR_S, TARGET_DB = 48, 6.0, -20.0
SILENT_DB = -90.0   # loudest 50 ms window below this = silent (Surge renders in float, no noise floor: a quiet
                    # note at -70 dBFS is a clean sound once level-normalised; output level is set after matching)
N_SAMPLES = int(DUR_S * SR)
N_FRAMES = (N_SAMPLES - N_FFT) // HOP + 1
SOURCES = ("perturb", "layered", "vae")
TASK_ROWS = 25

_sp = None


def init_worker():
    global _sp
    common.init_worker()
    import torch
    torch.set_num_threads(1)
    _sp = Space()


def build(spec):
    rng = np.random.default_rng(spec["seed"])
    src, extra = spec["source"], {}
    if src == "human":
        row = _sp.row(spec["base"])
    elif src == "perturb":
        row, extra = _sp.perturb(spec["base"], spec["strength"], rng)
    elif src == "layered":
        row, extra = _sp.layered(spec["label"], dict(zip(spec["u_cols"], spec["u"])), rng)
    else:
        row = _sp.vae_rows(spec["vae_run"], spec["label"], 1, spec["seed"])[0]
    return row, extra


def render_mel(row, seed, note=NOTE):
    """The stored view of a row: render at C3 (6 s at most), level-normalised, 64-band mel (559 x 64,
    float16, floored 80 dB below its peak). -> (mel, audio, gain_db, info dict with level / silent / nan)."""
    audio, info = common.render(row, note, seed, max_tail_s=DUR_S - 2.0)   # only the first 6 s are kept
    finite = bool(np.isfinite(audio).all())
    audio = np.nan_to_num(audio)
    lvl = _loudest_window_db(audio)
    silent = bool(info["silent"] or lvl < SILENT_DB)
    gain = 0.0 if silent else float(np.clip(TARGET_DB - lvl, -60, 60))
    x = audio[:, :N_SAMPLES] * 10 ** (gain / 20)
    if x.shape[1] < N_SAMPLES:
        x = np.pad(x, ((0, 0), (0, N_SAMPLES - x.shape[1])))
    return mel_view(x), audio, gain, dict(info, level_db=lvl, silent=silent, nan=not finite)


def mel_view(x):
    """(2, N_SAMPLES) level-normalised audio -> stored mel (559 x 64, float16, floored 80 dB below its peak)."""
    M = mel_db(x)
    return np.maximum(M, M.max() - FLOOR_DB).astype(np.float16)


def _task(specs):
    out = []
    for s in specs:
        row, extra = build(s)
        M, audio, gain, info = render_mel(row, s["seed"])
        lvl, silent, finite = info["level_db"], info["silent"], not info["nan"]
        if s["keep_audio"]:
            y = audio * 10 ** (gain / 20)
            y = y / max(1.0, float(np.abs(y).max()) / 0.98)
            sf.write(s["audio_path"], y.T, SR, format="FLAC", subtype="PCM_24")
        meta = dict(id=s["id"], order=s["order"], source=s["source"], label=s["label"], base=s.get("base", -1),
                    strength=s.get("strength", ""), seed=s["seed"], level_db=lvl, silent=silent, nan=not finite,
                    hot=info["n_over_0dbfs"] > 0, tail_capped=bool(info["tail_capped"]), total_s=info["total_s"],
                    switches=extra.get("switches", 0), structure_preset=extra.get("structure_preset", -1),
                    audio=s["keep_audio"])
        out.append((meta, np.array([row[f] for f in _sp.kept], dtype=np.float32), M))
    return out


def plan_chunk(sp, c, n, shares, args, outdir):
    rng = np.random.default_rng(1000 + c)
    specs = []
    if c == 0:
        for pid in sp.train_ids:
            specs.append(dict(source="human", base=int(pid), label=sp.label[int(pid)]))
    else:
        counts = np.floor(np.array(shares) * n).astype(int)
        counts[0] += n - counts.sum()
        by_label = {lab: sp.train_ids[sp.labels_tr == lab] for lab in LABELS}
        for j in range(counts[0]):
            lab = LABELS[j % len(LABELS)]
            specs.append(dict(source="perturb", label=lab, base=int(rng.choice(by_label[lab])),
                              strength=str(rng.choice(list(STRENGTH_P), p=list(STRENGTH_P.values())))))
        cols, U = sp.copula_draw(max(counts[1], 1), seed=1000 + c)
        for j in range(counts[1]):
            specs.append(dict(source="layered", label=LABELS[j % len(LABELS)], u_cols=cols, u=U[j].tolist()))
        for j in range(counts[2]):
            specs.append(dict(source="vae", label=LABELS[j % len(LABELS)], vae_run=args.vae_run))
        specs = [specs[i] for i in rng.permutation(len(specs))]
    (outdir / "audio").mkdir(parents=True, exist_ok=True)
    for j, s in enumerate(specs):
        s["id"] = c * 1_000_000 + j
        s["order"] = j
        s["seed"] = s["id"] + 7
        s["keep_audio"] = bool(rng.random() < args.audio_share)
        s["audio_path"] = str(outdir / "audio" / f"{s['id']}.flac")
    return specs


def render_chunks(args):
    sp = Space()
    root = OUT / "stage1" / args.name
    root.mkdir(parents=True, exist_ok=True)
    (root.parent / ".metadata_never_index").touch()          # keep Spotlight off the mel files
    shares = [float(x) for x in args.shares.split(",")]
    n_chunks = math.ceil(args.n / args.chunk)
    for c in range(n_chunks + 1):
        d = root / f"chunk_{c:03d}"
        if (d / "meta.parquet").exists():
            print(f"chunk {c}: exists, skipped")
            continue
        n = min(args.chunk, args.n - (c - 1) * args.chunk) if c else 0
        specs = plan_chunk(sp, c, n, shares, args, d)
        tasks = [specs[i:i + TASK_ROWS] for i in range(0, len(specs), TASK_ROWS)]
        t0, res = time.time(), []
        for r in run_pool(_task, tasks, args.workers, f"chunk {c}", init=init_worker):
            res += r
        res.sort(key=lambda r: r[0]["order"])
        meta = pd.DataFrame([r[0] for r in res])
        pd.DataFrame(np.vstack([r[1] for r in res]), columns=sp.kept).assign(id=meta.id.values).to_parquet(d / "codes.parquet", index=False)
        np.save(d / "mel.npy", np.stack([r[2] for r in res]))
        meta.to_parquet(d / "meta.parquet", index=False)
        el = time.time() - t0
        print(f"chunk {c}: {len(meta)}/{len(specs)} rows in {el:.0f}s ({len(meta) / el:.1f}/s); invalid "
              f"{(meta.silent | meta.nan).mean():.1%}; by source {meta.source.value_counts().to_dict()}", flush=True)


# ------------------------------------------------------------------------------------------- coverage ---
def load(name):
    root = OUT / "stage1" / name
    metas, mels = [], []
    for d in sorted(root.glob("chunk_*")):
        if not (d / "meta.parquet").exists():
            continue
        m = pd.read_parquet(d / "meta.parquet")
        m["chunk"] = int(d.name.split("_")[1])
        m["pos"] = np.arange(len(m))                        # row in mel.npy
        metas.append(m)
        mels.append(np.load(d / "mel.npy", mmap_mode="r"))
    meta = pd.concat(metas, ignore_index=True)
    return meta, mels


def exact(A, B):
    """mel_distance_db on stored (floored) mels: A (T, F), B (k, T, F) -> (k,)."""
    floor = np.maximum(A.max(), B.max(axis=(1, 2))) - FLOOR_DB
    Ac = np.maximum(A[None], floor[:, None, None])
    Bc = np.maximum(B, floor[:, None, None])
    aud = (Ac > floor[:, None, None]) | (Bc > floor[:, None, None])
    d = np.abs(Ac - Bc) * aud
    return d.sum(axis=(1, 2)) / np.maximum(aud.sum(axis=(1, 2)), 1)


def coverage(meta, mels, k=10, block=2000, seed=0):
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    meta = meta.sort_values(["chunk", "order"]).reset_index(drop=True)
    valid = ~(meta.silent | meta.nan)
    # stored mels (float32 on demand)
    chunk_ix = {c: i for i, c in enumerate(sorted(meta.chunk.unique()))}
    get = lambda r: np.asarray(mels[chunk_ix[meta.chunk.iat[r]]][meta.pos.iat[r]], dtype=np.float32)
    idx = np.flatnonzero(valid.to_numpy())
    print(f"coverage over {len(idx)} valid renders")
    pool = lambda M: M[: (N_FRAMES // 4) * 4].reshape(N_FRAMES // 4, 4, -1).mean(1).ravel()
    rng = np.random.default_rng(seed)
    fit = rng.choice(idx, min(4000, len(idx)), replace=False)
    pca = PCA(64, svd_solver="randomized", random_state=seed).fit(np.stack([pool(get(r)) for r in fit]))
    E = np.zeros((len(idx), 64), np.float32)
    for i in range(0, len(idx), 2000):
        E[i:i + 2000] = pca.transform(np.stack([pool(get(r)) for r in idx[i:i + 2000]]))
    nn = np.full(len(idx), np.inf)
    nn_id = np.full(len(idx), -1)
    recall = []
    t0 = time.time()
    for b0 in range(0, len(idx), block):
        b1 = min(b0 + block, len(idx))
        mel_b = [get(r) for r in idx[b0:b1]]
        cand = [[] for _ in range(b1 - b0)]
        if b0 > 0:
            knn = NearestNeighbors(n_neighbors=min(k, b0)).fit(E[:b0])
            _, J = knn.kneighbors(E[b0:b1])
            for i, js in enumerate(J):
                cand[i] += js.tolist()
        D = ((E[b0:b1, None] - E[None, b0:b1]) ** 2).sum(-1)
        for i in range(1, b1 - b0):
            cand[i] += (b0 + np.argsort(D[i, :i])[:k]).tolist()
        for i in range(b1 - b0):
            if not cand[i]:
                continue
            js = np.array(cand[i])
            B = np.stack([mel_b[j - b0] if j >= b0 else get(idx[j]) for j in js])
            d = exact(mel_b[i], B)
            nn[b0 + i], nn_id[b0 + i] = d.min(), idx[js[d.argmin()]]
        if b0 >= 4 * block and len(recall) < 100:             # recall of k vs 200 candidates
            for i in rng.choice(b1 - b0, 5, replace=False):
                knn = NearestNeighbors(n_neighbors=min(200, b0)).fit(E[:b0 + i])
                _, J = knn.kneighbors(E[b0 + i:b0 + i + 1])
                B = np.stack([get(idx[j]) for j in J[0]])
                best = exact(mel_b[i], B).min()
                recall.append((bool(best >= nn[b0 + i] - 1e-3), bool((best > CHARACTER_DB) == (nn[b0 + i] > CHARACTER_DB))))
        print(f"  coverage: {b1}/{len(idx)} ({time.time() - t0:.0f}s)", flush=True)
    out = meta.copy()
    out["nn_db"], out["nn_row"] = np.nan, -1
    out.loc[idx, "nn_db"] = nn
    out.loc[idx, "nn_row"] = nn_id
    out["new"] = out.nn_db > CHARACTER_DB
    # recall: share of checks where k candidates find the nearest of 200; and where the new / not-new call agrees
    return out, (dict(nearest=float(np.mean([r[0] for r in recall])), new_call=float(np.mean([r[1] for r in recall])),
                      k=k, n=len(recall)) if recall else None)


def report(cov, recall, name):
    rd = RUNS / f"stage1_{name}"
    rd.mkdir(parents=True, exist_ok=True)
    syn = cov[cov.source != "human"]
    v = cov[~(cov.silent | cov.nan)]
    by = lambda df, keys: df.groupby(keys).agg(n=("id", "size"), new=("new", "mean"), nn_median_db=("nn_db", "median")).round(3)
    rep = dict(
        name=name, n_rows=int(len(cov)), recall_vs_k200=recall,
        invalid=cov.assign(inv=cov.silent | cov.nan).groupby("source").inv.mean().round(4).to_dict(),
        hot=cov.groupby("source").hot.mean().round(4).to_dict(),
        tail_capped=cov.groupby("source").tail_capped.mean().round(4).to_dict(),
        coverage_by_source=by(v, "source").to_dict(orient="index"),
        coverage_by_strength=by(v[v.source == "perturb"], "strength").to_dict(orient="index"),
        coverage_by_source_label={f"{a}/{b}": r for (a, b), r in by(v, ["source", "label"]).to_dict(orient="index").items()},
    )
    # coverage curve: share new per 2,000 synthetic rows in generation order, per source
    s = v[v.source != "human"].sort_values(["chunk", "order"]).reset_index(drop=True)
    s["bin"] = np.arange(len(s)) // 2000
    curve = s.groupby(["bin", "source"]).new.mean().unstack()
    rep["coverage_curve"] = {src: curve[src].round(3).tolist() for src in curve.columns}
    (rd / "report.json").write_text(json.dumps(rep, indent=1, default=float))
    plot(curve, v, rd)
    print(json.dumps({k: rep[k] for k in ("recall_vs_k200", "invalid", "hot", "tail_capped")}, indent=1, default=float))
    pd.set_option("display.width", 200)
    print("\ncoverage by source (share > 6.7 dB from every earlier render; nn = distance to the nearest earlier):")
    print(by(v, "source").to_string())
    print("\nperturb by strength:")
    print(by(v[v.source == "perturb"], "strength").to_string())
    print("\nshare new by source x category:")
    print(v.groupby(["label", "source"]).new.mean().unstack().round(2).to_string())
    print("\ncoverage curve (share new per 2,000 synthetic rows):")
    print(curve.round(2).to_string())
    return rep


def plot(curve, v, rd):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1f1f1d", "#6b6a63", "#ecebe6"
    COL = {"perturb": "#2a78d6", "layered": "#d9822b", "vae": "#7a5bbf", "human": "#8a8980"}
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.4), facecolor=SURFACE)
    for ax in (a, b):
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.grid(color=GRID, linewidth=.6)
        ax.set_axisbelow(True)
    x = (curve.index + 1) * 2000
    for src in curve.columns:
        a.plot(x, curve[src], color=COL.get(src, INK), linewidth=2, marker="o", markersize=4)
        a.annotate(src, (x[-1], curve[src].iloc[-1]), xytext=(6, 0), textcoords="offset points", color=INK, fontsize=8, va="center")
    a.set_ylim(0, 1)
    a.set_xlabel("synthetic renders so far", color=MUTED, fontsize=8.5)
    a.set_ylabel("share new (> 6.7 dB from every earlier render)", color=MUTED, fontsize=8.5)
    a.set_title("Coverage curve per source", loc="left", fontsize=10, color=INK)
    bins = np.linspace(0, 25, 51)
    for src in ["human"] + list(SOURCES):
        d = v[v.source == src].nn_db.dropna().clip(upper=25)
        if len(d):
            b.hist(d, bins=bins, histtype="step", density=True, linewidth=1.6, color=COL[src], label=src)
    b.axvline(CHARACTER_DB, color=MUTED, linestyle="--", linewidth=.8)
    b.set_xlabel("distance to the nearest earlier render (dB, level-matched mel)", color=MUTED, fontsize=8.5)
    b.set_title("How far each render is from everything before it", loc="left", fontsize=10, color=INK)
    b.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(rd / "coverage.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _base_task(specs):
    out = []
    for s in specs:
        audio, _ = common.render(_sp.row(s["base"]), NOTE, s["seed"], max_tail_s=DUR_S - 2.0)
        audio = np.nan_to_num(audio)
        y = audio * 10 ** (float(np.clip(TARGET_DB - _loudest_window_db(audio), -60, 60)) / 20)
        sf.write(s["path"], (y / max(1.0, float(np.abs(y).max()) / 0.98)).T, SR, format="FLAC", subtype="PCM_24")
        out.append(s["path"])
    return out


def listen(name, workers=4):
    """data/synth/stage1/<name>/listen/: the saved FLACs (~1%) under descriptive names
    (<source>_<category>_<strength>_<id>.flac) + index.csv; for perturbed rows also the base preset's
    template rebuild with the same seed and level rule (<...>_base.flac), to hear what changed."""
    root = OUT / "stage1" / name
    out = root / "listen"
    out.mkdir(exist_ok=True)
    cov_path = root / "coverage.parquet"
    meta, _ = load(name)
    if cov_path.exists():
        meta = meta.merge(pd.read_parquet(cov_path)[["id", "nn_db", "new"]], on="id", how="left")
    names = pd.read_parquet(OUT.parent / "presets.parquet").set_index("preset_id").name
    rows, base_specs = [], []
    for _, m in meta[meta.audio].iterrows():
        src = Path_(root, m)
        tag = "_".join(x for x in (m.source, m.label, m.strength) if x)
        fn = f"{tag}_{m.id}.flac"
        link = out / fn
        if not link.exists():
            link.symlink_to(src)
        base = int(m.base) if m.base >= 0 else (int(m.structure_preset) if m.structure_preset >= 0 else -1)
        r = dict(file=fn, id=m.id, source=m.source, category=m.label, strength=m.strength, switches=m.switches,
                 base_preset=base, base_name=names.get(base, "") if base >= 0 else "",
                 nearest_earlier_db=round(float(m.get("nn_db", np.nan)), 2), new=m.get("new"), hot=m.hot,
                 cut_off_at_6s=m.tail_capped, base_file="")
        if m.source == "perturb":
            r["base_file"] = f"{tag}_{m.id}_base.flac"
            if not (out / r["base_file"]).exists():
                base_specs.append(dict(base=base, seed=int(m.seed), path=str(out / r["base_file"])))
        rows.append(r)
    if base_specs:
        tasks = [base_specs[i:i + 10] for i in range(0, len(base_specs), 10)]
        for _ in run_pool(_base_task, tasks, workers, "base renders", init=init_worker):
            pass
    idx = pd.DataFrame(rows).sort_values(["source", "category", "strength", "id"])
    idx.to_csv(out / "index.csv", index=False)
    print(f"{len(idx)} renders in {out} (+ {len(base_specs)} base renders); index.csv lists source, category, "
          f"strength, base preset, distance to the nearest earlier render")
    print(idx.groupby("source").size().to_dict())


def Path_(root, m):
    return root / f"chunk_{int(m.chunk):03d}" / "audio" / f"{m.id}.flac"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="pilot")
    ap.add_argument("--n", type=int, default=50000, help="synthetic rows (the human chunk comes on top)")
    ap.add_argument("--chunk", type=int, default=5000)
    ap.add_argument("--shares", default="0.5,0.3,0.2", help="perturb,layered,vae")
    ap.add_argument("--vae-run", default="h256_d01")
    ap.add_argument("--audio-share", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--listen", action="store_true", help="only build the listening folder")
    ap.add_argument("--k", type=int, default=40, help="nearest-neighbour candidates per render for coverage")
    args = ap.parse_args()
    if args.listen:
        listen(args.name, args.workers)
        return
    if not args.analyse_only:
        render_chunks(args)
    meta, mels = load(args.name)
    cov, recall = coverage(meta, mels, k=args.k)
    cov.to_parquet(OUT / "stage1" / args.name / "coverage.parquet", index=False)
    report(cov, recall, args.name)


if __name__ == "__main__":
    main()
