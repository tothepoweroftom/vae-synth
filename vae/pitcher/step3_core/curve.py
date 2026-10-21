"""Step 3.4: version B of the round-trip - rebuild presets on the TEMPLATE from their top-K core
groups only, render, and compare with the original (PIPELINE_VAE_SURGE.md, Step 3.4).

For every val preset and K, the writer gets:
  - core features (first K groups of data/core_ranking.csv): the preset's own codes
  - flags frozen in Step 2.1 ("mode"): their most common value
  - everything else: not written -> template value, or Surge's default for the written type
Two template variants (Step 3.3 decision):
  static   Init Saw, no modulation routings (option A)
  routes   + the routing features kept in Step 2.1, with the preset's depths (option B: the 13
           standard routings in Round 12, the routing matrix from Round 15 on)
Score: mel distance to the original render; "transparent" = distance <= max(1 dB, 1.5 x the
distance between two renders of the unchanged original) (re-render distance from roundtrip_A).

Writes  data/core_curve.csv (one row per variant, K, preset), runs/step3/core_curve.png
Run:    .venv/bin/python -m pitcher.step3_core.curve [--ks 16,30,...] [--split val]
"""

import argparse
import json
import time
from multiprocessing import Pool, TimeoutError

import numpy as np
import pandas as pd
import soundfile as sf

from pitcher.core.audio_metrics import mel_distance_db, rms_db
from pitcher.core.codec import decode
from pitcher.step2_explore.reference import DEFAULT_TEMPLATE
from pitcher.core.surge_io import PITCHER_ROOT, SurgeIO

DATA, OUT = PITCHER_ROOT / "data", PITCHER_ROOT / "runs" / "step3"
TRANSPARENT = 1.0
_io = _schema = None


def _init():
    global _io, _schema
    _io = SurgeIO()
    _schema = json.loads((DATA / "schema.json").read_text())


def _one(task):
    key, pid, row = task
    _io.load(DEFAULT_TEMPLATE)
    _io.write_plan(decode(row, _schema))
    audio, _ = _io.render()
    orig, _ = sf.read(DATA / "audio" / "fixed" / f"{pid}.wav", dtype="float32")
    return key, pid, mel_distance_db(orig.T, audio), rms_db(audio) - rms_db(orig.T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="16,30,45,60,90,150,250,452")
    ap.add_argument("--split", default="val")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--variants", default="static,routes")
    ap.add_argument("--out", default="core_curve.csv")
    args = ap.parse_args()
    ks = [int(k) for k in args.ks.split(",")]

    schema = json.loads((DATA / "schema.json").read_text())
    names = [f["name"] for f in schema["features"]]
    ranking = pd.read_csv(DATA / "core_ranking.csv")
    pruning = json.loads((DATA / "pruning.json").read_text())
    mode_frozen = [k for k, v in pruning["frozen"].items() if v["how"] == "mode"]
    routes = [n for n in pruning["kept"] if n.startswith("route/")]  # routing features kept in Step 2.1
    codes, pruned = pd.read_parquet(DATA / "codes.parquet"), pd.read_parquet(DATA / "codes_pruned.parquet")
    sp = pd.read_parquet(DATA / "splits.parquet").set_index("preset_id")
    ids = list(sp.index) if args.split == "all" else list(sp.index[sp.split == args.split])
    rer = pd.read_parquet(DATA / "roundtrip_A.parquet").set_index("preset_id")["d_rerender"].reindex(sp.index)

    tasks, sizes = [], {}
    for variant in args.variants.split(","):
        for k in ks:
            core = [f for feats in ranking.features[:k] for f in feats.split(";") if f in codes.columns]
            if variant == "routes":
                core += routes
            sizes[(variant, k)] = (min(k, len(ranking)) + (len(routes) if variant == "routes" else 0),
                                   sum(len(next(f for f in schema["features"] if f["name"] == n)["columns"])
                                       for n in core))
            for pid in ids:
                row = {n: np.nan for n in names}
                for n in mode_frozen:
                    row[n] = float(pruned.at[pid, n])
                for n in core:
                    row[n] = float(codes.at[pid, n])
                tasks.append(((variant, k), pid, row))

    t0, rows = time.time(), []
    with Pool(args.workers, initializer=_init) as pool:
        it = pool.imap_unordered(_one, tasks, chunksize=1)  # chunksize 1: iterator supports next(timeout)
        while True:
            try:
                key, pid, d, lv = it.next(timeout=180)
            except StopIteration:
                break
            except TimeoutError:
                print("  no result for 180 s - a worker probably crashed; partial results", flush=True)
                break
            rows.append(dict(variant=key[0], k=key[1], preset_id=pid, distance=d, level_diff_db=lv))
            if len(rows) % 500 == 0:
                print(f"  {len(rows)}/{len(tasks)}, {time.time() - t0:.0f}s", flush=True)
        pool.terminate()

    res = pd.DataFrame(rows).merge(sp[["label"]], left_on="preset_id", right_index=True)
    res["rerender"] = res.preset_id.map(rer)
    res["transparent"] = res.distance <= np.maximum(TRANSPARENT, 1.5 * res.rerender)
    res.to_csv(DATA / args.out, index=False)

    summary = res.groupby(["variant", "k"]).agg(transparent=("transparent", "mean"), median_db=("distance", "median"),
                                                p90_db=("distance", lambda d: d.quantile(.9)),
                                                silent=("level_diff_db", lambda v: int((v < -60).sum()))).reset_index()
    summary["groups"] = [sizes[(v, k)][0] for v, k in zip(summary.variant, summary.k)]
    summary["columns"] = [sizes[(v, k)][1] for v, k in zip(summary.variant, summary.k)]
    print(f"\n{len(ids)} {args.split} presets, {len(rows)} renders in {time.time() - t0:.0f}s")
    print(summary.round(3).to_string(index=False))
    plot(summary, OUT)


def plot(summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    SURFACE, INK, MUTED = "#fcfcfb", "#1f1f1d", "#6b6a63"
    colors = {"static": "#2a78d6", "routes": "#eb6834"}
    names = {"static": "A: template without routings", "routes": "B: + routing features"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), facecolor=SURFACE)
    for ax, col, title in [(axes[0], "transparent", "Share of val presets rebuilt indistinguishably"),
                           (axes[1], "median_db", "Median mel distance to the original (dB)")]:
        ax.set_facecolor(SURFACE)
        for v, g in summary.groupby("variant"):
            ax.plot(g.k, g[col] * (100 if col == "transparent" else 1), color=colors[v], linewidth=2,
                    marker="o", markersize=5, label=names[v])
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_xlabel("core parameter groups K (in ranking order)", color=MUTED, fontsize=9)
        ax.set_xscale("log")
        ax.minorticks_off()  # only the K values as ticks (minor log labels collided)
        ax.set_xticks(sorted(summary.k.unique()))
        ax.set_xticklabels([str(k) for k in sorted(summary.k.unique())])
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.grid(color="#ecebe6", linewidth=0.6)
    axes[0].set_ylabel("%", color=MUTED, fontsize=9)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out / "core_curve.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
