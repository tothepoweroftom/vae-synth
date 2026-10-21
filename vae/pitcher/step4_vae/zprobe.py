"""z probe (TEXT2PRESET_PLAN.md §6): does the VAE's latent z hold what prompts talk about?

A text -> z prior can only select a sound property if z encodes it. Per cleaned preset (train / val /
test of the VAE run) the template rebuild (all kept features + mode-frozen ones, i.e. what the VAE
learns from) is rendered at C3 with a fixed seed and measured with the Stable Audio post-processing's
own code (text2preset.postprocess: same trim + level, so the numbers compare with the pilot's word
effects):
  brightness  log2(spectral centroid / the note's fundamental), octaves
  attack      log10(attack time + 5 ms) (time to within 3 dB of the loudest 50 ms window)
  decay       level 1 s after the start, dB below the loudest window, clipped at -60 (pluck vs sustained)
  length      log2(duration until the tail is 60 dB down; hold is 2 s, so this is mostly release)
  noisiness   log10(median spectral flatness)
  percussive  harmonic-percussive split: percussive share of the energy

Probes (fitted on the train presets, scored on val + test, which the VAE never saw):
  label       the category one-hot alone (baseline: what the prior gets from the category word anyway)
  z           encoder mean mu (12 numbers)
  z + label   what the plugin chain has (the prior gives z, the category goes to the decoder too)
  params      the full encoder input + label (upper bound: what the preset's parameters say)
each with a ridge regression (linear) and a gradient-boosted model (non-linear); R^2 for the measures,
accuracy for the category (note: the CVAE's encoder gets the label, so z is pushed to carry what the
label doesn't; category-from-z is expected to be weak and that is fine).
Two references: the seed-to-seed R^2 (val + test rendered with a second seed: the measure's own noise,
the most any probe can reach) and the decoder path (val + test: mu -> decoder -> render -> measures,
R^2 against the original's; does the property survive the round trip through z?).

Writes  runs/vae/<run>/zprobe/measures.parquet (one row per render), probe.csv, summary.json, zprobe.png
Run:    .venv/bin/python -m pitcher.step4_vae.zprobe --run h256_d01 [--workers 4] [--analyse-only]
        .venv/bin/python -m pitcher.step4_vae.zprobe --run h256_d01_b0.02 --reuse h256_d01   (~3 min)
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from pitcher.core.surge_io import PITCHER_ROOT
from pitcher.step4_vae.data import LABELS, Spec, load
from pitcher.step4_vae.eval import rows_from
from pitcher.step4_vae.train import tensors
from pitcher.step6_sample.latent import load_model
from pitcher.synth_data.common import Base, render, run_pool
from pitcher.text2preset.postprocess import analyse, normalise

RUNS = PITCHER_ROOT / "runs" / "vae"
NOTE, SEED_A, SEED_B = 48, 7, 8
MEASURES = ["brightness", "attack", "decay", "length", "noisiness", "percussive"]


def measures(audio):
    x = np.asarray(audio, dtype=np.float32)
    x = np.vstack([x, x]) if x.ndim == 1 else x
    xn, info = normalise(x)
    m = analyse(xn)
    f0 = 440 * 2 ** ((m["main_midi"] - 69) / 12) if np.isfinite(m["main_midi"]) else 440 * 2 ** ((NOTE - 69) / 12)
    return dict(brightness=float(np.log2(m["centroid_hz"] / f0)) if np.isfinite(m["centroid_hz"]) else np.nan,
                attack=float(np.log10(m["attack_s"] + 0.005)), decay=float(np.clip(m["lvl_1s"], -60, 0)),
                length=float(np.log2(info["trimmed_s"])), noisiness=float(np.log10(m["flatness"] + 1e-6)),
                percussive=float(m["perc_share"]), main_midi=m["main_midi"], silent=bool(info["max_rms_db"] < -60))


def _task(task):
    kind, pid, seed, row = task
    audio, _ = render(row, NOTE, seed)
    return dict(kind=kind, preset_id=pid, seed=seed, **measures(audio))


def render_all(model, spec, data, base, workers, reuse=None):
    """reuse: measures of another run (same presets): its original renders are kept, only this run's
    decoded presets are rendered."""
    tasks = []
    for split in ("train", "val", "test"):
        for pid in ([] if reuse is not None else data[split]["ids"].tolist()):
            tasks.append(("orig", int(pid), SEED_A, base.row(int(pid))))
            if split != "train":
                tasks.append(("orig_seedB", int(pid), SEED_B, base.row(int(pid))))
        if split != "train":
            with torch.no_grad():
                t = tensors(data[split])
                mu, _ = model.encode(t["x_in"], t["y1h"])
                cont, bl, cats = model.decode(mu, t["y1h"])
            for pid, row in zip(data[split]["ids"].tolist(), rows_from(spec, cont, bl, cats)):
                tasks.append(("decoded", int(pid), SEED_A, row))
    print(f"{len(tasks)} renders + measurements")
    t0 = time.time()
    df = pd.DataFrame(list(run_pool(_task, tasks, workers, "zprobe")))
    if reuse is not None:
        df = pd.concat([reuse[reuse.kind != "decoded"], df], ignore_index=True)
    print(f"done in {time.time() - t0:.0f}s")
    return df


def latents(model, data):
    out = {}
    with torch.no_grad():
        for split in ("train", "val", "test"):
            t = tensors(data[split])
            mu, _ = model.encode(t["x_in"], t["y1h"])
            out[split] = np.array(mu.tolist())
    return out


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def probe(df, data, Z, kl):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression, RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    df = df.assign(decay=df.decay.clip(-60, 0))
    orig = df[df.kind == "orig"].set_index("preset_id")
    tr, te = data["train"], {k: np.concatenate([data["val"][k], data["test"][k]]) for k in ("ids", "y", "x_in")}
    Zte = np.vstack([Z["val"], Z["test"]])
    onehot = lambda y: np.eye(len(LABELS))[y]
    active = np.flatnonzero(np.array(kl) > 0.5)
    sets = {
        "label": (onehot(tr["y"]), onehot(te["y"])),
        "z": (Z["train"], Zte),
        f"z active ({len(active)})": (Z["train"][:, active], Zte[:, active]),
        "z + label": (np.hstack([Z["train"], onehot(tr["y"])]), np.hstack([Zte, onehot(te["y"])])),
        "params": (np.hstack([tr["x_in"], onehot(tr["y"])]), np.hstack([te["x_in"], onehot(te["y"])])),
    }
    rows = []
    for name, (A, B) in sets.items():
        lin = lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13)))
        for target in MEASURES:
            ya, yb = orig.loc[tr["ids"], target].to_numpy(), orig.loc[te["ids"], target].to_numpy()
            ka, kb = np.isfinite(ya), np.isfinite(yb)
            for model_name, make in (("linear", lin),
                                     ("boosted", lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=.05,
                                                                                       max_leaf_nodes=15, l2_regularization=1.0,
                                                                                       random_state=0))):
                m = make().fit(A[ka], ya[ka])
                rows.append(dict(features=name, target=target, model=model_name, score=r2(yb[kb], m.predict(B[kb]))))
        if name != "label" and not name.startswith("z +") and name != "params":
            for model_name, make in (("linear", lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))),
                                     ("boosted", lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=.05,
                                                                                        max_leaf_nodes=15, random_state=0))):
                m = make().fit(A, tr["y"])
                rows.append(dict(features=name, target="category", model=model_name, score=float((m.predict(B) == te["y"]).mean())))
    # category from the params alone (no label in the input)
    for model_name, make in (("linear", lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000))),):
        m = make().fit(tr["x_in"], tr["y"])
        rows.append(dict(features="params", target="category", model=model_name, score=float((m.predict(te["x_in"]) == te["y"]).mean())))
    P = pd.DataFrame(rows)

    # references on val + test
    ref = {}
    b = df[df.kind == "orig_seedB"].set_index("preset_id")
    d = df[df.kind == "decoded"].set_index("preset_id")
    ids = te["ids"]
    for target in MEASURES:
        y = orig.loc[ids, target].to_numpy()
        for name, other in (("seed_to_seed", b), ("decoded", d)):
            o = other.loc[ids, target].to_numpy()
            k = np.isfinite(y) & np.isfinite(o)
            ref[(name, target)] = r2(y[k], o[k])
        # how much of the within-category spread the label already explains (for reading the table)
    R = pd.Series(ref).unstack(0)
    return P, R


def plot(P, R, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1f1f1d", "#6b6a63", "#ecebe6"
    COLORS = {"label": "#c9c8c1", "z": "#2a78d6", "z + label": "#1b4f8f", "params": "#8a6bbf"}
    best = P[P.target != "category"].groupby(["features", "target"]).score.max().unstack(0)
    fig, ax = plt.subplots(figsize=(10, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    keys = [k for k in COLORS if k in best.columns]
    w = 0.8 / (len(keys) + 2)
    x = np.arange(len(MEASURES))
    for i, k in enumerate(keys):
        ax.bar(x + i * w, best.loc[MEASURES, k].clip(lower=0), w - .02, color=COLORS[k], label=f"probe: {k}")
    ax.bar(x + len(keys) * w, R.loc[MEASURES, "decoded"].clip(lower=0), w - .02, color="#e08a3c", label="decoder round trip")
    ax.scatter(x + (len(keys) - .5) * w, R.loc[MEASURES, "seed_to_seed"], marker="_", s=900, color=INK, zorder=3,
               label="seed-to-seed (noise ceiling)")
    ax.set_xticks(x + (len(keys)) * w / 2, MEASURES)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("R² on val + test presets (best of linear / boosted)", color=MUTED, fontsize=8.5)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.5, ncol=3, markerscale=.35, loc="upper left", bbox_to_anchor=(0, -.1))
    ax.set_title("z probe: how much of each sound measure can be read from the latent z", loc="left", fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(out / "zprobe.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="h256_d01")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--reuse", help="take the original renders from this run's zprobe (renders only the decoded presets)")
    args = ap.parse_args()
    out = RUNS / args.run / "zprobe"
    out.mkdir(exist_ok=True)
    model, cfg = load_model(args.run)
    spec, data = load(Spec())
    kl = json.loads((RUNS / args.run / "best.json").read_text())["kl_per_dim"]
    if args.analyse_only:
        df = pd.read_parquet(out / "measures.parquet")
    else:
        reuse = pd.read_parquet(RUNS / args.reuse / "zprobe" / "measures.parquet") if args.reuse else None
        df = render_all(model, spec, data, Base(), args.workers, reuse)
        df.to_parquet(out / "measures.parquet", index=False)
    Z = latents(model, data)
    P, R = probe(df, data, Z, kl)
    P.to_csv(out / "probe.csv", index=False)
    pd.set_option("display.width", 200)
    best = P.groupby(["target", "features"]).score.max().unstack(1)
    print("\nbest score (R² for measures, accuracy for category), val + test:")
    print(best.round(3).to_string())
    print("\nreferences (R² on val + test):")
    print(R.round(3).to_string())
    silent = df[df.kind == "orig"].silent.mean()
    summ = dict(run=args.run, n_train=int(len(data["train"]["ids"])), n_eval=int(len(data["val"]["ids"]) + len(data["test"]["ids"])),
                silent_share=float(silent), best=best.round(4).to_dict(), references=R.round(4).to_dict(),
                active_dims=[i for i, k in enumerate(kl) if k > 0.5])
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    plot(P, R, out)
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
