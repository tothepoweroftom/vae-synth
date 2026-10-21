"""Step 8.1 do generated presets have the requested category, and are they new? (uses the Step 8.0 tagger)

Groups, each tagged by data/tagger.pkl:
  original        real val presets, original patch (renders from Step 1.4)
  template        the same val presets' own codes rebuilt on the template (core features + writer):
                  the generator's audio path, so this is the realistic ceiling for samples
  samples:<s>     VAE samples per requested label, one group per sampler (pitcher.step6_sample.latent):
                  prior (z ~ tau x N(0, I)) and gmm (per-label mixture on the train presets' latents)
Label agreement = share whose top tag is the requested / folder label (balanced = mean over labels);
also the mean probability the tagger gives that label. Silent renders (< -60 dBFS) count as wrong.
Novelty = 1 - cosine similarity of the CLAP fingerprint to the nearest TRAIN preset (original renders);
the val presets' own value is the reference for "as new as a real unseen preset". Near-copies:
share below the 5th percentile of the val presets' novelty.

Writes  runs/step8/<run>/tau<tau>/label_check.csv (one row per sound), label_check.json,
        audio/<group>/<label>_<i>.wav
Run:    .venv/bin/python -m pitcher.step8_eval.label_check --run h256_d01 [--n 50 --tau 1.0 --samplers prior,gmm]
"""

import argparse
import json
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import soundfile as sf

from pitcher.core.audio_metrics import rms_db
from pitcher.core.codec import decode
from pitcher.core.surge_io import PITCHER_ROOT, SR, SurgeIO
from pitcher.step2_explore.reference import DEFAULT_TEMPLATE
from pitcher.step4_vae.data import LABELS, Spec, load
from pitcher.step6_sample.latent import decode_rows, get_gmms, load_model, sample_z
from pitcher.step8_eval.clap import cache_file
from pitcher.step8_eval.tagger import Tagger

DATA, RUNS = PITCHER_ROOT / "data", PITCHER_ROOT / "runs"
_io = _schema = None


def _init():
    global _io, _schema
    _io = SurgeIO()
    _schema = json.loads((DATA / "schema.json").read_text())


def _render(task):
    path, row = task
    _io.load(DEFAULT_TEMPLATE)
    _io.write_plan(decode(row, _schema))
    audio, _ = _io.render()
    sf.write(path, audio.T, SR, subtype="FLOAT")
    return path


def target_rows(spec, d):
    """The presets' own (pruned) codes as decoder-style outputs -> rows for the template."""
    cats = [np.eye(n)[np.clip(d["x_cat"][:, j], 0, None)] for j, n in enumerate(spec.cat_sizes)]
    return [spec.to_codes(d["x_cont"][i], d["x_bool"][i] * 2 - 1, [c[i] for c in cats]) for i in range(len(d["ids"]))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="h256_d01")
    ap.add_argument("--n", type=int, default=50, help="samples per label and sampler")
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--samplers", default="prior,gmm")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    samplers = a.samplers.split(",")
    out = RUNS / "step8" / a.run / f"tau{a.tau:g}"
    spec, data = load(Spec())
    model, cfg = load_model(a.run)
    gmms = get_gmms(a.run, model, data) if "gmm" in samplers else None
    sp = pd.read_parquet(DATA / "splits.parquet").set_index("preset_id")
    rng = np.random.default_rng(0)

    sounds, tasks = [], []  # (group, label, key, path or None for an original render)
    va = data["val"]
    for pid in va["ids"].tolist():
        sounds.append(("original", sp.label[pid], str(pid), None))
    (out / "audio" / "template").mkdir(parents=True, exist_ok=True)
    for pid, row in zip(va["ids"].tolist(), target_rows(spec, va)):
        path = str(out / "audio" / "template" / f"{pid}.wav")
        sounds.append(("template", sp.label[pid], str(pid), path)); tasks.append((path, row))
    for s in samplers:
        (out / "audio" / s).mkdir(parents=True, exist_ok=True)
        for lab in LABELS:
            z = sample_z(s, lab, a.n, a.tau, cfg["z"], gmms, rng)
            for i, row in enumerate(decode_rows(model, spec, z, lab)):
                path = str(out / "audio" / s / f"{lab}_{i:02d}.wav")
                sounds.append((f"samples:{s}", lab, f"{lab}_{i:02d}", path)); tasks.append((path, row))

    t0 = time.time()
    with Pool(a.workers, initializer=_init) as pool:
        for k, _ in enumerate(pool.imap_unordered(_render, tasks, chunksize=1)):
            if (k + 1) % 200 == 0:
                print(f"  rendered {k + 1}/{len(tasks)}  {time.time() - t0:.0f} s", flush=True)
    print(f"{len(tasks)} renders in {time.time() - t0:.0f} s; tagging {len(sounds)} sounds", flush=True)

    tagger = Tagger()
    E_all = pd.read_parquet(cache_file(tagger.clap_name)).set_index("preset_id")
    E_train = E_all.loc[sp.index[sp.split == "train"]].to_numpy()
    df = pd.DataFrame(sounds, columns=["group", "label", "key", "path"])
    emb, rms = np.zeros((len(df), E_train.shape[1]), np.float32), np.zeros(len(df))
    is_orig = df.path.isna().to_numpy()
    emb[is_orig] = E_all.loc[df.key[is_orig].astype(int)].to_numpy()
    rms[is_orig] = pd.read_parquet(DATA / "audio_index.parquet").query("mode == 'fixed'").set_index("preset_id") \
        .max_rms_dbfs.reindex(df.key[is_orig].astype(int)).to_numpy()  # originals were curated non-silent
    rest = np.flatnonzero(~is_orig)
    for i in range(0, len(rest), 64):
        idx = rest[i:i + 64]
        clips = [sf.read(df.path[j], dtype="float32")[0].T for j in idx]
        emb[idx] = tagger.clap.audio(clips)
        rms[idx] = [rms_db(c) for c in clips]
    P = tagger.proba_from(emb)
    df["tagged"] = P.idxmax(axis=1)
    df["p_label"] = [P.at[i, l] for i, l in enumerate(df.label)]
    df["rms_dbfs"] = rms
    df["silent"] = (df.rms_dbfs < -60) & ~is_orig
    df.loc[df.silent, "tagged"] = "(silent)"
    sim = emb @ E_train.T
    sim[is_orig] = np.where(sim[is_orig] > 0.99999, -1, sim[is_orig])  # (val presets are never in train anyway)
    df["novelty"] = 1 - sim.max(1)
    df = pd.concat([df, P.add_prefix("p_")], axis=1)
    df.to_csv(out / "label_check.csv", index=False)

    groups = ["original", "template"] + [f"samples:{s}" for s in samplers]
    d = df.assign(agree=df.tagged == df.label)
    tab = d.groupby(["label", "group"]).agree.mean().unstack("group")[groups].loc[LABELS]
    ref = df[df.group == "original"].novelty
    near = ref.quantile(0.05)
    summ = pd.DataFrame({
        "agreement (balanced)": tab.mean(),
        "mean P(label)": d.groupby(["group", "label"]).p_label.mean().groupby("group").mean(),
        "silent": d.groupby("group").silent.mean(),
        "novelty median": d.groupby("group").novelty.median(),
        "near-copies": d.groupby("group").novelty.apply(lambda v: (v < near).mean()),
    }).loc[groups]
    print(f"\nLABEL AGREEMENT (top tag = requested / folder label), run {a.run}, tau {a.tau}, {a.n} samples per label")
    print(tab.map(lambda v: f"{v:.0%}").to_string())
    print()
    print(summ.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"(novelty = 1 - cosine to nearest train preset; near-copy = below {near:.3f}, the 5th percentile of real val presets)")
    confs = {}
    for s in samplers:
        g = df[df.group == f"samples:{s}"]
        conf = pd.crosstab(g.label, g.tagged, normalize="index").reindex(LABELS).fillna(0)
        confs[s] = conf.to_dict(orient="index")
        print(f"\nsamples:{s}: requested label (rows) -> tagged as (columns)")
        print(conf.map(lambda v: f"{v:.0%}").to_string())
    (out / "label_check.json").write_text(json.dumps(dict(
        run=a.run, tau=a.tau, n=a.n, samplers=samplers, summary=summ.to_dict(orient="index"),
        per_label=tab.to_dict(), samples_confusion=confs, near_copy_threshold=near), indent=1))


if __name__ == "__main__":
    main()
