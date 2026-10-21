"""Text -> preset dataset, matching (TEXT2PRESET_PLAN.md §5): Stable Audio clips -> Surge presets.

Per kept clip (post-processing: single held note, data/text2preset/<name>/clips.parquet):
  1. clean: 48 kHz, leading silence trimmed and loudest 50 ms window at -20 dBFS (postprocess.normalise,
     the Surge render rule), first 6 s -> the stored mel view (stage1.mel_view)
  2. candidates: N from the inverter (candidate 0 = most likely, others with sampled switches) + the
     preset of the nearest stored training sound (retrieval); all rendered at C3 (seed 7), the closest
     (level-matched mel distance, stage1.exact) is kept
  3. octave: the clips are on C2..C5 (Stable Audio's own octave choice). The match is made as the clip
     sounds, so a preset matched to a C4 clip plays C4 when C3 is pressed. Reset: scene octave moved down
     by k = round(semitones from C3 / 12) steps (any remainder outside -3..+3 onto the active oscillators'
     octaves); check: the reset preset rendered at the clip's own note (C3 + 12k) against the clip
     (keytracked filters / envelopes can change a little)
  4. keep when the distance is within the threshold (MATCH_DB = 7.3 dB, from the clip-vs-match listening
     test, Round 35: <= 7.3 dB 6 of 10 pairs same character, 7.3-8.5 dB 1 of 12; the Surge-vs-Surge
     6.7 dB of Round 14 kept only 15% of the clips at the same quality)
Not yet: the short search from the best candidate, the second-CLAP check (§5 steps 4 and 6).

Writes  data/text2preset/<name>/matched.parquet (one row per clip: distances, octave, kept, source of
        the match) + matched_codes.parquet (kept features of the matched, octave-reset preset)
Run:    .venv/bin/python -m pitcher.text2preset.match --name pilot_small-music --audio data/small-music --run v0
        [--rekeep: only re-apply --threshold to an existing matched.parquet, no renders]
        [--joint j48: candidates from the joint VAE (sound code + category -> z -> preset) instead of inverter v0]
"""

import argparse
import json
import time

import numpy as np
import pandas as pd

from pitcher.core.surge_io import PITCHER_ROOT
from pitcher.inverter.data import MelSet, full_row
from pitcher.inverter.evaluate import candidates, load_model, retrieve
from pitcher.step4_vae.data import Spec
from pitcher.synth_data.common import run_pool
from pitcher.synth_data.stage1 import N_SAMPLES, NOTE, exact, init_worker, mel_view, render_mel

DATA = PITCHER_ROOT / "data"
SEED = 7
OCT_STEP = 1 / 6          # scene / oscillator octave: -3..+3 as codes 0..1
MATCH_DB = 7.3            # clip -> preset keep threshold (listening test, Round 35)


def clip_mel(path):
    from pitcher.text2preset.postprocess import load_48k, normalise
    x, _ = normalise(load_48k(path))
    x = x[:, :N_SAMPLES]
    if x.shape[1] < N_SAMPLES:
        x = np.pad(x, ((0, 0), (0, N_SAMPLES - x.shape[1])))
    return mel_view(x)


def _clip_task(paths):
    return [(p, clip_mel(p)) for p in paths]


def reset_octave(row, k):
    """Play k octaves lower: scene octave first, the rest on every active oscillator. -> (row, ok)."""
    row = dict(row)
    c = row["scene/octave"]
    want = c - k * OCT_STEP
    new = min(max(want, 0.0), 1.0)
    row["scene/octave"] = round(new / OCT_STEP) * OCT_STEP
    rest = round((want - new) / OCT_STEP)
    if rest:
        moved = False
        for o in range(3):
            f = f"scene/osc/{o}/octave"
            if f in row and row[f] == row[f]:
                row[f] = min(max(row[f] + rest * OCT_STEP, 0.0), 1.0)
                moved = True
        return row, moved
    return row, True


def _task(items):
    out = []
    for key, row, note, target in items:
        M, _, _, info = render_mel(row, SEED, note=note)
        d = float(exact(target.astype(np.float32), M.astype(np.float32)[None])[0]) if not info["silent"] else np.nan
        out.append((key, d))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="pilot_small-music")
    ap.add_argument("--audio", default="data/small-music")
    ap.add_argument("--run", default="v0", help="inverter run")
    ap.add_argument("--joint", default="", help="joint VAE run: use it instead of the inverter for the candidates")
    ap.add_argument("--data", default="pilot", help="stage-1 set for retrieval")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--threshold", type=float, default=MATCH_DB)
    ap.add_argument("--rekeep", action="store_true", help="re-apply the threshold to matched.parquet only")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    root = DATA / "text2preset" / args.name
    sfx = f"_{args.joint}" if args.joint else ""          # joint-VAE matches go to separate files
    if args.rekeep:
        M = pd.read_parquet(root / f"matched{sfx}.parquet")
        M["kept"] = keep_rule(M, args.threshold)
        M.to_parquet(root / f"matched{sfx}.parquet", index=False)
        write_summary(M, root, args.threshold, sfx)
        return
    clips = pd.read_parquet(root / "clips.parquet")
    clips = clips[clips.keep].reset_index(drop=True)
    if args.limit:
        clips = clips.iloc[:args.limit]
    audio = PITCHER_ROOT / args.audio
    paths = [str(audio / f) for f in clips.file]
    t0 = time.time()
    mels = {}
    for res in run_pool(_clip_task, [paths[i:i + 20] for i in range(0, len(paths), 20)], args.workers, "clip mels", init=init_worker):
        mels.update(dict(res))
    mel = np.stack([mels[p] for p in paths])
    print(f"{len(clips)} clips cleaned in {time.time() - t0:.0f}s", flush=True)

    spec = Spec()
    cands = []
    if args.joint:
        from pitcher.joint_vae import evaluate as jev
        from pitcher.joint_vae.data import JSpec, label_index
        from pitcher.joint_vae.sound_code import SoundCode
        from pitcher.joint_vae.train import load_model as load_joint
        js, (jm, _) = JSpec(spec), load_joint(args.joint)
        codes_a, ys = SoundCode()(mel), label_index(clips.category)
        for i in range(0, len(mel), 64):
            cands += jev.candidates(jm, js, codes_a[i:i + 64], ys[i:i + 64], args.n, seed=i)
    else:
        model, _ = load_model(args.run)
        for i in range(0, len(mel), 64):
            cands += candidates(model, spec, mel[i:i + 64], args.n, seed=i)
    ds = MelSet(args.data, spec)
    nn = retrieve(ds, mel)
    kept = list(ds.codes.columns.drop("id"))
    k_oct = np.round(clips.semis_from_c3.fillna(0).to_numpy() / 12).astype(int)

    # 1) all candidates at C3
    tasks = []
    for q in range(len(clips)):
        ret = full_row(spec, {f: float(ds.codes.iloc[nn[q]][f]) for f in kept})
        items = [((q, "retrieval"), ret, NOTE, mel[q])] + [((q, k), r, NOTE, mel[q]) for k, r in enumerate(cands[q])]
        tasks.append(items)
    D = {}
    for res in run_pool(_task, tasks, args.workers, "match"):
        D.update(dict(res))
    best = []
    for q in range(len(clips)):
        keys = ["retrieval"] + list(range(args.n))
        d = np.array([D[(q, k)] for k in keys], dtype=float)
        j = int(np.nanargmin(d)) if np.isfinite(d).any() else 0
        row = (full_row(spec, {f: float(ds.codes.iloc[nn[q]][f]) for f in kept}) if keys[j] == "retrieval" else cands[q][keys[j]])
        best.append(dict(key=keys[j], d=d[j], d_inv=np.nanmin(d[1:]) if np.isfinite(d[1:]).any() else np.nan,
                         d_ret=d[0], row=row))
    # 2) octave reset + check at the clip's own note
    tasks, reset = [], {}
    for q, b in enumerate(best):
        r, ok = reset_octave(b["row"], k_oct[q]) if k_oct[q] else (b["row"], True)
        reset[q] = (r, ok)
        tasks.append([((q, "reset"), r, int(NOTE + 12 * k_oct[q]), mel[q])])
    for res in run_pool(_task, tasks, args.workers, "octave check"):
        D.update(dict(res))

    rows, codes = [], []
    for q, b in enumerate(best):
        r, ok = reset[q]
        d_reset = D[(q, "reset")]
        c = clips.iloc[q]
        rows.append(dict(prompt_id=int(c.prompt_id), seed=int(c.seed), file=c.file, category=c.category, noun=c.noun,
                         split=c.split, octave_shift=int(k_oct[q]),
                         source="retrieval" if b["key"] == "retrieval" else ("joint" if args.joint else "inverter"),
                         candidate=str(b["key"]), d_match=b["d"], d_inverter=b["d_inv"], d_retrieval=b["d_ret"],
                         d_after_reset=d_reset, reset_ok=ok))
        codes.append({f: r.get(f, np.nan) for f in kept})
    M = pd.DataFrame(rows)
    M["kept"] = keep_rule(M, args.threshold)
    M.to_parquet(root / f"matched{sfx}.parquet", index=False)
    pd.DataFrame(codes).assign(prompt_id=M.prompt_id.values, seed=M.seed.values).to_parquet(root / f"matched_codes{sfx}.parquet", index=False)
    write_summary(M, root, args.threshold, sfx)
    print(f"total {time.time() - t0:.0f}s; written: {root / f'matched{sfx}.parquet'}")


def keep_rule(M, threshold):
    d = M[["d_match", "d_after_reset"]].max(axis=1)
    return M.reset_ok & np.isfinite(M.d_after_reset) & (d <= threshold)


def write_summary(M, root, threshold, sfx=""):
    summ = dict(n=len(M), threshold_db=threshold, kept=float(M.kept.mean()), n_kept=int(M.kept.sum()),
                median_match_db=float(M.d_match.median()), median_inverter_db=float(M.d_inverter.median()),
                median_retrieval_db=float(M.d_retrieval.median()), median_after_reset_db=float(M.d_after_reset.median()),
                source=M.source.value_counts().to_dict(),
                kept_by_category=M.groupby("category").kept.mean().round(3).to_dict(),
                kept_by_octave=M.groupby("octave_shift").kept.mean().round(3).to_dict())
    (root / f"matched_summary{sfx}.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
