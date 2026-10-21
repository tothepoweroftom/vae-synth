"""Step 1.4: render every kept preset to audio (PIPELINE_VAE_SURGE.md, Step 1.4).

Two renders per preset, MIDI note 48 (C3), velocity 100, 48 kHz, 32-bit float WAV:
  data/audio/fixed/<preset_id>.wav    DATASET render: key held exactly 2 s, then the release is
                                      recorded until the 50 ms RMS stays 60 dB below the loudest
                                      window (min 0.5 s, max 12 s). Same performance for every preset.
  data/audio/natural/<preset_id>.wav  LISTENING ONLY: key held until the sound is steady or has died
                                      out (0.5-6 s), same release rule. Not used for training or
                                      evaluation (the hold length would encode the category).
  data/audio_index.parquet            one row per preset and render: lengths, hold reason, levels, flags

Run:  .venv/bin/python -m pitcher.step1_dataset.render [--limit N] [--workers 4]
"""

import argparse
import time
from multiprocessing import Pool

import pandas as pd
import soundfile as sf

from pitcher.core.surge_io import DATA_HOME, PITCHER_ROOT, SR, SurgeIO

DATA = PITCHER_ROOT / "data"
AUDIO = DATA / "audio"
NOTE, VELOCITY = 48, 100
_io = None


def _init_worker():
    global _io
    _io = SurgeIO()


def _render_one(task):
    pid, file = task
    rows = []
    for mode, natural in (("fixed", False), ("natural", True)):
        _io.load(DATA_HOME / file)  # reload: resets FX, so no tail of the previous render leaks in
        t0 = time.time()
        audio, info = _io.render(note=NOTE, velocity=VELOCITY, natural=natural)
        out = AUDIO / mode / f"{pid}.wav"
        sf.write(out, audio.T, SR, subtype="FLOAT")
        rows.append(dict(preset_id=pid, mode=mode, file=str(out.relative_to(DATA)), note=NOTE,
                         velocity=VELOCITY, cpu_s=round(time.time() - t0, 3), **info))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only the first N presets (testing)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    presets = pd.read_parquet(DATA / "presets.parquet")
    if args.limit:
        presets = presets.head(args.limit)
    for mode in ("fixed", "natural"):
        (AUDIO / mode).mkdir(parents=True, exist_ok=True)

    tasks = list(zip(presets["preset_id"], presets["file"]))
    t0, rows = time.time(), []
    with Pool(args.workers, initializer=_init_worker) as pool:
        for n, r in enumerate(pool.imap_unordered(_render_one, tasks, chunksize=4), 1):
            rows.extend(r)
            if n % 200 == 0:
                print(f"  {n}/{len(tasks)} presets, {time.time() - t0:.0f}s")

    idx = pd.DataFrame(rows).sort_values(["preset_id", "mode"]).merge(
        presets[["preset_id", "label", "name"]], on="preset_id")
    idx.to_parquet(DATA / "audio_index.parquet", index=False)

    print(f"\nrendered {len(tasks)} presets x 2 in {time.time() - t0:.0f}s with {args.workers} workers")
    size_gb = sum(f.stat().st_size for f in AUDIO.rglob("*.wav")) / 1e9
    print(f"audio on disk: {size_gb:.2f} GB")
    for mode, g in idx.groupby("mode"):
        print(f"\n[{mode}] total length: median {g.total_s.median():.1f}s, max {g.total_s.max():.1f}s; "
              f"tail capped at 12 s: {g.tail_capped.sum()}; silent: {g.silent.sum()}; "
              f"samples > 0 dBFS in {(g.n_over_0dbfs > 0).sum()} presets; "
              f"max flush residual RMS {g.flush_residual_rms.max():.1e}")
        if mode == "natural":
            print("  hold reason:", g.hold_reason.value_counts().to_dict())
        print("  per label (median hold / tail / total s):")
        print(g.groupby("label")[["hold_s", "tail_s", "total_s"]].median().round(2).to_string())


if __name__ == "__main__":
    main()
