"""Writer round-trip test, version A (PIPELINE_VAE_SURGE.md, "Check for Step 1", item 3).

For every preset, its encoded features (data/codes.parquet, the format the VAE will output) are
decoded (codec.decode) and written back into a Surge engine (SurgeIO.write_plan, Step 7.2 order).

  A1  exactness   load the patch, reset ALL parameters incl. types to defaults, write the features,
                  read back and re-encode: every written code must come out unchanged.
  A2  sound       load the patch, reset only the VALUES (types kept, so modulation routings
                  survive: a type change clears them), write the features, render as in Step 1.4,
                  and compare with the original render. Reference: the distance between the
                  original render and a second render of the unchanged patch (randomness).
                  A2 measures what the canonical form loses: masked (inaudible) values reset to
                  defaults, FX moved to canonical slots/order, second FX instances, scene B -> A.

Writes  data/roundtrip_A.parquet   one row per preset
Run:    .venv/bin/python -m pitcher.step1_dataset.roundtrip [--limit N] [--workers 4]
"""

import argparse
import json
import os
import time
from collections import Counter
from multiprocessing import Pool, TimeoutError

import numpy as np
import pandas as pd
import soundfile as sf

from pitcher.core.audio_metrics import mel_distance_db, rms_db
from pitcher.core.codec import decode
from pitcher.core.schema import EXCLUDED_FX, FX_TYPES
from pitcher.core.surge_io import DATA_HOME, PITCHER_ROOT, SurgeIO

DATA = PITCHER_ROOT / "data"
TRANSPARENT = 1.0  # dB, mel_distance_db: below ~1 dB is barely audible
PROGRESS = DATA / "roundtrip_progress"  # per-worker stage log, to locate crashes
_io = _schema = _codes = None


def _init(codes_file="codes.parquet"):
    global _io, _schema, _codes
    _io = SurgeIO()
    _schema = json.loads((DATA / "schema.json").read_text())
    _codes = pd.read_parquet(DATA / codes_file)


def _check_a1(rb):
    bad = []
    for path, nat, got in rb["types"]:
        if int(round(got)) != int(nat):
            bad.append((path, nat, got))
    for name, path, val, got in rb["flags"]:
        if (bool(got) != bool(val)) if isinstance(val, bool) else int(got) != int(val):
            bad.append((name, val, got))
    for name, path, kind, code, nat, got, lo, hi in rb["values"]:
        if kind == "cont":
            c2 = (got - lo) / (hi - lo) if hi > lo else 0.0
            ok = abs(min(max(c2, 0.0), 1.0) - code) <= 1e-4
        elif kind == "ord":
            ok = int(round(got)) == int(round(lo + code * (hi - lo)))
        elif kind == "bool":
            ok = (got >= 0.5) == (code >= 0.5)
        else:
            ok = int(round(got)) == int(nat)
        if not ok:
            bad.append((name, round(code, 5) if kind != "cat" else nat, round(got, 5)))
    for name, want, got in rb.get("routings", []):
        if abs(got - want) > 1e-3:
            bad.append((name, round(want, 4), round(got, 4)))
    for where, want, got in rb.get("wavetables", []):
        if got != want:
            bad.append((f"wavetable {where}", want, got))
    return bad


def _stage(pid, stage):
    with open(PROGRESS / f"{os.getpid()}.log", "a") as f:
        f.write(f"{pid} {stage}\n")


def _last_stage(pid):
    stages = [line.split(" ", 1)[1].strip() for f in PROGRESS.glob("*.log")
              for line in f.read_text().splitlines() if line.split(" ", 1)[0] == str(pid)]
    return stages[-1] if stages else "not started"


def _one(task):
    pid, file = task
    fxp = DATA_HOME / file
    _stage(pid, "decode")
    row = {k: float(v) for k, v in _codes.loc[pid].items()}
    plan = decode(row, _schema, wavetable_fallback=False)  # written onto the preset's own patch

    # A1: full reset, write, read back
    _stage(pid, "A1 load+neutralise+write")
    _io.load(fxp)
    _io.neutralise(reset_types=True)
    _io.write_plan(plan)
    bad = _check_a1(_io.read_back(plan))
    n_items = (len(plan.types) + len(plan.subtypes) + len(plan.flags) + len(plan.values)
               + len(plan.routings) + len(plan.wavetables))

    # A2: keep types, reset values, write, render; reference = second render of the original
    _stage(pid, "A2 load+neutralise+write")
    _io.load(fxp)
    orig_slots = [int(_io.value(f"fx/{s}/type")) for s in range(16)]
    disabled = int(_io.value("fx_disable"))
    excluded_fx = sorted({FX_TYPES[t] for s, t in enumerate(orig_slots)
                          if t and FX_TYPES[t] in EXCLUDED_FX and not (disabled >> s) & 1})
    _io.neutralise(reset_types=False)
    _io.write_plan(plan)
    new_slots = [int(_io.value(f"fx/{s}/type")) for s in range(16)]
    _stage(pid, "A2 render rebuilt")
    rebuilt, _ = _io.render()
    _stage(pid, "A2 render original again")
    _io.load(fxp)
    again, _ = _io.render()
    _stage(pid, "done")
    orig, _ = sf.read(DATA / "audio" / "fixed" / f"{pid}.wav", dtype="float32")
    orig = orig.T

    return dict(preset_id=pid, a1_items=n_items, a1_mismatches=len(bad), a1_examples=str(bad[:5]),
                plan_warnings="; ".join(plan.warnings),
                d_rebuilt=mel_distance_db(orig, rebuilt), d_rerender=mel_distance_db(orig, again),
                rms_diff_db=rms_db(rebuilt) - rms_db(orig),
                fx_layout_changed=orig_slots != new_slots, excluded_fx=",".join(excluded_fx))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--codes", default="codes.parquet", help="feature file in data/ (e.g. codes_pruned.parquet)")
    ap.add_argument("--out", default="roundtrip_A.parquet", help="result file in data/")
    args = ap.parse_args()

    splits = pd.read_parquet(DATA / "splits.parquet")
    presets = pd.read_parquet(DATA / "presets.parquet").merge(splits[["preset_id", "split"]], on="preset_id")
    if args.limit:
        presets = presets.groupby("label").head(max(1, args.limit // 8))
    tasks = list(zip(presets["preset_id"], presets["file"]))

    # A crash inside the C++ engine kills a worker without a Python error, and Pool then waits
    # forever for its task. So: wait with a timeout, then re-run unfinished presets one by one in
    # fresh processes; a preset that still doesn't finish is recorded as a crash.
    PROGRESS.mkdir(exist_ok=True)
    for f in PROGRESS.glob("*.log"):
        f.unlink()
    t0, rows = time.time(), {}
    pool = Pool(args.workers, initializer=_init, initargs=(args.codes,))
    it = pool.imap_unordered(_one, tasks, chunksize=1)
    while True:
        try:
            r = it.next(timeout=180)
        except StopIteration:
            break
        except TimeoutError:
            print("  no result for 180 s: a worker probably crashed", flush=True)
            break
        rows[r["preset_id"]] = r
        if len(rows) % 200 == 0:
            print(f"  {len(rows)}/{len(tasks)}, {time.time() - t0:.0f}s", flush=True)
    pool.terminate()

    crashed = []
    for pid, file in [t for t in tasks if t[0] not in rows]:
        with Pool(1, initializer=_init, initargs=(args.codes,)) as single:
            try:
                r = single.apply_async(_one, ((pid, file),)).get(timeout=180)
                rows[pid] = r
            except TimeoutError:
                last = _last_stage(pid)
                crashed.append((pid, file, last))
                print(f"  CRASH: preset {pid} ({file}), last stage: {last}", flush=True)
            single.terminate()
    pd.DataFrame(crashed, columns=["preset_id", "file", "last_stage"]).to_csv(DATA / "roundtrip_crashes.csv", index=False)

    res = pd.DataFrame(list(rows.values())).merge(presets[["preset_id", "label", "name", "active_scene", "split"]], on="preset_id")
    res.to_parquet(DATA / args.out, index=False)

    print(f"\n{len(res)} presets in {time.time() - t0:.0f}s")
    # A1
    print(f"\nA1 exactness: {res.a1_items.sum()} written items, {res.a1_mismatches.sum()} mismatches, "
          f"in {(res.a1_mismatches > 0).sum()} presets")
    if res.a1_mismatches.sum():
        ex = Counter()
        for s in res.a1_examples[res.a1_mismatches > 0]:
            for item in eval(s):
                ex[str(item[0]).split("/")[0] + "/" + "/".join(str(item[0]).split("/")[1:3])] += 1
        print("  mismatches by feature group:", dict(ex.most_common(12)))
    print(f"  plan warnings: {(res.plan_warnings != '').sum()} presets")
    # A2
    res["transparent"] = res.d_rebuilt <= np.maximum(TRANSPARENT, 1.5 * res.d_rerender)
    print(f"\nA2 sound: distance original vs rebuilt: median {res.d_rebuilt.median():.4f}, "
          f"90th pct {res.d_rebuilt.quantile(.9):.4f}; original vs re-render (randomness): "
          f"median {res.d_rerender.median():.4f}, 90th pct {res.d_rerender.quantile(.9):.4f}")
    print(f"  transparent (rebuilt <= max({TRANSPARENT}, 1.5 x re-render)): {res.transparent.mean():.1%}")
    print("  transparent share by FX layout changed:", res.groupby("fx_layout_changed").transparent.agg(["mean", "size"]).round(3).to_dict())
    res["has_excluded_fx"] = res.excluded_fx != ""
    print("  transparent share by excluded FX (Airwindows etc.) present:", res.groupby("has_excluded_fx").transparent.agg(["mean", "size"]).round(3).to_dict())
    clean = res[~res.has_excluded_fx & (res.active_scene == 0)]
    print(f"  transparent share without excluded FX and scene A: {clean.transparent.mean():.1%} of {len(clean)}")
    print("  transparent share by active scene:", res.groupby("active_scene").transparent.agg(["mean", "size"]).round(3).to_dict())
    print("  transparent share by label:", res.groupby("label").transparent.mean().round(3).to_dict())
    print("  level change rebuilt vs original (dB): median %.2f, 5th pct %.2f, 95th pct %.2f" %
          (res.rms_diff_db.median(), res.rms_diff_db.quantile(.05), res.rms_diff_db.quantile(.95)))
    worst = res.sort_values("d_rebuilt", ascending=False).head(10)
    print(worst[["preset_id", "label", "name", "d_rebuilt", "d_rerender", "rms_diff_db", "excluded_fx", "active_scene"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
