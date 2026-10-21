"""Step 1.1-1.2: load every patch through Surge's own loader and dump the raw dataset.

Writes (long format, see PIPELINE_VAE_SURGE.md Step 1 / Route 2):
  data/presets.parquet     one row per kept preset: id, file, label, author, revision, active scene
  data/raw_values.parquet  one row per (preset, parameter): value, min, max, default, type, flags
  data/routings.parquet    one row per modulation routing of a kept preset
  data/params.parquet      one row per parameter path: synth id, OSC address
  data/dropped.csv         every patch that was not kept, with the reason

Run:  .venv/bin/python -m pitcher.step1_dataset.extract
"""

import time
from collections import Counter
from pathlib import Path

import pandas as pd

from pitcher.core.surge_io import DATA_HOME, PITCHER_ROOT, SurgeIO, patch_revision

OUT = PITCHER_ROOT / "data"

# Folder name -> merged label (Step 1.2). Folders not listed here are dropped.
LABELS = {
    "Basses": "Bass", "Bass": "Bass",
    "Leads": "Lead",
    "Pads": "Pad",
    "Atmospheres": "Atmosphere", "Ambiance": "Atmosphere", "Ambiances": "Atmosphere",
    "Soundscapes": "Atmosphere", "Textures": "Atmosphere", "Drones": "Atmosphere",
    "Keys": "Keys", "Organs": "Keys",
    "Plucks": "Pluck", "Guitars": "Pluck", "Bells": "Pluck", "Mallets": "Pluck",
    "Polysynths": "Synth", "Synths": "Synth", "Chords": "Synth",
    "Brass": "Orchestral", "Winds": "Orchestral", "Woodwinds": "Orchestral",
    "Strings": "Orchestral",
}
DROP_PACKS = {"Exquis MPE", "LinnStrument MPE"}  # rely on MPE gestures
OT_AUDIOINPUT = 4  # osc_type enum, SurgeStorage.h:296


def folder_info(fxp):
    """(raw folder category, author) from the path; the label comes from the folder, not <meta>."""
    parts = fxp.relative_to(DATA_HOME).parts
    if parts[0] == "patches_factory":
        return parts[1], "Surge Team"
    if len(parts) < 4:
        return None, parts[1]
    return parts[2], parts[1]


def main():
    io = SurgeIO()
    print(f"surgepy {__import__('surgepy').getVersion()}, {len(io.params)} parameters, "
          f"extras={'yes' if io.has_extras else 'no'}, data={DATA_HOME}")

    files = sorted(DATA_HOME.glob("patches_*/**/*.fxp"))
    presets, values, routings, dropped = [], [], [], []
    t0 = time.time()

    for n, fxp in enumerate(files, 1):
        rel = str(fxp.relative_to(DATA_HOME))
        raw_cat, author = folder_info(fxp)

        def drop(reason):
            dropped.append(dict(file=rel, folder_category=raw_cat, author=author, reason=reason))

        if author in DROP_PACKS:
            drop("mpe pack")
            continue
        label = LABELS.get(raw_cat)
        if label is None:
            drop(f"category: {raw_cat}")
            continue

        if not io.load(fxp):
            drop("load failed")
            continue
        if io.value("scenemode") != 0:
            drop("scenemode not single")
            continue
        sc = int(io.value("scene_active"))
        if any(io.value(f"scene/{sc}/osc/{o}/type") == OT_AUDIOINPUT for o in range(3)):
            drop("audio input oscillator")
            continue

        pid = len(presets)
        wt = io.read_wavetable_names(sc)
        presets.append(dict(
            preset_id=pid, file=rel, name=fxp.stem, label=label, folder_category=raw_cat,
            author=author, revision=patch_revision(fxp), active_scene=sc,
            wt_osc1=wt[0], wt_osc2=wt[1], wt_osc3=wt[2],
        ))
        values.extend(dict(preset_id=pid, **r) for r in io.read_values())
        routings.extend(dict(preset_id=pid, **r) for r in io.read_routings())

        if n % 250 == 0:
            print(f"  {n}/{len(files)} files, {len(presets)} kept, {time.time() - t0:.0f}s")

    OUT.mkdir(exist_ok=True)
    pd.DataFrame(presets).to_parquet(OUT / "presets.parquet", index=False)
    pd.DataFrame(values).to_parquet(OUT / "raw_values.parquet", index=False)
    pd.DataFrame(routings).to_parquet(OUT / "routings.parquet", index=False)
    pd.DataFrame(io.param_catalog()).to_parquet(OUT / "params.parquet", index=False)
    pd.DataFrame(dropped).to_csv(OUT / "dropped.csv", index=False)

    print(f"\n{len(files)} files in {time.time() - t0:.0f}s -> {len(presets)} kept, {len(dropped)} dropped")
    print("kept per label:", dict(Counter(p["label"] for p in presets).most_common()))
    print("dropped:", dict(Counter(d["reason"].split(":")[0] for d in dropped).most_common()))


if __name__ == "__main__":
    main()
