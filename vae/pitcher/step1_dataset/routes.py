"""Step 1.3 extension (Round 15): choose the routing matrix - which (source -> destination) pairs
become depth features. Selected on the TRAIN split only.

Sources: audible in a plain note render (LFOs, velocity, keytrack, envelopes, random, ...; no mod
wheel / macros / aftertouch, which sit at 0 when a note is just played).
Destinations: any scene parameter of the active scene. Routings onto oscillator knobs (osc/<o>/p/<i>)
change meaning with the oscillator type, so they are split per type, like the knob blocks.
A pair is kept if used by >= MIN_SHARE of train presets; an oscillator-knob pair per type if that
(pair, type) combination occurs in >= MIN_TYPE_PRESETS train presets.

Writes  data/routing_matrix.json   [{source, source_id, dest, osc_type (or null), train_share}]
Run:    .venv/bin/python -m pitcher.step1_dataset.routes   (then pitcher/step1_dataset/encode.py, pitcher/step2_explore/reference.py, pitcher/step2_explore/prune.py)
"""

import json
import re

import pandas as pd

from pitcher.core.schema import OSC_TYPES
from pitcher.core.surge_io import PITCHER_ROOT

DATA = PITCHER_ROOT / "data"
MIN_SHARE, MIN_TYPE_PRESETS = 0.01, 5
AUDIBLE = r"(Voice|Scene) LFO \d|Velocity|Keytrack|Filter EG|Amp EG|Random|Alternate|Release Velocity|Lowest|Highest|Latest"


def main():
    r = pd.read_parquet(DATA / "routings.parquet")
    sp = pd.read_parquet(DATA / "splits.parquet")
    presets = pd.read_parquet(DATA / "presets.parquet").set_index("preset_id")
    raw = pd.read_parquet(DATA / "raw_values.parquet", columns=["preset_id", "path", "value"])
    tr = set(sp.preset_id[sp.split == "train"])
    n = len(tr)
    r = r[r.preset_id.isin(tr) & r.scope.isin(["scene", "voice"]) & r.source.str.fullmatch(AUDIBLE)]
    r = r[r.scene == r.preset_id.map(presets.active_scene)]
    r = r.assign(dest=r.dest.str.replace(r"^scene/\d/", "", regex=True))
    usage = r.groupby(["source", "dest"]).preset_id.nunique() / n
    source_id = r.groupby("source").source_id.first().to_dict()

    osc_type = {}  # (preset, o) -> type name, for the active scene
    types = raw[raw.path.str.fullmatch(r"scene/\d/osc/\d/type")]
    for pid, path, v in zip(types.preset_id, types.path, types.value):
        sc, o = int(path.split("/")[1]), int(path.split("/")[3])
        if pid in tr and sc == presets.at[pid, "active_scene"]:
            osc_type[(pid, o)] = OSC_TYPES[int(v)]

    matrix = []
    for (source, dest), share in usage[usage >= MIN_SHARE].sort_values(ascending=False).items():
        m = re.fullmatch(r"osc/(\d)/p/(\d)", dest)
        if not m:
            matrix.append(dict(source=source, source_id=int(source_id[source]), dest=dest, osc_type=None,
                               train_share=round(float(share), 4)))
            continue
        users = r[(r.source == source) & (r.dest == dest)].preset_id.unique()
        per_type = pd.Series([osc_type.get((p, int(m[1]))) for p in users]).value_counts()
        for t_name, cnt in per_type.items():
            if cnt >= MIN_TYPE_PRESETS:
                matrix.append(dict(source=source, source_id=int(source_id[source]), dest=dest, osc_type=t_name,
                                   train_share=round(cnt / n, 4)))

    (DATA / "routing_matrix.json").write_text(json.dumps(matrix, indent=1))
    plain = [m for m in matrix if m["osc_type"] is None]
    print(f"{len(matrix)} routing features: {len(plain)} plain + {len(matrix) - len(plain)} oscillator-knob (per type)")
    print("sources:", pd.Series([m["source"] for m in matrix]).value_counts().to_dict())
    print("top 10:", [f"{m['source']} -> {m['dest']}{' [' + m['osc_type'] + ']' if m['osc_type'] else ''} ({m['train_share']:.1%})"
                      for m in matrix[:10]])


if __name__ == "__main__":
    main()
