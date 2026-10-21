"""Step 3.4: audio for the listening test that calibrates "same sound / same character".

20 val presets (Bass, Lead, Pad, Keys x3; Pluck, Synth, Atmosphere, Orchestral x2). Per preset three
rebuilt versions from three distance bands, close (< 6 dB), medium (6-12 dB) and far (>= 12 dB),
taken from versions that already exist:
  A        original patch as base, pruned features written (round-trip A)
  B<K>     template + 13 standard routings, top-K core groups (round-trip B, data/core_curve.csv)
plus 4 control pairs: the original against a fresh render of itself (Surge's own randomness).
Each chosen version is re-rendered and its distance re-measured. Per preset one gain (peak -3 dBFS
over the original and its versions) keeps relative loudness; files are AAC in .mp4 for the page.
Blind: pairs in random order with neutral ids (pair-NN) and file names (ref-NN for originals).

Writes  runs/step3/listening/audio/*.mp4, runs/step3/listening/pairs.json + index.html (from
        pitcher/step3_core/listening_page.template.html), data/listening_key.csv (the key: page id, version, band, distance)
Run:    .venv/bin/python -m pitcher.step3_core.listening
"""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from pitcher.core.audio_metrics import mel_distance_db
from pitcher.core.codec import decode
from pitcher.step2_explore.reference import DEFAULT_TEMPLATE
from pitcher.core.surge_io import DATA_HOME, PITCHER_ROOT, SR, SurgeIO

DATA, OUT = PITCHER_ROOT / "data", PITCHER_ROOT / "runs" / "step3" / "listening"
QUOTA = {"Bass": 3, "Lead": 3, "Pad": 3, "Keys": 3, "Pluck": 2, "Synth": 2, "Atmosphere": 2, "Orchestral": 2}
BANDS = [("close", 0, 6, 3.5), ("medium", 6, 12, 8.5), ("far", 12, 99, 16.0)]
N_CONTROL, SEED = 4, 7


def to_mp4(audio, path):
    """AAC in an MPEG-4 container; .mp4 because the artifact host serves .mp4 but not .m4a."""
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        sf.write(tmp.name, audio.T, SR, subtype="PCM_16")
        subprocess.run(["afconvert", "-f", "mp4f", "-d", "aac", "-b", "160000", tmp.name, str(path)],
                       check=True, capture_output=True)


def main():
    rng = np.random.default_rng(SEED)
    schema = json.loads((DATA / "schema.json").read_text())
    names = [f["name"] for f in schema["features"]]
    sp = pd.read_parquet(DATA / "splits.parquet").set_index("preset_id")
    presets = pd.read_parquet(DATA / "presets.parquet").set_index("preset_id")
    curve = pd.read_csv(DATA / "core_curve.csv")
    curve = curve[curve.variant == "routes"]
    rt_a = pd.read_parquet(DATA / "roundtrip_pruned.parquet").set_index("preset_id")
    ranking = pd.read_csv(DATA / "core_ranking.csv")
    pruning = json.loads((DATA / "pruning.json").read_text())
    mode_frozen = [k for k, v in pruning["frozen"].items() if v["how"] == "mode"]
    routes = [n for n in names if n.startswith("route/")]
    codes, pruned = pd.read_parquet(DATA / "codes.parquet"), pd.read_parquet(DATA / "codes_pruned.parquet")

    # candidate versions with their logged distances
    cands = {}
    for pid in sp.index[sp.split == "val"]:
        c = [("A", None, float(rt_a.at[pid, "d_rebuilt"]))] if pid in rt_a.index else []
        c += [("B", int(r.k), float(r.distance)) for r in curve[curve.preset_id == pid].itertuples()]
        cands[pid] = c

    def pick(pid):
        chosen = []
        for band, lo, hi, target in BANDS:
            pool = [x for x in cands[pid] if lo <= x[2] < hi and x not in chosen]
            if not pool:
                return None
            chosen.append(min(pool, key=lambda x: abs(x[2] - target)) + (band,))
        return chosen

    selection = []
    for label, n in QUOTA.items():
        ids = [p for p in sp.index[(sp.split == "val") & (sp.label == label)] if pick(p)]
        for p in rng.choice(ids, size=min(n, len(ids)), replace=False):
            selection.append((int(p), pick(int(p))))
    controls = set(rng.choice([p for p, _ in selection], size=N_CONTROL, replace=False).tolist())

    def row_for(kind, k, pid):
        if kind == "A":
            return {n: float(v) for n, v in pruned.loc[pid].items()}
        row = {n: np.nan for n in names}
        for n in mode_frozen:
            row[n] = float(pruned.at[pid, n])
        for n in [f for feats in ranking.features[:k] for f in feats.split(";") if f in codes.columns] + routes:
            row[n] = float(codes.at[pid, n])
        return row

    io = SurgeIO()
    (OUT / "audio").mkdir(parents=True, exist_ok=True)
    pairs, key = [], []
    for pid, chosen in selection:
        orig, _ = sf.read(DATA / "audio" / "fixed" / f"{pid}.wav", dtype="float32")
        orig = orig.T
        versions = []
        for kind, k, logged, band in chosen:
            if kind == "A":
                io.load(DATA_HOME / presets.at[pid, "file"])
                io.neutralise(reset_types=False)
            else:
                io.load(DEFAULT_TEMPLATE)
            io.write_plan(decode(row_for(kind, k, pid), schema, wavetable_fallback=(kind != "A")))
            audio, _ = io.render()
            versions.append((f"{kind}{k or ''}", band, audio))
        if pid in controls:
            io.load(DATA_HOME / presets.at[pid, "file"])
            audio, _ = io.render()
            versions.append(("R", "control", audio))
        peak = max(float(np.abs(a).max()) for a in [orig] + [v[2] for v in versions])
        gain = 10 ** (-3 / 20) / max(peak, 1e-9)
        ref = f"audio/ref-{len({p['original'] for p in pairs}) + 1:02d}.mp4"
        to_mp4(orig * gain, OUT / ref)
        for vname, band, audio in versions:
            pair_id = f"p{pid}-{vname}"
            pairs.append(dict(id=pair_id, preset_id=pid, name=presets.at[pid, "name"], label=presets.at[pid, "label"],
                              original=ref, audio=audio * gain))
            key.append(dict(id=pair_id, preset_id=pid, label=presets.at[pid, "label"], name=presets.at[pid, "name"],
                            version=vname, band=band, distance_db=mel_distance_db(orig, audio)))
        print(f"  {presets.at[pid, 'label']:10s} {presets.at[pid, 'name'][:28]:28s} "
              + "  ".join(f"{k['version']}:{k['distance_db']:.1f}" for k in key if k["preset_id"] == pid), flush=True)

    # blind: random order, neutral ids and file names; the key maps page ids to versions/distances
    order = rng.permutation(len(pairs))
    pairs = [pairs[i] for i in order]
    page_id = {}
    for n, p in enumerate(pairs, 1):
        page_id[p["id"]] = f"pair-{n:02d}"
        to_mp4(p.pop("audio"), OUT / "audio" / f"pair-{n:02d}.mp4")
        p.update(id=f"pair-{n:02d}", rebuilt=f"audio/pair-{n:02d}.mp4")
    (OUT / "pairs.json").write_text(json.dumps(pairs, indent=1))
    keydf = pd.DataFrame(key)
    keydf["page_id"] = keydf["id"].map(page_id)
    keydf.to_csv(DATA / "listening_key.csv", index=False)
    blind = [{k: p[k] for k in ("id", "name", "label", "original", "rebuilt")} for p in pairs]
    (OUT / "index.html").write_text((Path(__file__).with_name("listening_page.template.html")).read_text()
                                    .replace("__PAIRS__", json.dumps(blind, ensure_ascii=False)))
    size = sum(f.stat().st_size for f in (OUT / "audio").glob("*.mp4")) / 1e6
    k = pd.DataFrame(key)
    print(f"\n{len(pairs)} pairs ({len(selection)} presets, {N_CONTROL} controls), audio {size:.1f} MB")
    print(k.groupby("band").distance_db.describe()[["count", "min", "50%", "max"]].round(1).to_string())


if __name__ == "__main__":
    main()
