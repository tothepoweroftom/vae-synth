"""Checks for Step 1 (PIPELINE_VAE_SURGE.md, "Check for Step 1" and Appendix A).

1. Label counts match the Step 1.2 table.
2. Parameters missing from a .fxp got the Appendix A.2 fill-in value from Surge's loader.
3. Parameters present in a .fxp: the value surgepy reports equals the value in the XML,
   except where a revision upgrade step (Appendix A.4) explains the difference.

Run after pitcher/step1_dataset/extract.py:  .venv/bin/python -m pitcher.step1_dataset.verify
"""

import re
import struct
from pathlib import Path

import numpy as np
import pandas as pd

from pitcher.core.surge_io import DATA_HOME, PITCHER_ROOT, SURGE_ROOT

DATA = PITCHER_ROOT / "data"
EXPECTED = {"Bass": 412, "Lead": 396, "Pad": 326, "Keys": 254, "Pluck": 248,
            "Synth": 188, "Atmosphere": 151, "Orchestral": 147}

# Appendix A.2: storage-name pattern -> (expected natural value, only for revisions <= this or None)
FILL_INS = [
    (r"[ab]_send_fx_[34]", 0.0, None),
    (r"volume_FX[34]", 1.0, None),
    (r"[ab]_env[12]_mode", 0.0, None),
    (r"character", 0.0, 9),
    (r"fx\d+_type", 0.0, None),
]
XML_PARAM = re.compile(r'<(\w+) type="(\d)" value="([^"]*)"')

# Classes the engine spawns per type (dsp/Oscillator.cpp spawn_osc, dsp/Effect.cpp spawn_effect)
OSC_CLASSES = ["ClassicOscillator", "SineOscillator", "WavetableOscillator", "SampleAndHoldOscillator",
               "AudioInputOscillator", "FM3Oscillator", "FM2Oscillator", "WindowOscillator",
               "ModernOscillator", "StringOscillator", "TwistOscillator", "AliasOscillator"]
FX_CLASSES = ["-", "DelayEffect", "Reverb1Effect", "PhaserEffect", "RotarySpeakerEffect",
              "DistortionEffect", "ParametricEQ3BandEffect", "FrequencyShifterEffect",
              "ConditionerEffect", "ChorusEffect", "VocoderEffect", "Reverb2Effect", "FlangerEffect",
              "RingModulatorEffect", "AirWindowsEffect", "NeuronEffect", "GraphicEQ11BandEffect",
              "ResonatorEffect", "CHOWEffect", "ExciterEffect", "BBDEnsembleEffect",
              "CombulatorEffect", "NimbusEffect", "TapeEffect", "TreemonsterEffect",
              "WaveShaperEffect", "MSToolEffect", "SpringReverbEffect", "BonsaiEffect",
              "AudioInputEffect", "FloatyDelayEffect", "ConvolutionEffect"]


def max_streaming_threshold(cls):
    """Highest patch revision that the class's handleStreamingMismatches still converts."""
    dsp = SURGE_ROOT / "src" / "common" / "dsp"
    best = -1
    for f in list(dsp.rglob(f"{cls}.cpp")) + list(dsp.rglob(f"{cls}.h")) + list(dsp.rglob(f"{cls}Impl.h")):
        for op, n in re.findall(r"streamingRevision\s*(<=|<)\s*(\d+)", f.read_text(errors="ignore")):
            best = max(best, int(n) - (op == "<"))
    return best


def float_from_int_bits(x):
    try:
        return struct.unpack("<f", struct.pack("<i", int(x)))[0]
    except (struct.error, OverflowError, ValueError):
        return np.nan


def explain(r, osc_type, fx_type, osc_max, fx_max):
    """Name the loader conversion that explains a loaded-vs-XML difference, or None."""
    n, rev, x, v = r.storage_name, r.revision, r.xml_value, r.value
    if abs(x) > 1e6 and np.isclose(float_from_int_bits(x), v, atol=1e-5):
        return "old file stored a float as its int bit pattern"
    if n == "polylimit" and rev <= 15 and x == 8:
        return "polylimit 8 -> 16 (SurgePatch.cpp:2719)"
    if n == "volume" and rev < 17:
        return "global volume ignored for rev < 17 (SurgePatch.cpp:1954)"
    if re.fullmatch(r"[ab]_filter\d_(sub)?type", n) and (rev < 15 or rev <= 26):
        return "filter remap rev < 15 (SurgePatch.cpp:2616) / legacy subtypes rev <= 26 (:2783)"
    if re.fullmatch(r"[ab]_lfo\d+_magnitude", n) and rev <= 25 and np.isclose(v, 2 * x - 1, atol=1e-5):
        return "LFO magnitude 2v-1 for rev <= 25 (SurgePatch.cpp:2768)"
    m = re.fullmatch(r"([ab])_osc(\d)_(\w+)", n)
    if m:
        t = osc_type.get((r.preset_id, m[1], int(m[2])))
        if t == 10 and m[3] == "pitch" and rev <= 27:
            return "Twist pitch +4.61 cents for rev <= 27 (SurgePatch.cpp:2808)"
        if t == 1 and m[3] == "param0" and rev <= 27:
            return "Sine waveform renumbering for rev <= 27 (SurgePatch.cpp:2808)"
        if t is not None and rev <= osc_max[t]:
            return f"{OSC_CLASSES[t]}::handleStreamingMismatches (rev <= {osc_max[t]})"
    m = re.fullmatch(r"fx(\d+)_p(\d+)", n)
    if m:
        t = fx_type.get((r.preset_id, int(m[1])))
        if (x < r.min or x > r.max) and np.isclose(v, min(max(x, r.min), r.max), atol=1e-5):
            return "FX value outside its range, clamped by loadFx (SurgeSynthesizer.cpp:3383)"
        if t == 13 and m[2] == "0" and rev <= 27:
            return "Ring modulator waveform renumbering for rev <= 27 (SurgePatch.cpp:2858)"
        if t and rev <= fx_max[t]:
            return f"{FX_CLASSES[t]}::handleStreamingMismatches (rev <= {fx_max[t]})"
    return None


def xml_values(fxp):
    text = fxp.read_bytes().decode("latin1")
    block = text[text.index("<parameters>"): text.index("</parameters>")]
    return {name: float(v) for name, _, v in XML_PARAM.findall(block)}


def main():
    presets = pd.read_parquet(DATA / "presets.parquet")
    raw = pd.read_parquet(DATA / "raw_values.parquet", columns=["preset_id", "path", "value", "min", "max", "valtype"])
    params = pd.read_parquet(DATA / "params.parquet")
    assert params["storage_name"].notna().all(), "build surgepy with the Step 0.3 extras"
    raw = raw.merge(params[["path", "storage_name"]], on="path")

    # --- 1. label counts -------------------------------------------------------------------
    counts = presets["label"].value_counts().to_dict()
    print("1. Label counts (kept vs expected):")
    for lab, exp in EXPECTED.items():
        print(f"   {lab:11s} {counts.get(lab, 0):4d}  expected {exp:4d}  {'OK' if counts.get(lab, 0) == exp else 'DIFF'}")
    print(f"   total {len(presets)}  expected {sum(EXPECTED.values())}")
    print(f"   rows in raw_values: {len(raw)} = {len(presets)} presets x {raw['path'].nunique()} params; "
          f"NaN values: {raw['value'].isna().sum()}")

    # --- XML of every kept preset ------------------------------------------------------------
    xml_rows = []
    for pid, file in zip(presets["preset_id"], presets["file"]):
        for name, v in xml_values(DATA_HOME / file).items():
            xml_rows.append((pid, name, v))
    xml = pd.DataFrame(xml_rows, columns=["preset_id", "storage_name", "xml_value"])
    m = raw.merge(xml, on=["preset_id", "storage_name"], how="left")
    m = m.merge(presets[["preset_id", "revision", "file"]], on="preset_id")

    # --- 2. Appendix A fill-ins --------------------------------------------------------------
    print("\n2. Missing from the file -> Appendix A.2 fill-in:")
    missing = m[m["xml_value"].isna()]
    for pattern, expected, max_rev in FILL_INS:
        sel = missing[missing["storage_name"].str.fullmatch(pattern)]
        if max_rev is not None:
            sel = sel[sel["revision"] <= max_rev]
        bad = sel[~np.isclose(sel["value"], expected)]
        print(f"   {pattern:20s} fill {expected:<4}  missing in {sel['preset_id'].nunique():4d} presets "
              f"({len(sel)} values), mismatches: {len(bad)}")
        if len(bad):
            print(bad[["file", "storage_name", "value"]].head().to_string(index=False))
    other = missing[~missing["storage_name"].str.fullmatch("|".join(p for p, _, _ in FILL_INS))
                    & ~missing["storage_name"].str.match(r"fx\d+_p\d+")]
    print(f"   other missing (not FX slot params, not in table): {other['storage_name'].nunique()} names "
          f"{sorted(other['storage_name'].unique())[:12]}")

    # --- 3. present in the file: XML value vs loaded value ------------------------------------
    present = m[m["xml_value"].notna()].copy()
    tol = 1e-4 * np.maximum(1.0, np.abs(present["xml_value"]))
    present["match"] = np.abs(present["value"] - present["xml_value"]) <= tol
    print("\n3. Present in the file -> loaded value equals XML value:")
    by_rev = present.groupby(present["revision"].apply(lambda r: "< 15" if r < 15 else ">= 15"))["match"]
    for k, g in by_rev:
        print(f"   revision {k:5s}: {g.size:8d} values, mismatches {(~g).sum():6d} ({(~g).mean():.3%})")
    bad = present[~present["match"]].copy()

    # Context needed to explain mismatches: oscillator and FX type per slot, as loaded
    types = raw[raw["path"].str.fullmatch(r"scene/\d/osc/\d/type|fx/\d+/type")]
    osc_type, fx_type = {}, {}
    for pid, path, v in zip(types["preset_id"], types["path"], types["value"]):
        parts = path.split("/")
        if parts[0] == "scene":
            osc_type[(pid, "ab"[int(parts[1])], int(parts[3]) + 1)] = int(v)
        else:
            fx_type[(pid, int(parts[1]) + 1)] = int(v)
    osc_max = [max_streaming_threshold(c) for c in OSC_CLASSES]
    fx_max = [-1] + [max_streaming_threshold(c) for c in FX_CLASSES[1:]]

    bad["cause"] = [explain(r, osc_type, fx_type, osc_max, fx_max) for r in bad.itertuples()]
    print(f"   explained by a loader conversion: {bad['cause'].notna().sum()} of {len(bad)}")
    print(bad["cause"].fillna("UNEXPLAINED").str.replace(r" \(rev <= \d+\)", "", regex=True)
          .value_counts().to_string())
    unexplained = bad[bad["cause"].isna()]
    if len(unexplained):
        print("\n   UNEXPLAINED mismatches:")
        print(unexplained[["file", "revision", "storage_name", "xml_value", "value"]].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
