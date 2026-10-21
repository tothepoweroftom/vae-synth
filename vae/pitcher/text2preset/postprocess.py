"""Text -> preset dataset, step 4: analyse and filter the Stable Audio clips (TEXT2PRESET_PLAN.md, 2.5).

Per clip (44.1 kHz stereo FLAC from generate_sa3.py):
  same conditions as the Surge renders  resample to 48 kHz; trim leading silence and the tail once the
                     50 ms RMS stays 60 dB below the loudest window (the render rule, Step 1.4); scale so
                     the loudest 50 ms window matches the median Surge render (-20 dBFS RMS)
  pitch              pYIN (librosa) on the audible frames: main note, distance to C3 (MIDI 48), number of
                     distinct held notes (a melody / arpeggio = several), harmonic fit of the spectrum
  envelope           attack time, level after 1 / 2 / 4 s, level at the very end (cut off at 6 s?)
  drums / rhythm     percussive share (harmonic-percussive split) and number of onsets after 0.3 s
  CLAP               fingerprint (laion/clap-htsat-unfused, as the tagger), the category tagger, the
                     similarity to the clip's own description and its rank among all descriptions, the
                     nearest Surge preset (how reachable the sound is), seed-to-seed consistency
  domain alignment   to CLAP every Stable Audio clip partly sounds like "a Stable Audio clip" (the two sets'
                     mean fingerprints have cosine 0.44, Round 28): *_aligned columns use the fingerprint
                     shifted by (mean of the Surge train renders - mean of all clips), renormalised; no
                     labels involved. Raw columns are kept.
Flags: silent, multi_note, rhythmic (reject); off_pitch, chord_like, cut_off, clipped (keep, but marked).
Thresholds are calibrated on Surge renders analysed with the same code (--controls N; single notes at
C3 by construction), so a flag fires on only a few % of real Surge presets.

Writes  data/text2preset/<name>/clips.parquet   one row per clip: metrics, flags, keep, tagger, CLAP scores
        data/text2preset/<name>/clap.parquet    prompt_id, seed + e0..e511 (normalised clip)
        data/text2preset/<name>/surge_controls.parquet  the same metrics for N Surge renders
        runs/text2preset/<name>/report.md + plots
Run:    .venv/bin/python -m pitcher.text2preset.postprocess --audio data/small-music --name pilot_small-music
"""

import argparse
import json
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import yaml

from pitcher.core.surge_io import PITCHER_ROOT, SR

DATA = PITCHER_ROOT / "data"
C3 = 48
SR_PITCH, HOP = 22050, 256
WIN, WIN_HOP = int(0.05 * SR), int(0.01 * SR)  # 50 ms RMS windows every 10 ms
TARGET_RMS_DB = -20.0                          # median loudest-window RMS of the Surge fixed renders
REL_FLOOR_DB = 60.0                            # render rule: tail ends 60 dB below the loudest window


# ---------------------------------------------------------------- audio --------------------------------
def load_48k(path):
    """-> float32 (2, n) at 48 kHz."""
    import librosa
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.T
    if x.shape[0] == 1:
        x = np.vstack([x, x])
    return librosa.resample(x, orig_sr=sr, target_sr=SR, res_type="soxr_hq") if sr != SR else x


def rms_env_db(mono):
    if len(mono) < WIN:
        mono = np.pad(mono, (0, WIN - len(mono)))
    w = np.lib.stride_tricks.sliding_window_view(mono, WIN)[::WIN_HOP]
    return 20 * np.log10(np.sqrt((w.astype(np.float64) ** 2).mean(1)) + 1e-12)


def normalise(x):
    """Trim + level like the Surge renders. -> (x, info)."""
    env = rms_env_db(x.mean(0))
    top = env.max()
    loud = np.flatnonzero(env > top - REL_FLOOR_DB)
    a = max(loud[0] * WIN_HOP - int(0.005 * SR), 0)
    b = min(loud[-1] * WIN_HOP + WIN, x.shape[1])
    b = max(b, min(a + int(0.5 * SR), x.shape[1]))
    gain = TARGET_RMS_DB - top
    y = x[:, a:b] * 10 ** (gain / 20)
    return y.astype(np.float32), dict(lead_s=a / SR, trimmed_s=(b - a) / SR, gain_db=gain, max_rms_db=top)


# ---------------------------------------------------------------- analysis -----------------------------
def _runs(seq):
    """run lengths of equal consecutive values: [(value, length), ...]"""
    out, start = [], 0
    for i in range(1, len(seq) + 1):
        if i == len(seq) or seq[i] != seq[start]:
            out.append((seq[start], i - start)); start = i
    return out


def spectral_peaks(y, sr, frames, fmin):
    """Peaks (freqs, energies) of the average spectrum over the given frames, fmin..5 kHz, within 40 dB."""
    from scipy.signal import find_peaks
    n_fft = 8192
    if len(frames) == 0:
        return np.array([]), np.array([])
    idx = frames[np.linspace(0, len(frames) - 1, min(len(frames), 24)).astype(int)]
    segs = [y[i * HOP:i * HOP + n_fft] for i in idx]
    segs = [np.pad(s, (0, n_fft - len(s))) for s in segs]
    mag = np.mean([np.abs(np.fft.rfft(s * np.hanning(n_fft))) for s in segs], axis=0)
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    band = (freqs > fmin) & (freqs < 5000)
    db = 20 * np.log10(mag + 1e-12)
    pk, _ = find_peaks(np.where(band, db, -300), height=db[band].max() - 40, prominence=6)
    return freqs[pk], mag[pk] ** 2


def harmonic_fit(f, e, f0):
    """Energy share of the peaks within 40 cents of a harmonic of f0."""
    if len(f) == 0:
        return np.nan
    k = np.maximum(np.round(f / f0), 1)
    return float(e[1200 * np.abs(np.log2(f / (k * f0))) < 40].sum() / e.sum())


def reattacks(env, top, after=0.3):
    """Times where the 50 ms RMS rises >= 6 dB within 50 ms (and is within 30 dB of the peak); >= 80 ms apart."""
    rise = env[5:] - np.minimum.reduce([env[i:len(env) - 5 + i] for i in range(5)])
    hits = np.flatnonzero((rise >= 6) & (env[5:] > top - 30)) + 5
    times, last = [], -1.0
    for h in hits * 0.01:
        if h - last >= 0.08:
            times.append(h)
        last = h
    return [t for t in times if t > after]


def analyse(x, raw_peak=None):
    """Metrics of one clip x (2, n) at 48 kHz (already normalised)."""
    import librosa
    mono = x.mean(0)
    out = {}
    env = rms_env_db(mono)
    top = env.max()
    t = lambda s: env[int(s / 0.01)] - top if int(s / 0.01) < len(env) else -100.0
    out["attack_s"] = float(np.argmax(env >= top - 3) * 0.01)
    out["lvl_1s"], out["lvl_2s"], out["lvl_4s"] = t(1.0), t(2.0), t(4.0)
    out["dur_s"] = len(mono) / SR
    side, mid = 0.5 * (x[0] - x[1]), 0.5 * (x[0] + x[1])
    out["width_db"] = float(10 * np.log10((side ** 2).sum() / ((mid ** 2).sum() + 1e-12) + 1e-12))

    y = librosa.resample(mono, orig_sr=SR, target_sr=SR_PITCH, res_type="soxr_hq")
    f0, voiced, _ = librosa.pyin(y, fmin=32.7, fmax=2093.0, sr=SR_PITCH, frame_length=2048, hop_length=HOP)
    fenv = librosa.amplitude_to_db(librosa.feature.rms(y=y, frame_length=2048, hop_length=HOP)[0], ref=np.max)
    n = min(len(f0), len(fenv))
    audible = fenv[:n] > -30
    v = voiced[:n] & audible & np.isfinite(f0[:n])
    out["voiced_frac"] = float(v.sum() / max(audible.sum(), 1))
    out.update(main_midi=np.nan, semis_from_c3=np.nan, pc_offset=np.nan, n_notes=0, main_share=np.nan,
               pitch_std_cents=np.nan, harmonic_fit=np.nan, low_partial=np.nan)
    if v.sum() >= 10 and out["voiced_frac"] >= 0.3:
        midi = librosa.hz_to_midi(f0[:n])
        qv = np.round(midi[v])
        vals, counts = np.unique(qv, return_counts=True)
        main = float(np.median(midi[v][qv == vals[counts.argmax()]]))  # the most common note (never empty)
        q = np.round(np.where(v, midi - main, 0)).astype(int)
        d = ((q + 6) % 12) - 6                      # semitones from the main note, octave errors folded
        seq = np.where(v, d, 99)                    # 99 = break (unvoiced / quiet)
        held = Counter()
        for val, ln in _runs(list(seq)):
            if val != 99 and ln * HOP / SR_PITCH >= 0.15:
                held[val] += ln
        tot = v.sum()
        notes = [k for k, c in held.items() if c / tot >= 0.10]
        on_main = v & (d == 0)
        # pYIN can report a "missing fundamental" below the sound (a sub-octave, or C1 for a C3 major chord):
        # the lowest strong partial (within 25 dB of the strongest) is taken as the heard note, and the
        # harmonic fit is measured against it (a chord's other notes don't fit; a single note's do)
        f0 = librosa.midi_to_hz(main)
        f, e = spectral_peaks(y, SR_PITCH, np.flatnonzero(on_main), 0.8 * f0)
        if len(f):
            strong = f[e >= e.max() * 10 ** (-25 / 10)]
            k = max(int(round(strong.min() / f0)), 1)
            main, f0 = main + 12 * np.log2(k), f0 * k
            out["low_partial"] = k
            out["harmonic_fit"] = harmonic_fit(f[f > 0.8 * f0], e[f > 0.8 * f0], f0)
        out.update(main_midi=main, semis_from_c3=main - C3, pc_offset=float(((main - C3 + 6) % 12) - 6),
                   n_notes=max(len(notes), 1), main_share=float(on_main.sum() / tot),
                   pitch_std_cents=float(np.std(100 * (midi[on_main] - np.median(midi[on_main])))) if on_main.sum() > 2 else np.nan)

    h, p = librosa.effects.hpss(y)
    out["perc_share"] = float((p ** 2).sum() / ((h ** 2).sum() + (p ** 2).sum() + 1e-12))
    out["n_onsets"] = len(reattacks(env, top))
    out["flatness"] = float(np.median(librosa.feature.spectral_flatness(y=y, hop_length=HOP)))
    loud = fenv > -30
    cen = librosa.feature.spectral_centroid(y=y, sr=SR_PITCH, hop_length=HOP)[0][:len(fenv)]
    out["centroid_hz"] = float(np.median(cen[loud[:len(cen)]])) if loud.any() else np.nan
    return out


def _sa3_worker(path):
    x = load_48k(path)
    raw = x
    xn, info = normalise(x)
    m = analyse(xn)
    env = rms_env_db(raw.mean(0))
    orig, _ = sf.read(path, dtype="float32")  # count clipping in the file itself (resampling overshoots)
    m.update(info, n_clip=int((np.abs(orig) >= 0.9999).sum()),
             end_db=float(env[-25:].mean() - env.max()))
    return m


def _surge_worker(pid):
    x, sr = sf.read(DATA / "audio" / "fixed" / f"{pid}.wav", dtype="float32", always_2d=True)
    x = x.T if x.shape[1] <= 2 else x
    if x.shape[0] == 1:
        x = np.vstack([x, x])
    xn, info = normalise(x)
    m = analyse(xn)
    m.update(info, preset_id=pid)
    return m


# ---------------------------------------------------------------- flags --------------------------------
RULES = {  # name: (description, reject?)
    "silent": ("loudest 50 ms below -50 dBFS", True),
    "multi_note": ("two or more distinct held notes (melody, arpeggio, chord change)", True),
    "rhythmic": ("percussive share and onsets both above the Surge 98th percentile (drums, loops)", True),
    "off_pitch": ("tonal, main note more than 1 semitone from any C", False),
    "chord_like": ("partials don't fit one note (harmonic fit below the Surge 5th percentile): chord or inharmonic", False),
    "cut_off": ("still within 20 dB of its peak at the end of the 6 s clip", False),
    "clipped": ("more than 10 samples at full scale", False),
}


def thresholds(ctrl):
    """Calibrated on Surge renders: rhythmic only above their 98th percentiles."""
    return dict(perc=float(np.percentile(ctrl.perc_share, 98)), onsets=float(np.percentile(ctrl.n_onsets, 98)),
                fit=float(np.nanpercentile(ctrl.harmonic_fit, 5)))


def add_flags(df, th, sa3=True):
    df["silent"] = df.max_rms_db < -50 if sa3 else False
    df["multi_note"] = df.n_notes >= 2
    df["rhythmic"] = (df.perc_share > th["perc"]) & (df.n_onsets > th["onsets"])
    df["off_pitch"] = df.pc_offset.abs() > 1
    df["chord_like"] = df.harmonic_fit < th["fit"]
    df["cut_off"] = (df.end_db > -20) if sa3 else False
    df["clipped"] = (df.n_clip > 10) if sa3 else False
    df["keep"] = ~df[[k for k, (_, rej) in RULES.items() if rej]].any(axis=1)
    return df


# ---------------------------------------------------------------- CLAP ---------------------------------
def clap_part(meta, audio_dir, prompts, suffix):
    from pitcher.step8_eval.clap import Clap, cache_file
    from pitcher.step8_eval.tagger import Tagger
    tagger = Tagger()
    clap = tagger.clap
    emb = []
    for i in range(0, len(meta), 64):
        emb.append(clap.audio([normalise(load_48k(audio_dir / f))[0] for f in meta.file.iloc[i:i + 64]]))
        print(f"  CLAP {i + len(emb[-1])}/{len(meta)}", flush=True)
    E = np.vstack(emb)
    P = tagger.proba_from(E)
    out = pd.DataFrame(index=meta.index)
    out["tagged"] = P.idxmax(axis=1).to_numpy()
    for c in P.columns:
        out[f"p_{c}"] = P[c].to_numpy()
    # text: the description without the fixed suffix; rank of the clip's own description among all
    ids = sorted(prompts)
    desc = [prompts[i]["sa3_prompt"].replace(", " + suffix, "") for i in ids]
    T = clap.text(desc)
    S = E @ T.T
    own = meta.prompt_id.map({p: j for j, p in enumerate(ids)}).to_numpy()
    out["text_sim"] = S[np.arange(len(S)), own]
    out["text_rank"] = (S > out.text_sim.to_numpy()[:, None]).sum(1) + 1
    # nearest Surge preset (all curated renders)
    G = pd.read_parquet(cache_file(tagger.clap_name)).set_index("preset_id")
    N = E @ G.to_numpy().T
    out["surge_nn_sim"] = N.max(1)
    out["surge_nn_id"] = G.index.to_numpy()[N.argmax(1)]
    # seed consistency: mean similarity to the other seeds of the same prompt
    cons = np.full(len(E), np.nan)
    for _, ix in meta.groupby("prompt_id").indices.items():
        if len(ix) > 1:
            s = E[ix] @ E[ix].T
            cons[ix] = (s.sum(1) - 1) / (len(ix) - 1)
    out["seed_consistency"] = cons
    # domain alignment (mean shift) -> tagger + nearest Surge preset again
    from pitcher.step4_vae.data import LABELS  # noqa: F401  (tagger labels come from the pickle)
    sp = pd.read_parquet(DATA / "splits.parquet")
    Ea = align(E, G.loc[sp[sp.split == "train"].preset_id].to_numpy())
    Pa = tagger.proba_from(Ea)
    out["tagged_aligned"] = Pa.idxmax(axis=1).to_numpy()
    for c in Pa.columns:
        out[f"pa_{c}"] = Pa[c].to_numpy()
    Na = Ea @ G.to_numpy().T
    out["surge_nn_sim_aligned"] = Na.max(1)
    out["surge_nn_id_aligned"] = G.index.to_numpy()[Na.argmax(1)]
    return out, E, G


def align(E, target):
    """Shift fingerprints so their mean matches the target set's mean; renormalise."""
    X = E - E.mean(0) + target.mean(0)
    return (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32)


# ---------------------------------------------------------------- main ---------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default="data/small-music")
    ap.add_argument("--name", default="pilot_small-music")
    ap.add_argument("--prompts", default="data/text2preset/prompts_pilot.jsonl")
    ap.add_argument("--vocab", default="pitcher/text2preset/vocabulary_v2.yaml")
    ap.add_argument("--controls", type=int, default=300, help="Surge renders analysed for calibration")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    audio_dir = PITCHER_ROOT / a.audio
    out_dir = DATA / "text2preset" / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = {p["prompt_id"]: p for p in map(json.loads, open(PITCHER_ROOT / a.prompts))}
    suffix = yaml.safe_load(open(PITCHER_ROOT / a.vocab))["suffix"]

    meta = pd.DataFrame([json.loads(l) for l in open(audio_dir / "metadata.jsonl")])
    meta = meta.drop_duplicates("file", keep="last").sort_values(["prompt_id", "seed"]).reset_index(drop=True)
    meta = meta.iloc[:a.limit]
    for col in ("kind", "category", "noun", "split"):
        meta[col] = meta.prompt_id.map(lambda i: prompts[i][col])
    meta["tags"] = meta.prompt_id.map(lambda i: json.dumps(prompts[i]["tags"]))
    print(f"{len(meta)} clips, {meta.prompt_id.nunique()} prompts")

    t0 = time.time()
    with Pool(a.workers) as pool:
        ctrl_ids = pd.read_parquet(DATA / "splits.parquet").preset_id.sample(a.controls, random_state=0).tolist()
        ctrl = pd.DataFrame(pool.map(_surge_worker, ctrl_ids, chunksize=4))
        print(f"  {len(ctrl)} Surge controls analysed, {time.time() - t0:.0f} s", flush=True)
        rows = []
        for i, m in enumerate(pool.imap(_sa3_worker, [audio_dir / f for f in meta.file], chunksize=4)):
            rows.append(m)
            if (i + 1) % 200 == 0:
                print(f"  analysed {i + 1}/{len(meta)}, {time.time() - t0:.0f} s", flush=True)
    df = pd.concat([meta, pd.DataFrame(rows)], axis=1)
    th = thresholds(ctrl)
    df, ctrl = add_flags(df, th), add_flags(ctrl, th, sa3=False)

    cl, E, _ = clap_part(df, audio_dir, prompts, suffix)
    df = pd.concat([df, cl], axis=1)
    df.to_parquet(out_dir / "clips.parquet", index=False)
    ctrl.to_parquet(out_dir / "surge_controls.parquet", index=False)
    emb = pd.DataFrame(E, columns=[f"e{j}" for j in range(E.shape[1])])
    emb.insert(0, "seed", df.seed.to_numpy()); emb.insert(0, "prompt_id", df.prompt_id.to_numpy())
    emb.to_parquet(out_dir / "clap.parquet", index=False)
    (out_dir / "thresholds.json").write_text(json.dumps(dict(th, n_descriptions=len(prompts)), indent=1))
    print(f"wrote {out_dir.relative_to(PITCHER_ROOT)}/ ({time.time() - t0:.0f} s)")

    from pitcher.text2preset.pilot_report import report
    report(a.name)


if __name__ == "__main__":
    main()
