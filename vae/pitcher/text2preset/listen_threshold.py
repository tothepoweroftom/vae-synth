"""Listening test for the matching threshold (TEXT2PRESET_PLAN.md §5): is a Surge match at 7-9 dB still OK?

The 6.7 dB "same character" threshold was calibrated on Surge vs Surge pairs (Round 13/14). Matching the
Stable Audio clips keeps 15% at 6.7 dB but 60% at 8 dB, so the threshold for clip -> preset is checked by ear.
Pairs: the cleaned Stable Audio clip (as matched: 48 kHz, levelled, first 6 s) vs its matched, octave-reset
Surge preset rendered at the clip's own note (C3 + 12k, seed 7: the distance the keep rule used).
Sample: 30 clips over distance bands (max of the match and after-reset distances), spread over the
categories: <= 6.7 dB (kept, control) 4, 6.7-7.3 6, 7.3-7.9 6, 7.9-8.5 6, 8.5-9.5 6, > 9.5 2.
Blind: random order, no distances on the page; one gain per pair (peak -3 dBFS) keeps relative loudness.
Ratings: same sound / same character / different / broken, saved in the artifact's db (ratings/<pair id>).

Writes  runs/text2preset/listen_threshold/{index.html, pairs.json, audio/*.mp4}
        data/text2preset/<name>/listen_key.csv (page id -> clip, distances, band)
Run:    .venv/bin/python -m pitcher.text2preset.listen_threshold [--name pilot_small-music]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pitcher.core.surge_io import PITCHER_ROOT
from pitcher.inverter.data import full_row
from pitcher.step3_core.listening import to_mp4
from pitcher.step4_vae.data import Spec
from pitcher.synth_data.stage1 import N_SAMPLES, NOTE, exact, init_worker, render_mel
from pitcher.text2preset.match import SEED, clip_mel

OUT = PITCHER_ROOT / "runs" / "text2preset" / "listen_threshold"
BANDS = [(0, 6.7, 4), (6.7, 7.3, 6), (7.3, 7.9, 6), (7.9, 8.5, 6), (8.5, 9.5, 6), (9.5, 99, 2)]
TEMPLATE = PITCHER_ROOT / "pitcher" / "step3_core" / "listening_page.template.html"


def page(pairs):
    """The Round 13 page with this test's wording (same rating scale and db collection)."""
    html = TEMPLATE.read_text()
    swaps = [
        ("Each pair is one Surge XT preset: the original, and a version rebuilt from the features the VAE will use. "
         "Play both, then rate how close the rebuilt one is. Use headphones at a steady volume; the pairs are in random "
         "order and don't show how they were made.",
         "Each pair is a Stable Audio clip and the Surge XT preset the matcher found for it, played at the same note. "
         "Play both, then rate how close the Surge preset is to the clip. Use headphones at a steady volume; the pairs "
         "are in random order and don't show their distances."),
        ("<b>Same character</b>Clearly the same preset, a bit plainer or different",
         "<b>Same character</b>Clearly the same kind of sound, a bit plainer or different"),
        ('player(p.original, "Original"), player(p.rebuilt, "Rebuilt")', 'player(p.original, "Stable Audio"), player(p.rebuilt, "Surge match")'),
        ('"pitcher-ratings"', '"pitcher-match-ratings"'),
        ("Pitcher Listening Test", "Pitcher Match Test"),
    ]
    for a, b in swaps:
        assert a in html, a[:40]
        html = html.replace(a, b)
    return html.replace("__PAIRS__", json.dumps(pairs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="pilot_small-music")
    ap.add_argument("--audio", default="data/small-music")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from pitcher.text2preset.postprocess import load_48k, normalise
    root = PITCHER_ROOT / "data" / "text2preset" / args.name
    M = pd.read_parquet(root / "matched.parquet")
    C = pd.read_parquet(root / "matched_codes.parquet")
    M["d"] = M[["d_match", "d_after_reset"]].max(axis=1)
    M["catg"] = M.category.fillna("open")
    rng = np.random.default_rng(args.seed)

    picks = []
    for lo, hi, n in BANDS:
        B = M[(M.d > lo) & (M.d <= hi) & ~M.index.isin([p for p, _ in picks])]
        # round-robin over categories, random within each
        by_cat = {c: list(rng.permutation(g.index)) for c, g in B.groupby("catg")}
        order = list(rng.permutation(sorted(by_cat)))
        got = []
        while len(got) < n and any(by_cat.values()):
            for c in order:
                if by_cat[c] and len(got) < n:
                    got.append(by_cat[c].pop())
        picks += [(i, f"{lo:g}-{hi:g}" if hi < 99 else f">{lo:g}") for i in got]
    rng.shuffle(picks)

    init_worker()
    spec = Spec()
    kept = [c for c in C.columns if c not in ("prompt_id", "seed")]
    (OUT / "audio").mkdir(parents=True, exist_ok=True)
    for f in (OUT / "audio").glob("*.mp4"):
        f.unlink()
    pairs, key = [], []
    for n, (i, band) in enumerate(picks, 1):
        m = M.loc[i]
        path = PITCHER_ROOT / args.audio / m.file
        x, _ = normalise(load_48k(path))
        x = x[:, :N_SAMPLES]
        note = int(NOTE + 12 * m.octave_shift)
        row = full_row(spec, {f: float(C.loc[i, f]) for f in kept})
        mel, audio, gain, info = render_mel(row, SEED, note=note)
        y = audio * 10 ** (gain / 20)
        d = float(exact(clip_mel(path).astype(np.float32), mel.astype(np.float32)[None])[0])
        g = 10 ** (-3 / 20) / max(float(np.abs(x).max()), float(np.abs(y).max()), 1e-9)
        pid = f"pair-{n:02d}"
        to_mp4(x * g, OUT / "audio" / f"{pid}-a.mp4")
        to_mp4(y * g, OUT / "audio" / f"{pid}-b.mp4")
        noun = str(m.noun).replace("_", " ")
        pairs.append(dict(id=pid, name=noun, label=m.catg, original=f"audio/{pid}-a.mp4", rebuilt=f"audio/{pid}-b.mp4"))
        key.append(dict(pair=pid, file=m.file, prompt_id=int(m.prompt_id), category=m.catg, noun=noun, band=band,
                        octave_shift=int(m.octave_shift), note=note, source=m.source, d_match=float(m.d_match),
                        d_after_reset=float(m.d_after_reset), d=float(m.d), d_remeasured=d, kept=bool(m.kept)))
        print(f"{pid} {band:>8} {m.catg:<11} {noun:<22} d {m.d:.2f} (re-measured {d:.2f})", flush=True)
    (OUT / "pairs.json").write_text(json.dumps(pairs, indent=1))
    (OUT / "index.html").write_text(page(pairs))
    pd.DataFrame(key).to_csv(root / "listen_key.csv", index=False)
    size = sum(f.stat().st_size for f in (OUT / "audio").glob("*.mp4")) / 1e6
    print(f"{len(pairs)} pairs, audio {size:.1f} MB -> {OUT}")


if __name__ == "__main__":
    main()
