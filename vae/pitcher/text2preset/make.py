"""Text -> Surge preset, v0 (TEXT2PRESET_PLAN.md §7 "retrieval prior"): a sentence in, .fxp presets out.

Index (built once per matched set, cached): every user-text paraphrase of every kept matched clip
(match.py: data/text2preset/<name>/matched.parquet + the prompts' user_texts), embedded with CLAP's text
tower (laion/clap-htsat-unfused, the model the tagger uses) -> text_index.npz.
Query: the sentence is embedded the same way; each clip scores with its best-matching paraphrase; the top
clips' matched presets are the candidates (one per prompt, so 4 seeds of one prompt don't crowd out the rest).
Selection: each candidate is rendered at C3; score = text similarity of its paraphrase + CLAP text-audio
similarity of the render to the sentence (a second, independent opinion); the best N are written as
.fxp (Surge XT opens them) + .wav previews + info.json (which prompt / clip each came from, the scores).
The learned prior (text -> z -> decoder, propose / render / select) replaces the retrieval later.

Writes  runs/text2preset/make/<sentence>/{1_<score>.fxp, 1_<score>.wav, ..., info.json}
Run:    .venv/bin/python -m pitcher.text2preset.make "dark bass sound" [--n 4 --name pilot_small-music]
"""

import argparse
import json
import re

import numpy as np
import pandas as pd
import soundfile as sf

from pitcher.core.codec import Plan, decode
from pitcher.core.surge_io import PITCHER_ROOT, SR
from pitcher.inverter.data import full_row
from pitcher.step2_explore.reference import DEFAULT_TEMPLATE
from pitcher.step4_vae.data import Spec
from pitcher.synth_data import common
from pitcher.synth_data.stage1 import exact, render_mel

DATA = PITCHER_ROOT / "data"
OUT = PITCHER_ROOT / "runs" / "text2preset" / "make"
CLAP_MODEL = "laion/clap-htsat-unfused"


def prompts_file(name):
    vocab = name.split("_")[0]                     # pilot_small-music -> prompts_pilot.jsonl
    return DATA / "text2preset" / f"prompts_{vocab}.jsonl"


def build_index(name, clap, kept_only=True):
    root = DATA / "text2preset" / name
    M = pd.read_parquet(root / "matched.parquet")
    M["row"] = np.arange(len(M))
    if kept_only:
        M = M[M.kept]
    P = {p["prompt_id"]: p for p in map(json.loads, open(prompts_file(name)))}
    texts, rows = [], []
    for _, m in M.iterrows():
        for t in P[m.prompt_id]["user_texts"]:
            texts.append(t)
            rows.append(int(m.row))
    E = np.vstack([clap.text(texts[i:i + 64]) for i in range(0, len(texts), 64)])
    np.savez(root / "text_index.npz", E=E, rows=np.array(rows), texts=np.array(texts), kept_only=kept_only)
    return E, np.array(rows), np.array(texts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("--name", default="pilot_small-music")
    ap.add_argument("--n", type=int, default=4, help="presets to write")
    ap.add_argument("--k", type=int, default=12, help="candidates rendered")
    ap.add_argument("--all-matches", action="store_true", help="also use clips above the matching threshold")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    from pitcher.step8_eval.clap import Clap
    clap = Clap(CLAP_MODEL)
    root = DATA / "text2preset" / args.name
    idx = root / "text_index.npz"
    kept_only = not args.all_matches
    if idx.exists() and not args.rebuild and bool(np.load(idx)["kept_only"]) == kept_only:
        z = np.load(idx)
        E, rows, texts = z["E"], z["rows"], z["texts"]
    else:
        E, rows, texts = build_index(args.name, clap, kept_only)
    M = pd.read_parquet(root / "matched.parquet")
    C = pd.read_parquet(root / "matched_codes.parquet")
    q = clap.text([args.text])[0]
    sim = E @ q
    best = pd.DataFrame(dict(row=rows, sim=sim, text=texts)).sort_values("sim", ascending=False)
    best = best.drop_duplicates("row")
    best["prompt_id"] = M.prompt_id.to_numpy()[best.row.to_numpy()]
    cand = best.drop_duplicates("prompt_id").head(args.k)

    spec = Spec()
    kept = [c for c in C.columns if c not in ("prompt_id", "seed")]
    common.init_worker()
    schema = common._schema
    out = OUT / re.sub(r"[^a-z0-9]+", "_", args.text.lower()).strip("_")
    out.mkdir(parents=True, exist_ok=True)
    renders = []
    for _, c in cand.iterrows():
        row = full_row(spec, {f: float(C.iloc[int(c.row)][f]) for f in kept})
        _, audio, gain, info = render_mel(row, 7)
        y = audio * 10 ** (gain / 20)
        renders.append((c, row, y))
    A = clap.audio([y for _, _, y in renders])
    scores = []
    for (c, row, y), a in zip(renders, A):
        m = M.iloc[int(c.row)]
        scores.append(dict(score=float(c.sim + a @ q), text_sim=float(c.sim), audio_text_sim=float(a @ q), paraphrase=c.text,
                           prompt_id=int(m.prompt_id), clip=m.file, noun=m.noun,
                           category=m.category if isinstance(m.category, str) else "open",   # category-free prompts
                           match_db=float(m.d_match), octave_shift=int(m.octave_shift), source=m.source))
    order = np.argsort([-s["score"] for s in scores])[:args.n]
    io = common._io
    written = []
    for rank, i in enumerate(order, 1):
        c, row, y = renders[i]
        s = scores[i]
        stem = f"{rank}_{s['category']}_{s['noun']}"
        io.load(DEFAULT_TEMPLATE)
        io.write_plan(decode(row, schema))
        io.save_patch(out / f"{stem}.fxp")
        sf.write(out / f"{stem}.wav", (y / max(1.0, float(np.abs(y).max()) / 0.98)).T, SR, subtype="PCM_24")
        # check: the written .fxp, reloaded, sounds like the row
        a2, _ = io.render_plan(Plan(), out / f"{stem}.fxp", seed=7, note=48, max_tail_s=4.0)
        s["fxp_check_db"] = float(exact(render_mel(row, 7)[0].astype(np.float32),
                                        _mel(a2)[None].astype(np.float32))[0])
        written.append(dict(file=f"{stem}.fxp", **s))
    (out / "info.json").write_text(json.dumps(dict(text=args.text, presets=written), indent=1))
    print(f'"{args.text}" -> {len(written)} presets in {out}')
    for w in written:
        print(f"  {w['file']:<34} score {w['score']:.3f} (text {w['text_sim']:.2f}, audio {w['audio_text_sim']:.2f}) "
              f"from \"{w['paraphrase']}\"  match {w['match_db']:.1f} dB, fxp check {w['fxp_check_db']:.1f} dB")


def _mel(audio):
    from pitcher.synth_data.common import _loudest_window_db
    from pitcher.synth_data.stage1 import N_SAMPLES, TARGET_DB, mel_view
    x = np.nan_to_num(audio)
    x = x * 10 ** (float(np.clip(TARGET_DB - _loudest_window_db(x), -60, 60)) / 20)
    x = x[:, :N_SAMPLES]
    if x.shape[1] < N_SAMPLES:
        x = np.pad(x, ((0, 0), (0, N_SAMPLES - x.shape[1])))
    return mel_view(x)


if __name__ == "__main__":
    main()
