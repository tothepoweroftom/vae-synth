"""Step 5/6 first audio check of a trained CVAE: reconstructions and prior samples, rendered.

  reconstruction  val presets -> encoder mean -> decoder -> template + writer -> render; mel distance
                  to the original, compared with the ceiling (the preset's own codes on the template,
                  data/core_curve_all_kall.csv) at the listening threshold (6.7 dB)
  samples         per label, z ~ tau x N(0, I) -> decoder -> template + writer -> render;
                  validity: silent (RMS < -60 dBFS), clipping (> 0.1% samples |x| >= 1), NaN
Writes  runs/vae/<run>/eval_recon.csv, eval_samples.csv, samples/<label>_<i>.wav (a few per label)
Run:    .venv/bin/python -m pitcher.step4_vae.eval --run h256_d01 [--n 24 --tau 1.0]
"""

import argparse
import json
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F

from pitcher.core.audio_metrics import mel_distance_db, rms_db
from pitcher.core.codec import decode
from pitcher.step4_vae.model import CVAE
from pitcher.step2_explore.reference import DEFAULT_TEMPLATE
from pitcher.core.surge_io import PITCHER_ROOT, SR, SurgeIO
from pitcher.step4_vae.train import tensors
from pitcher.step4_vae.data import LABELS, Spec, load

DATA, RUNS = PITCHER_ROOT / "data", PITCHER_ROOT / "runs" / "vae"
_io = _schema = None


def _init():
    global _io, _schema
    _io = SurgeIO()
    _schema = json.loads((DATA / "schema.json").read_text())


def _render(task):
    key, row, ref_path, save = task
    _io.load(DEFAULT_TEMPLATE)
    _io.write_plan(decode(row, _schema))
    audio, info = _io.render()
    out = dict(key=key, rms_dbfs=rms_db(audio), peak=float(np.abs(audio).max()),
               clip_share=float((np.abs(audio) >= 1.0).mean()), nan=bool(np.isnan(audio).any()))
    if ref_path:
        orig, _ = sf.read(ref_path, dtype="float32")
        out["distance"] = mel_distance_db(orig.T, audio)
    if save:
        sf.write(save, audio.T, SR, subtype="FLOAT")
    return out


def rows_from(spec, cont, bl, cats):
    cont, bl = cont.tolist(), bl.tolist()
    cats = [c.tolist() for c in cats]
    return [spec.to_codes(cont[i], bl[i], [c[i] for c in cats]) for i in range(len(cont))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="h256_d01")
    ap.add_argument("--n", type=int, default=24, help="samples per label")
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--recon_split", default="val", help="split to reconstruct (train = memorisation check)")
    args = ap.parse_args()
    run = RUNS / args.run
    cfg = json.loads((run / "config.json").read_text())
    spec, data = load(Spec())
    model = CVAE(cfg["n_in"], cfg["n_cont"], cfg["n_bool"], cfg["cat_sizes"], len(LABELS), cfg["z"], cfg["hidden"], cfg["dropout"])
    model.load_state_dict(torch.load(run / "model.pt"))
    model.eval()
    torch.manual_seed(0)

    tasks = []
    with torch.no_grad():
        va = tensors(data[args.recon_split])
        mu, _ = model.encode(va["x_in"], va["y1h"])
        recon_rows = rows_from(spec, *model.decode(mu, va["y1h"])[:2], model.decode(mu, va["y1h"])[2])
        for pid, row in zip(data[args.recon_split]["ids"].tolist(), recon_rows):
            tasks.append((("recon", pid), row, str(DATA / "audio" / "fixed" / f"{pid}.wav"), None))
        (run / "samples").mkdir(exist_ok=True)
        for li, lab in enumerate(LABELS):
            z = args.tau * torch.randn(args.n, cfg["z"])
            y1h = F.one_hot(torch.full((args.n,), li), len(LABELS)).float()
            cont, bl, cats = model.decode(z, y1h)
            for i, row in enumerate(rows_from(spec, cont, bl, cats)):
                tasks.append((("sample", f"{lab}_{i}"), row, None, str(run / "samples" / f"{lab}_{i:02d}.wav") if i < 4 else None))

    t0, res = time.time(), []
    with Pool(args.workers, initializer=_init) as pool:
        for r in pool.imap_unordered(_render, tasks, chunksize=1):
            res.append(r)
    rec = pd.DataFrame([dict(preset_id=r["key"][1], **{k: v for k, v in r.items() if k != "key"}) for r in res if r["key"][0] == "recon"])
    smp = pd.DataFrame([dict(sample=r["key"][1], label=r["key"][1].rsplit("_", 1)[0], **{k: v for k, v in r.items() if k != "key"})
                        for r in res if r["key"][0] == "sample"])
    rec.to_csv(run / "eval_recon.csv", index=False)
    smp.to_csv(run / "eval_samples.csv", index=False)

    t80 = json.loads((DATA / "listening_threshold.json").read_text())["same_character_p80"]
    ceil = pd.read_csv(DATA / "core_curve_all_kall.csv").set_index("preset_id")["distance"]
    rec["ceiling"] = rec.preset_id.map(ceil)
    print(f"{len(tasks)} renders in {time.time() - t0:.0f}s (run {args.run}, tau {args.tau})")
    print(f"\nRECONSTRUCTION of {len(rec)} {args.recon_split} presets on the template (same character = <= {t80:.1f} dB):")
    print(f"  VAE reconstruction: {(rec.distance <= t80).mean():.1%} same character, median {rec.distance.median():.1f} dB")
    print(f"  ceiling (own codes): {(rec.ceiling <= t80).mean():.1%} same character, median {rec.ceiling.median():.1f} dB")
    silent = smp.rms_dbfs < -60
    clip = smp.clip_share > 0.001
    print(f"\nSAMPLES z ~ {args.tau} x N(0, I), {args.n} per label: silent {silent.mean():.1%}, clipping {clip.mean():.1%}, "
          f"NaN {smp.nan.mean():.1%}, valid {(~silent & ~clip & ~smp.nan).mean():.1%}")
    print(smp.assign(silent=silent, clip=clip).groupby("label")[["silent", "clip"]].mean().round(2).T.to_string())
    print(f"\nexample samples (4 per label): {run / 'samples'}")


if __name__ == "__main__":
    main()
