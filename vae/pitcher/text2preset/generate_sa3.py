"""Text -> preset dataset, step 3: render prompts with Stable Audio 3 on a GPU machine.

STANDALONE: runs in the stable-audio-3 environment (torch 2.7 + CUDA), not in the Pitcher .venv.
Needs only: torch, numpy, soundfile, stable_audio_3 (github.com/Stability-AI/stable-audio-3).
Copy this file + data/text2preset/prompts_<name>.jsonl to the GPU machine.

Setup on the GPU machine (once):
  git clone https://github.com/Stability-AI/stable-audio-3 && cd stable-audio-3 && uv sync
  (medium also needs Flash Attention 2: see the repo README; then use `uv sync --inexact`)
  accept the model conditions on huggingface.co/stabilityai/stable-audio-3-medium (free account), then
  uv run hf auth login        (free access token)
  uv pip install soundfile

FIRST run --inspect: the README doesn't document generate()'s seed/steps/CFG arguments or its return
type; this prints the real signature and the output of one prompt, so the defaults below can be checked.
  uv run python generate_sa3.py --inspect --model small-music
Pilot (500 prompts x 4 seeds, 6 s each), resumable (existing files are skipped):
  uv run python generate_sa3.py --prompts prompts_pilot.jsonl --out audio_pilot --model small-music --seeds 4
Compare models on a subset:  --model small-sfx --limit 100   (and small-music)
Split over several GPUs:     --shard $SLURM_ARRAY_TASK_ID --n-shards 4   (prompt i -> shard i % 4;
                             each shard writes metadata_shard<i>.jsonl, so workers never share a file)

Writes  <out>/<model>/<prompt_id>_s<seed>.flac (stereo, 16-bit, native rate: 44.1 kHz)
        <out>/<model>/metadata.jsonl (metadata_shard<i>.jsonl when sharded) (one line per clip: prompt_id, seed, model, sa3_prompt, file,
        sample_rate, duration_s, peak, rms_dbfs, gen_s, generate kwargs)
Post-processing (resample to 48 kHz, pitch / polyphony checks, loudness, CLAP) happens back in Pitcher.
"""

import argparse
import inspect
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

DURATION = 6.0


def load(name):
    from stable_audio_3 import StableAudioModel
    return StableAudioModel.from_pretrained(name)


def sample_rate(model):
    for attr in ("sample_rate", "sr", "sampling_rate"):
        if hasattr(model, attr):
            return int(getattr(model, attr))
    return 44100  # README: the autoencoder outputs 44.1 kHz stereo


def to_numpy(audio):
    """generate() output -> float32 (samples, channels). Handles tensor/array, (ch, n) or (1, ch, n)."""
    if isinstance(audio, (list, tuple)):
        audio = audio[0]
    if isinstance(audio, dict):
        audio = audio.get("audio", next(iter(audio.values())))
    if torch.is_tensor(audio):
        audio = audio.detach().float().cpu()
        try:
            audio = audio.numpy()
        except RuntimeError:  # torch built without numpy 2 support (e.g. torch 2.2 on Intel macOS)
            audio = np.array(audio.tolist(), dtype=np.float32)
    a = np.asarray(audio, dtype=np.float32)
    while a.ndim > 2:
        a = a[0]
    if a.ndim == 1:
        a = a[None]
    if a.shape[0] > a.shape[1]:  # (n, ch) -> (ch, n)
        a = a.T
    return a.T


def gen_kwargs(model, prompt, seed, steps, cfg):
    """Pass seed / steps / cfg only if generate() accepts them (names vary); else seed torch globally."""
    params = inspect.signature(model.generate).parameters
    kw = dict(prompt=prompt, duration=DURATION)
    seeded = False
    for name in ("seed",):
        if name in params:
            kw[name] = seed; seeded = True
    for name in ("steps", "num_steps", "n_steps"):
        if steps and name in params:
            kw[name] = steps; break
    for name in ("cfg_scale", "guidance_scale", "cfg"):
        if cfg is not None and name in params:
            kw[name] = cfg; break
    if not seeded:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    return kw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="prompts_pilot.jsonl")
    ap.add_argument("--out", default="audio_pilot")
    ap.add_argument("--model", default="small-music", help="small-music | small-sfx | medium")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--steps", type=int, default=None, help="leave unset to use the model default")
    ap.add_argument("--cfg", type=float, default=None, help="leave unset to use the model default")
    ap.add_argument("--limit", type=int, default=None, help="first N prompts only (model comparison)")
    ap.add_argument("--shard", type=int, default=0, help="this worker's index (e.g. $SLURM_ARRAY_TASK_ID)")
    ap.add_argument("--n-shards", type=int, default=1, help="number of workers; prompt i goes to shard i % n")
    ap.add_argument("--inspect", action="store_true")
    a = ap.parse_args()

    model = load(a.model)
    if a.inspect:
        print("generate signature:", inspect.signature(model.generate))
        print("model attributes:", [x for x in dir(model) if not x.startswith("_")])
        p = "Dark gritty reese bass, single note, C3, one-shot, dry, no reverb, synthesizer, no drums, no melody"
        t0 = time.time()
        out = model.generate(**gen_kwargs(model, p, 0, a.steps, a.cfg))
        print(f"{time.time() - t0:.1f} s; output type {type(out)}",
              getattr(out, "shape", None) if not isinstance(out, (list, tuple)) else [getattr(x, "shape", None) for x in out])
        x = to_numpy(out)
        sr = sample_rate(model)
        print(f"as array {x.shape} at {sr} Hz = {x.shape[0] / sr:.2f} s, peak {np.abs(x).max():.3f}")
        sf.write(f"inspect_{a.model}.flac", x, sr, subtype="PCM_16")
        print(f"wrote inspect_{a.model}.flac: listen to it")
        return

    prompts = [json.loads(l) for l in open(a.prompts)][:a.limit]
    prompts = [p for p in prompts if p["prompt_id"] % a.n_shards == a.shard]
    out = Path(a.out) / a.model
    out.mkdir(parents=True, exist_ok=True)
    meta = open(out / (f"metadata_shard{a.shard}.jsonl" if a.n_shards > 1 else "metadata.jsonl"), "a")
    sr = sample_rate(model)
    todo = [(p, s) for p in prompts for s in range(a.seeds) if not (out / f"{p['prompt_id']}_s{s}.flac").exists()]
    print(f"{len(todo)} clips to generate ({len(prompts)} prompts x {a.seeds} seeds, model {a.model}, "
          f"shard {a.shard}/{a.n_shards})", flush=True)
    t_start = time.time()
    for i, (p, s) in enumerate(todo):
        kw = gen_kwargs(model, p["sa3_prompt"], s, a.steps, a.cfg)
        t0 = time.time()
        with torch.inference_mode():
            x = to_numpy(model.generate(**kw))
        gen_s = time.time() - t0
        path = out / f"{p['prompt_id']}_s{s}.flac"
        peak = float(np.abs(x).max())
        sf.write(path, np.clip(x, -1, 1), sr, subtype="PCM_16")
        rms = float(20 * np.log10(max(np.sqrt(np.mean(x ** 2)), 1e-12)))
        meta.write(json.dumps(dict(prompt_id=p["prompt_id"], seed=s, model=a.model, sa3_prompt=p["sa3_prompt"],
                                   file=path.name, sample_rate=sr, duration_s=round(len(x) / sr, 3), peak=round(peak, 4),
                                   rms_dbfs=round(rms, 2), gen_s=round(gen_s, 2),
                                   kwargs={k: v for k, v in kw.items() if k != "prompt"})) + "\n")
        meta.flush()
        if (i + 1) % 20 == 0 or i + 1 == len(todo):
            el = time.time() - t_start
            print(f"  {i + 1}/{len(todo)}  {el / 60:.1f} min, eta {el / (i + 1) * (len(todo) - i - 1) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
