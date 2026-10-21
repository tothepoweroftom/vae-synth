"""Step 8.0 the tagger's "ear": CLAP audio embeddings (and text embeddings for zero-shot).

CLAP (LAION, contrastive language-audio pretraining) maps a sound to a 512-number fingerprint;
similar sounds get nearby fingerprints, and a text such as "a synth bass sound" lands near matching
sounds. The model is used frozen (no training here).

Audio: the dataset renders (48 kHz, 2 s hold + release, data/audio/fixed), mixed to mono, cut to the
first 10 s (CLAP's own cropping of longer clips is random, which made 127 renders non-deterministic);
CLAP's feature extractor repeat-pads shorter clips to 10 s. Embeddings are L2-normalised.
torch 2.2 can't use numpy 2's C API, so inputs/outputs cross the boundary as Python lists.

Writes  data/clap_<model>.parquet (preset_id + e0..e511), for every curated preset (splits.parquet)
Run:    .venv/bin/python -m pitcher.step8_eval.clap [--model laion/clap-htsat-unfused]
"""

import argparse
import time

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from pitcher.core.surge_io import PITCHER_ROOT, SR

DATA = PITCHER_ROOT / "data"
MODELS = ["laion/clap-htsat-unfused", "laion/larger_clap_music", "laion/larger_clap_general"]


def cache_file(model):
    return DATA / f"clap_{model.split('/')[-1]}.parquet"


class Clap:
    def __init__(self, model="laion/clap-htsat-unfused"):
        from transformers import ClapModel, ClapProcessor
        self.processor = ClapProcessor.from_pretrained(model)
        self.model = ClapModel.from_pretrained(model).eval()
        self.sr = self.processor.feature_extractor.sampling_rate
        assert self.sr == SR, self.sr
        self.max_len = self.processor.feature_extractor.nb_max_samples  # 10 s

    @staticmethod
    def _norm(rows):
        e = np.asarray(rows, dtype=np.float32)
        return e / np.linalg.norm(e, axis=1, keepdims=True)

    @torch.no_grad()
    def audio(self, clips, batch=16):
        """clips: list of arrays (channels, n) or (n,) at 48 kHz -> (len, 512) normalised."""
        out = []
        for i in range(0, len(clips), batch):
            mono = [np.asarray(c, dtype=np.float32).mean(axis=0) if np.ndim(c) == 2 else np.asarray(c, np.float32)
                    for c in clips[i:i + batch]]
            mono = [m[:self.max_len] for m in mono]  # CLAP crops longer clips at a RANDOM offset; keep the start
            feats = self.processor.feature_extractor(mono, sampling_rate=self.sr, return_tensors=None)
            kw = {k: torch.tensor(np.asarray(v).tolist()) for k, v in feats.items()}
            out += self.model.get_audio_features(**kw).tolist()
        return self._norm(out)

    @torch.no_grad()
    def text(self, prompts):
        tok = self.processor.tokenizer(prompts, padding=True, return_tensors=None)
        kw = {k: torch.tensor(v) for k, v in tok.items()}
        return self._norm(self.model.get_text_features(**kw).tolist())


def read_render(preset_id):
    audio, _ = sf.read(DATA / "audio" / "fixed" / f"{preset_id}.wav", dtype="float32")
    return audio.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODELS[0])
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    ids = pd.read_parquet(DATA / "splits.parquet").preset_id.tolist()[:a.limit]
    t0 = time.time()
    clap = Clap(a.model)
    print(f"loaded {a.model} in {time.time() - t0:.0f} s; embedding {len(ids)} renders")
    emb, t0 = [], time.time()
    for i in range(0, len(ids), 64):
        emb.append(clap.audio([read_render(p) for p in ids[i:i + 64]]))
        print(f"  {i + len(emb[-1])}/{len(ids)}  {time.time() - t0:.0f} s", flush=True)
    emb = np.vstack(emb)
    df = pd.DataFrame(emb, columns=[f"e{j}" for j in range(emb.shape[1])])
    df.insert(0, "preset_id", ids)
    if a.limit is None:
        df.to_parquet(cache_file(a.model), index=False)
        print("wrote", cache_file(a.model).relative_to(PITCHER_ROOT))
    else:
        print(df.iloc[:, :6])


if __name__ == "__main__":
    main()
