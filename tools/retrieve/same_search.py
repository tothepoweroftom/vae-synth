#!/usr/bin/env python3
"""Experimental SAME-S retrieval. Model and audio work run outside the plug-in."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

FORMAT = 'vae-same-index-1'
POOLING = 'mean-std-cosine-1'
SAMPLES = 4 * 44100


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def pool(latents):
    """One unit vector per clip. Retain distribution, discard precise time order."""
    z = np.asarray(latents, dtype=np.float32)
    if z.ndim != 3 or z.shape[1] != 256 or z.shape[2] < 1 or not np.isfinite(z).all():
        raise ValueError('Expected finite SAME-S latents [batch, 256, time]')
    v = np.concatenate([z.mean(axis=-1), z.std(axis=-1)], axis=1)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError('Degenerate SAME embedding')
    return v / norms


def rank(query, embeddings, entries, count=5):
    if embeddings.shape != (len(entries), 512) or not np.isfinite(embeddings).all():
        raise ValueError('Invalid embedding matrix')
    if query.shape != (512,) or not np.isfinite(query).all():
        raise ValueError('Invalid query embedding')
    if not np.allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-4):
        raise ValueError('Index embeddings must be unit vectors')
    distances = 1 - embeddings @ query
    seen, results = set(), []
    for i in np.argsort(distances, kind='stable'):
        key = tuple(entries[int(i)]['values'])
        if key in seen:
            continue
        seen.add(key)
        results.append((int(i), float(max(0, distances[i]))))
        if len(results) == count:
            break
    return results


class Encoder:
    def __init__(self, device):
        import torch
        from stable_audio_3 import AutoencoderModel
        from stable_audio_3.model_configs import ae_models
        from stable_audio_3.model import load_autoencoder
        torch.set_num_threads(4)
        # Resolve once, load and fingerprint those exact files (avoid drifting cache identity).
        # Always use the standalone encoder: upstream resolve() prefers a cached
        # generator checkpoint, changing cache identity after the first generation.
        from huggingface_hub import hf_hub_download
        cfg = ae_models['same-s']
        config = hf_hub_download(cfg.ae_repo_id, cfg.ae_config_path)
        weights = hf_hub_download(cfg.ae_repo_id, cfg.ae_ckpt_path)
        self.identity = {'model': 'stabilityai/SAME-S', 'weights_sha256': digest(weights),
                         'config_sha256': digest(config), 'pooling': POOLING,
                         'preprocessing': 'stereo-onset-peak-44100-4s-v1'}
        self.model = AutoencoderModel(load_autoencoder(config, weights, device=device), 44100, device)
        self.model.autoencoder.eval().requires_grad_(False)

    def audio(self, path):
        import soundfile as sf
        import torch
        import torchaudio
        info = sf.info(str(path))
        if not (8000 <= info.samplerate <= 192000 and 0 < info.frames <= info.samplerate * 30
                and info.channels in (1, 2)):
            raise ValueError(f'Expected mono/stereo audio, 8–192 kHz, at most 30 seconds: {path}')
        audio, sr = sf.read(str(path), dtype='float32', always_2d=True)
        if not np.isfinite(audio).all():
            raise ValueError('Non-finite audio')
        peak = np.max(np.abs(audio))
        if peak < 1e-5:
            raise ValueError('Silent audio')
        first = np.flatnonzero(np.max(np.abs(audio), axis=1) > peak * .01)[0]
        x = torch.from_numpy(audio[first:].T.copy())
        if sr != 44100:
            x = torchaudio.functional.resample(x, sr, 44100)
        if x.shape[0] == 1:
            x = x.repeat(2, 1)
        x = x[:, :SAMPLES]
        peak = x.abs().max()
        if peak < 1e-5:
            raise ValueError('Silent retained audio')
        x = x / peak
        return torch.nn.functional.pad(x, (0, SAMPLES - x.shape[1]))

    def encode(self, paths):
        import torch
        with torch.inference_mode():
            z = self.model.encode(torch.stack([self.audio(p) for p in paths]), 44100)
        return pool(z.detach().float().cpu().numpy())


def read_manifest(path):
    from vae_schema import load
    schema = load()
    entries = json.loads(path.read_text())
    if not isinstance(entries, list) or not entries:
        raise ValueError('Empty manifest')
    for entry in entries:
        if (entry.get('format') != 'vae-preset-1' or entry.get('engine') != 'lite'
                or entry.get('schema_version') != schema.version or entry.get('note') not in (48, 60, 72)):
            raise ValueError('Incompatible manifest preset')
        values = entry.get('values', [])
        if len(values) != len(schema.controls) or any(
                isinstance(v, bool) or not isinstance(v, (float, int)) or not np.isfinite(v)
                or schema.sanitise(c, v) != v for c, v in zip(schema.controls, values)):
            raise ValueError('Invalid preset controls')
        audio = Path(entry.get('audio', ''))
        if audio.name != str(audio) or audio.suffix != '.wav':
            raise ValueError('Audio must be a WAV filename within the manifest directory')
    return entries


def build(args):
    entries = read_manifest(args.manifest)
    if args.output.exists():
        raise ValueError('Choose a new index path')
    encoder = Encoder(args.device)
    # Content hashes bind embeddings to both controls and the actual probe renders.
    audio_hashes = [digest(args.manifest.parent / e['audio']) for e in entries]
    metadata = {**encoder.identity, 'format': FORMAT, 'manifest_sha256': digest(args.manifest),
                'audio_sha256': audio_hashes}
    vectors = []
    for start in range(0, len(entries), args.batch_size):
        batch = entries[start:start + args.batch_size]
        vectors.append(encoder.encode([args.manifest.parent / e['audio'] for e in batch]))
        print(f'Encoded {min(start + len(batch), len(entries))}/{len(entries)}', flush=True)
    # Write only a complete cache; a failed encoding never looks like a valid index.
    tmp = args.output.with_suffix(args.output.suffix + '.tmp')
    try:
        with tmp.open('xb') as f:
            np.savez_compressed(f, embeddings=np.concatenate(vectors), metadata=json.dumps(metadata))
        os.replace(tmp, args.output)
    finally:
        tmp.unlink(missing_ok=True)


def search(args, encoder=None):
    entries = read_manifest(args.manifest)
    if args.output.exists():
        raise ValueError('Choose a new result directory')
    with np.load(args.index, allow_pickle=False) as cache:
        metadata = json.loads(str(cache['metadata']))
        embeddings = cache['embeddings'].copy()
    if (metadata.get('format') != FORMAT or metadata.get('pooling') != POOLING
            or metadata.get('manifest_sha256') != digest(args.manifest)):
        raise ValueError('Stale or incompatible index; rebuild it')
    encoder = encoder or Encoder(args.device)
    if any(metadata.get(k) != v for k, v in encoder.identity.items()):
        raise ValueError('Model or preprocessing differs from index; rebuild it')
    selected = rank(encoder.encode([args.reference])[0], embeddings, entries)
    # Check before creating results: previews must be the indexed audio, not stale renders.
    for i, _ in selected:
        if digest(args.manifest.parent / entries[i]['audio']) != metadata['audio_sha256'][i]:
            raise ValueError('Indexed preview audio changed; rebuild the index')
    args.output.mkdir(parents=True)
    results = []
    for n, (i, score) in enumerate(selected, 1):
        preset = {**entries[i], 'score': score, 'method': POOLING, 'model': metadata['model']}
        preset.pop('audio')
        (args.output / f'match-{n}.json').write_text(json.dumps(preset, indent=2))
        shutil.copyfile(args.manifest.parent / entries[i]['audio'], args.output / f'match-{n}.wav')
        results.append(preset)
        print(f"{n}: {preset['name']} (note {preset['note']}, cosine distance {score:.4f})")
    (args.output / 'results.json').write_text(json.dumps(results, indent=2))
    (args.output / 'provenance.json').write_text(json.dumps(
        {**metadata, 'reference_sha256': digest(args.reference)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    subs = parser.add_subparsers(dest='command', required=True)
    b = subs.add_parser('build')
    b.add_argument('manifest', type=Path)
    b.add_argument('output', type=Path)
    b.add_argument('--batch-size', type=int, default=8)
    s = subs.add_parser('search')
    s.add_argument('manifest', type=Path)
    s.add_argument('index', type=Path)
    s.add_argument('reference', type=Path)
    s.add_argument('output', type=Path)
    args = parser.parse_args()
    if getattr(args, 'batch_size', 1) < 1:
        parser.error('batch-size must be positive')
    try:
        (build if args.command == 'build' else search)(args)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
