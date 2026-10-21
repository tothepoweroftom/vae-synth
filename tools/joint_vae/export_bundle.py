#!/usr/bin/env python3
"""Export a student's matching checkpoint/schema/PCA directory for the native VAE Lab.

python3 tools/joint_vae/export_bundle.py --source /path/to/missing_files --out /path/to/bundle
Requires torch, onnx, onnxruntime, numpy. No Surge installation or training dataset needed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'vae'))
from pitcher.joint_vae.model import JointVAE

LABELS = ['Bass', 'Lead', 'Pad', 'Keys', 'Pluck', 'Synth', 'Atmosphere', 'Orchestral', 'Unknown']


def layout(schema, pruning):
    kept = set(pruning['kept'])
    features = [f for f in schema['features'] if f['name'] in kept]
    continuous = [f for f in features if f['kind'] in ('cont', 'ord')]
    stepped = [f for f in continuous if f['kind'] == 'ord'
               and round(f['natural_range'][1] - f['natural_range'][0]) + 1 <= 32]
    levels = [round(f['natural_range'][1] - f['natural_range'][0]) + 1 for f in stepped]
    numeric = [f for f in continuous if f not in stepped]
    boolean = [f for f in features if f['kind'] == 'bool']
    categorical = [f for f in features if f['kind'] == 'cat']
    sizes = [len(f['options']) for f in categorical] + levels
    ordered = numeric + boolean + categorical + stepped
    manifest = [dict(name=f['name'], min=0, max=len(f['options']) - 1 if f['kind'] == 'cat' else 1,
                     kind=f['kind']) for f in ordered]
    return numeric, boolean, categorical, levels, sizes, manifest


class Decoder(torch.nn.Module):
    def __init__(self, model, n_cat, levels):
        super().__init__()
        self.model = model
        self.n_cat = n_cat
        self.levels = levels

    def forward(self, latent, category):
        (num, bl, cats), _ = self.model.decode(latent, category)
        choices = []
        for i, cat in enumerate(cats):
            v = cat.argmax(-1).float()
            if i >= self.n_cat:
                v = v / (self.levels[i - self.n_cat] - 1)
            choices.append(v[:, None])
        return torch.cat([num, (bl > 0).float()] + choices, -1)


class Encoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.net = model.enc_a

    def forward(self, sound, category):
        mu, lv = self.net(torch.cat([sound, category], -1)).chunk(2, -1)
        precision = torch.exp(-lv.clamp(-8, 8))
        # Sound expert combined with the N(0,I) prior, exactly as posterior().
        return precision * mu / (1 + precision)


def export(source, out):
    with np.load(source / 'sound_pca.npz', allow_pickle=False) as pca:
        if pca['components'].shape != (48, 8896) or pca['mean'].shape != (8896,):
            raise ValueError('Expected the trained 48-component, 139 x 64 pooled mel PCA')
        if not all(np.isfinite(pca[key]).all() for key in ('components', 'mean', 'scale')) or float(pca['scale']) <= 0:
            raise ValueError('Invalid sound PCA values or scale')
    cfg = json.loads((source / 'config.json').read_text())
    schema = json.loads((source / 'schema.json').read_text())
    pruning = json.loads((source / 'pruning.json').read_text())
    num, boolean, cats, levels, sizes, features = layout(schema, pruning)
    if (cfg['z'], cfg['k'], cfg['n_num'], cfg['n_bool'], cfg['cat_sizes']) != (32, 48, len(num), len(boolean), sizes):
        raise ValueError('Checkpoint configuration and feature layout do not match the supported joint VAE')
    model = JointVAE(len(num), len(boolean), sizes, k_sound=48, n_cls=9, z_dim=32,
                     hidden=cfg['hidden'], dropout=cfg['dropout']).eval()
    model.load_state_dict(torch.load(source / 'model.pt', map_location='cpu', weights_only=True))
    decoder = Decoder(model, len(cats), levels).eval()
    encoder = Encoder(model).eval()
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(17)
    z, a, y = torch.randn(1, 32), torch.randn(1, 48), torch.eye(9)[:1]
    for name, module, inputs, names, output in [
        ('decoder', decoder, (z, y), ['latent', 'category'], 'codes'),
        ('sound_encoder', encoder, (a, y), ['sound', 'category'], 'latent'),
    ]:
        torch.onnx.export(module, inputs, str(out / f'{name}.onnx'), input_names=names,
                          output_names=[output], opset_version=17, dynamo=False)
    sessions = {n: ort.InferenceSession(str(out / f'{n}.onnx'), providers=['CPUExecutionProvider'])
                for n in ('decoder', 'sound_encoder')}
    errors = []
    references = []
    with torch.no_grad():
        for category in range(9):
            y = torch.eye(9)[category:category+1]
            for _ in range(4):
                z, a = torch.randn(1, 32), torch.randn(1, 48)
                enc = encoder(a, y)
                # Check the wrapper against the student's original posterior too.
                mean, _ = model.posterior(None, a, y, torch.zeros(1), torch.ones(1))
                torch.testing.assert_close(enc, mean)
                for name, inputs, expected in [
                    ('sound_encoder', {'sound': a.numpy(), 'category': y.numpy()}, enc.numpy()),
                    ('decoder', {'latent': z.numpy(), 'category': y.numpy()}, decoder(z, y).numpy()),
                ]:
                    actual = sessions[name].run(None, inputs)[0]
                    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=2e-5)
                    errors.append(float(np.max(np.abs(actual - expected))))
                references.append(dict(category=category, latent=z[0].tolist(), sound=a[0].tolist(),
                                       encoded_latent=enc[0].tolist(), codes=decoder(z, y)[0].tolist()))
    digest = hashlib.sha256()
    for name in ('model.pt', 'config.json', 'schema.json', 'pruning.json', 'sound_pca.npz'):
        digest.update((source / name).read_bytes())
    identity = cfg['run'] + '-' + digest.hexdigest()[:16]
    manifest = dict(format='pitcher.joint-vae', version=1, id=identity, latent_dimensions=32,
                    sound_dimensions=48, labels=LABELS, features=features,
                    notes='Codes ordered num/bool/cat/stepped; stepped indices converted to normalized codes. '
                          'Use schema.json and pruning.json with the student codec for full Surge patches.')
    for name in ('config.json', 'schema.json', 'pruning.json', 'sound_pca.npz'):
        shutil.copyfile(source / name, out / name)
    (out / 'bundle.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (out / 'parity.json').write_text(json.dumps(dict(cases=len(references), max_error=max(errors), references=references), indent=2) + '\n')
    (out / 'example-sound.json').write_text(json.dumps(dict(bundle_id=identity, category=0, sound=references[0]['sound']), indent=2) + '\n')
    print(f'Exported {identity}: {len(features)} features; {len(references)} parity cases; max error {max(errors):.3g}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    export(args.source.resolve(), args.out.resolve())
