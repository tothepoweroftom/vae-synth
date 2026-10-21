#!/usr/bin/env python3
"""Write xypad.json and xypad-parity.json into an exported joint VAE bundle from the student's xypad_v2.html.

python3 tools/joint_vae/export_xypad.py --html /path/to/xypad_v2.html --bundle models/joint-vae-j48w_b05
xypad.json is the page's #latent object without the decoder weights, activity stand-in and PCA (sound_pca.npz
has it), plus "knobs" = the page's #data object. xypad-parity.json holds numpy references of the page's
constrain / measures / forest / group math, for the native tests. Needs numpy and onnxruntime.
The bundle id is finalised here because it also hashes xypad.json.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import onnxruntime as ort

DROP = ('dec_p', 'dec_a', 'activity', 'pca')
CASES = 12


def page_json(html, ident):
    m = re.search(r'<script type="application/json" id="%s">(.*?)</script>' % ident, html, re.S)
    if not m:
        raise ValueError(f'No #{ident} block in the page')
    return json.loads(m.group(1))


def constrain(z0, axes, ts):
    # smallest change of z so that axis i reads ts[i]: G a = r, G = U U^T + 1e-6 I (page: constrain)
    U = np.array([a['u'] for a in axes])
    G = U @ U.T + 1e-6 * np.eye(len(axes))
    return z0 + np.linalg.solve(G, np.array(ts) - U @ z0) @ U


def measures(X, lt):
    # latent.py measures(), for one pooled mel (T, 64)
    X = X[None]
    FC, dt, f0, hold = np.array(lt['band_hz']), lt['dt'], lt['f0'], lt['hold_frames']
    P = 10 ** (X / 10)
    e = 10 * np.log10(P.sum(-1) + 1e-12)
    emax = e.max(1, keepdims=True)
    act = e >= emax - 30
    Pb = (P * act[..., None]).sum(1) / np.maximum(act.sum(1, keepdims=True), 1)
    d = np.abs(np.diff(X[:, :hold], axis=1))
    loud = np.maximum(X[:, 1:hold], X[:, :hold - 1]) > -50
    out = dict(brightness=np.log2((Pb * FC).sum(-1) / Pb.sum(-1) / f0),
               attack=np.argmax(e >= emax - 3, axis=1) * dt,
               length=(X.shape[1] - np.argmax((e >= emax - 40)[:, ::-1], axis=1)) * dt,
               noise=np.exp(np.log(Pb + 1e-12).mean(-1)) / Pb.mean(-1),
               movement=(d * loud).sum((1, 2)) / np.maximum(loud.sum((1, 2)), 1))
    return {k: float(v[0]) for k, v in out.items()}


def forest(trees, vals):
    # page predict(): a missing value walks as -1
    x = [-1.0 if v is None else v for v in vals]
    p = 0
    for t in trees:
        n = 0
        while t['l'][n] != -1:
            n = t['l'][n] if x[t['f'][n]] <= t['t'][n] else t['r'][n]
        p = p + np.array(t['v'][n])
    return (p / len(trees)).tolist()


def group_pos(group, knobs, index, vals):
    # page groupPos(): mean unit position of the group's present knobs, sign-flipped for negative signs
    us = []
    for name in group['knobs']:
        k, v = knobs[index[name]], vals[index[name]]
        if v is None:
            continue
        u = min(1.0, max(0.0, (v - k['p5']) / (k['p95'] - k['p5'] or 1e-6)))
        us.append(u if group['signs'][name] > 0 else 1 - u)
    return sum(us) / len(us) if us else None


def finalise_id(bundle, run):
    names = ['config.json', 'schema.json', 'pruning.json', 'sound_pca.npz', 'decoder.onnx', 'sound_encoder.onnx',
             'sound_decoder.onnx', 'xypad.json']
    digest = hashlib.sha256()
    for n in names:
        digest.update((bundle / n).read_bytes())
    ident = run + '-' + digest.hexdigest()[:16]
    manifest = json.loads((bundle / 'bundle.json').read_text())
    manifest['id'] = ident
    (bundle / 'bundle.json').write_text(json.dumps(manifest, indent=2) + '\n')
    example = json.loads((bundle / 'example-sound.json').read_text())
    example['bundle_id'] = ident
    (bundle / 'example-sound.json').write_text(json.dumps(example, indent=2) + '\n')
    return ident


def export(html, bundle):
    page = html.read_text()
    data, lt = page_json(page, 'data'), page_json(page, 'latent')
    run = json.loads((bundle / 'config.json').read_text())['run']
    if lt['run'] != run:
        raise ValueError(f'Page was built for {lt["run"]}, bundle is {run}')
    knobs, trees, groups = data['knobs'], data['forest']['trees'], data['groups']
    index = {k['name']: i for i, k in enumerate(knobs)}
    rng = np.random.default_rng(7)

    axes = [dict(id='s:' + d['key'], u=d['u'], mean=d['mean'], sd=d['sd']) for d in lt['directions']]
    for r in lt['dims']:
        u = [0.0] * lt['z_dim']
        u[r['d']] = 1.0
        axes.append(dict(id=f'z:{r["d"]}', u=u, mean=r['mean'], sd=r['sd']))
    # 1-3 axes, the first cases are the awkward ones: two correlated sound axes, a sound axis plus a z dim, three axes
    picks = [[0, 1], [0, 5], [5, 6], [2, 0, 5], [0, 1, 2], [3, 4, 0]] + \
            [list(rng.choice(len(axes), n, replace=False)) for n in (1, 2, 2, 3, 3, 3)]
    constrain_cases = []
    for pick in picks:
        sel = [axes[i] for i in pick]
        z0 = rng.standard_normal(lt['z_dim'])
        ts = [a['mean'] + (rng.random() - 0.5) * 5 * a['sd'] for a in sel]
        constrain_cases.append(dict(z0=z0.tolist(), axes=[a['id'] for a in sel], targets=ts,
                                    z=constrain(z0, sel, ts).tolist()))

    with np.load(bundle / 'sound_pca.npz', allow_pickle=False) as pca:
        comp, mean, scale = pca['components'], pca['mean'], float(pca['scale'])
    session = ort.InferenceSession(str(bundle / 'sound_decoder.onnx'), providers=['CPUExecutionProvider'])
    measure_cases = []
    for i in range(CASES):
        z, c = rng.standard_normal((1, lt['z_dim'])).astype(np.float32), i % 9
        y = np.eye(9, dtype=np.float32)[c:c + 1]
        sound = session.run(None, {'latent': z, 'category': y})[0][0]
        X = ((sound * scale) @ comp + mean).reshape(-1, 64)
        mel_max = float(X.max())
        measure_cases.append(dict(sound=sound.tolist(), category=c, latent=z[0].tolist(), mel_max=mel_max,
                                  measures=measures(np.maximum(X - mel_max, -80.0), lt)))

    rows = data['presets']['values']
    forest_cases = []
    for i, r in enumerate(rng.choice(len(rows), CASES, replace=False)):
        vals = list(rows[r])
        if i % 3 == 1:                                  # knob values with nulls, whatever the preset had
            vals = [None if rng.random() < 0.4 else v for v in vals]
        if i == 2:
            vals = [None] * len(vals)
        forest_cases.append(dict(knobs=vals, p=forest(trees, vals)))

    group_cases = []
    for i, gi in enumerate(rng.choice(len(groups), CASES, replace=False)):
        vals = list(rows[rng.integers(len(rows))])
        if i % 3 == 1:
            vals = [None if rng.random() < 0.5 else v for v in vals]
        group_cases.append(dict(group=groups[gi]['name'], vals=vals, pos=group_pos(groups[gi], knobs, index, vals)))

    out = {k: v for k, v in lt.items() if k not in DROP}
    out['knobs'] = data
    (bundle / 'xypad.json').write_text(json.dumps(out, separators=(',', ':')) + '\n')
    (bundle / 'xypad-parity.json').write_text(json.dumps(dict(constrain=constrain_cases, measures=measure_cases,
                                                            forest=forest_cases, groups=group_cases)) + '\n')
    ident = finalise_id(bundle, run)
    print(f'Wrote xypad for {ident}: {len(lt["directions"])} directions, {len(lt["dims"])} dims, '
          f'{len(lt["presets"]["label"])} presets, {len(knobs)} knobs, {len(groups)} groups, {len(trees)} trees')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--html', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    args = parser.parse_args()
    export(args.html.resolve(), args.bundle.resolve())
