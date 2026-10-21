#!/usr/bin/env python3
"""Import the student's p0/p1 text-to-latent folder as a prompt bundle for an installed joint VAE Lab bundle.

python3 -I tools/joint_vae/export_prompt.py --source ~/Downloads/p1 --lab models/joint-vae-j48w_b05 --out models/prompt-p1
Then: tools/joint_vae/prompt_reference.py --bundle models/prompt-p1 (writes prompt-parity.json).

Checks (any failure aborts and leaves --out untouched): the bank's feature set equals the Lab's, layout.json's
num/bool/cat lists are exactly the Lab's feature order with 435/85/111 entries and the Lab's category sizes,
labels match in order, layout.vae == the Lab's config run, and the supplied decoder.onnx gives the Lab decoder's
codes on 8 fixed (z, label) inputs. The supplied decoder is only compared, never copied.

Bank rows are the student's stored kept-feature codes (pitcher.inverter.data: NaN = inactive), in schema order.
bank.bin reorders them to Lab feature order and keeps NaN: StudentCodec::plan skips NaN exactly as
codec.decode(full_row(...)) does. Checked by codec_parity.py --bank.
The source folder is untrusted: numpy reads it with allow_pickle=False. Needs numpy and onnxruntime.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import onnxruntime as ort

COUNTS = (435, 85, 111)
CATEGORIES = 8  # Bass..Orchestral; index 8 (unknown) is never a bank-search category
# Our reading of tokenizer/ (see prompt_reference.py): the student's CLAP text call
# (pitcher/step8_eval/clap.py Clap.text) is processor.tokenizer(prompts, padding=True), so one prompt is unpadded;
# truncation at model_max_length (512 incl. <s> </s>) is ours, the student's call does not truncate.
TOKENIZER = dict(max_length=512, pad_id=1, bos=0, eos=2, padding='longest', truncation='longest_first')
FILES = ('text_encoder.onnx', 'tokenizer.json', 'prior.onnx', 'bank.bin', 'bank.json')


def fail(message):
    raise SystemExit('export_prompt: ' + message)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def check_tokenizer(folder):
    config = json.loads((folder / 'tokenizer_config.json').read_text())
    special = json.loads((folder / 'special_tokens_map.json').read_text())
    added = {int(k): v['content'] for k, v in config['added_tokens_decoder'].items()}
    want = {TOKENIZER['bos']: '<s>', TOKENIZER['pad_id']: '<pad>', TOKENIZER['eos']: '</s>'}
    if any(added.get(i) != t for i, t in want.items()) or config['model_max_length'] != TOKENIZER['max_length'] \
            or config['padding_side'] != 'right' or special['pad_token']['content'] != '<pad>' \
            or config.get('tokenizer_class') != 'RobertaTokenizer':
        fail('tokenizer metadata differs from the pinned RoBERTa/CLAP preprocessing')


def lab_layout(lab):
    manifest = json.loads((lab / 'bundle.json').read_text())
    schema = {f['name']: f for f in json.loads((lab / 'schema.json').read_text())['features']}
    names = [f['name'] for f in manifest['features']]
    sizes, levels = [], []
    for f in manifest['features']:
        s = schema[f['name']]
        if f['kind'] == 'cat':
            sizes.append(len(s['options']))
        elif f['kind'] == 'ord' and f['name'] not in names[:COUNTS[0]]:
            levels.append(round(s['natural_range'][1] - s['natural_range'][0]) + 1)
    return manifest, names, sizes + levels, levels


def to_codes(num, bool_logits, cat_logits, sizes, levels):
    # layout.json steps: bool = logit > 0, cat = argmax per block, stepped knob code = index / (levels - 1)
    blocks, start = [], 0
    for s in sizes:
        blocks.append(cat_logits[:, start:start + s].argmax(-1).astype(np.float32))
        start += s
    if start != cat_logits.shape[1]:
        fail('categorical logits do not cover the Lab category sizes')
    n_cat = len(sizes) - len(levels)
    for i, lv in enumerate(levels):
        blocks[n_cat + i] = blocks[n_cat + i] / (lv - 1)
    return np.hstack([num, (bool_logits > 0).astype(np.float32), np.stack(blocks, 1)])


def compare_decoders(source, lab, sizes, levels):
    supplied = ort.InferenceSession(str(source / 'decoder.onnx'), providers=['CPUExecutionProvider'])
    installed = ort.InferenceSession(str(lab / 'decoder.onnx'), providers=['CPUExecutionProvider'])
    rng = np.random.default_rng(12)
    z = rng.standard_normal((8, 32)).astype(np.float32)
    y = np.eye(9, dtype=np.float32)[np.arange(8)]
    out = supplied.run(None, {'z': z, 'label': y})
    got = to_codes(*out, sizes, levels)
    ref = np.vstack([installed.run(None, {'latent': z[i:i + 1], 'category': y[i:i + 1]})[0] for i in range(8)])
    n = COUNTS[0]
    numeric = float(np.abs(got[:, :n] - ref[:, :n]).max())
    discrete = int((got[:, n:] != ref[:, n:]).sum())
    if discrete or numeric > 1e-4:
        fail(f'supplied decoder differs from the Lab decoder: {discrete} bool/cat mismatches, numeric max {numeric:.3g}')
    return numeric


def export(source, lab, out):
    layout = json.loads((source / 'layout.json').read_text())
    config = json.loads((lab / 'config.json').read_text())
    manifest, names, sizes, levels = lab_layout(lab)
    lists = [layout['num'], layout['bool'], [c['name'] for c in layout['cat']]]
    if tuple(map(len, lists)) != COUNTS:
        fail(f'layout counts {tuple(map(len, lists))} are not {COUNTS}')
    if sum(lists, []) != names:
        fail("layout.json feature order differs from the Lab bundle's")
    if [c['size'] for c in layout['cat']] != sizes or [c['stepped_levels'] for c in layout['cat'] if c['stepped_levels']] != levels:
        fail("layout.json category sizes differ from the Lab schema's")
    # layout says "unknown", the Lab "Unknown": same label, order checked case-insensitively
    if [l.lower() for l in layout['labels']] != [l.lower() for l in manifest['labels']]:
        fail('labels differ from the Lab bundle')
    if layout['vae'] != config['run'] or (layout['z_dim'], layout['k']) != (32, 8):
        fail(f"layout is prior {layout['prior']} / vae {layout['vae']}, the Lab run is {config['run']}")
    check_tokenizer(source / 'tokenizer')

    with np.load(source / 'bank.npz', allow_pickle=False) as bank:
        z, codes, category, feats = (bank[k] for k in ('z', 'codes', 'category', 'feats'))
    feats = [str(f) for f in feats]
    rows = len(z)
    if set(feats) != set(names) or len(feats) != len(names):
        fail("bank features are not the Lab bundle's feature set")
    if z.shape != (rows, 32) or codes.shape != (rows, len(names)) or category.shape != (rows,) or not rows:
        fail('bank arrays have unexpected shapes')
    labels = manifest['labels']
    if not set(category.tolist()) <= set(labels[:CATEGORIES]):
        fail('bank has categories outside the Lab labels')
    codes = codes[:, [feats.index(n) for n in names]].astype(np.float32)
    lo = np.array([f['min'] for f in manifest['features']], np.float32)
    hi = np.array([f['max'] for f in manifest['features']], np.float32)
    seen = ~np.isnan(codes)
    whole = np.array([f['kind'] in ('cat', 'bool') for f in manifest['features']])
    if not np.isfinite(z).all() or np.isinf(codes).any() or (seen & ((codes < lo) | (codes > hi))).any() \
            or (seen[:, whole] & (codes[:, whole] != np.round(codes[:, whole]))).any():
        fail('bank z or codes are out of range (codes may be NaN = inactive, nothing else)')

    numeric = compare_decoders(source, lab, sizes, levels)

    tmp = out.with_name(out.name + '.tmp')
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    shutil.copyfile(source / 'text_encoder.onnx', tmp / 'text_encoder.onnx')
    shutil.copyfile(source / 'prior.onnx', tmp / 'prior.onnx')
    shutil.copyfile(source / 'tokenizer' / 'tokenizer.json', tmp / 'tokenizer.json')
    (tmp / 'bank.bin').write_bytes(np.ascontiguousarray(z, '<f4').tobytes() + np.ascontiguousarray(codes, '<f4').tobytes())
    label_of = [labels.index(c) for c in category.tolist()]
    (tmp / 'bank.json').write_text(json.dumps(dict(
        category=label_of, name=[f'{labels[c]} · {r:04d}' for r, c in enumerate(label_of)]), ensure_ascii=False) + '\n')
    hashes = {name: sha256(tmp / name) for name in FILES}
    prompt = dict(format='pitcher.prompt', version=1,
                  id=layout['prior'] + '-' + hashlib.sha256(''.join(hashes[n] for n in FILES).encode()).hexdigest()[:16],
                  lab_bundle_id=manifest['id'], prior=layout['prior'], vae=layout['vae'], labels=labels,
                  tokenizer=TOKENIZER, embedding=dict(dim=512, l2_normalise=True),
                  selection=dict(categories=CATEGORIES, k=8, min_weight=0.02, max_results=8),
                  bank=dict(rows=rows, feats_in_bundle_order=True, z_dim=32, codes=len(names), nan='inactive',
                            per_category={l: label_of.count(i) for i, l in enumerate(labels) if i in label_of}),
                  features=names, sha256=hashes, decoder_check=dict(cases=8, numeric_max_abs_diff=numeric))
    (tmp / 'prompt.json').write_text(json.dumps(prompt, indent=2, ensure_ascii=False) + '\n')
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    print(f"Exported {prompt['id']} for {manifest['id']}: {rows} bank rows; decoder check numeric max {numeric:.3g}, "
          'bool/cat identical')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--lab', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    export(args.source.expanduser().resolve(), args.lab.resolve(), args.out.resolve())
