#!/usr/bin/env python3
"""Offline reference runner for a p0 prompt bundle (export_prompt.py): writes prompt-parity.json.

python3 -I tools/joint_vae/prompt_reference.py --bundle models/prompt-p0

text -> tokenizer -> text encoder -> L2-normalised embedding -> prior -> mixture means -> nearest unused bank row
of the predicted category (squared distance in bank z) -> stored codes. This is OUR reading of the student's
preprocessing (tokenizer.json + prompt.json "tokenizer"); the folder has no golden outputs to check it against.
Needs tokenizers, onnxruntime and numpy.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

CASES = [
    ('empty', ''),
    ('long', ' '.join(['a warm evolving analog pad with slow shimmering movement'] * 80)),
    ('unicode', 'Basse sombre et chaude · 低音 🎛️ — naïve'),
    ('bass', 'deep sub bass'),
    ('lead', 'screaming saw lead'),
    ('pad', 'lush ambient pad'),
    ('keys', 'electric piano keys'),
    ('dark_bass', 'dark bass'),
    ('bright_bass', 'bright bass'),
    ('short_pad', 'short pad'),
    ('long_pad', 'long pad'),
    ('soft_lead', 'soft lead'),
    ('sharp_lead', 'sharp lead'),
    ('near_tie', 'acid lead'),  # second-nearest bank row 1.9e-4 farther; sharp_lead has a 7e-6 weight gap
    ('keys_forced', 'electric piano keys', 3),  # p0 never predicts 3..7 (logits ~ -10): forced no-bank case
]


class Reference:
    def __init__(self, bundle):
        self.meta = json.loads((bundle / 'prompt.json').read_text())
        t = self.meta['tokenizer']
        self.tok = Tokenizer.from_file(str(bundle / 'tokenizer.json'))
        self.tok.no_padding()  # one prompt per call: 'longest' padding pads nothing
        self.tok.enable_truncation(max_length=t['max_length'], strategy=t['truncation'])
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1  # reproducible float sums
        self.encoder = ort.InferenceSession(str(bundle / 'text_encoder.onnx'), opts, providers=['CPUExecutionProvider'])
        self.prior = ort.InferenceSession(str(bundle / 'prior.onnx'), opts, providers=['CPUExecutionProvider'])
        rows, n = self.meta['bank']['rows'], self.meta['bank']['codes']
        raw = np.fromfile(bundle / 'bank.bin', '<f4')
        if raw.size != rows * (32 + n):
            raise SystemExit('bank.bin size does not match prompt.json')
        self.z, self.codes = raw[:rows * 32].reshape(rows, 32), raw[rows * 32:].reshape(rows, n)
        self.category = np.array(json.loads((bundle / 'bank.json').read_text())['category'])

    def run(self, text, force=None):
        e = self.tok.encode(text)
        ids, mask = np.array([e.ids], np.int64), np.array([e.attention_mask], np.int64)
        raw = self.encoder.run(None, {'input_ids': ids, 'attention_mask': mask})[0][0]
        norm = float(np.linalg.norm(raw))
        if not np.isfinite(raw).all() or norm == 0:
            raise ValueError('nonfinite or zero text embedding')
        emb = (raw / norm).astype(np.float32)
        logits, mix, mu, logsd = self.prior.run(None, {'text': emb[None]})
        logits, mix, mu = logits[0], mix[0], mu[0]
        if not all(np.isfinite(a).all() for a in (logits, mix, mu, logsd)) or mu.shape != (8, 32):
            raise ValueError('nonfinite or misshaped prior output')
        sel = self.meta['selection']
        category = int(np.argmax(logits[:sel['categories']])) if force is None else force
        w = np.exp(mix - mix.max())
        w = w / w.sum()
        order = sorted((k for k in range(len(w)) if w[k] >= sel['min_weight']), key=lambda k: (-w[k], k))
        pool = np.flatnonzero(self.category == category)
        rows, dists, ties, used = [], [], [], set()
        for k in order:
            free = [r for r in pool if r not in used]
            if not free or len(rows) == sel['max_results']:
                break
            d = ((self.z[free] - mu[k]) ** 2).sum(1)
            j = min(range(len(free)), key=lambda i: (d[i], free[i]))  # equal distance -> lower row id
            used.add(free[j])
            rows.append(int(free[j]))
            dists.append(float(d[j]))
            ties.append(float(np.partition(d, 1)[1] - d[j]) if len(d) > 1 else None)
        return dict(text=text, ids=e.ids, mask=e.attention_mask, embedding=emb.tolist(), category_logits=logits.tolist(),
                    category=category, mix_weights=w.tolist(), components=order, mu=mu.tolist(),
                    error=None if len(pool) else 'no_bank_entries', rows=rows, distances=dists, distance_gaps=ties, forced_category=force,
                    codes=[[None if np.isnan(v) else float(v) for v in self.codes[r]] for r in rows])


def gaps(case):
    # smallest gap between eligible mixture weights (near-tie evidence; distance_gaps has the row ties)
    w = sorted((case['mix_weights'][k] for k in case['components']), reverse=True)
    return min((a - b for a, b in zip(w, w[1:])), default=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    bundle = parser.parse_args().bundle.resolve()
    ref = Reference(bundle)
    cases = []
    for name, text, *force in CASES:
        c = dict(name=name, **ref.run(text, *force))
        c['weight_gap'] = gaps(c)
        cases.append(c)
        print(f"{name:12s} tokens {len(c['ids']):3d} category {ref.meta['labels'][c['category']]:10s} "
              f"components {c['components']} rows {c['rows']} gap {c['weight_gap']}")
    out = dict(bundle_id=ref.meta['id'], tolerance=1e-4,
               note='Our interpretation of the preprocessing, not student golden outputs (none supplied).',
               cases=cases)
    (bundle / 'prompt-parity.json').write_text(json.dumps(out, ensure_ascii=False) + '\n')
    print(f'Wrote {len(cases)} cases to {bundle / "prompt-parity.json"}')


if __name__ == '__main__':
    main()
