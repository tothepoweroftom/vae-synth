#!/usr/bin/env python3
"""Write codec-parity.json into an exported joint VAE bundle: the student's own codec
(vae/pitcher/core/codec.py: full_row + decode) applied to every parity case's decoder codes.
The native Surge writer is checked against these plans.

python3 tools/joint_vae/codec_parity.py --bundle models/joint-vae-j48
Needs only the Python standard library (the codec is pure Python).

--bank SOURCE/bank.npz --prompt models/prompt-p0: instead writes <prompt>/bank-codec-parity.json for 20 bank rows:
codes = the converted bank.bin row (Lab feature order, null = NaN = inactive), plan = the student's codec on the
ORIGINAL npz row (schema order). Native check: VAE_CODEC_PARITY=<that file> VAE_JOINT_BUNDLE=<lab>/bundle.json
vae_ml_tests. Needs numpy for this mode (allow_pickle=False: the npz is untrusted).
"""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'vae'))
from pitcher.core.codec import decode  # noqa: E402


def full_row(schema, pruning, names, codes):
    # pitcher.inverter.data.full_row, without the pandas-backed Spec
    row = {f['name']: math.nan for f in schema['features']}
    for n, fr in pruning['frozen'].items():
        if fr['how'] == 'mode' and fr['code'] is not None:
            row[n] = fr['code']
        elif fr['how'] == 'absent' and n.endswith('/present'):
            row[n] = 0.0
    row.update(zip(names, codes))
    return row


def plan_json(plan):
    return dict(types=[[p, v] for p, v in plan.types],
                subtypes=[[p, v] for p, v in plan.subtypes],
                flags=[[p, m, v] for _, p, m, v in plan.flags],
                values=[[p, k, c, v] for _, p, k, c, v in plan.values],
                wavetables=[[o, n] for _, o, n in plan.wavetables],
                routings=[[s, p, d] for _, s, p, d in plan.routings])


def bank_cases(schema, pruning, names, bank, prompt):
    import numpy as np
    with np.load(bank, allow_pickle=False) as b:
        feats, original = [str(f) for f in b['feats']], b['codes']
    rows = len(original)
    raw = np.fromfile(prompt / 'bank.bin', '<f4')
    converted = raw[rows * 32:].reshape(rows, len(names))
    cases = []
    for r in np.linspace(0, rows - 1, 20).round().astype(int).tolist():
        # .tolist(): python floats, so codec._isnan sees NaN (np.float32 is not a float)
        plan = plan_json(decode(full_row(schema, pruning, feats, original[r].tolist()), schema))
        assert plan == plan_json(decode(full_row(schema, pruning, names, converted[r].tolist()), schema)), r
        cases.append(dict(row=r, codes=[None if math.isnan(v) else v for v in converted[r].tolist()], plan=plan))
    (prompt / 'bank-codec-parity.json').write_text(json.dumps(dict(cases=cases)) + '\n')
    print(f'Wrote {len(cases)} bank codec plans to {prompt / "bank-codec-parity.json"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--bank', type=Path)
    parser.add_argument('--prompt', type=Path)
    args = parser.parse_args()
    bundle = args.bundle
    schema = json.loads((bundle / 'schema.json').read_text())
    pruning = json.loads((bundle / 'pruning.json').read_text())
    names = [f['name'] for f in json.loads((bundle / 'bundle.json').read_text())['features']]
    if args.bank:
        return bank_cases(schema, pruning, names, args.bank, args.prompt)
    refs = json.loads((bundle / 'parity.json').read_text())['references']
    cases = [dict(codes=r['codes'], plan=plan_json(decode(full_row(schema, pruning, names, r['codes']), schema)))
             for r in refs]
    (bundle / 'codec-parity.json').write_text(json.dumps(dict(cases=cases)) + '\n')
    print(f'Wrote {len(cases)} codec plans to {bundle / "codec-parity.json"}')


if __name__ == '__main__':
    main()
