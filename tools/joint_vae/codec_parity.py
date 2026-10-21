#!/usr/bin/env python3
"""Write codec-parity.json into an exported joint VAE bundle: the student's own codec
(vae/pitcher/core/codec.py: full_row + decode) applied to every parity case's decoder codes.
The native Surge writer is checked against these plans.

python3 tools/joint_vae/codec_parity.py --bundle models/joint-vae-j48
Needs only the Python standard library (the codec is pure Python).
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    bundle = parser.parse_args().bundle
    schema = json.loads((bundle / 'schema.json').read_text())
    pruning = json.loads((bundle / 'pruning.json').read_text())
    names = [f['name'] for f in json.loads((bundle / 'bundle.json').read_text())['features']]
    refs = json.loads((bundle / 'parity.json').read_text())['references']
    cases = [dict(codes=r['codes'], plan=plan_json(decode(full_row(schema, pruning, names, r['codes']), schema)))
             for r in refs]
    (bundle / 'codec-parity.json').write_text(json.dumps(dict(cases=cases)) + '\n')
    print(f'Wrote {len(cases)} codec plans to {bundle / "codec-parity.json"}')


if __name__ == '__main__':
    main()
