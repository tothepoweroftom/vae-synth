#!/usr/bin/env python3
"""Extract the student sound decoder and preset coordinates; stdlib only.

Run from any directory. The original browser payload remains the source of truth.
No parameter decoder, audio engine, or text encoder is included.
"""
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parent
html = (root.parent / 'xy-pad/index.html').read_text()
match = re.search(r'<script[^>]*id="latent"[^>]*>(.*?)</script>', html, re.S)
if not match:
    raise SystemExit('Missing embedded latent payload')
source = json.loads(match.group(1))
keys = ('run', 'z_dim', 'n_cls', 'labels', 'directions', 'dims', 'dec_a',
        'pca', 'band_hz', 'dt', 'f0', 'hold_frames', 'presets', 'mean_z')
data = {key: source[key] for key in keys}
output = root / 'Assets/study.json'
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(data, separators=(',', ':')) + '\n')
print(f'{output}: {output.stat().st_size:,} bytes')
