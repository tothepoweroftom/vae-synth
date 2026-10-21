#!/usr/bin/env python3
"""Run both retrieval methods on references and write a local listening report."""
import argparse
import html
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('retriever', type=Path)
    p.add_argument('bank', type=Path)
    p.add_argument('manifest', type=Path)
    p.add_argument('index', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('references', type=Path, nargs='+')
    a = p.parse_args()
    if a.output.exists():
        p.error('Choose a new report directory')
    a.output.mkdir(parents=True)
    sections = []
    for n, ref in enumerate(a.references, 1):
        dest = a.output / str(n)
        dest.mkdir()
        shutil.copyfile(ref, dest / 'reference.wav')
        subprocess.run([str(a.retriever.resolve()), 'search', str(a.bank), str(ref), str(dest/'spectral')], check=True)
        subprocess.run([sys.executable, str(Path(__file__).with_name('same_search.py')),
                        'search', str(a.manifest), str(a.index), str(ref), str(dest/'same')], check=True)
        columns = []
        for method in ['spectral', 'same']:
            results = json.loads((dest/method/'results.json').read_text())
            rows = ''.join(
                f'<li>{html.escape(r["name"])} · MIDI {r["note"]}'
                f'<audio controls preload="none" src="{n}/{method}/match-{i}.wav"></audio>'
                f'<a href="{n}/{method}/match-{i}.json">Editable preset</a></li>'
                for i, r in enumerate(results, 1))
            columns.append(f'<div><h3>{method.upper()}</h3><ol>{rows}</ol></div>')
        sections.append(f'<section><h2>{html.escape(ref.stem)}</h2>'
                        f'<p>Reference</p><audio controls src="{n}/reference.wav"></audio>'
                        f'<div class="columns">{"".join(columns)}</div></section>')
    (a.output/'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8">
<title>SAME versus spectral retrieval</title><style>
body{font:16px system-ui;max-width:1000px;margin:40px auto;padding:0 20px;background:#faf9f6;color:#252525}
.columns{display:grid;grid-template-columns:1fr 1fr;gap:32px}audio{display:block;margin:10px 0;max-width:100%}
li{margin:24px 0}section{border-top:1px solid #ccc;margin-top:32px}a{color:#225a47}
@media(max-width:650px){.columns{grid-template-columns:1fr}}
</style><h1>SAME versus spectral retrieval</h1>
<p>Compare timbre and envelope by ear. The reference is audio; every candidate is a playable Lite preset.
These experimental rankings are not a claim of improved perceptual quality. Levels may differ.</p>
''' + ''.join(sections) + '</html>')
    print(a.output/'index.html')


if __name__ == '__main__':
    main()
