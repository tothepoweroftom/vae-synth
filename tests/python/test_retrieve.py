"""End-to-end retrieval, without Surge resources or model weights."""
import array
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import wave

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from vae_schema import load


def main():
    exe = sys.argv[1]
    schema = load()
    def run(*args, ok=True):
        result = subprocess.run([exe, *map(str, args)], capture_output=True, text=True)
        assert (result.returncode == 0) == ok, result.stdout + result.stderr
        return result

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        rows = []
        for name in ('sine', 'saw', 'square'):
            values = schema.defaults()
            control = schema['osc1_wave']
            values[control.index] = control.options.index(name)
            rows.append({'file': name + '.fxp', 'values': values})
        source = d / 'vectors.jsonl'
        source.write_text('\n'.join(map(json.dumps, rows + [rows[0], {'values': []}])))
        bank = d / 'bank.json'
        result = run('build', source, bank)
        assert '1 duplicate vectors' in result.stdout, result.stdout
        assert '1 invalid rows' in result.stdout, result.stdout
        assert len(json.loads(bank.read_text())['entries']) == 9
        run('export', bank, d / 'audio')
        manifest = json.loads((d / 'audio/manifest.json').read_text())
        assert len(manifest) == 9
        assert all((d / 'audio' / e['audio']).exists() for e in manifest)
        assert all(e['format'] == 'vae-preset-1' and e['engine'] == 'lite' for e in manifest)
        run('export', bank, d / 'audio', ok=False)
        # A synthetic, non-bank query exercises reference decoding at 44.1 kHz.
        ref = d / 'reference.wav'
        with wave.open(str(ref), 'wb') as w:
            w.setparams((1, 2, 44100, 0, 'NONE', 'not compressed'))
            w.writeframes(array.array('h', (int(8000 * math.sin(2 * math.pi * 261.6256 * i / 44100))
                                           for i in range(66150))).tobytes())
        run('search', bank, ref, d / 'matches')
        candidates = json.loads((d / 'matches/results.json').read_text())
        assert len(candidates) == 3  # distinct controls, not duplicate pitches
        assert candidates[0]['name'] == 'sine', candidates
        # Re-query the actual exported render: nearest controls must remain the same.
        run('search', bank, d / 'matches/match-1.wav', d / 'roundtrip')
        best = json.loads((d / 'roundtrip/match-1.json').read_text())
        assert best['values'] == candidates[0]['values']
        assert best['score'] < .05, best['score']
        run('search', bank, ref, d / 'matches', ok=False)  # never overwrite an audition
        with wave.open(str(ref), 'wb') as w:
            w.setparams((1, 2, 48000, 0, 'NONE', 'not compressed'))
            w.writeframes(b'\0\0' * 48000)
        run('search', bank, ref, d / 'silent', ok=False)
        data = json.loads(bank.read_text())
        data['schema_version'] = -1
        bank.write_text(json.dumps(data))
        run('search', bank, ref, d / 'bad-schema', ok=False)
    print('Retrieval integration tests passed')


if __name__ == '__main__':
    main()
