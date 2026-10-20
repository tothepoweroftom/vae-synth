"""Validate an upstream/native full-generation fixture pair and retrieved presets."""
import json
import sys
from pathlib import Path
import numpy as np
import soundfile as sf

root = Path(sys.argv[1])
expected, expected_rate = sf.read(root / 'python-reference.wav')
actual, actual_rate = sf.read(root / 'cpp/reference.wav')
assert expected_rate == actual_rate == 44100
assert expected.shape == actual.shape == (176400, 2)
assert np.isfinite(actual).all()
relative_rms = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
cosine = float(np.sum(actual * expected) / np.linalg.norm(actual) / np.linalg.norm(expected))
assert relative_rms < .002, relative_rms
assert cosine > .99999, cosine
results = json.loads((root / 'cpp/results.json').read_text())
assert len(results) == 5
assert len({tuple(result['values']) for result in results}) == 5
for result in results:
    assert result['schema_version'] == 2
    assert len(result['values']) == 93
    assert np.isfinite(result['values']).all()
report = dict(relative_rms=relative_rms, cosine=cosine,
              max_abs=float(np.max(np.abs(actual - expected))), matches=len(results))
(root / 'waveform-parity.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
