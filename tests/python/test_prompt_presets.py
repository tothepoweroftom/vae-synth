"""Pipeline contract tests with a fake generator; not model-quality verification."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools/retrieve'))
import prompt_presets as prompt


class PromptTests(unittest.TestCase):
    def test_validation_before_model_loading(self):
        pipeline = prompt.Pipeline(Path('missing'), Path('missing'))
        for text, model, seed in [('', 'small-sfx', 42), ('bell', 'unknown', 42), ('bell', 'small-sfx', -1)]:
            with self.assertRaises(ValueError):
                pipeline.run(text, Path('unused'), model, seed)

    def test_generated_reference_reaches_same_search_and_records_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'manifest').write_text('[]')
            (root/'index').write_text('placeholder')
            p = prompt.Pipeline(root/'manifest', root/'index')
            calls = []
            def generate(**kwargs):
                calls.append(kwargs)
                return torch.ones(1, 2, 4410)*.1
            p.generator = SimpleNamespace(generate=generate, model=SimpleNamespace(sample_rate=44100))
            p.generator_name = 'small-sfx'
            p.encoder = object()
            def search(args, encoder):
                self.assertTrue(args.reference.is_file())
                self.assertIs(encoder, p.encoder)
                args.output.mkdir()
                (args.output/'results.json').write_text('[{"name":"test fixture"}]')
            with patch.object(prompt.same_search, 'search', side_effect=search):
                result = p.run('bell', root/'run', seed=123, progress=lambda _: None)
            self.assertEqual(result[0]['name'], 'test fixture')
            self.assertEqual(calls, [dict(prompt='bell', duration=4, steps=8, seed=123)])
            metadata = json.loads((root/'run/request.json').read_text())
            self.assertEqual(metadata['status'], 'complete')
            with self.assertRaises(ValueError):
                p.run('bell', root/'run')
            p.generator.generate = lambda **kw: torch.full((1, 2, 20), float('nan'))
            with self.assertRaises(ValueError):
                p.run('bell', root/'bad', progress=lambda _: None)
            self.assertEqual(json.loads((root/'bad/request.json').read_text())['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
