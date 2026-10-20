"""Fast retrieval contract tests; no downloaded weights required."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import json
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
spec = importlib.util.spec_from_file_location('same_search', ROOT / 'tools/retrieve/same_search.py')
same = importlib.util.module_from_spec(spec)
spec.loader.exec_module(same)
from vae_schema import load


class SameTests(unittest.TestCase):
    def test_pool_and_rank_distinct_controls(self):
        rng = np.random.default_rng(3)
        vectors = same.pool(rng.normal(size=(4, 256, 44)))
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-6)
        entries = [{'values': [1]}, {'values': [1]}, {'values': [2]}, {'values': [3]}]
        ranked = same.rank(vectors[0], vectors, entries)
        self.assertEqual(ranked[0][0], 0)
        self.assertLess(ranked[0][1], 1e-6)
        self.assertEqual(len(ranked), 3)

    def test_invalid_latents(self):
        for z in (np.zeros((1, 256, 44)), np.ones((1, 128, 44)),
                  np.full((1, 256, 44), np.nan)):
            with self.assertRaises(ValueError):
                same.pool(z)

    def test_manifest_validation(self):
        schema = load()
        entry = dict(format='vae-preset-1', engine='lite', schema_version=schema.version,
                     note=60, audio='00000.wav', values=schema.defaults())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'manifest.json'
            path.write_text(json.dumps([entry]))
            self.assertEqual(len(same.read_manifest(path)), 1)
            for changes in ({'engine': 'surge'}, {'audio': '../outside.wav'}, {'values': [0]},
                            {'schema_version': -1}):
                path.write_text(json.dumps([{**entry, **changes}]))
                with self.assertRaises(ValueError):
                    same.read_manifest(path)

    def test_encoder_identity_does_not_switch_to_cached_generator(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, weights = Path(tmp)/'config', Path(tmp)/'weights'
            config.write_text('{}')
            weights.write_bytes(b'standalone encoder fixture')
            with patch('huggingface_hub.hf_hub_download', side_effect=[str(config), str(weights)]) as download, \
                 patch('stable_audio_3.model.load_autoencoder', return_value=MagicMock()):
                encoder = same.Encoder('cpu')
            self.assertEqual(download.call_args_list[0].args, ('stabilityai/SAME-S', 'model_config.json'))
            self.assertEqual(download.call_args_list[1].args, ('stabilityai/SAME-S', 'model.safetensors'))
            self.assertEqual(encoder.identity['weights_sha256'], same.digest(weights))

    def test_audio_gain_alignment_and_silence(self):
        import soundfile as sf
        encoder = same.Encoder.__new__(same.Encoder)  # preprocessing needs no model
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'audio.wav'
            x = np.sin(2*np.pi*440*np.arange(22050)/44100).astype('float32')
            sf.write(path, x, 44100, subtype='FLOAT')
            a = encoder.audio(path).numpy()
            sf.write(path, np.concatenate([np.zeros(1000), x*.1]), 44100, subtype='FLOAT')
            b = encoder.audio(path).numpy()
            np.testing.assert_allclose(a, b, atol=1e-6)
            self.assertEqual(a.shape, (2, same.SAMPLES))
            sf.write(path, np.zeros(44100), 44100)
            with self.assertRaises(ValueError):
                encoder.audio(path)


if __name__ == '__main__':
    unittest.main()
