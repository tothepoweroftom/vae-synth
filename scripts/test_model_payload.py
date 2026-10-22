import json
from pathlib import Path
import tempfile
import unittest

from model_payload import LAB_FILES, PROMPT_FILES, stage_models


class ModelPayloadTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.models = self.root / "source"
        self.lab = self.models / "joint-vae-j48w_b05"
        self.prompt = self.models / "prompt-p1"
        for directory, names in [(self.lab, LAB_FILES), (self.prompt, PROMPT_FILES)]:
            directory.mkdir(parents=True)
            for name in names:
                (directory / name).write_bytes(b"asset")
        (self.lab / "bundle.json").write_text(json.dumps({"id": "lab"}))
        (self.prompt / "prompt.json").write_text(json.dumps({"lab_bundle_id": "lab"}))
        (self.lab / "schema.json").write_text(json.dumps({"features": [
            {"name": "osc/wt", "options": ["Example", "other"]}]}))
        self.surge = self.root / "surge"
        for name in ["patches_factory/Templates/Init Saw.fxp", "wavetables/Example.wt",
                     "wavetables/Sine Octaves.wt", "wavetables/Unused.wt"]:
            path = self.surge / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"surge asset")
        self.destination = self.root / "payload"

    def stage(self):
        return stage_models(self.destination, self.models, self.surge)

    def test_complete_bundles_and_selected_surge_assets_are_packaged(self):
        (self.lab / "decoder.onnx.data").write_bytes(b"external tensors")
        self.stage()
        self.assertEqual((self.destination / "joint-vae/decoder.onnx.data").read_bytes(), b"external tensors")
        self.assertTrue((self.destination / "prompt-p1/tokenizer.json").is_file())
        self.assertTrue((self.destination / "surge-data/patches_factory/Templates/Init Saw.fxp").is_file())
        self.assertTrue((self.destination / "surge-data/wavetables/Example.wt").is_file())
        self.assertFalse((self.destination / "surge-data/wavetables/Unused.wt").exists())

    def test_missing_model_fails_before_staging(self):
        (self.prompt / "prior.onnx").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "prior.onnx"):
            self.stage()
        self.assertFalse(self.destination.exists())

    def test_mismatched_bundles_fail_before_staging(self):
        (self.prompt / "prompt.json").write_text(json.dumps({"lab_bundle_id": "wrong"}))
        with self.assertRaisesRegex(ValueError, "IDs do not match"):
            self.stage()
        self.assertFalse(self.destination.exists())

    def test_missing_wavetable_fails_before_staging(self):
        (self.surge / "wavetables/Example.wt").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Example"):
            self.stage()
        self.assertFalse(self.destination.exists())

    def test_optional_native_models_preserve_complete_bundles(self):
        native = self.root / "native"
        for folder, names in [("native-same", ["native-bank.json", "same.onnx"]),
                              ("native-generator", ["generator.json", "text.onnx", "dit.onnx",
                                                    "decoder.onnx", "tokenizer.json", "dit.onnx.data"])]:
            directory = native / folder
            directory.mkdir(parents=True)
            for name in names:
                (directory / name).write_bytes(b"asset")
        (native / "native-same/native-bank.json").write_text(json.dumps({"model_file": "same.onnx"}))
        stage_models(self.destination, self.models, self.surge, native)
        self.assertTrue((self.destination / "same/same.onnx").is_file())
        self.assertTrue((self.destination / "generator/dit.onnx.data").is_file())
