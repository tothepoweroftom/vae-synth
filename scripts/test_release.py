import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import plistlib

import release


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="vae release test ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.artifacts = self.root / "original"
        self.work = self.root / "work"
        self.work.mkdir()
        self.exe = self.artifacts / "Standalone/VAE Synth.exe"
        self.dll = self.artifacts / "Standalone/onnxruntime.dll"
        self.plugin = self.artifacts / "VST3/VAE Synth.vst3/Contents/x86_64-win/VAE Synth.vst3"
        self.plugin_dll = self.plugin.with_name("onnxruntime.dll")
        for path in (self.exe, self.dll, self.plugin, self.plugin_dll):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"original")
        self.args = argparse.Namespace(sign=True, notarize=False)
        self.models = self.root / "models"
        self.models.mkdir()
        (self.models / "model.onnx").write_bytes(b"model")
        model_mock = patch.object(release, "models", return_value=self.models)
        self.model_mock = model_mock.start()
        self.addCleanup(model_mock.stop)

    def test_windows_signs_copies_before_installer_and_preserves_runtime(self):
        events = []

        def sign(path):
            self.assertTrue(path.is_relative_to(self.work))
            path.write_bytes(b"signed")
            events.append(("sign", path.name))

        def run(*args):
            self.assertIn(f"/DModels={self.models}", args)
            events.append(("iscc", str(args[0])))
            (self.work / "installer.exe").write_bytes(b"installer")

        with patch.object(release, "windows_sign", side_effect=sign), \
                patch.object(release, "run", side_effect=run), \
                patch.object(release, "notices", return_value=self.work / "notices"):
            result = release.windows_package(self.artifacts, self.work, {"version": "0.1.0"}, self.args)
        self.assertEqual([event[0] for event in events], ["sign"] * 4 + ["iscc", "sign"])
        self.assertEqual(result.read_bytes(), b"signed")
        self.model_mock.assert_called_once()
        for original in (self.exe, self.dll, self.plugin, self.plugin_dll):
            self.assertEqual(original.read_bytes(), b"original")
            staged = self.work / "artifacts" / original.relative_to(self.artifacts)
            self.assertEqual(staged.read_bytes(), b"signed")

    def test_missing_runtime_stops_before_signing_or_packaging(self):
        self.dll.unlink()
        with patch.object(release, "windows_sign") as sign, patch.object(release, "run") as run:
            with self.assertRaises(FileNotFoundError):
                release.windows_package(self.artifacts, self.work, {"version": "0.1.0"}, self.args)
        sign.assert_not_called()
        run.assert_not_called()

    def test_sign_failure_stops_before_installer(self):
        with patch.object(release, "windows_sign", side_effect=RuntimeError("signing failed")), \
                patch.object(release, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "signing failed"):
                release.windows_package(self.artifacts, self.work, {"version": "0.1.0"}, self.args)
        run.assert_not_called()

    def test_signtool_uses_explicit_certificate_and_verifies(self):
        with patch.dict(release.os.environ, {
            "VAE_WINDOWS_CERT_SHA1": "certificate",
            "VAE_WINDOWS_TIMESTAMP_URL": "https://timestamp.example",
            "VAE_SIGNTOOL": "local signtool.exe",
            "VAE_WINDOWS_MACHINE_STORE": "1",
        }), patch.object(release, "run") as run:
            release.windows_sign(self.exe)
        sign, verify = [call.args for call in run.call_args_list]
        self.assertIn("/sm", sign)
        self.assertEqual(sign[sign.index("/sha1") + 1], "certificate")
        self.assertEqual(sign[sign.index("/fd") + 1], "SHA256")
        self.assertEqual(sign[sign.index("/td") + 1], "SHA256")
        self.assertEqual(verify, ("local signtool.exe", "verify", "/pa", "/all", self.exe))

    def test_mac_signs_runtime_before_bundle_and_notarizes_installer(self):
        for fmt, suffix in (("Standalone", ".app"), ("VST3", ".vst3"), ("AU", ".component")):
            runtime = self.artifacts / fmt / (release.PRODUCT + suffix) / "Contents/Frameworks/libonnxruntime.dylib"
            runtime.parent.mkdir(parents=True, exist_ok=True)
            runtime.write_bytes(b"runtime")
        notices = self.root / "notices"
        notices.mkdir()
        commands = []

        def run(*args):
            commands.append(args)
            if args[:2] == ("pkgbuild", "--analyze"):
                Path(args[-1]).write_bytes(plistlib.dumps([{"BundleIsRelocatable": True}]))

        self.args.notarize = True
        with patch.dict(release.os.environ, {
            "VAE_MAC_APPLICATION_IDENTITY": "application",
            "VAE_MAC_INSTALLER_IDENTITY": "installer",
            "VAE_MAC_NOTARY_PROFILE": "profile",
        }), patch.object(release, "run", side_effect=run), \
                patch.object(release, "notices", return_value=notices):
            release.mac_package(self.artifacts, self.work, {"version": "0.1.0"}, self.args)
        self.assertEqual((self.work / "root/Library/Application Support/VAE Synth/models/model.onnx").read_bytes(), b"model")
        signed = [cmd[-1] for cmd in commands if cmd[:2] == ("codesign", "--force")]
        self.assertEqual(len(signed), 6)
        for runtime, bundle in zip(signed[::2], signed[1::2]):
            self.assertEqual(runtime, bundle / "Contents/Frameworks/libonnxruntime.dylib")
        self.assertEqual(commands[-4][:3], ("xcrun", "notarytool", "submit"))
        self.assertEqual(commands[-3][:3], ("xcrun", "stapler", "staple"))
        self.assertEqual(commands[-2][:3], ("xcrun", "stapler", "validate"))
        self.assertEqual(commands[-1][:2], ("spctl", "--assess"))
        entries = plistlib.loads((self.work / "components.plist").read_bytes())
        self.assertFalse(entries[0]["BundleIsRelocatable"])


if __name__ == "__main__":
    unittest.main()
