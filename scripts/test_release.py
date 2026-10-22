import argparse
import json
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
            image = bytearray(256)
            image[:2] = b"MZ"
            image[0x3c:0x40] = (128).to_bytes(4, "little")
            image[128:134] = b"PE\0\0\x64\x86"
            path.write_bytes(image)
        self.original = self.exe.read_bytes()
        self.args = argparse.Namespace(sign=True, notarize=False, models_dir=None, without_models=True)
        self.redist_patch = patch.object(release, "windows_redist", return_value=self.work / "vc_redist.x64.exe")
        self.redist_patch.start()
        self.addCleanup(self.redist_patch.stop)
        self.info = {"version": "0.1.0", "ort_root": str(self.root / "ort")}

    def test_windows_signs_copies_before_installer_and_preserves_runtime(self):
        events = []

        def sign(path):
            self.assertTrue(path.is_relative_to(self.work))
            path.write_bytes(b"signed")
            events.append(("sign", path.name))

        def run(*args):
            events.append(("iscc", str(args[0])))
            (self.work / "installer.exe").write_bytes(b"installer")

        with patch.object(release, "windows_sign", side_effect=sign), \
                patch.object(release, "run", side_effect=run), \
                patch.object(release, "notices", return_value=self.work / "notices"):
            result = release.windows_package(self.artifacts, self.work, self.info, self.args)
        self.assertEqual([event[0] for event in events], ["sign"] * 4 + ["iscc", "sign"])
        self.assertEqual(result.read_bytes(), b"signed")
        for original in (self.exe, self.dll, self.plugin, self.plugin_dll):
            self.assertEqual(original.read_bytes(), self.original)
            staged = self.work / "artifacts" / original.relative_to(self.artifacts)
            self.assertEqual(staged.read_bytes(), b"signed")

    def test_missing_runtime_stops_before_signing_or_packaging(self):
        self.dll.unlink()
        with patch.object(release, "windows_sign") as sign, patch.object(release, "run") as run:
            with self.assertRaises(FileNotFoundError):
                release.windows_package(self.artifacts, self.work, self.info, self.args)
        sign.assert_not_called()
        run.assert_not_called()

    def test_sign_failure_stops_before_installer(self):
        with patch.object(release, "windows_sign", side_effect=RuntimeError("signing failed")), \
                patch.object(release, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "signing failed"):
                release.windows_package(self.artifacts, self.work, self.info, self.args)
        run.assert_not_called()

    def test_x86_binary_is_rejected(self):
        image = bytearray(self.original)
        image[132:134] = b"\x4c\x01"
        self.exe.write_bytes(image)
        with self.assertRaisesRegex(ValueError, "require x64"):
            release.windows_package(self.artifacts, self.work, self.info, self.args)

    def test_model_packaging_preserves_external_weights_and_rejects_lfs_pointers(self):
        source = self.root / "models"
        bundle = source / "same"
        bundle.mkdir(parents=True)
        (bundle / "native-bank.json").write_text('{}')
        (bundle / "same-encoder.onnx").write_bytes(b"model")
        (bundle / "weights.data").write_bytes(b"external weights")
        target = self.work / "models"
        release.stage_models(source, target)
        self.assertEqual((target / "same/weights.data").read_bytes(), b"external weights")
        self.assertTrue((target / "installed-manifest.json").is_file())
        (bundle / "same-encoder.onnx").write_text('version https://git-lfs.github.com/spec/v1\noid sha256:123\n')
        with self.assertRaisesRegex(ValueError, "Git LFS pointer"):
            release.stage_models(source, self.work / "bad-models")

    def test_incomplete_model_bundle_is_rejected(self):
        source = self.root / "models"
        (source / "generator").mkdir(parents=True)
        with self.assertRaisesRegex(FileNotFoundError, "Incomplete model bundle"):
            release.stage_models(source, self.work / "models")

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

    def test_azure_signing_reuses_truce_configuration_and_verifies_timestamp(self):
        metadata_seen = []

        def command(*args):
            if args[1] == "sign":
                metadata_seen.append(json.loads(Path(args[args.index("/dmdf") + 1]).read_text()))
                self.assertNotIn("/sha1", args)
                self.assertEqual(args[args.index("/dlib") + 1], "azure-signing.dll")
                self.assertEqual(args[args.index("/tr") + 1], "http://timestamp.acs.microsoft.com")

        with patch.dict(release.os.environ, {
            "TRUCE_AZURE_ACCOUNT": "existing-account", "TRUCE_AZURE_PROFILE": "juce",
            "TRUCE_AZURE_ENDPOINT": "weu",
        }, clear=True), patch.object(release, "windows_signtool", return_value="signtool.exe"), \
                patch.object(release, "windows_azure_dlib", return_value="azure-signing.dll"), \
                patch.object(release, "run", side_effect=command) as run:
            release.windows_sign(self.exe)
        self.assertEqual(metadata_seen, [{"Endpoint": "https://weu.codesigning.azure.net",
                                          "CodeSigningAccountName": "existing-account",
                                          "CertificateProfileName": "juce"}])
        self.assertEqual(run.call_args_list[-1].args,
                         ("signtool.exe", "verify", "/pa", "/all", "/tw", self.exe))
        self.assertEqual(list(self.exe.parent.glob("azure-sign-*")), [])

    def test_partial_azure_configuration_does_not_fall_back_to_another_identity(self):
        with patch.dict(release.os.environ, {"TRUCE_AZURE_ACCOUNT": "existing-account"}, clear=True):
            with self.assertRaisesRegex(ValueError, "both VAE_AZURE_ACCOUNT"):
                release.windows_azure_metadata()

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
