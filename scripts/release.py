#!/usr/bin/env python3
"""Build and optionally package VAE Synth locally. No third-party Python dependencies."""
import argparse
import json
import os
from pathlib import Path
import platform
import plistlib
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PRODUCT = "VAE Synth"


def run(*args):
    subprocess.run([str(arg) for arg in args], check=True)


def copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination, symlinks=True)
    else:
        shutil.copy2(source, destination)


def require_env(name):
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"Set {name} before signing.")
    return value


def notices(work, info):
    dest = work / "notices"
    copy(ROOT / "assets/fonts/Inter-LICENSE.txt", dest / "Inter-LICENSE.txt")
    copy(ROOT / "docs/LICENSING.md", dest / "LICENSING.md")
    ort = Path(info["ort_root"])
    copy(ort / "LICENSE", dest / "ONNX-Runtime-LICENSE.txt")
    copy(ort / "ThirdPartyNotices.txt", dest / "ONNX-Runtime-ThirdPartyNotices.txt")
    return dest


def mac_package(artifacts, work, info, args):
    stage = work / "root"
    for fmt, suffix, location in [
        ("Standalone", ".app", "Applications"),
        ("VST3", ".vst3", "Library/Audio/Plug-Ins/VST3"),
        ("AU", ".component", "Library/Audio/Plug-Ins/Components"),
    ]:
        bundle = stage / location / (PRODUCT + suffix)
        copy(artifacts / fmt / bundle.name, bundle)
        runtime = bundle / "Contents/Frameworks/libonnxruntime.dylib"
        if not runtime.is_file():
            raise FileNotFoundError(f"Missing bundled runtime: {runtime}")
        if args.sign:
            identity = require_env("VAE_MAC_APPLICATION_IDENTITY")
            # Sign inside out: the bundled runtime must use our Developer ID too.
            for binary in (runtime, bundle):
                run("codesign", "--force", "--options", "runtime", "--timestamp",
                    "--sign", identity, binary)
        run("codesign", "--verify", "--deep", "--strict", bundle)
    copy(notices(work, info), stage / "Library/Application Support/VAE Synth/Licenses")
    components = work / "components.plist"
    run("pkgbuild", "--analyze", "--root", stage, components)
    with components.open("rb") as stream:
        entries = plistlib.load(stream)
    for entry in entries:
        entry["BundleIsRelocatable"] = False
    with components.open("wb") as stream:
        plistlib.dump(entries, stream)
    payload = work / "payload.pkg"
    run("pkgbuild", "--root", stage, "--component-plist", components,
        "--identifier", "com.vaeproject.vaesynth.pkg", "--version", info["version"],
        "--install-location", "/", payload)
    installer = work / "installer.pkg"
    signing = ["--sign", require_env("VAE_MAC_INSTALLER_IDENTITY")] if args.sign else []
    run("productbuild", *signing, "--package", payload, installer)
    if args.sign:
        run("pkgutil", "--check-signature", installer)
    if args.notarize:
        run("xcrun", "notarytool", "submit", installer, "--keychain-profile",
            require_env("VAE_MAC_NOTARY_PROFILE"), "--wait")
        run("xcrun", "stapler", "staple", installer)
        run("xcrun", "stapler", "validate", installer)
        run("spctl", "--assess", "--type", "install", "--verbose=4", installer)
    return installer


def windows_sign(path):
    tool = os.environ.get("VAE_SIGNTOOL", "signtool")
    store = ["/sm"] if os.environ.get("VAE_WINDOWS_MACHINE_STORE") == "1" else []
    run(tool, "sign", *store, "/sha1", require_env("VAE_WINDOWS_CERT_SHA1"),
        "/fd", "SHA256", "/tr", require_env("VAE_WINDOWS_TIMESTAMP_URL"),
        "/td", "SHA256", path)
    run(tool, "verify", "/pa", "/all", path)


def windows_package(artifacts, work, info, args):
    stage = work / "artifacts"
    # Copy the entire standalone directory: ONNX must sit beside the EXE.
    for fmt in ("Standalone", "VST3"):
        copy(artifacts / fmt, stage / fmt)
    binaries = [
        stage / "Standalone" / (PRODUCT + ".exe"),
        stage / "Standalone/onnxruntime.dll",
        stage / "VST3" / (PRODUCT + ".vst3") / "Contents/x86_64-win" / (PRODUCT + ".vst3"),
        stage / "VST3" / (PRODUCT + ".vst3") / "Contents/x86_64-win/onnxruntime.dll",
    ]
    for binary in binaries:
        if not binary.is_file():
            raise FileNotFoundError(f"Missing release binary: {binary}")
    if args.sign:
        for binary in binaries:
            windows_sign(binary)
    run(os.environ.get("VAE_ISCC", "iscc"), f'/DProductVersion={info["version"]}',
        f"/DArtifacts={stage}", f"/DNotices={notices(work, info)}",
        f"/DOutputDir={work}", ROOT / "installer/windows.iss")
    installer = work / "installer.exe"
    if args.sign:
        windows_sign(installer)
    return installer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--with-surge", action="store_true", default=True,
                        help="Include the Surge backend (enabled by default)")
    parser.add_argument("--test", action="store_true", help="Build tests and require CTest to pass")
    parser.add_argument("--package", action="store_true", help="Create an installer after building")
    parser.add_argument("--package-only", action="store_true", help="Package an existing Release build")
    parser.add_argument("--sign", action="store_true", help="Sign using local certificates")
    parser.add_argument("--notarize", action="store_true", help="Submit signed macOS installer to Apple")
    parser.add_argument("--mac-arch", choices=("universal", "arm64", "x86_64"), default="universal")
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--cmake-arg", action="append", default=[], help="Extra configure argument; use --cmake-arg=-DNAME=value")
    args = parser.parse_args()
    system = platform.system()
    if system not in ("Darwin", "Windows"):
        parser.error("This release helper supports macOS and Windows.")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.notarize and (system != "Darwin" or not args.sign):
        parser.error("--notarize requires macOS and --sign")
    packaging = args.package or args.package_only
    if args.sign and not packaging:
        parser.error("--sign requires --package or --package-only")
    if args.package_only and (args.test or args.cmake_arg):
        parser.error("--package-only uses the existing build; omit build/test options")
    if args.sign:
        names = ("VAE_MAC_APPLICATION_IDENTITY", "VAE_MAC_INSTALLER_IDENTITY") if system == "Darwin" else (
            "VAE_WINDOWS_CERT_SHA1", "VAE_WINDOWS_TIMESTAMP_URL")
        for name in names:
            require_env(name)
    if args.notarize:
        require_env("VAE_MAC_NOTARY_PROFILE")
    variant = "surge" if args.with_surge else "lite"
    label = f"mac-{args.mac_arch}" if system == "Darwin" else "windows-x64"
    build = (args.build_dir or ROOT / f"build-release-{label}-{variant}").resolve()
    if not args.package_only:
        generator = ["-G", "Xcode", f'-DCMAKE_OSX_ARCHITECTURES={"arm64;x86_64" if args.mac_arch == "universal" else args.mac_arch}'] if system == "Darwin" else [
            "-G", "Visual Studio 17 2022", "-A", "x64"]
        run("cmake", "-S", ROOT, "-B", build, *generator,
            f'-DVAE_WITH_SURGE={"ON" if args.with_surge else "OFF"}',
            f'-DVAE_BUILD_TESTS={"ON" if args.test else "OFF"}',
            "-DVAE_PLUGIN_FORMATS=VST3;Standalone", *args.cmake_arg)
        run("cmake", "--build", build, "--config", "Release", "--parallel", args.jobs)
        if args.test:
            run("ctest", "--test-dir", build, "-C", "Release", "--output-on-failure", "--no-tests=error")
    artifacts = build / "VAESynth_artefacts/Release"
    print(f"Build artifacts: {artifacts}", flush=True)
    if packaging:
        info = json.loads((build / "release-info.json").read_text())
        variant = "surge" if info["surge"].upper() in ("ON", "TRUE", "1") else "lite"
        if system == "Darwin":
            arches = set(info["mac_architectures"].split(";"))
            arch = "universal" if arches == {"arm64", "x86_64"} else info["mac_architectures"] or platform.machine()
            label = f"mac-{arch}"
        status = "notarized" if args.notarize else "signed" if args.sign else "unsigned"
        dist = ROOT / "dist"
        dist.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="package-", dir=build) as temporary:
            work = Path(temporary)
            package = (mac_package if system == "Darwin" else windows_package)(artifacts, work, info, args)
            dest = dist / f'VAE-Synth-{info["version"]}-{label}-{variant}-{status}{package.suffix}'
            shutil.copy2(package, dest)
        print(f"Installer: {dest}", flush=True)


if __name__ == "__main__":
    main()
