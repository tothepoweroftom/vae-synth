#!/usr/bin/env python3
"""Build and optionally package VAE Synth locally. No third-party Python dependencies."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import plistlib
import shutil
import subprocess
import struct
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


def signing_env(name, fallback=None):
    return os.environ.get(name) or (os.environ.get(fallback) if fallback else None)


def windows_signtool():
    configured = os.environ.get("VAE_SIGNTOOL") or shutil.which("signtool")
    if configured:
        return configured
    sdk = Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Windows Kits/10/bin"
    candidates = sorted(sdk.glob("10.*/x64/signtool.exe"),
                        key=lambda p: tuple(int(n) for n in p.parent.parent.name.split(".")))
    if not candidates:
        raise FileNotFoundError("Install the Windows SDK or set VAE_SIGNTOOL to its x64 signtool.exe.")
    return str(candidates[-1])


def windows_azure_metadata():
    metadata_path = os.environ.get("VAE_AZURE_METADATA")
    if metadata_path:
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8-sig"))
        for key in ("Endpoint", "CodeSigningAccountName", "CertificateProfileName"):
            if not metadata.get(key):
                raise ValueError(f"Missing {key} in VAE_AZURE_METADATA")
        return metadata
    account = signing_env("VAE_AZURE_ACCOUNT", "TRUCE_AZURE_ACCOUNT")
    profile = signing_env("VAE_AZURE_PROFILE", "TRUCE_AZURE_PROFILE")
    if not account and not profile:
        return None
    if not account or not profile:
        raise ValueError("Set both VAE_AZURE_ACCOUNT and VAE_AZURE_PROFILE (or their TRUCE equivalents).")
    endpoint = signing_env("VAE_AZURE_ENDPOINT", "TRUCE_AZURE_ENDPOINT") or "weu"
    if not endpoint.startswith("https://"):
        if "://" in endpoint:
            raise ValueError("Azure signing requires an HTTPS endpoint.")
        endpoint = f"https://{endpoint.strip('/')}.codesigning.azure.net"
    return {"Endpoint": endpoint, "CodeSigningAccountName": account, "CertificateProfileName": profile}


def windows_azure_dlib():
    path = signing_env("VAE_AZURE_DLIB", "TRUCE_AZURE_DLIB")
    if not path:
        path = str(Path(os.environ.get("LOCALAPPDATA", "")) /
                   "Microsoft/MicrosoftArtifactSigningClientTools/Azure.CodeSigning.Dlib.dll")
    if not Path(path).is_file():
        raise FileNotFoundError("Set VAE_AZURE_DLIB to the x64 Azure.CodeSigning.Dlib.dll from the Microsoft signing client.")
    return path


def windows_sign(path):
    tool = windows_signtool()
    metadata = windows_azure_metadata()
    if metadata:
        dlib = windows_azure_dlib()
        timestamp = signing_env("VAE_WINDOWS_TIMESTAMP_URL", "TRUCE_TIMESTAMP_URL") or "http://timestamp.acs.microsoft.com"
        # Only public account/profile metadata goes on disk; authentication stays
        # with Azure.Identity and the user's existing credential provider.
        with tempfile.TemporaryDirectory(prefix="azure-sign-", dir=path.parent) as temporary:
            manifest = Path(temporary) / "metadata.json"
            manifest.write_text(json.dumps(metadata), encoding="utf-8")
            run(tool, "sign", "/fd", "SHA256", "/tr", timestamp, "/td", "SHA256",
                "/dlib", dlib, "/dmdf", manifest, path)
        run(tool, "verify", "/pa", "/all", "/tw", path)
        return
    store = ["/sm"] if os.environ.get("VAE_WINDOWS_MACHINE_STORE") == "1" else []
    run(tool, "sign", *store, "/sha1", require_env("VAE_WINDOWS_CERT_SHA1"),
        "/fd", "SHA256", "/tr", require_env("VAE_WINDOWS_TIMESTAMP_URL"),
        "/td", "SHA256", path)
    run(tool, "verify", "/pa", "/all", path)


def require_x64(path):
    """Check the PE header, including for --package-only builds."""
    with path.open("rb") as stream:
        if stream.read(2) != b"MZ":
            raise ValueError(f"Not a Windows PE binary: {path}")
        stream.seek(0x3c)
        offset = struct.unpack("<I", stream.read(4))[0]
        stream.seek(offset)
        if stream.read(4) != b"PE\0\0" or stream.read(2) != b"\x64\x86":
            raise ValueError(f"Windows releases require x64 binaries: {path}")


def windows_redist(work):
    override = os.environ.get("VAE_VC_REDIST")
    candidates = [Path(override)] if override else sorted(
        Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)"))
        .glob("Microsoft Visual Studio/2022/*/VC/Redist/MSVC/v143/vc_redist.x64.exe"))
    if not candidates or not candidates[0].is_file():
        raise FileNotFoundError("Set VAE_VC_REDIST to Microsoft's vc_redist.x64.exe (14.44.35211 or newer).")
    source = candidates[0].resolve()
    # Validate the publisher and minimum supported version before embedding it.
    shell_env = os.environ.copy()
    shell_env.pop("PSModulePath", None)  # Do not mix PowerShell 7 and Windows PowerShell modules.
    subprocess.run([shutil.which("pwsh") or "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
                    "-File", str(ROOT / "scripts/verify-redist.ps1"),
                    "-Path", str(source)], check=True, env=shell_env)
    dest = work / "vc_redist.x64.exe"
    copy(source, dest)
    return dest


def stage_models(source, destination):
    required = {
        "joint-vae": ("bundle.json", "decoder.onnx", "sound_encoder.onnx", "sound_decoder.onnx",
                      "schema.json", "pruning.json", "xypad.json", "sound_pca.npz"),
        "prompt-p0": ("prompt.json", "text_encoder.onnx", "prior.onnx", "tokenizer.json", "bank.json", "bank.bin"),
        "prompt-p1": ("prompt.json", "text_encoder.onnx", "prior.onnx", "tokenizer.json", "bank.json", "bank.bin"),
        "same": ("native-bank.json", "same-encoder.onnx"),
        "generator": ("generator.json", "text.onnx", "dit.onnx", "decoder.onnx", "tokenizer.json"),
        "surge-data": ("patches_factory/Templates/Init Saw.fxp",),
    }
    if not source.is_dir():
        raise FileNotFoundError(f"Model directory not found: {source}")
    bundles = [name for name in required if (source / name).is_dir()]
    if not bundles or bundles == ["surge-data"]:
        raise ValueError("Model directory must contain exported runtime bundles (joint-vae, prompt-p1, same, generator).")
    for bundle in bundles:
        for name in required[bundle]:
            if not (source / bundle / name).is_file():
                raise FileNotFoundError(f"Incomplete model bundle: {source / bundle / name}")
    manifest = {}
    for bundle in bundles:
        for path in sorted((source / bundle).rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Model bundles must contain files, not symlinks: {path}")
            if not path.is_file() or path.name == ".DS_Store":
                continue
            with path.open("rb") as stream:
                if stream.read(128).startswith(b"version https://git-lfs.github.com/spec/v1"):
                    raise ValueError(f"Unresolved Git LFS pointer: {path}. Run git lfs pull before packaging.")
                stream.seek(0)
                digest = hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else None
                if digest is None:
                    hasher = hashlib.sha256()
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        hasher.update(block)
                    digest = hasher.hexdigest()
            relative = path.relative_to(source)
            copy(path, destination / relative)
            manifest[relative.as_posix()] = digest
    (destination / "installed-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


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
        require_x64(binary)
    # Include provider/support DLLs from the exact ORT package used by CMake.
    for dll in (Path(info["ort_root"]) / "lib").glob("onnxruntime_*.dll"):
        require_x64(dll)
        for directory in (binaries[0].parent, binaries[2].parent):
            copy(dll, directory / dll.name)
    redist = windows_redist(work)
    models = work / "models"
    models.mkdir()
    if args.models_dir:
        stage_models(args.models_dir, models)
    elif not args.without_models:
        raise ValueError("Windows installers require --models-dir (or --without-models for a stub-only development installer).")
    if args.sign:
        for binary in binaries:
            windows_sign(binary)
    run(os.environ.get("VAE_ISCC", "iscc"), f'/DProductVersion={info["version"]}',
        f"/DArtifacts={stage}", f"/DNotices={notices(work, info)}",
        f"/DRedist={redist}", f"/DModels={models}", f"/DIncludeModels={int(bool(args.models_dir))}",
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
    models = parser.add_mutually_exclusive_group()
    models.add_argument("--models-dir", type=Path, help="Exported runtime bundles to embed in the Windows installer")
    models.add_argument("--without-models", action="store_true", help="Explicitly create a stub-only development installer")
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
    if system == "Windows" and packaging and not args.models_dir and not args.without_models:
        default_models = ROOT / "installer/models"
        if default_models.is_dir():
            args.models_dir = default_models
        else:
            parser.error("Copy the Mac runtime bundles to installer/models, or pass --models-dir. Use --without-models only for a stub-only installer.")
    if args.sign and not packaging:
        parser.error("--sign requires --package or --package-only")
    if args.package_only and (args.test or args.cmake_arg):
        parser.error("--package-only uses the existing build; omit build/test options")
    if args.sign:
        if system == "Darwin":
            names = ("VAE_MAC_APPLICATION_IDENTITY", "VAE_MAC_INSTALLER_IDENTITY")
        elif windows_azure_metadata():
            windows_signtool()
            windows_azure_dlib()
            names = ()
        else:
            names = ("VAE_WINDOWS_CERT_SHA1", "VAE_WINDOWS_TIMESTAMP_URL")
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
        if system == "Windows" and args.without_models:
            variant += "-stub"
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
