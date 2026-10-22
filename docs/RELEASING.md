# Local macOS and Windows builds

The local release helpers follow Ablaze.WASMSynth's native installer layout.
They use local signing tools and certificates, with no remote signing service
or CI dependency. Run them from any working directory.

## Prerequisites

- Python 3.9+, CMake 3.22+ and Git. First configure downloads JUCE and ONNX Runtime.
- macOS: full Xcode, selected with `xcode-select`. The default build is universal
  (Apple Silicon + Intel), targeting macOS 11 or newer.
- Windows: Visual Studio 2022 with Desktop development with C++, the Windows
  SDK, and the Python launcher (`py`). Builds target x64.
- Windows packaging: Inno Setup 6.3+; put `iscc.exe` on PATH or set `VAE_ISCC`
  to its full path. Signed builds also need `signtool.exe` on PATH or
  `VAE_SIGNTOOL` set to its full path.

## Build and test

```sh
# macOS: universal Release build
./build-mac.sh
./build-mac.sh --test

# Faster Apple Silicon development build
./build-mac.sh --mac-arch arm64
```

```powershell
# Windows PowerShell
.\build-windows.ps1
.\build-windows.ps1 --test
# If local execution policy blocks .ps1 files, invoke Python directly:
py -3 scripts/release.py --test
```

Both wrappers use `scripts/release.py`. Builds live in
`build-release-<platform>-<variant>/VAESynth_artefacts/Release`.
Default variant: Surge (`VAE_WITH_SURGE=ON`). The legacy `--with-surge` flag
is still accepted. See [licensing notes](LICENSING.md) before distributing it.
`--test` builds and runs the project's CTest suite, stopping on failure.
Use `--jobs 6` to change parallelism (default: 3).

Override dependency locations or other CMake settings with repeatable arguments:

```sh
./build-mac.sh --cmake-arg=-DVAE_JUCE_DIR=/path/to/JUCE \
  --cmake-arg=-DVAE_ORT_ROOT=/path/to/extracted/onnxruntime
```

Keep the release formats (`VST3;Standalone`, plus AU on macOS) and Release
configuration when packaging. Use a separate `--build-dir` if changing generators
or architectures. Windows packaging currently supports x64 only.

## Installers

```sh
./build-mac.sh --test --package
```

```powershell
.\build-windows.ps1 --test --package
```

Outputs appear in ignored `dist/`, with version, platform, architecture, engine
variant and signing status in their filenames. Repeating a release replaces
the same output filename. Version comes from `project(VAESynth VERSION ...)`
in `CMakeLists.txt`.

- macOS `.pkg`: app in `/Applications`, VST3 and AU in
  `/Library/Audio/Plug-Ins/{VST3,Components}`, notices in
  `/Library/Application Support/VAE Synth/Licenses`.
- Windows `.exe`: app and ONNX DLL in `Program Files\VAE Synth`, VST3 bundle
  (including its own ONNX DLL) in `Common Files\VST3`, Start menu shortcut
  and uninstaller.

Installers include the current Lab inference payload: the complete
`models/joint-vae-j48w_b05` and `models/prompt-p1` bundles (ONNX models,
external tensor files, tokenizer, preset bank and metadata), plus the Surge
Init Saw template and the wavetables referenced by the Lab schema. Packaging
fails if required files or wavetables are absent, or the prompt and Lab bundle
IDs disagree. Override the source directory with `--model-root /path/to/models`.
The configured Surge checkout is recorded in `release-info.json`; reconfigure
older builds before using `--package-only`.

Models install once for all plugin formats and users:

- macOS: `/Library/Application Support/VAE Synth/models`
- Windows: `%ProgramData%\VAE Synth\models`

The plugin prefers these shared bundles, falls back to per-user development
models, and keeps generated caches in the user's data folder. `VAE_DATA_DIR`
explicitly overrides both model lookup and caches for isolated testing.

The retired Prompt tab is disabled in the current UI. To also include its
SAME and Stable Audio assets, pass `--native-model-root /path/to/build-dir`
containing `native-same/` and `native-generator/` (approximately 3 GB extra).

The embedded stub model and fonts are already compiled into the synth.
Installers also include the Inter and ONNX Runtime license notices.
Windows machines need the Microsoft Visual C++ 2015–2022 x64 Redistributable
required by ONNX Runtime; the installer does not download or install it.
These scripts package binaries; they do not prepare corresponding source
archives for a GPL release.

To repackage an existing build without rebuilding:

```sh
./build-mac.sh --package-only --build-dir build-release-mac-universal-lite
```

Existing builds must be reconfigured once after adding these scripts to
generate `release-info.json`. Build settings and version are read from that file.
Packaging fails if required bundles, runtimes or notices are missing.
It stages temporary copies and never signs or modifies the original artifacts.

## Local macOS signing

Install your Developer ID Application and Developer ID Installer certificates
in your Keychain, then set their exact names (or identity hashes):

```sh
export VAE_MAC_APPLICATION_IDENTITY='Developer ID Application: Your Name (TEAMID)'
export VAE_MAC_INSTALLER_IDENTITY='Developer ID Installer: Your Name (TEAMID)'
./build-mac.sh --package --sign
```

This signs the nested ONNX library before each bundle, uses hardened runtime
and timestamps, verifies bundle signatures, and signs the installer.
For distribution, notarize and staple it as well. Store credentials once with
`xcrun notarytool store-credentials vae-synth` (interactive), then:

```sh
export VAE_MAC_NOTARY_PROFILE=vae-synth
./build-mac.sh --package --sign --notarize
```

`--notarize` uploads the installer to Apple, waits for the result, staples and
validates its ticket, and checks Gatekeeper acceptance. Without `--sign`,
macOS bundles retain their development ad-hoc signatures and the installer
is labeled unsigned.

## Local Windows signing

The default uses a certificate in the current user's `My` certificate store.
For a certificate in the local machine store, set `VAE_WINDOWS_MACHINE_STORE=1`.
Hardware-backed certificates work when their provider is configured locally
for SignTool.

```powershell
$env:VAE_WINDOWS_CERT_SHA1 = 'YOUR_CERTIFICATE_THUMBPRINT'
$env:VAE_WINDOWS_TIMESTAMP_URL = 'http://timestamp.digicert.com'
# Optional if tools are not on PATH:
$env:VAE_SIGNTOOL = 'C:\path\to\signtool.exe'
$env:VAE_ISCC = 'C:\Program Files (x86)\Inno Setup 6\ISCC.exe'
.\build-windows.ps1 --package --sign
```

The app, plugin, bundled ONNX DLLs and final installer are signed and verified.
SignTool uses SHA-256 file digests and RFC 3161 timestamps. Certificates and
passwords are not stored in this repository.

References: [Apple notarization](https://developer.apple.com/documentation/security/customizing-the-notarization-workflow),
[Microsoft SignTool](https://learn.microsoft.com/en-us/windows/win32/seccrypto/signtool).

## Release-helper checks

```sh
python3 -m unittest discover -s scripts -p 'test_*.py'
```

These cover staging and signing command order without certificates. They do
not replace testing an installed plugin in a DAW on both target platforms.
