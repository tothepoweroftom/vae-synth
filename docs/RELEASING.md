# Local macOS and Windows builds

The local release helpers follow Ablaze.WASMSynth's native installer layout.
They use local build tools and either local signing certificates or Azure
Trusted Signing on Windows. Run them from any working directory.

## Prerequisites

- Python 3.9+, CMake 3.22+ and Git. First configure downloads JUCE and ONNX Runtime.
- macOS: full Xcode, selected with `xcode-select`. The default build is universal
  (Apple Silicon + Intel), targeting macOS 11 or newer.
- Windows: Visual Studio 2022 with Desktop development with C++, the Windows
  SDK, and Python (`py` or `python` on PATH). Builds target x64.
- Rust/Cargo with the MSVC toolchain on Windows, for the native tokenizer.
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

Windows release installers embed the exported model bundles. Copy the Mac's
`~/Library/Application Support/VAE Synth/models/` contents into `installer/models/`
(or supply `--models-dir C:\path\to\models`). Preserve the runtime directory names:
`joint-vae`, `prompt-p1`/`prompt-p0`, `same`, `generator`, and, for Surge builds,
`surge-data`. Copy the complete bundles, including tokenizers, banks, manifests,
NPZ files and ONNX external weights. Run `git lfs pull` if these assets use LFS.
Packaging rejects missing required files and unresolved LFS pointers, and records
SHA-256 hashes of the embedded files in `installed-manifest.json`.
When the joint-vae and prompt-p1 manifests are present at configure time, CTest
also runs `vae_model_bundle_tests`: native ONNX inference and parity checks on
the exact bundled models, using an isolated per-user cache directory.

```powershell
.\build-windows.ps1 --test --package --models-dir C:\Models\VAE-Synth
# Explicit development-only installer with the embedded stub model:
.\build-windows.ps1 --package-only --without-models
```

The Windows installer requires Windows 10 version 2004 or newer and an x64-compatible
processor. Packaging validates PE headers to reject x86/ARM64 app, plugin or ONNX
DLLs, including when using `--package-only`. It embeds Microsoft's signed x64
Visual C++ Redistributable (14.44.35211 or newer), found in the VS2022 `v143`
redistributable directory. Alternatively set `VAE_VC_REDIST` to Microsoft's
`vc_redist.x64.exe`. Its publisher signature and version are checked before
packaging. Setup installs it before the synth, aborts on prerequisite failure,
skips an existing compatible version, and reports a required restart.

ONNX Runtime and its support DLLs are installed beside the standalone EXE and
inside the VST3 bundle's `Contents/x86_64-win` directory. Models install once in
`%ProgramData%\VAE Synth\models`, readable by all users; the app and plugin discover
them automatically. Per-user models in `%APPDATA%\VAE Synth\models` take precedence.
Optimized graphs and verification caches stay in per-user app data. `VAE_DATA_DIR`
isolates both models and caches from the shared install for testing. The installed
app requires no Python, Rust, Visual Studio, PATH changes or model downloads.

Before distributing, test in a clean Windows VM: install offline with no VC++
runtime present, launch the standalone, load the VST3 in a 64-bit host, and exercise
every included model bundle from a standard second user account. Repeat installation
with a newer VC++ runtime already installed, then test upgrade and uninstall.
Local build/CTest and installer compilation do not replace this clean-machine test.

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

The embedded stub model and fonts are already compiled into the synth.
Installers also include the Inter and ONNX Runtime license notices.
Windows installers include and install the Microsoft Visual C++ x64 Redistributable
required by the app and ONNX Runtime; installation can run offline.
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

Azure Trusted Signing (Artifact Signing) can reuse the same account, identity
and authentication as Truce. The native Windows signing works for JUCE binaries
as well as Rust binaries. To choose the JUCE certificate profile:

```powershell
$env:VAE_AZURE_ACCOUNT = 'YOUR_EXISTING_ACCOUNT'
$env:VAE_AZURE_PROFILE = 'juce'
$env:VAE_AZURE_ENDPOINT = 'weu'
.\build-windows.ps1 --package --sign
```

The release helper also accepts `TRUCE_AZURE_ACCOUNT`, `TRUCE_AZURE_PROFILE`,
`TRUCE_AZURE_ENDPOINT`, `TRUCE_AZURE_DLIB` and `TRUCE_TIMESTAMP_URL` as fallbacks.
`VAE_` values take precedence. Set `VAE_AZURE_DLIB` if the Microsoft signing client
is installed outside its default per-user location. SignTool is discovered in
the installed x64 Windows SDK, or selected with `VAE_SIGNTOOL`.

Alternatively set `VAE_AZURE_METADATA` to a local Azure signing metadata JSON
with `Endpoint`, `CodeSigningAccountName` and `CertificateProfileName`. No private
keys or passwords belong in that JSON. Azure.Identity uses the existing credential
provider (for example Azure CLI/Visual Studio login or the `AZURE_*` service-principal
environment variables). No authentication values are copied into this repository.
Signatures use SHA-256 and the Microsoft RFC 3161 timestamp service, and every
signed output is verified. Missing or partial Azure configuration fails the build.

For a certificate installed in the Windows certificate store instead:

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
