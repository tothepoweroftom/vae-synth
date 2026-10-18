# vae-synth

A JUCE synth for **prompt → preset** sound design. A prompt becomes a latent
vector, a decoder turns that into a fixed **control vector** (32 Surge-style
parameters), and a synth engine plays it.

The C++ counterpart to the Rust/Truce `VAE_SURGE` experiment, sharing its
schema v1 (same ids, order, defaults and mock decoder).

```
prompt ──► PromptEncoder ──► latent[8] ──► LatentDecoder ──► controls[32] ──► engine
           (stub: hash)                    (ONNX, or mock)    schema/controls.json
```

## Layout

| Path | What |
|---|---|
| `schema/controls.json` | **Source of truth** for the control vector (ids, order, defaults, Surge param names). CMake generates `ControlSchema.generated.h` from it. |
| `source/schema/` | `Snapshot`, `mockDecode`, and `SurgeCurves.h` (Surge's value curves, re-implemented) |
| `source/engine/SynthEngine.h` | Backend interface + factory |
| `source/engine/lite/` | **Lite** backend: Surge-shaped subset on the permissive DSP vendored from Ablaze.WASMSynth (`lite/ablaze/`) |
| `source/engine/surge/` | **Surge XT** backend: headless `surge-common` (only with `VAE_WITH_SURGE=ON`, GPL) |
| `source/ml/` | ONNX Runtime (loaded at runtime), generic `OnnxModel`, `PresetGenerator` (encoder → decoder on a worker thread) |
| `source/params/` | Clean-room parameter store (`ParamStore`, UI bindings), built from `docs/specs/parameter-manager.md` |
| `source/plugin/` | `AudioProcessor`, editor, look and feel |
| `models/` | `make_stub_decoder.py` and the committed `stub_decoder.onnx` (embedded in the plug-in) |
| `tests/` | `vae_params_tests`, `vae_engine_tests`, `vae_ml_tests` (CTest) |

No AudioFX.Core dependency.

## Build

Needs CMake ≥ 3.22, a C++20 compiler and network access on first configure
(JUCE, ONNX Runtime, and Surge if enabled are fetched).

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build
```

Options:

| Option | Default | |
|---|---|---|
| `VAE_WITH_SURGE` | `OFF` | Add the Surge XT backend. **Makes the binary GPL-3.0-or-later**; see `docs/LICENSING.md`. |
| `VAE_JUCE_DIR` | fetch `VAE_JUCE_TAG` (8.0.10) | Use a local JUCE checkout |
| `VAE_ORT_ROOT` | fetch `VAE_ORT_VERSION` (1.22.0) | Use an extracted ONNX Runtime package |
| `VAE_SURGE_DIR` | fetch `VAE_SURGE_REV` | Use a local Surge checkout (with submodules) |
| `VAE_PLUGIN_FORMATS` | `VST3;Standalone` | AU is added on macOS |

Linux needs the usual JUCE packages (`libasound2-dev libx11-dev libxrandr-dev
libxinerama-dev libxcursor-dev libfreetype-dev libfontconfig1-dev
libgl1-mesa-dev`).

### ONNX Runtime

The plug-in compiles against ORT's headers with `ORT_API_MANUAL_INIT` and
**loads the library at runtime**. It never links against it. The build copies
`libonnxruntime.{so,dylib}` / `onnxruntime.dll` next to each plug-in binary
(macOS: `Contents/Frameworks`). If the library is missing, the plug-in still
loads and falls back to the mock decoder. `VAE_ORT_LIBRARY=/path/to/lib`
overrides the search, for development.

Decoder contract: one float input `[batch, 8]`, first output float
`[batch, 32]` in 0..1, in schema order. Load one from **Model...** in the
editor. **Stub** returns to the embedded placeholder.

## Engines

The host parameters are the 32 schema controls. Both backends read the vector
with Surge's semantics, so a vector sounds broadly similar on either. The
backend is a hidden, persisted setting (the combo box at the top right). It is
not restored from presets.

- **Lite** (always built, permissive): Osc 1 → wavetable osc with unison,
  Osc 2 → PolyBLEP osc, Osc 3 → sub osc (−3..+3 oct), Surge filter types
  folded onto LP12/LP24/HP/BP/Notch, Surge envelope times. Filter balance and
  LFO 1 are not modelled. See `source/engine/lite/LiteEngine.h`.
- **Surge XT** (`VAE_WITH_SURGE`): each control is resolved by Surge parameter
  name at the pinned revision; the tests assert all 32 resolve.

## Known schema v1 issues (inherited from VAE_SURGE)

- Surge's default patch **mutes Osc 3 and Noise**, and v1 has no control to
  unmute them, so `osc3_*` and `noise_volume` are inaudible on Surge. Lite
  mirrors that for parity.
- `A Osc 1 Shape`, `Unison Voices` and `Unison Detune` are Classic-osc names.
  For other osc types the same slots mean other things.
- `filter_type` 0.0 is Surge's "Off", so the default vector is unfiltered.

These want a schema v2. Bump `schema_version` when changing ids, order or
semantics.
