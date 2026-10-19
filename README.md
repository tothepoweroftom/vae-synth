# vae-synth

A JUCE synth for **prompt → preset** sound design. A prompt becomes a latent
vector, a decoder turns that into a fixed **control vector** (93 Surge-style
controls), and a synth engine plays it.

The control vector follows the student's component list (schema v2, see
`docs/SCHEMA.md`). It is a Surge-shaped synth: voice/glide, 3 oscillators,
2 filters, 2 curved ADSRs, 2 LFOs and a fixed FX chain, with categorical
controls one-hot encoded for the model.

```
prompt ──► PromptEncoder ──► latent[8] ──► LatentDecoder ──► features[191] ──► controls[93] ──► engine
           (stub: hash)                    (ONNX, or mock)    (argmax/clamp)    schema/controls.json
```

## Layout

| Path | What |
|---|---|
| `schema/controls.json` | **Source of truth** for the control vector (ids, kinds, options, curves, defaults, Surge mapping notes). CMake generates `ControlSchema.generated.h` from it. |
| `docs/SCHEMA.md` | Value encoding, model features, decoder contract, Surge render/import rules |
| `source/schema/` | `Snapshot`, feature encode/decode, `mockDecode`, and `SurgeCurves.h` (curve units, display) |
| `tools/` | `vae_schema.py` (Python twin of the schema) and `surge_preset_to_vector.py` (Surge `.fxp` → vectors, for the dataset sampler) |
| `source/engine/SynthEngine.h` | Backend interface + factory |
| `source/engine/lite/` | **Lite** backend: Surge-shaped subset on the permissive DSP vendored from Ablaze.WASMSynth (`lite/ablaze/`) |
| `source/engine/surge/` | **Surge XT** backend: headless `surge-common` (only with `VAE_WITH_SURGE=ON`, GPL) |
| `source/ml/` | ONNX Runtime (loaded at runtime), generic `OnnxModel`, `PresetGenerator` (encoder → decoder on a worker thread) |
| `source/params/` | Clean-room parameter store (`ParamStore`, UI bindings), built from `docs/specs/parameter-manager.md` |
| `source/plugin/` | `AudioProcessor`, editor, look and feel |
| `models/` | `make_stub_decoder.py` and the committed `stub_decoder.onnx` (embedded in the plug-in) |
| `tests/` | `vae_params_tests`, `vae_engine_tests`, `vae_ml_tests`, `vae_schema_tools` (CTest) |

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
`[batch, 191]` features (see `docs/SCHEMA.md`). Load one from **Model...** in the
editor. **Stub** returns to the embedded placeholder.

## Engines

The host parameters are the 93 schema controls: continuous controls are
0..1 knobs, categorical ones are choices and toggles are switches. Both
backends read the vector with Surge's semantics, so a vector sounds broadly
similar on either. The backend is a hidden, persisted setting (the combo box
at the top right). It is part of the plug-in state, so loading a project or a
host preset restores it too (JUCE cannot tell the two apart).

- **Lite** (always built, permissive): its own implementation of the full v2
  feature set. See `source/engine/lite/`.
- **Surge XT** (`VAE_WITH_SURGE`): each control is mapped onto Surge scene A
  parameters, fixed FX slots and modulation routes. See
  `source/engine/surge/SurgeEngine.h`.

## Importing Surge presets

```sh
python3 tools/surge_preset_to_vector.py path/to/presets -o vectors.jsonl --stats
```

This converts every `.fxp` into raw values plus model features and lists what
did not map exactly. All 641 Surge factory presets convert.

## Open question: modulation routing

The `*_destination` / `*_depth` controls are **provisional**, with one route
per filter EG / LFO, until the student decides how routing should appear in
the vector. Options and factory-library statistics are in `docs/SCHEMA.md`.
