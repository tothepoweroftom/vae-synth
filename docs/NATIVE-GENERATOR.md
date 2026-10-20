# Native Stable Audio Small → playable presets

The JUCE **Generate** button now runs Stable Audio 3 Small-SFX locally, then encodes the reference with SAME and retrieves five editable Lite presets. It selects the first result automatically. No Python process, local HTTP server, network request, or student model is used at runtime.

## Use this build

Build the `VAESynth_Standalone` or `VAESynth_AU` target. Open the rebuilt standalone app (an already running copy still has the old code).

1. **SAME bank…** → `build-lite-au/native-same/native-bank.json`.
2. **Audio model…** → `build-lite-au/native-generator/generator.json`.
3. Wait for “Stable Audio ready”, enter a prompt, and press **Generate**.
4. Play using MIDI or the onscreen keyboard. Select another SAME match to audition its editable controls.

Initial loading is slow and has been reported to take longer than the earlier roughly-one-minute estimate. Stage timings and the bottleneck have not yet been established. The approximately 3 GB FP32 model package stays loaded between prompts. One measured four-second generation plus search took 22.1 seconds, excluding loading, with ORT restricted to one CPU thread. This prioritizes leaving CPU capacity for the synth over maximum generation speed. Model paths currently need selecting again after restarting the plugin. The applied preset and last prompt are saved in plugin state.

Reference WAVs and prompt/seed metadata are saved under JUCE's user application-data directory in `VAE Synth/generated`. On macOS this is `~/Library/VAE Synth/generated`. **Cancel search** also cancels generation between ONNX calls and suppresses stale callbacks. An in-progress ONNX call or model load finishes before cancellation takes effect. Inference, tokenization, file access and preset search all run on the serialized worker, never the audio callback. Preset controls are applied on the message thread through the existing parameter store.

**Random**, **Model…**, and **Stub** remain the separate latent-decoder experiment. The prompt **Generate** action uses the native Stable Audio route and does not fall back to a hash/stub when a package is missing.

## Native implementation

- `StableAudioGenerator`: three FP32 ONNX sessions, native tokenization, ping-pong diffusion, seeded normal noise, WAV output.
- Text encoder: T5Gemma, 256 tokens, learned padding, `[1,256,768]` output. IDs and masks enter as exactly representable FP32 integers and are cast inside the graph.
- DiT: Small-SFX, eight steps, CFG 1, fixed four-second duration conditioning, six seconds of generation headroom, 108 latent positions. Duration/global conditioning is baked into this version's graph.
- Decoder: SAME-S decoding with bottleneck and mask noise explicitly supplied by C++. Returns four seconds of stereo 44.1 kHz audio.
- Tokenizer: Hugging Face `tokenizers` 0.21.4, statically linked via a small Rust C ABI. Application orchestration and ONNX inference are C++; Rust is a build dependency for the native tokenizer only. No Rust executable is shipped or launched.
- `SameSearchWorker`: serializes package loading, generation and search. Completion and progress callbacks are delivered on the message thread and guarded by request IDs.
- Package loading checks SHA-256 hashes, tensor contracts and schedule endpoints; failed loads retain the previous generator.

Production noise uses `std::mt19937` and `std::normal_distribution<float>`. A seed is repeatable within this build, but is not promised to match PyTorch or a different C++ standard library. Integration tests inject the exact upstream random draws to compare arithmetic independently of RNG algorithms. FP32 text encoding is deliberate: the upstream loader otherwise retains BF16 text tensors, which this CPU ORT path cannot execute.

## Export and build

Python is used only for preparing and validating model assets. The existing authenticated `.cache/same-venv` and pinned Stable Audio checkout are used. Model weights remain external to source control.

```sh
for stage in dit text decoder; do
  .cache/same-venv/bin/python tools/retrieve/export_generator_onnx.py \
    build-lite-au/native-generator --stage "$stage"
done
.cache/same-venv/bin/python tools/retrieve/package_generator.py build-lite-au/native-generator
cmake --build build-lite-au --target VAESynth_Standalone VAESynth_AU vae_generate -j 4
```

Cargo is required when building retrieval/plugin targets; its dependency versions are pinned in `source/ml/tokenizer/Cargo.lock`. This integration has been tested on native macOS arm64, not cross-compiled or qualified on Windows/Linux or universal macOS builds. ONNX Runtime remains the existing dynamically loaded 1.22.0 runtime. The exporter decomposes RMSNorm into mathematically equivalent primitive operations because the PyTorch 2.7 legacy exporter lacks that symbolic operation.

The package manifest is written only after all graphs have parity reports matching their hashes. All four sibling files (`text.onnx`, `dit.onnx`, `decoder.onnx`, `tokenizer.json`) must accompany `generator.json`. These assets retain their upstream model/tokenizer license terms.

## Validation

On the development machine:

- DiT single-step relative RMS error: `6.83e-6`.
- Text encoder relative RMS error: `2.37e-6`.
- Decoder with identical random inputs: `6.84e-4`.
- Native tokenizer IDs and masks exactly match Transformers on six prompts covering whitespace, Unicode, special tokens and truncation.
- Independent complete upstream FP32 generation versus C++ generation with identical noise: relative waveform RMS error `9.59e-5`, cosine `0.9999999955`, maximum absolute sample error `6.19e-5` (including 24-bit WAV quantization).
- Full native generation/search returns five schema-valid, distinct editable presets.

Fixture and CLI commands:

```sh
.cache/same-venv/bin/python tools/retrieve/generator_reference_fixture.py build-lite-au/native-generator-validation
build-lite-au/tools/retrieve/vae_generate_artefacts/Release/vae_generate \
  build-lite-au/native-generator/generator.json \
  build-lite-au/native-same/native-bank.json \
  'A bright metallic bell, single strike, clean decay' \
  build-lite-au/native-generator-validation/cpp \
  build-lite-au/native-generator-validation/noise.raw
```

Omit the final noise-file argument for entirely native random generation. Choose a fresh output directory for each run. CLI output includes the reference WAV and five loadable preset JSON files. Native generation is creative reference synthesis; the final playable sound is still constrained by the existing 626-preset search bank. It does not yet optimize or synthesize new parameter vectors from the reference.

For the distilled lessons, evidence boundaries and prioritized next steps, see [the hackathon guide](HACKATHON-ONNX-SAME-LEARNINGS.md).
