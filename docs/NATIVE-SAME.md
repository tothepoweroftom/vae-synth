# Native SAME inference and preset search

Reference-audio retrieval now runs inside the JUCE plug-in and standalone in C++.
Python is needed only to prepare the distributable bank/model bundle. Runtime
inference uses the existing ONNX Runtime 1.22.0 wrapper. The prompt generator also now runs natively; see
[Native Stable Audio generator](NATIVE-GENERATOR.md) for its setup and validation.

## Use the built standalone

1. Open `build-lite-au/VAESynth_artefacts/Release/Standalone/VAE Synth.app`.
2. Click **SAME bank...** and select `build-lite-au/native-same/native-bank.json`.
3. Once ready, click **Find from audio...** and select a reference WAV, AIFF or FLAC.
4. Choose a result in the adjacent dropdown. Its controls are applied to Lite;
   play with MIDI or the on-screen keyboard and edit the parameters normally.

A real generated reference is available at
`build-lite-au/prompt-runs/20261020-154717-1636d2/reference.wav`.

The bank stays loaded while the processor exists, including when closing/reopening
the editor. Its file path and the candidate list are not yet persisted in host state;
the selected synth controls use the existing host-state mechanism. Reload the bank
after restarting the plug-in. **Cancel search** discards pending/stale results; it
does not interrupt an ONNX kernel already executing. The existing preset remains
unchanged until a candidate is explicitly selected.

## Threading and validation

`SameSearch` owns the model and immutable bank and provides synchronous operations
for non-audio threads. `SameSearchWorker` serialises load/search jobs on a dedicated
single-thread pool and delivers results on the JUCE message thread. Request IDs
suppress stale and cancelled callbacks. Shutdown waits for an active job before
releasing its resources. ONNX inference itself uses the existing one-thread runtime
configuration to limit contention with the DAW.

No model loading, audio file reading, resampling, embedding allocation or ranking is
added to `processBlock`. Candidate selection uses the existing host-notified control
store. A rejected bank preserves the previous loaded model and entries.

The loader validates schema and format versions, engine, preprocessing/pooling IDs,
control bounds, finite unit embeddings, model SHA-256 and exact ONNX tensor shapes.
The model must be a sibling of the bank JSON. The current runtime supports Lite
banks only. The bundle is trusted local content; hashes detect mismatched files,
not provenance or publisher authenticity.

## Python parity

Native preprocessing follows the experiment: reject non-finite/silent/oversized
input; align the first sample over 1% of peak; resample to 44.1 kHz using a rational
Hann-windowed sinc (width 6, rolloff .99); repeat mono to stereo; crop/pad to four
seconds; peak-normalise. The FIR parameters follow torchaudio's default
`sinc_interp_hann` implementation. Only non-zero tap support is evaluated.

ONNX returns `[1,256,44]`. Per-channel population mean/std produce the same 512-value
unit vector. Search uses cosine distance, stable ordering and distinct control
vectors. Very close ties can change order because PyTorch, ONNX and C++ reductions
have small floating-point differences.

The real-model integration test passed on the 48 kHz bank saw render, a 44.1 kHz
synthetic sine and the generated stereo bell reference. Maximum preprocessing error
was 3e-7 on these files; embedding cosine agreement was effectively 1.0, and all
returned scores/rankings agreed within the test's 3e-5 tie tolerance. The bell's
five results matched the Python run. This verifies the port, not perceptual quality.

## Build / package

```sh
cmake --build build-lite-au --target VAESynth_Standalone vae_same_search vae_plugin_tests
```

The standalone is built and tested locally. Rebuild your desired plug-in format
before testing it in a DAW; no installed AU/VST3 is replaced automatically.

The exporter requires the existing Python SAME index and an ONNX parity report
that fingerprints the exact graph. Existing reports without a graph hash need a
new `export_same_onnx.py --verify-existing` run first.

```sh
.cache/same-venv/bin/python tools/retrieve/export_same_onnx.py --verify-existing \
  build-lite-au/same-encoder-fp32.onnx reference.wav
.cache/same-venv/bin/python tools/retrieve/export_native_bank.py \
  build-lite-au/same-audio/manifest.json build-lite-au/same-index.npz \
  build-lite-au/same-encoder-fp32.onnx build-lite-au/native-same
```

Choose a new bundle output directory. Keep `native-bank.json` and
`same-encoder.onnx` together. The bank stores embeddings/controls directly; native
runtime search needs neither the original rendered WAVs, NumPy, Python nor a server.
The plain JSON bank favours inspection over compactness for this prototype.

## Tests and diagnostics

```sh
ctest --test-dir build-lite-au \
  -R 'vae_plugin_tests|vae_retrieve_smoke|vae_format_check' --output-on-failure
.cache/same-venv/bin/python tests/python/test_native_same.py \
  build-lite-au/tools/retrieve/vae_same_search_artefacts/Release/vae_same_search \
  build-lite-au/native-same/native-bank.json \
  build-lite-au/same-audio/manifest.json build-lite-au/same-index.npz \
  reference.wav another-reference.wav
```

`vae_same_search native-bank.json reference.wav new-output-directory` invokes the
same core as JUCE and exports presets plus diagnostic preprocessed audio/embedding
raw tensors. It performs an extra encoding for diagnostics; it is not a latency
benchmark. Plug-in tests cover malformed latent rejection, message-thread delivery
and cancellation. Real-model tests are optional because model assets are not committed.

Next steps and reusable engineering lessons are recorded in
[the hackathon guide](HACKATHON-ONNX-SAME-LEARNINGS.md).

The generator port is now implemented: see [Native Stable Audio generator](NATIVE-GENERATOR.md) for the fully local JUCE prompt-to-preset route.
