# Model loading and generation performance

Recorded 21 October 2026 on the development Mac (Apple Silicon, macOS 15). Measured with
`vae_generate`, the bell prompt, installed models, fixed four-second Small-SFX generation. These
are single-machine numbers, not a cross-platform benchmark.

## Results

| | Before | First load | Later loads |
|---|---|---|---|
| Load generator + SAME bank | ~60 s | ~60 s | **~3 s** |
| Generation + search | 22.1 s | ~7–8 s | **~6 s** |

## What the profile showed

A `sample` of a load showed most of the time in `juce::SHA256` reading the ~3 GB package. Every load
re-hashed every file, then ONNX Runtime re-optimized every graph. Generation was slow because ORT
was restricted to one intra-op thread.

## Changes

1. **Hash once per file.** `matchesSha256` ([StableAudioGenerator.cpp](../source/ml/StableAudioGenerator.cpp))
   writes a stamp to `cache/verified/<sha256>` holding the file's path, size and mtime. A matching
   stamp skips the hash. The stamp uses `|` separators because `File::replaceWithText` rewrites
   `\n` as `\r\n`. An earlier version used newlines, so the stamp never matched and every load still
   hashed.
2. **Optimize once.** `OnnxModel::load (file, { threads, cacheKey })` ([OnnxModel.cpp](../source/ml/OnnxModel.cpp))
   saves ORT's optimized graph to `cache/optimized/<sha256>-ort<version>.onnx`, with weights in a
   `.data` sidecar to stay under protobuf's 2 GB limit. Later loads open it with graph optimization
   disabled. The cache is keyed by content and runtime version, so a changed model or ORT never
   reuses a stale graph. It is written to a `.partial` name and renamed only on success. An
   unreadable cache is deleted and rebuilt, and a failed save falls back to a plain load.
3. **More threads.** Generator sessions use `clamp (physical cores / 2, 1, 4)` intra-op threads.
   `session.intra_op.allow_spinning = 0` makes idle workers sleep instead of busy-waiting next to
   the audio thread. The SAME encoder and the latent decoder stay at one thread.
4. **Installed location and preload.** `scripts/install-models.sh` copies the assets to
   `~/Library/Application Support/VAE Synth/models` (`appDataDir()`; JUCE's
   `userApplicationDataDirectory` is `~/Library` on macOS, so the subfolder is added explicitly).
   The editor starts a background preload when it opens, and a spinner shows while the worker is
   busy. `VAE_DATA_DIR` overrides the folder. The ML and plugin tests set it so they never load the
   user's 3 GB models.

## Costs and risks

- **Disk:** the optimized cache is ~3.1 GB, on top of the ~3.2 GB installed models.
- **Integrity:** the stamp catches corruption and replaced files, not deliberate tampering that
  preserves the path, size and mtime.
- **Hardware-specific cache:** ORT warns that `ORT_ENABLE_ALL` graphs may be specific to the
  hardware that optimized them. If the cache is migrated to another machine and fails to load, it
  is rebuilt; this path has not been tested.
- **CPU contention:** four threads during generation have not been measured against a busy DAW
  session. Check for dropouts.
- **Memory:** each plugin instance preloads ~3 GB when its editor opens.

## Options not yet tried

- **CoreML execution provider (macOS).** The bundled ORT 1.22.0 `osx-universal2` library includes
  `CoreMLExecutionProvider` (checked in the binary). Try `MLProgram` format with a
  `ModelCacheDirectory`, falling back to CPU. It may speed up generation and move work off the CPUs
  the audio thread needs. Expect a slow first compile of the 1.8 GB DiT, CPU fallback for
  unsupported ops (e.g. the decomposed RMSNorm), and fp16 arithmetic that needs fresh parity and
  listening tests.
- **DirectML (Windows).** Needs the DirectML ORT package instead of the current pin in
  [VaeOnnxRuntime.cmake](../cmake/VaeOnnxRuntime.cmake).
- **`stabilityai/stable-audio-3-optimized`.**
  - *TensorRT:* NVIDIA only.
  - *ONNX exports* (`onnx/t5gemma/encoder.onnx`, `onnx/sa3-sm-sfx/dit_fp16.onnx`, `onnx/same-*/dec_dynamic_*.onnx`):
    made for TensorRT. They are fp16 (poorly supported by the ORT CPU provider), and their tensor
    contracts likely differ from our sampler's.
  - *LiteRT/TFLite int8* (~1.2 GB in total: DiT 0.5 GB, T5Gemma 0.6 GB, SAME-S decoder 0.09 GB):
    the closest fit for a portable CPU build. It needs a new runtime dependency. Its int8 DiT gives
    different samples, and its README advises clips of 20 s or longer.
  - *MLX:* fastest on Apple Silicon (~1 s per 10 s clip on M1), but the model code is Python and
    would need porting to MLX C++.
