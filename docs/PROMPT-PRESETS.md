# Prompt to playable presets

> Historical Python prototype / export investigation. Subsequent native implementation and current status: [hackathon guide](HACKATHON-ONNX-SAME-LEARNINGS.md), [native generator](NATIVE-GENERATOR.md), [native SAME](NATIVE-SAME.md). Statements below about native work still being pending describe the earlier experiment.

`tools/retrieve/prompt_presets.py` connects Stable Audio 3 Small generation to the
existing SAME search and editable preset format. A local browser UI lets you enter
a prompt, choose Small-SFX or Small-Music, set a seed, audition the generated reference
and five matches, and download each match's JSON for **Preset...** in VAE Synth.
This is the prototype front end; generation and SAME search currently run outside
JUCE. The returned instrument sound is synthesised from controls, not sampled audio.

## Run

From the repository root, after building the SAME bank/index:

```sh
.cache/same-venv/bin/python tools/retrieve/prompt_presets.py serve
```

Open `http://127.0.0.1:8765`. The server binds only to loopback, runs one generation
at a time, and keeps loaded models warm. Output goes to unique directories under
`build-lite-au/prompt-runs/`. Closing the terminal/server stops the service. Defaults
can be overridden with `--manifest`, `--index` and `--device` before `serve`; use
`serve --port` and `serve --output` for the server options.

The first use downloads the chosen model. Accept its Hugging Face access terms
and log in locally if requested:

```sh
.cache/same-venv/bin/hf auth login
```

- [Small-SFX access page](https://huggingface.co/stabilityai/stable-audio-3-small-sfx)
- [Small-Music access page](https://huggingface.co/stabilityai/stable-audio-3-small-music)

Do not paste tokens into the prompt UI. Use Hugging Face's local authentication.

For command-line use:

```sh
.cache/same-venv/bin/python tools/retrieve/prompt_presets.py run \
  --prompt 'An isolated metallic bell hit with a long decay, no background music' \
  --model small-sfx --seed 42 --output build-lite-au/my-bell
```

The exact prompt is sent without hidden rewriting. Generation uses four seconds,
eight diffusion steps, and a recorded seed. Reference audio is stored as a float
WAV to preserve the generated signal. The existing SAME preprocessing handles
normalisation. `request.json` records settings and success/failure; matched presets
and previews live in `matches/`. Search failures preserve the reference for diagnosis.

## ONNX and native JUCE plan

**SAME-Small is proven through the actual C++ runtime wrapper.** We exported a
fixed-shape FP32 graph (opset 17), ran it in ONNX Runtime 1.22.0, and then ran the
same graph through `vae::ml::OnnxModel` in the `vae_same_onnx_probe` executable.
The C++ and Python ONNX outputs were identical on the metallic-bell probe.

Input: preprocessed stereo audio `[1, 2, 176400]` at 44.1 kHz. The graph pads to the
model's minimum-length multiple internally. Output: `[1, 256, 44]` latents. On three
local test sounds (saw, sine, inharmonic bell), relative latent RMS error versus
PyTorch was at most 2.4e-5 and pooled cosine similarity was effectively 1.0. Python
ORT CPU runs with four intra-op threads took approximately 0.27–0.31 seconds each,
excluding model loading and preprocessing. This is a small feasibility test, not
a cross-platform performance guarantee.

The initial strict per-element tolerance rejected a few near-zero latent values
(maximum discrepancy on those entries about 1.6e-4). Verification therefore reports
raw maximum error and uses relative RMS <= 1e-4 plus pooled cosine >= 0.99999 as the
retrieval-focused acceptance criteria. Do not treat the graph as bit-identical to
PyTorch. Recheck ranking parity on broader audio before switching the production
retriever. The current Python pipeline continues to use the original SAME model.

Implementation sequence:

1. **Native retrieval worker:** port sample-rate conversion/onset handling, fixed
   four-second stereo preparation, mean/std pooling, cosine ranking and bank loading
   into C++. Verify preprocessing and ranking against Python. Use the tested SAME
   ONNX graph through the existing runtime and keep inference off the audio thread.
2. **Native prompt generator:** validate the tokenizer/text encoder, diffusion
   transformer, iterative sampler and SAME decoder separately. These require multiple
   inputs and integer token/mask tensors, so extend `OnnxModel` beyond its current
   single-float-input interface. Preserve a Python reference for component parity.
3. **JUCE interface and packaging:** connect prompt/progress/cancel, reference audition
   and candidate selection to background jobs; pass fixed control snapshots to the
   existing parameter path. Package model files and runtime dependencies after the
   selected runtime is validated. Account for worker CPU contention while audio plays.

The official repository publishes ONNX files under its TensorRT workflow. Their
FP16/BF16/FP8 precision and TensorRT-oriented build recipes do not establish that
all components run unchanged on portable ONNX Runtime. Prefer FP32 exports first;
benchmark CoreML/other execution providers later. TFLite is a fallback if a generator
component cannot be made reliable in ORT. No generator ONNX graph has been validated
in JUCE yet.

## Reproduce the encoder experiment

```sh
uv pip install --python .cache/same-venv/bin/python -r tools/retrieve/requirements-onnx.txt
.cache/same-venv/bin/python tools/retrieve/export_same_onnx.py \
  build-lite-au/same-encoder-fp32.onnx reference.wav another-reference.wav
cmake --build build-lite-au --target vae_same_onnx_probe
```

Choose a new model output path, or use `--verify-existing` to validate an existing
export. The exporter writes a `.parity.json` report only after checks pass.
The native probe accepts raw little-endian float32, channel-major preprocessed audio,
and writes raw float32 latents. It is a developer test utility, not the final plug-in
inference adapter. The model and generated outputs stay in ignored build directories.

Tests:

```sh
.cache/same-venv/bin/python tests/python/test_prompt_presets.py
.cache/same-venv/bin/python tests/python/test_same_search.py
```

The prompt contract tests deliberately use a fake generator and do not establish
real generation quality or model access. Live generation must be checked separately.

Sources:

- [Official generation API](https://github.com/Stability-AI/stable-audio-3/blob/77e0398c26f7d22d801ec1bb036307e8b231f741/stable_audio_3/model.py)
- [Official ONNX/TensorRT recipes](https://github.com/Stability-AI/stable-audio-3/blob/77e0398c26f7d22d801ec1bb036307e8b231f741/optimized/tensorRT/build/build_from_onnx.py)
- [ONNX Runtime C++ API](https://onnxruntime.ai/docs/get-started/with-cpp.html)

## Live verification (2026-10-20)

After local Hugging Face login, Small-SFX generated the bell prompt shown in the
UI. The completed run is `build-lite-au/prompt-runs/20261020-154717-1636d2/`:
stereo 44.1 kHz, exactly four seconds, followed by five valid distinct preset
results and WAV previews. Generation plus SAME retrieval took 4.34 seconds on this
run, excluding model loading. Browser verification confirmed reference playback
controls, all five candidates and their preset links. Musical similarity still
requires listening judgement. Small-Music is selectable but has not been live-tested.

The integration exposed an upstream cache preference: once generator weights are
cached, its autoencoder resolver prefers the generator checkpoint over standalone
SAME-S weights. Retrieval now explicitly resolves `stabilityai/SAME-S` so its model
identity remains consistent with the existing index. A regression test covers this.
Five SAME tests, two prompt contract tests and the selected C++ regression checks pass.

The Python service below remains a development prototype. The JUCE prompt route now has a [native ONNX implementation](NATIVE-GENERATOR.md), including native tokenization and diffusion sampling; it no longer needs this service at runtime.
