# SAME-Small retrieval experiment

> Historical Python prototype / export investigation. Subsequent native implementation and current status: [hackathon guide](HACKATHON-ONNX-SAME-LEARNINGS.md), [native generator](NATIVE-GENERATOR.md), [native SAME](NATIVE-SAME.md). Statements below about native work still being pending describe the earlier experiment.

SAME-S now provides an alternate retrieval path, independent of the student and
any trained synth decoder. It is an **offline Python experiment**, not yet inference
inside the JUCE plug-in. The output is the same editable JSON preset format consumed
by **Preset...**. Stable Audio text-to-audio generation is available through the separate
[prompt prototype](PROMPT-PRESETS.md).

## Reproduce

From the repository root, using an existing spectral bank built as described in
[RETRIEVAL.md](RETRIEVAL.md):

```sh
uv venv --python 3.11 .cache/same-venv
uv pip install --python .cache/same-venv/bin/python -r tools/retrieve/requirements-same.txt
cmake --build build-lite-au --target vae_retrieve

build-lite-au/tools/retrieve/vae_retrieve_artefacts/Release/vae_retrieve \
  export build-lite-au/factory-bank.json build-lite-au/same-audio

.cache/same-venv/bin/python tools/retrieve/same_search.py build \
  build-lite-au/same-audio/manifest.json build-lite-au/same-index.npz

.cache/same-venv/bin/python tools/retrieve/same_search.py search \
  build-lite-au/same-audio/manifest.json build-lite-au/same-index.npz \
  reference.wav build-lite-au/same-results
```

All output paths must be new. The model weights download through the official
`stabilityai/SAME-S` model configuration and use the normal Hugging Face cache.
No text generator weights are needed. Model code is pinned in requirements;
transitive packages follow upstream constraints. `--device cpu` is the default;
`--device mps` or `--device cuda` must precede the subcommand and are optional,
not required for this experiment. The bank build uses batches of eight; lower
`--batch-size` if memory is constrained.

An export of this factory bank uses about 1 GB of WAV data. The pooled feature
cache is only a few MB. Python dependencies and model weights occupy additional
space. Generated artifacts stay in ignored build/cache directories.

## Representation and comparison

The official `AutoencoderModel.encode` API returns `[batch, 256, time]` for
SAME-S. The adapter resolves the model files once, fingerprints their contents,
and loads those exact files using the pinned upstream loader. It resamples to
44.1 kHz, repeats mono to stereo, aligns the first sample exceeding 1% of peak,
crops/pads to four seconds and peak-normalises. Input is limited to 30 seconds,
one or two channels, and 8–192 kHz; silence/non-finite input is rejected.

We concatenate each channel's temporal mean and standard deviation, yielding
512 values, then L2-normalise. Search uses cosine distance and returns five
distinct control vectors. This is a **heuristic retrieval representation**, not
a trained preset mapper or a documented recommendation from Stability. It loses
precise temporal order. SAME audio latents do not directly represent synth knobs.

The `.npz` cache uses no pickle. Metadata records weights/config hashes, preprocessing
and pooling versions, the manifest hash, and all rendered-audio hashes. Search
rejects mismatched model/preprocessing/manifest identities and verifies the selected
WAV previews against the indexed content. Rebuild after changing DSP, schema,
model, preprocessing or pooling. Re-export audio after DSP changes. A failed build
does not publish a partial index; rerunning currently starts encoding from scratch.

`match-N.wav` is copied from the actual indexed render. `match-N.json` is playable
in the existing plug-in and records source, MIDI note, method and cosine distance.
`provenance.json` records model/index identity and the reference hash. Distance is
not a calibrated confidence score, and is not comparable numerically with the
spectral baseline's distance.

## Evaluate before choosing a default

Run both methods on the same isolated reference notes. Compare the five candidates
by ear and judge timbre, envelope and playability separately. A same-render query
should recover its source (unless another preset produces effectively identical
sound); that verifies wiring, not generalisation. Synthetic references outside the
bank exercise retrieval but cannot establish broad musical usefulness. Generated
Stable Audio references and user listening judgements are still needed.

```sh
.cache/same-venv/bin/python tests/python/test_same_search.py
ctest --test-dir build-lite-au -R 'vae_retrieve_smoke|vae_format_check' --output-on-failure
```

## Native JUCE direction

The official repository includes SAME-S encoder/decoder TFLite exports, plus MLX
encoding on Apple Silicon. Its current TFLite pipeline uses fixed-length rung
models and LiteRT >= 2.2.0, with Python dispatch. This is evidence for a native
runtime path, not a tested JUCE C++ integration. We still need to validate the
native LiteRT API, reproduce preprocessing and latent parity, handle fixed input
lengths and package models/runtime dependencies. Do that after retrieval quality
is useful. Inference, indexing and ranking must stay off the audio callback.

Sources inspected at commit `77e0398c26f7d22d801ec1bb036307e8b231f741`:

- [Official autoencoder implementation](https://github.com/Stability-AI/stable-audio-3/blob/77e0398c26f7d22d801ec1bb036307e8b231f741/stable_audio_3/model.py)
- [Official model configuration](https://github.com/Stability-AI/stable-audio-3/blob/77e0398c26f7d22d801ec1bb036307e8b231f741/stable_audio_3/model_configs.py)
- [TFLite runtime and encoder exports](https://github.com/Stability-AI/stable-audio-3/blob/77e0398c26f7d22d801ec1bb036307e8b231f741/optimized/tflite/README.md)
- [MLX audio encoding](https://github.com/Stability-AI/stable-audio-3/blob/77e0398c26f7d22d801ec1bb036307e8b231f741/optimized/mlx/models/defs/audio_encoding.py)

## Listening report

This helper runs both search paths on each supplied reference and produces a
local HTML page with reference audio, both sets of previews, and preset links:

```sh
.cache/same-venv/bin/python tools/retrieve/compare_searches.py \
  build-lite-au/tools/retrieve/vae_retrieve_artefacts/Release/vae_retrieve \
  build-lite-au/factory-bank.json build-lite-au/same-audio/manifest.json \
  build-lite-au/same-index.npz build-lite-au/listening-report \
  reference.wav another-reference.wav
```

Use the SAME environment to run this helper. Open the resulting `index.html` in
a browser. Each result JSON is also loadable through the instrument's **Preset...**
button. The script refuses to overwrite an existing report directory.

## Local experiment result (2026-10-20)

The full factory bank encoded successfully on CPU: **1,878 × 512** finite,
unit-normalised feature vectors covering **626** distinct playable control vectors.
The compressed index is approximately **3.4 MB**. Four Python contract tests and
the C++ export/retrieval smoke and formatting checks passed.

A query using the indexed Init Saw render recovered its exact control vector at
rank 2 with cosine distance below 1e-5. Init Modern ranked first with effectively
the same feature vector; acoustically equivalent controls can tie. This verifies
the model/cache/export path, not general sound-matching quality.

A synthetic sine and an inharmonic decaying bell were also queried against both
methods. Their rankings differ. The local listening report is at
`build-lite-au/same-listening-report/index.html`, with five candidates per method
per reference. No human listening judgement has been collected, so neither method
is declared better. These references were constructed locally, not generated by
Stable Audio. That experiment did not include prompt generation. See [the prompt prototype and
ONNX feasibility results](PROMPT-PRESETS.md) for subsequent work.
