# Hackathon guide: Stable Audio Small, SAME and native ONNX

Recorded 20 October 2026. This is the reusable engineering handoff for the independent `vae-synth` backup. It describes our measured implementation, not a general performance or quality guarantee for these models.

## What we have achieved

A prompt can produce a generated reference and five editable, playable synth presets without Python at runtime:

```text
Prompt
  → native Hugging Face tokenizer
  → T5Gemma text encoder (ONNX)
  → Small-SFX diffusion transformer (ONNX, eight C++ sampling steps)
  → SAME audio decoder (ONNX)
  → four-second stereo reference
  → standalone SAME-S encoder (ONNX)
  → pooled embedding → preset-bank cosine search
  → selected Lite control snapshot → MIDI-playable JUCE synth
```

The same native generation/search core has completed real command-line runs. Standalone and AU builds succeed, and the rebuilt interface exposes this route through **Generate**. The user's full interactive load/generate/play experience still needs confirmation: loading was reported as slow. We have not established a startup-time bound or diagnosed that delay.

This route is independent of the student's deliverables. Python remains an asset-export, bank-building and validation tool, not a runtime service. The old localhost browser prototype is separate from the native plugin.

**Accurate demo claim:** “Describe a sound; generate an audio reference; find a playable, editable synth patch that approximates it.” Do not claim direct text-to-parameters, exact sound reconstruction, arbitrary durations, or newly generated synth parameters.

## What SAME is doing—and what it is not

SAME appears twice: the generator uses an audio decoder to turn diffusion latents into a waveform; retrieval uses a separately packaged standalone SAME-S encoder to describe that waveform. These are different roles. Do not assume matching names mean interchangeable checkpoint files.

**Yes, the query and preset bank use the same retrieval encoder identity and preprocessing contract.** The bank was encoded in PyTorch; the query is encoded by its validated ONNX export. Model hashes and parity tests connect the two implementations.

SAME produces audio latents, not our 93 synth controls. Our retrieval descriptor concatenates temporal mean and population standard deviation for each of 256 channels, producing 512 values, then L2-normalizes. Cosine distance compares descriptors. This pooling is our heuristic, not a learned preset mapper or a proven best perceptual metric. It discards precise temporal order.

The bank currently covers **626 distinct control vectors**, rendered at MIDI notes **48, 60 and 72**, giving **1,878 searchable examples**. That is enough to exercise the pipeline and demonstrate retrieval. It does not prove coverage of every requested timbre, and it is not evidence that we have enough data to train a good inverse model. Multiple note renders improve pitch coverage; they do not add new patches. Results are deduplicated by control vector.

The system is creative in reference generation and subsequent musical editing. The final automatic choice is currently retrieval from a finite bank. A productive extension is to mutate a retrieved patch, render it through the actual synth, encode it, and keep improvements. This is proposed work, not implemented behavior.

## Lessons worth carrying into the hackathon

| Finding | What to do next time |
| --- | --- |
| An upstream ONNX/TensorRT recipe is not proof that the model runs on CPU ONNX Runtime. | Start with FP32 and the intended execution provider. Test an actual exported graph before committing to deployment claims. |
| Generation is a pipeline, not one neural-network call. | Separate tokenizer, text encoder, denoiser, sampler and audio decoder. Validate each boundary before wiring the UI. |
| PyTorch 2.7's legacy exporter could not export `aten::rms_norm` at opset 17. | Decompose it into equivalent primitive arithmetic, then measure error. Never assume an export workaround preserves behavior. |
| The text encoder retained BF16 tensors despite the generator being requested in full precision. A BF16 `Where` failed on our CPU runtime. | Explicitly convert the text encoder to FP32. Check actual tensor types rather than relying on a top-level flag. |
| The audio decoder adds noise at inference, as well as the sampler doing so. | Supply decoder noise explicitly as ONNX inputs. Different random draws can look like a large numerical-porting bug. |
| A seed does not specify the same random sequence across PyTorch and C++ libraries. | Inject identical saved noise for parity tests. Record production seeds, but only promise reproducibility within the tested build. |
| Tokenizer details matter: Unicode, whitespace, special tokens and truncation. | Reuse the native Hugging Face tokenizer. We linked its Rust library through a small C ABI and compared exact token IDs/masks. No Python tokenizer is launched. |
| Four output seconds require more than four seconds of diffusion context. | Preserve the upstream six-second headroom and alignment. Our fixed configuration uses 108 latent positions and then trims to 176,400 samples per channel. |
| Upstream SAME resolution could prefer a cached generator checkpoint after generation was used. | Explicitly resolve the standalone retrieval checkpoint and fingerprint the files actually loaded. Cache availability must not silently change index identity. |
| Tiny elementwise errors can be misleading near zero and can reorder near-ties. | Report relative RMS, maximum error, embedding cosine and retrieval rankings. Allow a documented tolerance for genuinely tied results. |
| Correct inference does not imply useful sound matching. | Keep numerical parity tests separate from listening evaluations. Compare envelope, timbre and playability by ear. |

### The preprocessing contract is part of the model

For both bank and query: preserve stereo or duplicate mono; align onset at 1% of peak; resample to 44.1 kHz with the matching Hann-windowed sinc; crop/pad to four seconds; peak-normalize. The encoder graph pads internally to its required length multiple and produces `[1,256,44]`.

Our native resampler reproduces torchaudio's width-6, rolloff-0.99 behavior. An arbitrary JUCE resampler is not automatically equivalent. Reject silent and non-finite input. Changing the encoder, pooling or preprocessing means rebuilding the index. Changing the synth DSP means rerendering the bank and re-encoding it. Parameter schema/control changes also require compatibility checks.

### Preserve real-time safety

All tokenization, ONNX calls, audio-file access, allocations and ranking belong on a worker. Our serialized worker prevents concurrent operations on shared sessions. Progress/results return to the message thread; request IDs suppress stale callbacks. Applying a preset uses the existing host-notified parameter path.

No model work was added to `processBlock`. Keep future inference optimizations out of that callback too. Cancellation currently takes effect between ONNX calls, not inside an executing call or model load. Shutdown waits for active work. These are practical UX limits to remember during a demo.

## Evidence we can show

| Check | Observed result |
| --- | --- |
| Native tokenizer vs Transformers | Exact IDs and masks on six prompts, including Unicode and truncation |
| Text encoder ONNX vs PyTorch | Relative RMS `2.37e-6` |
| DiT single step ONNX vs PyTorch | Relative RMS `6.83e-6` |
| Decoder, identical noise | Relative RMS `6.84e-4` |
| Complete native waveform vs independent upstream FP32 generation, identical noise | Relative RMS `9.59e-5` (0.0096%); cosine `0.9999999955` |
| Native SAME preprocessing/ranking | Three reference cases passed; preprocessing error at most `3e-7`, ranking tie tolerance `3e-5` |
| End-to-end native generation and search | Bell and bass prompts each returned five distinct playable preset vectors |
| Generation plus search, excluding model loading | About 22.1 seconds for the bell fixture and 18.9 seconds for native-random bass, on this development Mac |
| Regression checks | ML, plugin, retrieval smoke and formatting checks passed |

These are small local tests, not broad model-quality, real-time-load or cross-platform benchmarks. The generator comparisons use FP32 text encoding; they do not establish bit-identical results with the earlier BF16-text Python prototype.

The generator package is approximately 3 GB and stays loaded between prompts. Update (21 October): a profile showed loading dominated by re-hashing the package and re-optimizing each graph on every load. Hash results and ORT's optimized graphs are now cached per machine, so warm loads take about 3 s (first load about 60 s). Generator sessions now use up to four non-spinning threads, cutting generation plus search to about 6 s; CPU contention with a busy DAW during generation is not yet measured. Details and untried options: [MODEL-LOADING-PERFORMANCE.md](MODEL-LOADING-PERFORMANCE.md).

## Prioritized next steps

1. **Make loading understandable and dependable.** Instrument hash verification, each ONNX session load/optimization, and tokenizer loading separately. Add stage progress and elapsed time. Measure cold and warm loads and peak memory. Then evaluate cached optimized graphs or session settings against parity tests; do not guess the bottleneck or remove integrity checks blindly.
2. **Finish the actual demo acceptance test.** Let the user load both packages, generate, choose each result and play MIDI. Verify cancellation and responsiveness while audio is running. CLI success and an AU build are not a completed DAW test. Preload before presenting and keep a known reference/preset available as a fallback.
3. **Improve everyday usability.** Remember package/bank paths, show readiness independently for both, and add reference audition plus matched-preset preview. The native UI currently selects the first match; it does not yet provide the browser prototype's reference-audio audition controls. Keep prompt/seed provenance and expose a reproducible seed when useful.
4. **Measure retrieval quality.** Collect a small representative set of prompts and reference notes across bass, lead, pad, pluck and percussion. Listen blind to SAME and the existing spectral baseline. Record top-five usefulness, envelope similarity and editability. Use failures to choose whether to broaden the bank, alter descriptors or add reranking.
5. **Add controlled creativity beyond retrieval.** Start from a good retrieved patch, vary a small set of continuous controls, render and rerank. Handle categorical controls deliberately; naive averaging of every parameter is not a reliable synth morph. Keep originals and require a measured improvement before promoting a candidate.
6. **Optimize and distribute only after the above.** Benchmark thread counts or hardware providers with audio active. Treat quantization, FP16 and CoreML as experiments requiring fresh parity/listening tests. Qualify other platforms and package the runtime, tokenizer and model assets with their upstream terms. Small-Music is not validated by this Small-SFX port.

For a hackathon deadline, prioritize steps 1–3 and a small listening set. Reuse the working fixed four-second CPU path before expanding model options or duration support.

## Reuse map and demo runbook

From the `vae-synth` repository:

- Native core: [StableAudioGenerator.cpp](../source/ml/StableAudioGenerator.cpp), [SameSearch.cpp](../source/ml/SameSearch.cpp), [OnnxModel.cpp](../source/ml/OnnxModel.cpp).
- Native tokenizer: [Rust C ABI](../source/ml/tokenizer/src/lib.rs), [locked dependencies](../source/ml/tokenizer/Cargo.lock).
- Asset preparation: [generator export](../tools/retrieve/export_generator_onnx.py), [package validation](../tools/retrieve/package_generator.py), [SAME export](../tools/retrieve/export_same_onnx.py), [bank packaging](../tools/retrieve/export_native_bank.py).
- Reference fixtures: [upstream generation recorder](../tools/retrieve/generator_reference_fixture.py), [full waveform test](../tests/python/test_native_generation_parity.py), [tokenizer test](../tests/python/test_native_generator.py), [SAME integration test](../tests/python/test_native_same.py).
- Setup details: [native generator](NATIVE-GENERATOR.md), [native SAME](NATIVE-SAME.md). The Python prototype remains documented in [PROMPT-PRESETS.md](PROMPT-PRESETS.md).

Open the rebuilt standalone at `build-lite-au/VAESynth_artefacts/Release/Standalone/VAE Synth.app`. Load `build-lite-au/native-same/native-bank.json` through **SAME bank…**, and `build-lite-au/native-generator/generator.json` through **Audio model…**. Wait for readiness, then type an isolated-sound prompt and press **Generate**. Play the selected result and try alternatives. An already running old binary does not acquire new code when the app is rebuilt.

Keep each JSON manifest beside its model files. Runtime generation needs neither Hugging Face login nor the Python server once those packages exist. Asset acquisition/export does need the previously authorized model access. Generated build/cache assets are ignored by Git: moving the source repository alone will not move the models or bank. Model paths must currently be selected again after restarting.

Known test outputs are in `build-lite-au/native-generator-validation/cpp/` and `build-lite-au/native-generator-validation/native-bass/`. The waveform parity report is `build-lite-au/native-generator-validation/waveform-parity.json`. For a quick regression check:

```sh
ctest --test-dir build-lite-au \
  -R 'vae_ml_tests|vae_plugin_tests|vae_retrieve_smoke|vae_format_check' --output-on-failure
```

Preserve the validated dependency baseline: upstream Stable Audio commit `77e0398c26f7d22d801ec1bb036307e8b231f741`, PyTorch 2.7.1, ONNX opset 17, ORT 1.22.0, and the Cargo lockfile. Re-run the appropriate parity checks when changing it.
