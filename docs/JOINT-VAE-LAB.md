# Joint VAE Lab

The **VAE Lab** tab is an isolated experiment alongside **Synth** and **Prompt**.
Its model/data types live in `source/experiments/joint_vae/` and do not depend on
our 93-control schema. The existing Prompt → Stable Audio → SAME retrieval flow
is unchanged.

## Working now

- Load the student's real `j48` sound encoder and parameter decoder through ONNX.
- Encode an imported 48-value sound code to its posterior mean (including the
  Gaussian prior expert), or start from a standard-normal random latent.
- Explore a two-dimensional slice around the anchor. X/Y select distinct latent
  dimensions; radius controls the distance. The centre preserves the complete
  anchor and all other dimensions remain fixed. Axes have no claimed semantic
  meaning. Decode on mouse release, off the audio thread.
- Inspect all 631 decoded features and export/import a latent point as JSON.
- A failed model replacement preserves the previous model. Inference errors
  never substitute the old mock decoder.
- **Full Surge playback** (`VAE_WITH_SURGE=ON` builds). Every decoded point becomes
  a complete Surge patch: `StudentCodec` (port of the student's `full_row` +
  `codec.decode`) turns the 631 codes into a write plan, `SurgePatchBuilder` (port
  of `SurgeIO.write_plan`) executes it on a private headless Surge loaded with the
  `Init Saw` template and factory wavetables, and the patch data goes to the Surge
  engine, which is selected. All Surge features the student models are kept: 12 osc
  types, 32 wavetables, 16 FX slots, filter models, flags and the 95 routings.
- Loading a retrieval preset or generating from the Prompt tab returns the Surge
  engine to the 93-control schema. While a VAE patch plays, the Synth tab's knobs
  do not reach Surge.

## Not connected yet

Text → audio → sound-code preprocessing is not implemented in the native tab.
The sound-code input must use the matching PCA, not SAME embeddings. Builds
without Surge decode and show the codes but cannot play them.

The handover uses Surge's own patch queue (`enqueuePatchForLoad`): Surge loads the
patch on the audio thread at the next block, as Surge XT does. That load parses
and allocates, so a decode can glitch audio for one block. It happens on mouse
release only, not while dragging.

The tab's session lasts while the editor is open; export a point before closing
it. It is not included in host project state yet. Importing a point restores it
as a new anchor; axis assignments/radius are UI settings, not part of the file.

## Build a bundle

The supplied directory must contain matching `model.pt`, `config.json`,
`schema.json`, `pruning.json`, and `sound_pca.npz`. Export uses CPU PyTorch and
`weights_only=True` checkpoint loading. Install torch, numpy, onnx and onnxruntime
in the development Python environment.

```sh
python3 tools/joint_vae/export_bundle.py \
  --source /path/to/missing_files \
  --out models/joint-vae-j48
```

Open **VAE Lab → Load VAE bundle…** and select that directory's `bundle.json`.
Generated bundles are ignored by Git. No weights are automatically installed or
bundled into releases. The exporter checks 36 vectors across all nine categories
against PyTorch. `parity.json` holds reference outputs for native validation.
`example-sound.json` contains a **synthetic test vector**, not an encoded real
sound; use it only to check the import path.

## Contracts

`decoder.onnx`: float inputs `latent[1,32]`, `category[1,9]`; float output
`codes[1,N]`. Values are in `features` order: continuous values, thresholded
booleans, categorical argmax indices, then stepped knobs converted back to 0–1.
The current checkpoint has N=631. Frozen/omitted features remain in
`pruning.json`; the decoder output alone is not a complete Surge write plan.
`StudentCodec` restores them (mode codes; "reference" features keep the template's
value) before planning.

`sound_encoder.onnx`: float inputs `sound[1,48]`, `category[1,9]`; float output
`latent[1,32]` (posterior mean). Categories: Bass, Lead, Pad, Keys, Pluck, Synth,
Atmosphere, Orchestral, Unknown. Unknown support is not evidence of good
category-free generation.

Import format:

```json
{"bundle_id":"ID from bundle.json","category":0,"sound":[48 numbers]}
```

Or use `latent` with 32 numbers. These array descriptions are placeholders, not
literal valid JSON. Exported point files include real values and can be imported
directly. Bundle IDs fingerprint weights and metadata to reject mismatched data.

## Verification

```sh
cmake --build build --target vae_ml_tests vae_plugin_tests
VAE_JOINT_BUNDLE="$PWD/models/joint-vae-j48/bundle.json" \
  ctest --test-dir build -R 'vae_ml_tests|vae_plugin_tests' --output-on-failure
```

With `VAE_WITH_SURGE=ON`, also write the Python codec's plans into the bundle first
(standard library only); the tests compare the native codec with them exactly and
build Surge patches from them:

```sh
python3 tools/joint_vae/codec_parity.py --bundle models/joint-vae-j48
cmake -S . -B build-surge -G Ninja -DCMAKE_BUILD_TYPE=Release -DVAE_WITH_SURGE=ON
cmake --build build-surge
VAE_JOINT_BUNDLE="$PWD/models/joint-vae-j48/bundle.json" ctest --test-dir build-surge
```

No inference, model loading, or experiment UI logic runs in the audio callback.
The Surge patch load itself does (see above).

Surge's factory data (template, wavetables) is read from the build tree
(`VAE_SURGE_DATA_DIR`, the fetched Surge checkout). A shipped build needs it
installed next to the models.

## Agent handoff — 2026-10-21

### User intent and agreed boundaries

The intended end result is a separate tab for **text → Stable Audio reference →
joint-VAE sound encoder → latent → parameter decoder → playable editable preset**,
with an XY pad exploring variations around that latent. Keep the existing Synth
and Prompt/retrieval paths available so the user can compare approaches. The
current inspector is an intermediate milestone, not completion of that goal.
The user explicitly authorized proceeding while the student supplied assets;
those assets have now arrived. No further permission is needed for ordinary
implementation/testing within that scope.

The student's existing `vae/pitcher/xypad/` HTML demo is a different experiment:
functional parameter macros and a random-forest category overlay, not a VAE
latent map. Its embedded data is in `vae/runs/xypad/xypad.html`. Do not conflate
that map with this tab's local latent slice or present its preset dots as latent
coordinates. User discussion moved toward the latent-space workflow above.

### Local assets and tested build

Workspace: `/Users/thomaspower/Developer/Hackathon/vae-synth`.

Original artifacts supplied by the user are in
`/Users/thomaspower/Downloads/missing_files/`:

- `model.pt`, `config.json`: run `j48`, z=32, hidden=512, dropout=0.1, k=48.
- `schema.json`, `pruning.json`: matching feature definitions and frozen values.
- `sound_pca.npz`: components `[48,8896]`, mean `[8896]`, scalar scale. The pooled
  spectrogram is **139 × 64**, not 140 × 64: 559 frames are truncated to 556
  before pooling groups of four. Some original docstrings round this incorrectly.

The checkpoint loaded with `weights_only=True`; exported layout is 435 numeric,
85 boolean and 111 categorical/stepped targets, totaling 631 output values.
The original weights remain in Downloads; they were not copied into Git.

Generated bundle: `models/joint-vae-j48/bundle.json`, id
`j48-278dbe441234e620`, with both ONNX models, matching metadata/PCA, reference
vectors and example import. This directory is ignored by Git, so another
checkout needs the original assets or a copy of the exported bundle.

The standalone built successfully at:
`build/VAESynth_artefacts/Release/Standalone/VAE Synth.app`.
This build uses `VAE_WITH_SURGE=OFF`. AU/VST3 were not rebuilt or installed for
this milestone. The new tab does not auto-load a model: use its bundle picker.

### Code map

| File | Responsibility |
|---|---|
| `tools/joint_vae/export_bundle.py` | Reconstructs the student's layout from schema/pruning, loads weights, exports deterministic decoder and sound-only posterior mean, verifies Python/ONNX parity. No Surge installation needed. |
| `source/experiments/joint_vae/JointVae.h/.cpp` | Validated bundle loading; native encoder/decoder; local-slice math; single-worker execution with latest-request cancellation and message-thread callbacks. |
| `source/experiments/joint_vae/JointVaePanel.h/.cpp` | Bundle/import/export pickers, category and axis selectors, random anchors, pad, parameter readout. Owns its worker; session is editor-local. |
| `source/plugin/PluginEditor.h/.cpp` | Minimal integration: three-view enum, VAE Lab button, panel visibility/layout. Existing processor/audio code was not changed. |
| `CMakeLists.txt` | Separate `vae_joint_experiment` interface target linked into the plug-in. |
| `tests/ml/JointVaeTests.cpp` | Slice correctness, rejection, cancellation, optional real-bundle native parity and failed replacement behavior. |
| `tests/plugin/PluginTests.cpp` | Tab visibility/layout and unchanged synth parameters when switching views; optional screenshot export. |

The decoder export makes categorical decisions inside ONNX (argmax), thresholds
boolean logits at zero, and rescales stepped categories back to normalized codes.
It does not export the auxiliary sound decoder. The sound encoder includes the
prior expert: `precision * mu / (1 + precision)`, with log variance clamped to
[-8,8]. It exports only the posterior mean, not uncertainty or sampling. Adding
candidate generation from the posterior would require an additional variance
output/export version.

### Verification actually performed

- Export: 36 deterministic cases across all nine category slots; maximum
  Python/ONNX absolute discrepancy `1.43e-6`. Sound wrapper also compared against
  the original `JointVAE.posterior` implementation.
- Native tests: all 36 reference latent/code and sound/latent cases checked with
  absolute tolerance `2e-5` using the exported bundle.
- Final `vae_ml_tests` and `vae_plugin_tests`: both passed with
  `VAE_JOINT_BUNDLE` set. Without that environment variable, the private real-model
  parity section is skipped (basic tests still run).
- Standalone target built; editor screenshot inspected. Screenshot is available
  at `/tmp/vae-lab.png`; test export is enabled by `VAE_LAB_SCREENSHOT`.
- Changed C++ files passed clang-format and `git diff --check`.
- Sandbox-only plug-in tests initially failed because JUCE creates fixtures in
  the macOS user cache. They passed when rerun with filesystem access to that
  temporary cache location. Do not interpret missing-fixture errors as a model
  regression.
- No actual reference-audio preprocessing or decoded audio playback was verified.

### Next implementation work, in dependency order

1. **Exact audio-to-sound-code frontend.** Reference
   `vae/pitcher/synth_data/stage1.py` (`render_mel`, `mel_view`),
   `vae/pitcher/core/audio_metrics.py`, and `joint_vae/sound_code.py`.
   Match the training view: 48 kHz, six seconds padded/truncated, loudest-window
   normalization, the student's mel filterbank/STFT (2048/512, 64 bands), dB floor,
   float16 stored-mel quantization, four-frame pooling, peak-relative floor, then
   saved PCA and global scale. Compare intermediate arrays against Python;
   generic library defaults and SAME features are not interchangeable. Determine
   how generated clip pitch/onset handling should align with the training note.
2. ✅ **Full generated-patch representation and writer** (`StudentCodec`,
   `SurgePatchBuilder`; plans match the Python codec exactly on all 36 parity
   cases). Original notes: Follow
   `joint_vae/data.py:JSpec.to_row` → `inverter/data.py:full_row` →
   `core/codec.py:decode` → `core/surge_io.py:write_plan`. Restore frozen values,
   honor inactive/type-dependent blocks, oscillator/filter choices, flags,
   wavetable fallback, FX slot allocation and routing-depth conversion. Do not
   truncate these codes into the existing 93-control schema and call it faithful.
   Inspect the pruning template reference and obtain/locate the actual template
   and wavetable resources; those assets were not validated in this milestone.
3. ✅ **Surge playback and handover**, via Surge's own patch queue rather than a
   lock-free swap (see "Not connected yet"). Still open: compare renders with the
   student's writer (needs their patched surgepy). Original notes: The existing Surge backend still
   targets the smaller schema/fixed FX chain. Extend it or introduce an isolated
   full-patch backend. Its current header notes allocation on some oscillator,
   FX and routing changes: do not apply arbitrary patch construction in the audio
   callback. Prepare resources off-thread; design and test a lock-free handover
   and off-thread retirement, then compare renders with the student's writer.
   Preserve complete patch state, not only visible UI controls.
4. **Text pipeline wiring.** Reuse `StableAudioGenerator` on a worker, passing its
   audio into the new frontend/encoder. `SameSearchWorker::generate` currently
   also performs retrieval; avoid changing that behavior for the existing Prompt
   tab. Use a separate controller for this experiment. Keep category selection
   explicit until a category inference strategy is tested.
5. **Persist and compare.** Move experiment state out of the editor into an
   appropriate processor/controller-owned state before adding DAW recall. Persist
   bundle identity, anchor/current latent, category, axes/radius and full patch.
   Add reference/decoded audition and explicit comparison with retrieval; ensure
   the user always knows which path and sound they are hearing.
6. **Latent UX.** Current axes are selectable raw dimensions. A PCA basis or
   measured perceptual directions can be added later without discarding the
   anchor. A global 2D projection needs an explicit lifting/inverse strategy;
   never assume a 2D coordinate uniquely determines all 32 dimensions. Discrete
   decoded choices can jump even along a smooth latent path.

The student's saved results are an important quality constraint: see
`vae/IMPLEMENTATION_LOG.md` rounds 36–39 and
`vae/runs/joint_vae/j48/{eval,roundtrip}/summary.json`. Native parity establishes
faithful inference, not good sound matching. Do not promise that successful
integration fixes the decoder's measured reconstruction weaknesses.

### Working-tree cautions

This work was not committed. Check `git status` before continuing. At handoff,
`docs/HACKATHON-ONNX-SAME-LEARNINGS.md` was modified and
`docs/MODEL-LOADING-PERFORMANCE.md` was untracked by other ongoing work; neither
was authored or changed by this VAE task. Preserve them. Model bundles are local
ignored artifacts; source, tests, CMake changes and this guide need normal review.
