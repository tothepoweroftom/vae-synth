# Joint VAE → full Surge patches: results and next steps

*21 Oct 2026. Covers the work after [JOINT-VAE-LAB.md](JOINT-VAE-LAB.md): decoded points from the student's
joint VAE (`j48`) now play as complete Surge XT patches in the plug-in.*

## What works now

```
VAE Lab tab: latent z[32] + category (one of 8 labels, or Unknown)
   │  decoder.onnx
   ▼
631 decoded values
   │  StudentCodec       C++ copy of the student's full_row + codec.decode
   ▼
list of Surge settings   osc types, wavetables, FX slots, filter models, flags, routings
   │  SurgePatchBuilder  the student's SurgeIO.write_plan, on a private headless Surge with "Init Saw"
   ▼
Surge patch data         wavetables embedded
   │  PluginProcessor::loadSurgePatch → SurgeEngine::loadPatchData
   ▼
plays on the Surge engine (selected automatically)
```

- Nothing from the student's feature set is dropped: 12 oscillator types, 32 named wavetables, 16 FX slots
  (insert / send / global), filter models, per-parameter flags (temposync, extend, …) and 95 modulation routings.
  Converting to our 93-control schema instead would have cost a median 11.5 dB spectral difference, above
  the student's 6.7 dB "same character" threshold.
- Loading a retrieval preset or generating from the Prompt tab returns the Surge engine to the 93 controls.
- Needs a `VAE_WITH_SURGE=ON` build (GPL-3.0-or-later). Builds without Surge still decode and display.

### Measured

| Check | Result |
|---|---|
| Native ONNX vs PyTorch (36 cases, all 9 categories) | max error 1.4e-6 (from the earlier export) |
| C++ codec vs the student's Python codec (36 cases) | identical plans: every type, subtype, flag, value, wavetable and routing |
| Plan → Surge, read back | every type, subtype and choice lands as planned; the same plan gives identical patch bytes |
| Decoded patches played on a Surge **without** factory data (as in the plug-in) | 6 of 6 make sound at C3: −13 to −22 dBFS RMS |
| Engine patch mode | survives a re-prepare; ignores the 93 controls while active; returns to them cleanly |
| Test suites | Surge build 8/8, default build 7/7 (`vae_ml: 43,720 checks`) |

Not measured: whether our renders sound the same as the student's renders of the same points, and how good
the decoded sounds are. The student's own results (IMPLEMENTATION_LOG Rounds 37–39) say a preset round
trip through the decoder keeps its character only 1–5% of the time, so many points will sound generic or
odd. That is a limit of the decoder, not of this playback path.

### Build fixes found on the way

The Surge backend had never been built on macOS:
- `enable_language(OBJCXX)`: surge-common has Objective-C++ sources.
- `-Wl,-no_compact_unwind` on Apple: C++, Objective-C, LuaJIT (Surge) and Rust (tokenizer) are four exception
  personalities; Apple's compact unwind fits three, so the app did not link.

## Files

| File | What |
|---|---|
| `source/experiments/joint_vae/StudentCodec.{h,cpp}` | Decoded values → Surge settings (port of `vae/pitcher/core/codec.py`) |
| `source/experiments/joint_vae/SurgePatchBuilder.{h,cpp}` | Settings → Surge patch data (port of `SurgeIO.write_plan`) |
| `source/engine/surge/SurgeEngine.{h,cpp}` | `loadPatchData`, `useSchemaControls`, `hasPatchData` |
| `source/plugin/PluginProcessor.{h,cpp}` | `loadSurgePatch`; `applyControls` leaves patch mode |
| `source/experiments/joint_vae/JointVae*.{h,cpp}` | Each decode also builds a patch; the panel plays it |
| `tools/joint_vae/codec_parity.py` | Runs the student's Python codec on the parity vectors → `codec-parity.json` |
| `cmake/VaeSurge.cmake` | macOS build fixes; `VAE_SURGE_DATA_DIR` |
| `tests/ml/JointVaeTests.cpp`, `tests/engine/SurgeEngineTests.cpp` | The checks above |

## Run it

```sh
python3 tools/joint_vae/codec_parity.py --bundle models/joint-vae-j48     # once per bundle
cmake -S . -B build-surge -G Ninja -DCMAKE_BUILD_TYPE=Release -DVAE_WITH_SURGE=ON
cmake --build build-surge
VAE_JOINT_BUNDLE="$PWD/models/joint-vae-j48/bundle.json" ctest --test-dir build-surge
scripts/install-models.sh      # copies the bundle + Surge template and 32 wavetables (760 KB) to app data
open "build-surge/VAESynth_artefacts/Release/Standalone/VAE Synth.app"
```

The Lab loads `~/Library/Application Support/VAE Synth/models/joint-vae/bundle.json` when the window opens,
like the Prompt tab's models (**Load VAE bundle…** still overrides it). Surge's template and wavetables come
from `models/surge-data/` there, or the build tree if not installed.

In the app: type a prompt in the Lab ("dark reese bass") and press **Sound from label**, then play. **Random
anchor**, the category menu, the radius and a pad release each decode a new patch.

## Known limits

- **Click on decode.** Surge loads the patch on the audio thread (its own patch queue, as Surge XT does), which
  parses and allocates: expect a possible one-block glitch on each pad release. Not while dragging.
- **Spurious routings.** The decoder outputs all 95 routings with near-zero depths, so a patch carries ~78 weak
  routings (true presets have ~3). This matches the student's code; a snap-to-zero below a threshold is a
  one-line change once the student confirms the threshold they used.
- **Synth tab knobs** don't reach Surge while a VAE patch plays.
- **Lab state** isn't saved with the plug-in's project.

## Next steps

### 1. Prompt → z (the student's proposal: prompt → CLAP → z)

**Makes sense, and it is the plan in `vae/TEXT2PRESET_PLAN.md` §7.** A small model maps the CLAP text embedding of
a prompt to a point (or a distribution) in the joint VAE's z. In the plug-in this needs no Stable Audio at run
time: text → CLAP text encoder → prior → z → decoder → Surge patch.

Points to agree with the student:

- **Training pairs come cheaply from the joint VAE.** The 46,220 kept Stable Audio clips each have a prompt. Their
  sound codes (mel → PCA 48) go through the joint VAE's sound encoder to give z directly, without the slow clip →
  preset matching. Caveat: Stable Audio clips sit apart from Surge renders (CLAP cosine 0.44 between the means),
  so z encoded from clips may land off the area the decoder knows. z from matched presets is cleaner but there
  are far fewer (653 kept pilot clips; the main run isn't matched).
- **Predict a distribution, not one point.** One prompt has many valid sounds (4 seeds per prompt differ). A plain
  regression averages them into a generic sound. The plan already says a mixture density network (~8 components)
  or a small flow; the XY pad can then explore around a sample.
- **Category.** The decoder also needs one of the 9 categories. The prior should output category probabilities
  too (or we take them from the prompt's noun, below).
- **CLAP text tower in the plug-in.** A second text model (`laion/clap-htsat-unfused`, ~125 M parameters) next to
  Stable Audio's T5Gemma. It must be exported to ONNX with its tokenizer. Alternative to discuss: train the prior
  on the T5Gemma embedding we already run natively, which avoids a second text model.
- **Baselines it must beat:** label-only sampling (step 2 below) and retrieval (nearest training prompt → its z).
  The student's Round 18 found CLAP's zero-shot text matching weak (20% on categories); a trained head can still
  work, but it needs measuring on held-out prompts.

**Ask the student for** (contract for the plug-in):
1. `prior.onnx`: input the text embedding `[1, D]`; output mixture weights, means `[K, 32]` and scales
   `[K, 32]` (or samples), plus category probabilities `[9]`.
2. Which text encoder and checkpoint, and its exact preprocessing (tokenizer, max length, normalisation), so we can
   export it and check parity like the decoder.
3. Held-out results: do word swaps ("dark" → "bright") move the rendered sound the right way, and does it beat
   label-only sampling.

**Fallback that needs nothing new from the student:** prompt → native Stable Audio (already in the plug-in) →
the student's mel + PCA 48 in C++ → `sound_encoder.onnx` → z. Slower (~6 s per prompt) but uses only what we
have. The mel must match the training view exactly (see JOINT-VAE-LAB.md, next work item 1).

### 2. Label demo with the 8 labels (built)

The 8 labels are **Bass, Lead, Pad, Keys, Pluck, Synth, Atmosphere, Orchestral** (+ Unknown), the student's
Surge folder labels; the decoder is conditioned on them, and the Lab already has the menu.

Built (the Lab's prompt box, **Sound from label**), using only what exists:
- **Prompt → label** by the student's own vocabulary: `vae/pitcher/text2preset/vocabulary_v2.yaml` lists the
  nouns per category ("reese", "808", "supersaw", "e-piano", …). The first noun found picks the label; none →
  Unknown.
- **Label → sound**: decode z = 0 (the label's typical sound); **Random anchor** gives variations within the
  label and the XY pad explores around the current anchor. Longest noun wins ("synth strings" → Orchestral, not
  Synth); whole words only ("subtle" is not "sub").
- **Better anchors (ask the student):** per-label latent statistics for `j48` (mean and covariance, or a small
  per-label mixture, of z for the train presets). The existing `latent_gmm.pkl` belongs to the old VAE, not `j48`.
  With those, "Random" samples from the label's own region instead of N(0, I).
- Honest framing: this picks a category, not adjectives. "Dark" and "bright" bass get the same distribution until
  the prompt → z model exists.

### 3. Other open items, in order

1. Compare our renders with the student's for a few decoded points (needs their patched surgepy build or renders).
2. Meaningful XY axes: directions from the student's z-probe (brightness, decay, length), scaled by the encoder's
   spread per dimension.
3. Anchor + residual decoding: play `anchor + (decode(z) − decode(z_anchor))` on continuous knobs only, so the
   pad centre is exactly the anchor preset and the decoder's errors cancel.
4. Persist Lab state in the plug-in project.
5. Off-thread patch handover, to remove the click on decode.
