# VAE Lab follow-up: p0 prompt-to-preset integration

*21 October 2026. Proposed implementation specification; queued after the current VAE Lab v2 work. No integration is implemented by this document.*

## Outcome and sequencing

The user describes a sound in the native Lab and receives several real presets selected by the student's learned text-to-latent model. Selecting a result moves the pad marker to that preset's latent position and plays its stored parameters through the existing Lab audio paths.

Implement after [VAE Lab v2](2026-10-21-vae-lab-xypad-design.md) and its [implementation plan](../plans/2026-10-21-vae-lab-xypad.md) are complete and passing their required checks. Rebase implementation decisions on the completed controller and worker interfaces, rather than modifying the current work in parallel. This adds the text-to-z follow-up; it does not silently include the separately deferred anchor/residual decoder work.

Preserve the current Lab decisions: Lite for interactive preview, explicit **Send to Surge** for the full patch, and no changes to the separate Prompt/retrieval tab. This spec targets the native plug-in, not the browser design study.

## Evidence and supplied assets

The supplied folder is `/Users/thomaspower/Downloads/p0`. Treat that as an import source, never a runtime dependency.

| Asset | Observed contract |
|---|---|
| `text_encoder.onnx` | `input_ids`, `attention_mask`: int64 `[batch, tokens]`; output `text`, fed to a 512-dimensional prior input |
| `tokenizer/` | RoBERTa tokenizer with CLAP processor metadata; BOS 0, padding 1, EOS 2; advertised maximum length 512 |
| `prior.onnx` | Input `text`: float `[batch,512]`; outputs `category_logits`, `mix_logits`, `mu`, `logsd` |
| `layout.json` | Prior `p0`, VAE `j48w_b05`, z dimension 32, eight mixture components, nine category labels |
| `bank.npz` | 1,227 rows: `z[1227,32]`, `codes[1227,631]`, category strings and ordered feature names |
| `decoder.onnx` | Inputs `z[batch,32]`, `label[batch,9]`; outputs 435 numeric codes, 85 boolean logits and 1,057 categorical logits across 111 categorical features |

Bank coverage is Bass 485, Lead 447 and Pad 295. Other categories exist in the classifier but have no bank entries. The bank has no preset display names; use stable row-based names unless a separate verified name mapping is supplied. Do not match rows by index to the existing XY cloud: the datasets have different row counts.

The student's instructions explicitly specify nearest-bank snapping as the default and decoding new points **“after the decoder fix.”** Reported ONNX/PyTorch differences are export-parity evidence, not evidence of prompt relevance or audio quality. Training provenance, held-out quality and exact text preprocessing are not fully specified in this folder.

## User flow

1. The existing Lab prompt box accepts a sound description. A compact **Find sounds** action requests a new set of suggestions. Empty input is a no-op.
2. Display a loading status while preserving the playing patch and pad state. No per-keystroke inference.
3. On success, show up to eight distinct candidates in mixture-weight order and select the first. Use compact labels such as `Bass · 0142`; do not invent original preset names. Do not display mixture probability as a sound-match percentage.
4. Selection restores the bank row's full codes, category and z as one state change. The marker uses the bank z, not the unsnapped mixture mean. Preview uses the existing code-to-Lite translation. **Send to Surge** uses those same full codes through StudentCodec and the patch builder, bypassing VAE decoding.
5. Show a brief status: `Selected a bank preset · Lite preview`. Existing category-match readouts remain separate from the prior's requested category. Expected-sound plots remain labelled model predictions, not measurements of the playing preset.
6. If the predicted category has no bank entries, show `This bundle has no Keys presets` (or the corresponding category). Keep the current patch and do not silently substitute Bass, Lead or Pad. Offer the existing word-list starting-point path explicitly.
7. A missing or invalid p0 bundle leaves the existing Lab available. Clearly identify when the user is using the word-list path instead of learned prompt search.

### Exploration after selecting a bank preset

> **Superseded by user decision, 21 Oct 2026 (plan Task 20):** Find highlights its candidates as numbered dots on the pad. Clicking a dot or chip plays the exact bank preset. Exploring is never locked: the first move decodes from the bank z and dragging continues. Re-selecting restores the exact codes. There is no "Explore generated variations" button. The bullets below describe the original design.

The bank patch and `decode(bank.z)` are not equivalent. Re-decoding immediately would discard the benefit of selecting a real preset.

- While a bank result is selected, retain its exact codes. Camera changes, axis assignment and display-only updates must not decode or change audio.
- Knob-pad edits may operate on those codes through the current translation path. Returning to the candidate restores its original codes.
- Before a latent edit, expose an explicit **Explore generated variations** action with a short explanation that this switches to the model's approximation and may change the sound. Until activated, latent sound-edit gestures are disabled; inspection remains available.
- Activating this action decodes the selected bank z once, then uses the completed v2 live-drag behavior. Selecting a candidate again restores bank mode. No seamless or faithful interpolation claim.
- Anchor/residual decoding can replace this transition in a separate follow-up, after its behavior and validation are specified.

### Follow-up words

Preserve v2's deterministic adjective nudges without applying them twice. A **Find sounds** request sends the entire prompt through the learned model and does not subsequently apply word-list adjectives. An explicit nudge action uses the existing word interpreter only; in bank mode it requires the same generated-exploration transition. Mixed or ambiguous text defaults to Find sounds. The UI must make these two actions distinguishable without introducing a second chat interface.

## Inference and selection contract

All processing is offline after bundle installation:

`text → tokenizer → text encoder → L2-normalized embedding → prior → mixture means → category-filtered bank search → stored codes`

- Confirm special tokens, truncation, sequence length, padding, attention masks and normalization against a student reference before accepting the bundle. Metadata's 512-token maximum alone does not establish the training-time inference recipe.
- Text input tensors are int64. Reject nonfinite outputs and zero-norm embeddings before normalization.
- Pick category by argmax over logits 0–7, as instructed by the student; Unknown is not a bank-search fallback.
- Softmax the eight mixture logits. Use means whose weight is at least 0.02, highest weight first. This version is deterministic: no temperature or random sampling controls.
- For each eligible mean, choose the nearest unused bank row within the predicted category using squared Euclidean distance in the supplied 32-dimensional coordinates. Break equal weights by component index and equal distances by bank row index. Do not whiten, normalize z, or project into screen coordinates for search.
- Return up to eight distinct rows. Validate finite weights, means and bank values. `logsd` is part of the validated model contract but is unused for deterministic selection.
- Return structured candidates containing bundle identity, bank row id, category, z and full codes. Keep presentation names separate from parameter identity.

## Bundle packaging and compatibility

Add a versioned optional prompt bundle referencing the installed Lab bundle. Include model/tokenizer hashes, prior and VAE identities, labels, dimensions, preprocessing settings, bank metadata and ordered feature names. Import NPZ offline into a simple validated native format; Python is not a plug-in runtime dependency. Large model assets remain outside tracked source, following existing model-install conventions.

Validate all assets before replacing a working bundle. Require exact feature ordering and compatible codec/schema, pruning values and category ordering; matching `j48w_b05` text alone is insufficient. Compare supplied decoder output with the installed v2 decoder on fixed vectors before declaring compatibility. Do not overwrite v2's decoder, sound decoder or axes with the supplied decoder automatically.

Verify the unresolved pruning assumption from v2 before claiming full Surge fidelity. A mismatch must disable this integration with an actionable error, leaving the previous Lab state intact.

## Implementation boundaries

- Add a worker-only prompt model component responsible for loading, tokenization, inference and bank selection. Its public seam is text in, ranked candidates or a typed error out.
- Reuse ONNX runtime and the current native tokenizer infrastructure where appropriate. The current Rust tokenizer wrapper hardcodes 256 tokens, pad id 0 and float outputs for the existing model. Add a separate configurable/int64 path for CLAP; preserve existing callers and their behavior.
- Extend the completed Lab controller's start-point seam to accept stored codes and provenance as well as z/category. A `(z, category)` result alone cannot preserve a bank preset.
- Reuse StudentCodec, code-to-Lite translation and Send to Surge. Do not reduce the bank to the 93-control schema before storing/restoring the selected patch.
- Keep text inference off both the audio thread and the live decode queue, so a slow prompt cannot stall pad movement. Bound thread counts and reuse loaded sessions; measure memory and CPU contention with audio running.
- Use latest-request-wins results. New prompts, manual sound edits, candidate selection, reset, bundle changes and panel destruction invalidate obsolete results. Late completion must not overwrite newer state.
- No loading, inference, patch construction, allocation, locks or file I/O in the audio callback. Use the v2 handover path. This work does not claim to fix the separately documented Surge patch-loading glitch.
- Inference errors preserve the last playable sound and return a concise actionable status. Do not present a synthetic success or an automatic word-list fallback as model output.

## Validation and acceptance

### Reference and parity

Create an offline reference runner and fixtures covering token ids/masks, normalized embeddings, logits, mixture parameters, selected row ids and selected codes. Prefer the student's inference runner or golden outputs; a runner we write only establishes consistency with our interpretation, not proof of his preprocessing. Record this distinction.

Include empty/long input, Unicode, Bass/Lead/Pad descriptions, unsupported categories and adjective swaps. Assert token equality, finite dimensions, numerical parity and deterministic row selection. Start with a documented 1e-4 absolute ONNX/native tolerance; investigate deviations rather than loosening it without evidence. Near-tied rankings need explicit tie fixtures, not floating-point equality assumptions.

### Behavioral checks

1. A successful prompt selects the expected fixture bank row, plays its codes through Lite and places its marker at the matching z.
2. Selecting an alternative restores that row's codes exactly. Send to Surge receives the same codes through the existing codec; no decoder call intervenes.
3. Display/axis changes preserve a bank patch. Generated exploration requires the explicit transition. Returning to a candidate restores its exact stored codes.
4. No-bank categories, corrupt assets, feature-order mismatch and inference failures preserve the previous sound and explain the problem.
5. Rapid prompts, prompt-then-drag, bundle changes and editor close cannot apply stale results or access deleted UI.
6. The CLAP tokenizer path does not change existing Prompt-tab tokenization or inference. Existing v2 and engine tests remain green.
7. Profiling with held notes confirms text inference runs off the audio thread and does not starve live controls. Record cold/warm latency, peak memory and audio underruns on the target Mac; do not invent a latency promise before measurement.

### Musical evaluation

Before declaring prompt search an improvement, compare it with category-only and existing word-list starting points on a fixed prompt set. Include contrasting pairs such as dark/bright bass, short/long pad and soft/sharp lead. Listen through Surge as well as Lite, record coverage and failures, and keep the report alongside the integration docs. These are quality measurements, separate from numerical parity. If results do not support semantic claims, ship only as an explicitly experimental option.

## Delivery order

1. Finish and verify Lab v2; obtain/confirm the text preprocessing reference and compatible codec assets.
2. Package/validate p0 and produce reference fixtures. Resolve unsupported-category behavior before UI wiring.
3. Implement isolated prompt inference and deterministic bank selection with parity tests.
4. Wire candidates, exact bank restoration, explicit exploration transition and stale-result handling into the completed Lab.
5. Run integration/regression checks, profile under audio load, perform the musical comparison and update installation/user documentation.

## Out of scope

Training or fine-tuning, favourites-driven personalization, a decoder fix, anchor/residual interpolation, generated-patch suggestions from mixture samples, additional bank categories, an LLM, browser deployment, host-project Lab persistence and replacement of the separate Prompt tab. Existing v2 sound-axis nudges remain available; p0 is not assumed to learn independent musical axes.
