# VAE Lab v2: the student's XY pad in the plug-in

*21 Oct 2026. Replaces the current VAE Lab screen (tagged `vae-lab-v1`, commit `8360749`).*

## Intent

The user describes a sound in words, the plug-in plays a preset for it at once, and an XY pad shows where
that preset sits in the joint VAE's latent space, so the user can tweak it to taste by dragging or by typing
follow-ups ("brighter", "longer").

The standard is the student's web demo, `pitcher/xypad/page_v2.template.html`, built into
`runs/xypad/xypad_v2.html` (source: `/Users/thomaspower/Downloads/new_xy/`). It felt right because the
sound follows the dot **instantly and smoothly** while dragging. The plug-in must match that feel. It does
not have to reproduce the exact Surge preset.

## Goals

1. Prompt box: text gives a starting point and a playing preset. Follow-ups nudge the dot along the named
   sound axes.
2. Latent XY pad with everything the demo has: named axes (sound axes plus latent dimensions), real presets
   as dots (click one to jump to it), a coloured background map, the expected-sound spectrogram and
   measures, Lock structure, an optional Z axis, and the knob pad.
3. Audio follows the dot live while dragging, on the **Lite** engine, with no clicks.
4. **Send to Surge**: build the full Surge patch for the current point (the existing path) for anyone who
   wants the faithful preset.

## Non-goals (this version)

- Starting from a real preset with anchor + residual decoding. This is the biggest known quality lever
  (decoded "typical" points sound generic: decoder round trip keeps character only 1–4%). It is deferred
  by the user's decision and is the first follow-up.
- A language model for the prompt. Word lists only. The student is training a text → z encoder that
  will replace the first-prompt step later.
- Text → Stable Audio → sound encoder → z.
- Saving Lab state in the host project.
- The Web Audio preview synth itself: Lite replaces it.

## Anti-goal

Sounds that aren't "in the ballpark" of the words. Mostly a model limit, but two levers are ours: the
decoded-values → Lite translation must not make things worse than necessary, and the starting points
(deferred above).

## Decisions made with the user

- **Lite follows the dot.** It doesn't have to match the Surge preset exactly. Surge stays available via
  Send to Surge.
- **Model: `j48w_b05`**, checkpoint `/Users/thomaspower/Downloads/model.pt` (11.5 MB, sha1 `61f0a8d6…`;
  config at `vae/runs/joint_vae/j48w_b05/config.json`: z 32, hidden 512, β 0.05, same 435/85/111 layout
  as j48). The axes table belongs to this model, so the Lab bundle moves to it.
  - **Checked:** the PCA embedded in the student's page equals `~/Downloads/missing_files/sound_pca.npz`
    (components within float16 rounding, same scale 213.48), and both configs have identical
    `cat_sizes`.
  - `pruning.json` (the frozen values): the user believes it is the same for `j48w_b05`, so we use the
    j48 file. It is unconfirmed by the student, and a wrong one would give wrong patches silently.
- **The new screen replaces the current Lab** completely: raw-dim slice, radius, 631-value list,
  import/export buttons.
- **Layout: two columns** above the existing keyboard, in the 1240 px editor:
  - **Left (~2/3 width):** the XY pad, taking most of the height. Under it are the X label, the Z slider
    (when a Z axis is set) and the **prompt / chat box**. Then one short strip with the expected-sound
    spectrogram, the measures (brightness, attack, length, noise), the decoded preset (osc types,
    effects, cutoff, amp ADSR) and the category guess.
  - **Right (~1/3):** the axis list. Sound axes first, then latent dimensions ranked by dB impact, with
    those under 2 dB folded away. Each row has X/Y/Z buttons, its own slider, a description and section
    tags. In knob-pad mode it lists the knob groups instead.
  - **Toolbar:** Latent pad / Knob pad · Start from (Typical ⟨label⟩, Random real preset) · Category ·
    Lock structure · Reset · Background map (Category guess, Brightness, Length, Attack, Noise, None) ·
    Preset dots on/off · Send to Surge.
  - Mockup: `.superpowers/brainstorm/12456-1792589358/content/lab-layout-v3.html` (prompt box not drawn;
    it goes under the pad, above the readout strip).
- **Prompt box = word lists.**
  - The first message picks the label from the student's nouns (`vocabulary_v2.yaml`; `labelFor`
    already exists) and starts from Typical ⟨label⟩. Adjectives in the same message nudge the axes.
  - Follow-ups nudge along the five sound axes by adjective: dark/bright, short/long, snappy/slow attack,
    tonal/noisy, static/moving, with "more/less/a lot/slightly" as step sizes.
  - Every message gets a one-line reply saying what it did and which words it ignored, e.g.
    "Bass · Dark ↔ Bright −1 step · didn't understand 'gritty'". It never silently guesses.
  - Keep the prompt → start-point step behind one function so the text → z encoder can replace it.
    Its interface: text in, (z, category) out, or "no start found".
  - *My call:* a new noun in a later message ("now make it a pad") restarts from Typical ⟨that label⟩,
    then applies that message's adjectives. A first message with no noun ("something dark") starts from
    the mean z of all 1,375 presets with category Unknown, and the reply says no sound type was found.
  - *Known weakness:* in the student's data, "noisy" and "moving" barely change the sound (+0.024 and
    +0.028 over two sd; movement R² 0.3). Those words are still accepted, but their replies say the
    effect is small. Brightness (1.35 oct), length (0.51 s) and attack (0.21 s) are the axes that
    carry.

## Behaviour to copy from the demo (the reference for any doubt)

| Piece | Demo source (page_v2 / latent.py) |
|---|---|
| Axis value ↔ z | `tOf`, `posOf`, `RANGE = 5` (pad spans ±2.5 sd) |
| Pad move → z | `constrain()`: smallest change of z so the 2–3 assigned axes read their targets; other directions untouched |
| Axis slider in the list | moves along that axis, keeps the pad's axes fixed |
| Lock structure | freezes activity, on/off switches and choices at lock time; only continuous values follow |
| Click a preset dot | jumps z and category to that real preset (a tap of < 3 px) |
| Background map | 24×24 grid, recomputed ~150 ms after movement stops; category map via the forest, measure maps via the sound decoder |
| Expected sound | sound decoder → 48 codes → PCA → peak-relative mel floored at −80 dB → `measures()` |
| Knob pad | groups of continuous knobs moved together with signs; switching keeps the decoded preset |
| Arrow keys on the pad | ±0.02, Shift ±0.1 |

Axis names, descriptions, "minor" folding (< 2 dB), effect wording (`fmtEff`, `EFF_MIN`) and
parameter naming (`pname`) follow the demo.

## Data the plug-in needs

All of it is per model. It ships next to the ONNX files in the bundle, so it can't drift from the decoder.

- From `latent.json` (embedded in `xypad_v2.html` as `#latent`, or regenerated by `latent.py`): sound
  axis directions (u[32], mean, sd, r², effect), latent dims (impact, sections, top features, mean, sd),
  per-label mean z, the 1,375 human presets' z, labels and names, band frequencies, dt, f0, hold frames.
  **Drop** its float16 decoder weights and the activity stand-in: the plug-in uses the real ONNX decoder
  and the codec's own activity.
- From `knobs.json` (embedded as `#data`): the 123 knobs (ranges, p5/p95, ctrltype), the 61 groups, the
  per-preset knob values, and the category forest (used by the category guess and the category map).
- New ONNX export: the **sound decoder** (z + category → 48 codes). `export_bundle.py` skips it today. The
  PCA is already in the bundle.

The embedded data in the shipped HTML was built for `j48w_b05`. It can be extracted directly, so no
student re-run is needed.

## Audio path

```
pad / prompt / slider / dot  ->  z (+ category)
   -> decoder.onnx (worker thread, latest request wins)
   -> StudentCodec plan (existing)                         [Lock structure: discrete part from the lock]
   -> plan -> 93 Lite controls  (NEW translation)
   -> Lite, smoothed, live while dragging
Send to Surge: plan -> SurgePatchBuilder -> Surge (existing path, Surge builds only)
```

- **Translation** (new): map the codec's Surge-valued plan onto the 93 schema controls by the same rules
  `tools/surge_preset_to_vector.py` uses for real Surge presets (docs/SCHEMA.md). Those rules are the
  measured reference: schema loss is a median 11.5 dB on factory presets, which is the known ceiling.
- **The Lab requires a `VAE_WITH_SURGE=ON` build (user decision).** The translation may use the headless
  Surge (`SurgePatchBuilder`) to get each parameter's real value and range for the current osc or FX type
  and the `Init Saw` template's values. No offline table of Surge data is needed. A non-Surge build shows
  the Lab as unavailable.
- **Lite gets smoothed parameters (user decision).** Today `LiteEngine::setControl` applies values at
  once; only the output gain ramps. Add per-control smoothing for continuous controls and a short
  crossfade for discrete changes (osc wave, filter type, FX on/off, octave, play mode), so drags never
  click or zipper. This also changes Synth-tab knob moves, which is intended. Steady-state sound must not
  change: re-run the Lite-vs-Surge comparison (docs/COMPARE.md) once to confirm.
- Decoding is under 1 ms, so the decode rate is limited only by mouse events. Check the translation's
  per-move cost (headless Surge write and read-back) early. If it can't keep up, decode at a capped rate.
- The Lab selects Lite when it plays. Send to Surge selects Surge with the full patch. The Synth tab's
  knobs show the Lite controls the Lab wrote, so fine-tuning there also works.

## Constraints

- No inference, model loading or UI logic on the audio thread. The existing `JointVae` worker (single
  worker, latest request wins, message-thread callbacks) is the pattern.
- Existing Synth and Prompt/retrieval tabs keep working as before. The only change is Lite's new
  smoothing.
- Bundle loading stays validated (bundle id fingerprint). A bad bundle keeps the previous one.
- Native parity with the Python/JS reference: decoder (as now), sound decoder, `constrain`, `measures`,
  forest predictions, checked against values from the demo page (`window.__xypad` exposes `decodeP`,
  `decodeA`, `measures`, `AX`).

## Left to the builder

- C++ structure, component split, and how state is owned between panel and controller.
- How live control writes avoid flooding host automation (the existing `applyControls` wraps every write
  in a gesture).
- Map-recompute scheduling and cancellation; grid size may shrink if the decoder pass is too slow.
- Colours and type follow the plug-in's existing look, with the demo's category colours.
- Exact adjective list and step sizes, within the five axes.
- Whether the knob pad edits the student's 123 knobs (then translated to Lite) or Lite controls
  directly. The demo edits the student's knobs.

## Later

1. Start from a real preset + anchor/residual decoding (the main quality lever).
2. The student's text → z encoder replaces the word-list start point.
3. Measure Lite vs Surge renders along each sound axis: does "brighter" actually get brighter?
4. Persist Lab state in the project. Off-thread Surge handover for Send to Surge.
