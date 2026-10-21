# Joint VAE Lab (v2)

The **VAE Lab** tab turns a few words into a playing preset and lets you move it around the student's joint VAE
(`j48w_b05`) in a 3D orbit view of the preset cloud. The sound follows the handle live on the **Lite** engine. **Send to Surge** builds the full
Surge patch for the current point. The design is in `docs/superpowers/specs/2026-10-21-vae-lab-xypad-design.md`.

The old Lab (raw latent slice, radius, 631-value list, import/export) is tagged **`vae-lab-v1`**. Its Surge playback
path, which v2 still uses for Send to Surge, is described in [JOINT-VAE-SURGE-RESULTS.md](JOINT-VAE-SURGE-RESULTS.md).
The Lab needs a `VAE_WITH_SURGE=ON` build; the default build shows "VAE Lab needs a Surge build".

## Layout (Task 17)

The Lab is laid out as the brutalist design study (`docs/design/PITCHER-BRUTALIST-JUCE.md`, reference image
`docs/design/pitcher-native-study.png`), in three columns on one continuous `#252525` workspace that runs up under
the editor's Synth / VAE Lab header. The two sidebars have equal widths and heights; the centre column is flexible.

- **Left:** the pitcher mark, **CATEGORY MATCH** (the knob forest's guess at the current sound, "Closest: <label> ·
  <n>%" and eight rows with 3 px bars), then the controls: Latent pad / Knob pad, Start from, Category, Lock structure,
  Reset and Send to Surge. **Expected sound** (the spectrogram, 0 s / 2 s / 6 s) sits at the bottom; hover it for the
  details: brightness, attack, length and noise, then the decoded preset.
- **Centre:** the orbit view (or, on the knob pad, the flat pad with its category map), the Replace banner and Find's
  chips when there are any, the single-line prompt with its submit button, a line with Find sounds / Word list, the
  "experimental" tag and the status, and the 3-octave keyboard (C3-C6, MIDI 48-84), which replaces the editor's
  keyboard while the Lab is shown.
- **Right:** the axis list under one **AXES** tab (**KNOB GROUPS** on the knob pad): a flat list, sound axes first,
  then the latent dimensions in bundle order, each row with X / Y / Z buttons and, when assigned, its low-high slider.
  It scrolls within its height. There is no Presets tab and no favourites.

Clicking a CATEGORY MATCH row highlights that category's presets in the cloud (and dims the rest); clicking it again
clears it. It is visual only: the decoder's category and the sound stay.

The reusable controls come from the study's `pitcher_study_ui` target (`design-study/juce/Source/ui`):
`PitcherLookAndFeel`, `PitcherKeyboard`, `AxisRow`, `OrbitView` / `OrbitMath`, `Spectrum` and `PitcherLayout`. The
Lab's worker, forest, prompt and data are its own; none of the study's data or approximations are used. The
colours are defined once, in `source/experiments/joint_vae/LabTheme.h`.

### Orbit gestures

| Gesture | Result |
|---|---|
| Drag empty space | Orbits the camera. Sound and handle stay. |
| Drag the handle | Moves it in the camera-facing plane; Shift-drag moves it in depth. |
| Click the handle, then drag an X / Y / Z arrow | Moves along that axis only. |
| Wheel, - / + | Zoom. Reset view (or Home) resets the camera only. |
| Click a numbered dot | Selects that Find match, as its chip does. |
| Click a preset dot of the current category | Jumps to that real preset. |
| Arrow keys / Page Up, Page Down | Move the handle in X and Y / in Z: 0.02 per press, Shift 0.1. |

Camera gestures never decode or play. A handle move sets the three assigned axes to the handle's X, Y, Z, then
constrains z to them and decodes on the worker, like the old pad; the handle then sits at those axes' positions of
the new z. Orbit always has a Z axis: on load it is the first latent dimension, and Z's button cannot clear it (on
the knob pad Z again still clears Z).

## Two views, one sound

The editor has two views, **Synth** and **VAE Lab**, and they show the same sound. The Lab plays by writing the
schema controls, so the Synth view's knobs show the Lab's preset and you can tweak it there.

- **Synth edits carry back.** A change to any control that does not come from the Lab (a Synth-view knob, Init, host
  automation) marks the Lab's sound edited, and its status line reads "Edited in Synth view". The next Lab edit
  (handle, slider, prompt, base, category, lock or preset dot) first asks "Replace your Synth edits?" in a banner
  above the Lab's prompt box, and the status line says a question is waiting until you answer, whatever else reports (R21: no separate dialog window, which some
  hosts open behind the plug-in). Other Lab edits are ignored until you answer. **Replace** applies the Lab edit;
  **Keep editing** leaves the sound and the Lab as they were.
- **The gate survives closing the editor.** The processor keeps the Lab's last write for the session, so a reopened
  editor starts "edited" when the sound changed while it was closed (or, before any Lab write, when it is not the
  default sound). A Lab decode that a Synth or host edit overtakes is dropped, so it never overwrites that edit.
- **After Send to Surge** the Synth view shows "Playing the full Surge patch from the Lab. Knobs return to Lite", also
  after a reopen (the state is read from the Surge engine). The first knob move goes back to Lite with the knobs' values.
- **The Prompt tab is compiled out** (Task 16). Its code stays behind `VAE_PROMPT_TAB`, default 0; build with
  `-DVAE_PROMPT_TAB=1` to bring it back. The Stable Audio / SAME code in the processor still builds.

## Workflow: prompt, preset, pad, tweak

1. **Prompt.** Type a sound ("deep sub bass") in the box under the orbit view. Two actions share the box, chosen by
   the Find sounds / Word list switch under it; Return runs the chosen one:
   - **Find sounds** (experimental; the default with the prompt bundle installed) sends the whole text to the student's text
     model and lists up to 8 real presets from its bank, like `Bass · 0142`. The first is selected. No word-list
     adjectives are applied on top.
   - **Word list** (the only action without a prompt bundle) picks a label from the student's nouns and starts from
     Typical <label>; its adjectives nudge the axes. The reply says what it did and which words it ignored.

   **Matches in the cloud.** Find marks its presets in the orbit view as numbered dots (a green ring, 1-8, above the
   preset dots) at their z on the current axes; they follow axis changes and stay until the next Find, Reset, a base change
   or a bundle change. Clicking a numbered dot or its chip selects that preset: it plays its exact stored settings
   (Lite preview; Send to Surge builds the same codes, no decoder) and the handle moves to it. Exploring is never
   locked (there is no Explore button): the first handle, axis-slider, category or word-list move decodes from that preset's z ("Exploring from
   <name>"; the model's approximation, so the sound may change), dragging continues live, and Send to Surge then
   sends the decoded patch. The knob pad edits a copy of the bank settings. Selecting the preset again restores them
   exactly. A new prompt, any Lab edit, Reset or a bundle change drops a prompt result that has not arrived yet, as
   does an open "Replace your Synth edits?" question; the status then reads "Find cancelled".
2. **Preset.** The decoded point plays at once on Lite, once you have acted in the Lab (see limits).
3. **Orbit.** Drag the handle. X, Y and Z are any of the five sound axes or the latent dimensions. The sound
   follows while you drag, without clicks. Drag empty space to turn the cloud.
4. **Tweak.** Type follow-ups ("brighter", "a lot longer", "slightly less noisy"), move an axis slider, click a
   preset dot to jump to that real preset, or switch to the knob pad for fine adjustments.

## Controls

All controls but the axis list are in the left sidebar, under CATEGORY MATCH.

| Control | What it does |
|---|---|
| Latent pad / Knob pad | The latent pad is the orbit view over sound axes and latent dimensions. The knob pad is a flat XY pad that moves groups of the student's knobs together, over a category map. |
| Start from | Typical <label>, or a random real preset. |
| Category | The category the decoder is conditioned on (8 labels plus Unknown). |
| Lock structure | Freezes oscillator types, on/off switches and choices at the lock time. Only continuous values follow the handle. |
| Reset | Back to the start point (the same preset for Random real preset; choosing that item again draws a new one). |
| Send to Surge | Builds the full Surge patch for this point and plays it on the Surge engine. |
| Axis list (right) | Sound axes first, then every latent dimension, flat. X/Y/Z buttons; an assigned row's slider moves that axis (on the knob pad, that group). |
| Expected sound | The spectrogram at the handle; hover it for the measures and the decoded preset. |
| Keyboard | C3-C6, playing the plug-in's keyboard state; the computer keys play C3 up. |

The preset dots are always shown (the old Preset dots switch and the latent pad's background maps are gone: Orbit is
the only view). The worker's map request remains for the knob pad's category map.

Sound axes: Dark/Bright, Snappy/Slow attack, Short/Long, Tonal/Noisy, Static/Moving. Brightness, length and attack
carry well; "noisy" and "moving" change the sound little, and the reply says so.

## Bundles

The Lab loads `~/Library/Application Support/VAE Synth/models/joint-vae/bundle.json` when the window opens, and the
prompt bundle from `.../models/prompt-p1/` (else `prompt-p0/`) if present. `scripts/install-models.sh` installs
`models/joint-vae-j48w_b05` (and `models/prompt-p1`, else `models/prompt-p0`, when it exists), plus the Surge template and the 32 wavetables
the model can pick. Running it overwrites the installed Lab bundle. Models are git-ignored.

Lab bundle contents: `bundle.json` (id fingerprints everything below), `config.json`, `schema.json`,
`pruning.json`, `decoder.onnx`, `sound_encoder.onnx`, `sound_decoder.onnx`, `sound_pca.npz`, `xypad.json` (axes,
latent dimensions, per-label means, preset z, knobs, category forest), plus reference data for the tests
(`parity.json`, `codec-parity.json`, `xypad-parity.json`, `example-sound.json`).
Prompt bundle (`models/prompt-p1`, or the older `prompt-p0`): `prompt.json`, `text_encoder.onnx` (501 MB), `prior.onnx`, `tokenizer.json`,
`bank.json` / `bank.bin` (p1: 4,429 real presets in all 8 categories; p0: 1,227, Bass/Lead/Pad only; Lab feature order), and parity files. It is installed by
`scripts/install-models.sh` to `.../models/prompt-p1/` (or `prompt-p0/`) and must match the Lab bundle's id.

The prompt model is **lazy** (R20): opening the editor only checks that the bundle is installed and shows Find (with
an "experimental" tag); the model loads on the first Find, which shows "Loading prompt model..." and then runs. A
failed load turns Find off with the reason and leaves the Word list.

### Export order

Needs numpy, onnxruntime and, for the first step, torch and onnx. The sources are the student's files (checkpoint,
`schema.json`, `pruning.json`, `sound_pca.npz`, `xypad_v2.html`, the `p0`/`p1` folder).

```sh
python3 tools/joint_vae/export_bundle.py --source <model dir> --out models/joint-vae-j48w_b05
python3 tools/joint_vae/export_xypad.py  --html <xypad_v2.html> --bundle models/joint-vae-j48w_b05
python3 tools/joint_vae/codec_parity.py  --bundle models/joint-vae-j48w_b05
# prompt model p1 (optional; p0 works the same):
python3 -I tools/joint_vae/export_prompt.py --source <p1 dir> --lab models/joint-vae-j48w_b05 --out models/prompt-p1
python3 tools/joint_vae/prompt_reference.py --bundle models/prompt-p1
python3 tools/joint_vae/codec_parity.py --bundle models/joint-vae-j48w_b05 --bank <p1 dir>/bank.npz --prompt models/prompt-p1
```

`export_xypad.py` recomputes the bundle id, so run it before `codec_parity.py`, not after.

## Tests

```sh
cmake --build build-surge
VAE_JOINT_BUNDLE="$PWD/models/joint-vae-j48w_b05/bundle.json" \
VAE_PROMPT_BUNDLE="$PWD/models/prompt-p1/prompt.json" \
  ctest --test-dir build-surge --output-on-failure
ctest --test-dir build-default        # no Surge: checks the "needs a Surge build" screen
```

- `VAE_JOINT_BUNDLE`: Lab bundle. Without it the real-model parity sections are skipped.
- `VAE_PROMPT_BUNDLE`: `prompt.json` of the prompt bundle (p1 or p0), for the prompt model parity and Lab prompt search tests.
- `VAE_CODEC_PARITY`: an extra codec plan file to compare the native codec with, e.g. the bank's `bank-codec-parity.json`.
- `VAE_LAB_SCREENSHOT=<file.png>`: `vae_plugin_tests` writes a snapshot of the editor there.
- `VAE_LAB_ASK_SCREENSHOT=<file.png>`, `VAE_LAB_KNOB_SCREENSHOT=<file.png>`: the Lab with the Replace banner, and on the
  knob pad (need `VAE_JOINT_BUNDLE`).
- `VAE_LAB_BANK_SCREENSHOT=<file.png>`: a snapshot of the Lab with the "deep sub bass" matches highlighted (needs both bundles).
- `VAE_SYNTH_SCREENSHOT=<file.png>`: a snapshot of the Synth view after a Lab pad move.

No inference, model loading or UI logic runs in the audio callback. Surge's own patch load does, once per Send to
Surge.

## Known limits

- Going back to Lite after Send to Surge cuts a held note once. Dragging never cuts notes.
- The Lab plays nothing until you act in it, so opening the editor never changes your sound.
- Before you have acted in the Lab, a first prompt with no sound noun starts from the general Unknown average.
- Lite approximates the decoded Surge patch: median 11.5 dB schema loss on factory presets. Use Send to Surge for
  the faithful sound.
- The decoder keeps a preset's character only 1-4% of the time (the student's measurement), so decoded points can
  sound generic.
- The knob pad has no preset dots and no axis captions on the pad (the axis list shows which group is on X and Y).
- In the orbit view the arrow keys move the handle (as the old pad); the study's arrow-key camera rotation is not
  available from the keyboard (drag to orbit, Home to reset).
- The Lab needs the editor's 1240 px width; `PitcherLayout` targets at least 1000 px and has no compact layout.
- `pruning.json` for `j48w_b05` is assumed identical to `j48`'s. The student has not confirmed it, and a wrong one
  would give wrong patches silently.
- The p0/p1 text-encoder preprocessing is our reading of the student's code; there are no student golden outputs.
- p0's prior only predicted Bass, Lead and Pad and its bank held only those; p1 (the installed model) has 4,429
  rows in all 8 categories. A category with no bank presets (a p0 bundle, or a bank that lacks one) comes back as "This bundle has no <label> presets"; the sound is kept and the
  Word list can still pick a start.
- **Find is experimental.** With p1 the first candidate differs for dark/bright bass and short/long pad but is the same row (892) for
  soft/sharp lead (p0 returned the same row for all three pairs); that is a selection fact, not evidence that it follows adjectives. It is
  labelled "experimental" in the UI and here until listening notes in `docs/PROMPT-EVAL.md` support semantic claims.
- Cost of p1 (R22; measured 2026-10-21 with `vae_p0_eval profile` and `mem` on `models/prompt-p1`, headless, Apple
  Silicon, Release): the model loads on the first Find in about 0.5 s (499 ms) and adds about 0.94 GB of resident
  memory per editor that has searched: 1.02 GB for one editor with p1 loaded against 75 MB without, 2.01 GB for two
  against 113 MB. A warm Find takes about 13 ms (21 ms mean, 29 ms max while a Lab drag runs) on its own thread; the
  cold Find, with the load, about 0.52 s. With a held note rendering on Lite and 10 of 20 Finds during a drag,
  `process()` averaged 0.02 ms (max 0.4 ms) against a 5.3 ms block budget: no blocks over budget. p0 measured the
  same within noise (the text encoder is identical). Sharing one prompt model across the process, so several editors
  pay for it once, is deferred.
  Closing the editor during a load or Find still waits for it (`shortcut:` in `PromptWorker`'s destructor; the fix is
  a process-wide or detached runner).
- Exploring from a match is a jump, not a faithful continuation: the decoded z can sound different from the bank
  preset.
- Lab state is not saved in the host project. The Lab's last write (the edited gate) lasts for the session only, and
  a reopened Lab starts from its first base, not where the pad was left.

## Try it

Run `scripts/install-models.sh`, then open `build-surge/VAESynth_artefacts/Release/Standalone/VAE Synth.app`, go to
VAE Lab, hold a note, and check each line. Note anything that differs.

- [ ] Type "dark reese bass": you hear Lite, and the reply names Bass and the dark step.
- [ ] Drag the orbit handle with a held note: the sound follows, no clicks.
- [ ] Drag empty space, zoom, Reset view: the cloud turns and the sound does not change.
- [ ] Type "longer": the handle moves along the Short/Long axis.
- [ ] Lock structure, then drag: oscillator types and switches stay the same.
- [ ] Click a preset dot of the current category: jumps to that preset.
- [ ] Click a CATEGORY MATCH row: the cloud highlights it; the sound does not change.
- [ ] Knob pad: fine-tune and hear it.
- [ ] Send to Surge: the Surge patch plays (a held note may cut once on the switch).
- [ ] Drag again: back on Lite.
- [ ] Close and reopen the editor: your sound is unchanged until you act in the Lab.
- [ ] Move a knob in the Synth view, then drag the Lab handle: it asks "Replace your Synth edits?"; Keep editing keeps
      your knob, Replace plays the pad.
- [ ] Drag the handle, close the editor, move a knob in the host (or change nothing), reopen and drag: it asks only
      when something changed.
- [ ] Send to Surge, close and reopen: the Synth view still says the Surge patch is playing; a knob returns to Lite.

## Rendering and profiling the prompt feature

`vae_p0_eval` (Surge builds; `tools/p0_eval`) renders the fixed prompt set and profiles Find. Run from the repo
root with `B=models/joint-vae-j48w_b05/bundle.json` and `P=models/prompt-p1/prompt.json`:

```sh
T=build-surge/tools/p0_eval/vae_p0_eval_artefacts/Release/vae_p0_eval
$T eval build-surge/p0-eval $B $P    # WAVs + results.md (see docs/PROMPT-EVAL.md)
$T profile $B $P                     # 20 Finds, half during a Lab drag, held note on Lite
$T mem 2 1 $B $P                     # resident memory: <editors> <load p0 0|1>
```
