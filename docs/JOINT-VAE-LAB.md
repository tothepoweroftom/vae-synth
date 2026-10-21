# Joint VAE Lab (v2)

The **VAE Lab** tab turns a few words into a playing preset and lets you move it around the student's joint VAE
(`j48w_b05`) on an XY pad. The sound follows the dot live on the **Lite** engine. **Send to Surge** builds the full
Surge patch for the current point. The design is in `docs/superpowers/specs/2026-10-21-vae-lab-xypad-design.md`.

The old Lab (raw latent slice, radius, 631-value list, import/export) is tagged **`vae-lab-v1`**. Its Surge playback
path, which v2 still uses for Send to Surge, is described in [JOINT-VAE-SURGE-RESULTS.md](JOINT-VAE-SURGE-RESULTS.md).
The Lab needs a `VAE_WITH_SURGE=ON` build; the default build shows "VAE Lab needs a Surge build".

## Workflow: prompt, preset, pad, tweak

1. **Prompt.** Type a sound ("dark reese bass") in the box under the pad. The first message picks a label from the
   student's nouns and starts from Typical <label>; its adjectives nudge the axes. A one-line reply says what it did
   and which words it ignored. With the p0 prompt bundle installed, the start comes from the student's text model
   instead of the word list.
2. **Preset.** The decoded point plays at once on Lite, once you have acted in the Lab (see limits).
3. **Pad.** Drag the dot. X, Y and an optional Z are any of the five sound axes or the latent dimensions. The sound
   follows while you drag, without clicks.
4. **Tweak.** Type follow-ups ("brighter", "a lot longer", "slightly less noisy"), move an axis slider, click a
   preset dot to jump to that real preset, or switch to the knob pad for fine adjustments.

## Controls

| Control | What it does |
|---|---|
| Latent pad / Knob pad | Pad axes are sound axes and latent dimensions. The knob pad moves groups of the student's knobs together. |
| Start from | Typical <label>, or a random real preset. |
| Category | The category the decoder is conditioned on (8 labels plus Unknown). |
| Lock structure | Freezes oscillator types, on/off switches and choices at the lock time. Only continuous values follow the dot. |
| Reset | Back to the start point. |
| Background map | Colours the pad by Category guess, Brightness, Length, Attack, Noise, or none. Recomputed shortly after you stop moving. |
| Preset dots | Shows or hides the student's real presets as dots (latent pad only). Click one to jump to it. |
| Send to Surge | Builds the full Surge patch for this point and plays it on the Surge engine. |
| Axis list | Sound axes first, then latent dimensions by dB impact (under 2 dB folded away). X/Y/Z buttons and a slider per row. |
| Z slider | Appears when a Z axis is set. |
| Readout strip | Expected-sound spectrogram, brightness / attack / length / noise, the decoded preset, and the category guess. |
| Arrow keys on the pad | 0.02 per press, Shift 0.1. |

Sound axes: Dark/Bright, Snappy/Slow attack, Short/Long, Tonal/Noisy, Static/Moving. Brightness, length and attack
carry well; "noisy" and "moving" change the sound little, and the reply says so.

## Bundles

The Lab loads `~/Library/Application Support/VAE Synth/models/joint-vae/bundle.json` when the window opens, and the
prompt bundle from `.../models/prompt-p0/` if present. `scripts/install-models.sh` installs
`models/joint-vae-j48w_b05` (and `models/prompt-p0` when it exists), plus the Surge template and the 32 wavetables
the model can pick. Running it overwrites the installed Lab bundle. Models are git-ignored.

Lab bundle contents: `bundle.json` (id fingerprints everything below), `config.json`, `schema.json`,
`pruning.json`, `decoder.onnx`, `sound_encoder.onnx`, `sound_decoder.onnx`, `sound_pca.npz`, `xypad.json` (axes,
latent dimensions, per-label means, preset z, knobs, category forest), plus reference data for the tests
(`parity.json`, `codec-parity.json`, `xypad-parity.json`, `example-sound.json`).
Prompt bundle (`models/prompt-p0`): `prompt.json`, `text_encoder.onnx` (501 MB), `prior.onnx`, `tokenizer.json`,
`bank.json` / `bank.bin` (1,227 real presets in Lab feature order), and parity files.

### Export order

Needs numpy, onnxruntime and, for the first step, torch and onnx. The sources are the student's files (checkpoint,
`schema.json`, `pruning.json`, `sound_pca.npz`, `xypad_v2.html`, the `p0` folder).

```sh
python3 tools/joint_vae/export_bundle.py --source <model dir> --out models/joint-vae-j48w_b05
python3 tools/joint_vae/export_xypad.py  --html <xypad_v2.html> --bundle models/joint-vae-j48w_b05
python3 tools/joint_vae/codec_parity.py  --bundle models/joint-vae-j48w_b05
# p0 (optional):
python3 -I tools/joint_vae/export_prompt.py --source <p0 dir> --lab models/joint-vae-j48w_b05 --out models/prompt-p0
python3 tools/joint_vae/prompt_reference.py --bundle models/prompt-p0
python3 tools/joint_vae/codec_parity.py --bundle models/joint-vae-j48w_b05 --bank models/prompt-p0 --prompt models/prompt-p0
```

`export_xypad.py` recomputes the bundle id, so run it before `codec_parity.py`, not after.

## Tests

```sh
cmake --build build-surge
VAE_JOINT_BUNDLE="$PWD/models/joint-vae-j48w_b05/bundle.json" \
VAE_PROMPT_BUNDLE="$PWD/models/prompt-p0/prompt.json" \
  ctest --test-dir build-surge --output-on-failure
ctest --test-dir build-default        # no Surge: checks the "needs a Surge build" screen
```

- `VAE_JOINT_BUNDLE`: Lab bundle. Without it the real-model parity sections are skipped.
- `VAE_PROMPT_BUNDLE`: `prompt.json` of the p0 bundle, for the prompt model parity tests.
- `VAE_CODEC_PARITY`: an extra codec plan file to compare the native codec with, e.g. the bank's `bank-codec-parity.json`.
- `VAE_LAB_SCREENSHOT=<file.png>`: `vae_plugin_tests` writes a snapshot of the editor there.

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
- The category map takes about 0.75 s to compute; the brightness map about 70 ms.
- The knob pad has no preset dots.
- `pruning.json` for `j48w_b05` is assumed identical to `j48`'s. The student has not confirmed it, and a wrong one
  would give wrong patches silently.
- The p0 text-encoder preprocessing is our reading of the student's code; there are no student golden outputs.
- Lab state is not saved in the host project.

## Try it

Run `scripts/install-models.sh`, then open `build-surge/VAESynth_artefacts/Release/Standalone/VAE Synth.app`, go to
VAE Lab, hold a note, and check each line. Note anything that differs.

- [ ] Type "dark reese bass": you hear Lite, and the reply names Bass and the dark step.
- [ ] Drag on the pad with a held note: the sound follows, no clicks.
- [ ] Type "longer": the dot moves up the Short/Long axis.
- [ ] Lock structure, then drag: oscillator types and switches stay the same.
- [ ] Click a preset dot: jumps to that preset.
- [ ] Knob pad: fine-tune and hear it.
- [ ] Send to Surge: the Surge patch plays (a held note may cut once on the switch).
- [ ] Drag again: back on Lite.
- [ ] Close and reopen the editor: your sound is unchanged until you act in the Lab.
