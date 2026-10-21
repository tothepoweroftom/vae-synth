# Reference audio to an editable preset

This independent backup needs no student model, VAE decoder, training run, Python ML
packages, or Stable Audio weights. The first stage accepts a reference WAV. Stable
Audio 3 Small can now supply that WAV through the separate
[prompt prototype](PROMPT-PRESETS.md).

The offline `vae_retrieve` executable builds a bank from imported control vectors,
then searches it and exports up to five distinct presets and audio previews. The
plug-in's **Preset...** button loads a result JSON, selects its engine, and applies
its controls through the existing parameter store. Play it with MIDI or the on-screen
keyboard; the reference audio is not sampled by the instrument. Host state saves the
loaded controls as usual.

## Build and run

Run these commands from the repository root. `build-retrieval` is an example build
directory; any existing build can build the same target. Surge DSP is not needed.

```sh
cmake -S . -B build-retrieval -G Ninja -DCMAKE_BUILD_TYPE=Release -DVAE_BUILD_TESTS=ON
cmake --build build-retrieval --target vae_retrieve VAESynth_Standalone vae_plugin_tests
```

Import an existing local Surge preset directory. Our local checkout has 641 factory
and 2,920 third-party `.fxp` files. Start with the factory bank; importing all the
third-party presets is optional. An import is a lossy mapping into our 93 controls,
not an exact reproduction of the original Surge patch.

```sh
python3 tools/surge_preset_to_vector.py \
  build-surge/_deps/surge-src/resources/data/patches_factory \
  -o build-retrieval/factory-vectors.jsonl --stats

build-retrieval/tools/retrieve/vae_retrieve_artefacts/Release/vae_retrieve \
  build build-retrieval/factory-vectors.jsonl build-retrieval/factory-bank.json

build-retrieval/tools/retrieve/vae_retrieve_artefacts/Release/vae_retrieve \
  search build-retrieval/factory-bank.json reference.wav build-retrieval/audition-1
```

Bank and result paths must be new, to avoid overwriting previous work. Rebuilding
the bank is necessary after DSP, schema, probe, or descriptor changes. The bank
format is versioned but does not automatically fingerprint the engine binary.

Open the newly built standalone (or rebuild the plug-in format you use). Click
**Preset...**, choose `audition-1/match-1.json`, and play. Each `match-N.wav` is the
candidate played at the matched probe pitch, with a 1.5-second key hold and a
2.5-second release window. The `note` field records the MIDI note. `results.json`
is a ranked summary; load the individual `match-N.json` files in the instrument.

## What the search does

- Deduplicates identical control vectors and renders each through **Lite**, at
  MIDI notes 48, 60 and 72, velocity 100. It excludes silent or failed renders.
- Converts mono/stereo references at 8–192 kHz to 48 kHz with windowed-sinc
  resampling. Rejects empty, silent, non-finite, or longer-than-30-second input.
- Aligns the first sample above 1% of peak, takes at most four seconds after that
  onset, pads shorter sounds, and normalises peak amplitude on both sides.
- Reuses the comparison tool's 40-band mean mel spectrum and adds a 24-bin
  normalised amplitude envelope. Distance is spectrum RMS / 20 dB plus 0.5 times
  envelope RMS. These are initial heuristic weights, not learned or calibrated
  perceptual scores; smaller is closer, not a confidence percentage.
- Returns five distinct control vectors rather than five pitches of one preset.

No heavy work runs in the audio callback: rendering, file I/O, resampling, and
search happen in a separate command-line process. Preset parsing happens on the
message thread and rejects malformed values before applying any controls.

## Scope and limitations

This stage is audio retrieval. It does not invent new parameter combinations.
The intended creative workflow uses Stable Audio Small to generate an original
reference from a prompt, then retrieves an editable approximation. A subsequent
bounded parameter-refinement step could move beyond existing presets; that
refinement step is not implemented yet. An experimental
[SAME-Small ranking path](SAME-RETRIEVAL.md) is now available alongside this baseline.

Use isolated notes, plucks, bells, basses and simple pads for the first demo.
Pitch, performance, long attacks and stereo cancellation can affect ranking.
Three probe pitches help but do not make the metric pitch invariant. Long releases
are truncated, and complex sequences, speech and layered music cannot be reliably
reconstructed by this search. Random oscillator/noise behaviour may change previews
slightly between renders. Results always use Lite so the indexed sound and playable
engine agree. The current schema cannot preserve all Surge modulation, wavetable,
effects or scene features.

Useful next experiments: audition held-out generated references, compare the top
five by ear, then evaluate SAME-Small ranking against this spectral baseline. Do
not claim a quality improvement until that listening comparison supports it.

## Verification

```sh
ctest --test-dir build-retrieval \
  -R 'vae_retrieve_smoke|vae_plugin_tests|vae_schema_tools|vae_format_check' \
  --output-on-failure
```

The retrieval test needs neither Surge assets nor models. It checks duplicate
removal, invalid rows, a 44.1 kHz reference, known-tone ranking, exported-render
retrieval, distinct candidates, silence rejection, schema rejection, and protection
against overwriting results. Processor tests check loading without a trained model,
state persistence, and rejecting invalid presets without changing current controls.
