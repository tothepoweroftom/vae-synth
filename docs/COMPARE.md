# Comparing Lite, Surge and the original presets

Two questions this answers:

1. **How different are the two engines for the same control vector?**
   (the "engine gap")
2. **Is that small next to what the schema already loses** when a real Surge
   preset is squeezed into the 93 controls? (the "schema loss")

If the engine gap is well below the schema loss, a preset made for the schema
sounds about as right on Lite as on Surge.

## Run it

Needs a `VAE_WITH_SURGE=ON` build (`vae_compare` is only built then) and
numpy for the report.

```sh
# 1. presets -> control vectors
python3 tools/surge_preset_to_vector.py <surge>/resources/data/patches_factory \
    <surge>/resources/data/patches_3rdparty -o vectors.jsonl

# 2. render every preset three ways and measure (a few minutes per 1000 presets)
build/tools/compare/vae_compare_artefacts/Release/vae_compare vectors.jsonl -o compare.csv --jobs 8

# 3. report, plus the worst presets as WAVs for listening
python3 tools/compare_report.py compare.csv vectors.jsonl -o report.md --wav-list worst.txt --worst 20
build/tools/compare/vae_compare_artefacts/Release/vae_compare vectors.jsonl -o /dev/null \
    --wav-dir wav --wav-list worst.txt
```

## What is rendered

Every preset is played with the same MIDI: note 48 for 1.5 s, 1 s of
release, then a C major chord (60/64/67) for 1 s and 1 s of release. That is
4.5 s at 48 kHz, summed to mono, three times:

| render | what |
|---|---|
| `orig` | the original `.fxp` loaded into headless Surge |
| `svec` | the preset's control vector on the Surge backend |
| `lite` | the same vector on the Lite backend |

A fresh engine is used for every render, so nothing carries over between
presets.

## What the numbers mean

The audio is cut into short overlapping windows (2048 samples, 512 apart),
and each window gets a 40-band "mel" spectrum: energy per band on a pitch-like
scale. Everything quieter than 60 dB below the loudest point is ignored.
A render quieter than −60 dB RMS counts as silent (Surge leaves inaudible
residue where Lite is exactly 0), and presets whose original is silent are left
out of the distances.

| column | plain words |
|---|---|
| `frame_mel_db` | how different the two renders' spectra are, window by window, in dB. The main "do they sound alike" number: it sees timbre and how it changes over time |
| `mel_db` | the same for the long-term average spectrum (overall tone colour, ignoring timing). Loudness is removed first. Very pure sounds (a sine) score high on small, inaudible residue; trust `frame_mel_db` for those |
| `env_corr` | how alike the loudness curves are, from -1 to 1 (1 = same shape) |
| `d_level_db` | loudness difference in dB |
| `d_centroid_oct` | brightness difference in octaves (the spectral centroid is the "centre of mass" of the spectrum) |
| `d_flatness` | difference in noisiness (0 = pure tone, 1 = white noise) |
| `attack_s`, `release_s` | time to reach 90 % of the first note's peak, and to fall 40 dB after its release (capped at 1 s) |

Rough guide for `frame_mel_db`: under ~2 dB is very close, 2–5 dB the same
kind of sound with audible timbre differences, above ~8 dB clearly different.

The report also gives:

- **Rank agreement:** do the engines agree on which presets are brighter,
  louder, slower to attack? This matters more to a model than absolute values.
- **Category test:** a simple nearest-neighbour classifier, trained on Surge
  renders, guesses each preset's folder category (pad, bass, lead...) from
  its audio. If it does about as well on Lite renders, the category survives
  the engine change.
- **Where Lite diverges most:** the engine gap grouped by control values
  (filter type, wave, effects...), pointing at the Lite parts to improve.
- **Worst presets**, to listen to.
