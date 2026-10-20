# Lite vs Surge: fix list

From the `vae_compare` run over 3,561 Surge presets (see `docs/COMPARE.md`).

**TL;DR:** Lite's gap to Surge (median 4.9 dB spectral) is less than half
of what the 93-control schema already loses from the original presets
(11.4 dB). Ordering across presets mostly holds (brightness and loudness
0.86, release 0.93); attack doesn't (0.42). The biggest single problem is in
the **importer**, not the engines. Fix in this order.

## 1. Importer: fold modulation into osc level and amp sustain (35% of the data)

- **Where:** `tools/surge_preset_to_vector.py`, `Converter.oscillators()` (~L172-176) and `Converter.envelopes()` (~L242-247).
- **Problem:** 1,242 presets set an osc level, or the amp sustain, to 0 and drive it with an LFO, aftertouch or a controller. The importer keeps the static 0, so the vector is silent. This explains 212 of the 218 "Lite silent, Surge not" presets, plus 2 of the 4 "Surge silent" ones.
- **Fix:** if `level_oN` (or `envN_sustain`) has incoming `modrouting`, use static + max(positive depth), clamped to 0..1, and add a `lossy` note. `Patch.routes[name]` already holds `(source, depth)`.
- **Proper fix:** add osc level as a modulation destination. That's part of the student's routing decision.
- **Check:** importer stats; the silence mismatch count in the compare report should drop from 222 to around 10.

## 2. Compare report: realistic silence threshold

- **Where:** `tools/compare/Features.cpp:85` (`silent = peak < 1e-4`).
- **Problem:** Surge's inaudible residue (median −71 dB RMS) counts as sound, while Lite's exact zero counts as silent.
- **Fix:** `silent = levelDb < -60.0f`. Also exclude presets whose *original* is silent (Basses/Fat Growl, Brass/Batbrass) from the comparisons.

## 3. Lite: sine is ~6 dB quieter than Surge

- **Where:** `source/engine/lite/LiteDsp.h` ~L267 (`Wave::Sine`).
- **Evidence:** Init Sine renders at −27.7 dB in Lite vs −21.7 dB in Surge.
- **Fix:** scale the sine output (≈ ×2) so its level matches Surge; check Init Saw stays matched (RMS 0.080 on both).

## 4. Lite: filter self-oscillation

- **Where:** `source/engine/lite/LiteDsp.h:203`: `k = 2 * (1 - 0.985 * res)`.
- **Problem:** the damping never reaches 0, so Lite's filter can't ring on its own. Surge rings at −26 to −44 dB, which matters for percussion presets (Synth Tom 1-3, Ghost Notes, Tuned Resonance Harmonics).
- **Fix:** let `k` reach ~0 near res = 1; add `tanh` soft saturation in the feedback path, plus a tiny noise excitation so it can start ringing.
- **Test:** HP24 at res 0.92 with oscs near zero should give audible output, and stay finite.

## 5. Lite: noise character and attack timing

- **Noise:** the worst presets are noise-based (Noise Crash 1/2, Fireworks, Crackling). Lite's noise is much flatter (more broadband) than Surge's S&H noise.
  - Fix: match Surge's S&H noise behaviour: it holds per sample period, with correlation, plus the sync +48 st the Surge backend uses.
- **Attack:** rank agreement for attack time is only 0.42 (e.g. Gliss Pluck Distorted: Surge 0.91 s, Lite 0.01 s).
  - Compare `ShapedEnvelope` attack (curve powers 0.5/1/2) with Surge's attack shapes. Check that LFO fade-in and glide aren't feeding into it.

## 6. Lite: wavetable and effects (largest per-component gaps)

Median engine gap in dB (overall 4.9):

| component | gap |
|---|---|
| wavetable osc | 9.3 |
| distortion on | 7.8 |
| chorus on | 7.1 |
| notch filter | 7.0 |
| mod FX on | 6.7 |
| delay on | 6.5 |
| reverb on | 6.2 |

- **Wavetable:** ship a few real tables, and use the same ones in both engines, since the Surge backend also uses a built-in table.
- **Effects:** bring distortion, chorus, mod FX, delay and reverb closer to Surge's (Reverb 2, peaking EQ bands).

## 7. Category test is inconclusive

The kNN category classifier is too weak to judge: 25.6% on Surge itself vs 16% chance; Lite 18.0%, originals 20.4%. Replace the features with CLAP embeddings, or use the student's tags, before reading anything into it.

## Re-run after fixes

```sh
python3 tools/surge_preset_to_vector.py <surge>/resources/data/patches_factory <surge>/resources/data/patches_3rdparty -o vectors.jsonl
build-surge/tools/compare/vae_compare_artefacts/Release/vae_compare vectors.jsonl -o compare.csv --jobs 8
python3 tools/compare_report.py compare.csv vectors.jsonl -o report.md --wav-list worst.txt
```

Medians. "Before" is the run this list came from; "after" has fixes 1-4,
the noise part of 5, the shared wavetable and the gains below applied.

| measure | engine gap before | engine gap after | schema loss after |
|---|---|---|---|
| spectrum over time | 4.9 dB | 3.5 dB | 11.5 dB |
| brightness | 0.31 oct | 0.15 oct | 0.79 oct |
| loudness | 4.0 dB | 1.3 dB | 6.7 dB |
| envelope correlation | 0.95 | 0.96 | 0.80 |
| svec/lite silence mismatch | 222 | 36 | |

Also done along the way: the noise mixer channel imports into a free osc slot
as `noise` (≈110 presets were silent only because of it); Surge's Sine,
Wavetable and FM2 oscs run ~6 dB above its Classic saw (Lite `kHotOscGain`);
Lite FM index 0.68 rad matches the backend's FM2 M1 amount 0.3; Lite osc
start phases are random like Surge's without retrigger. Still open: attack
rank (0.46, the metric is fragile with delay/distortion tails), effects
(distortion 7.6 dB, mod FX 5.5, delay 4.8, chorus 4.7, reverb 4.4), item 7.
