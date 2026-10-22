# Lite vs Surge: fix list

From the `vae_compare` run over 3,561 Surge presets (see `docs/COMPARE.md`).

**TL;DR:** Lite's gap to Surge (median 4.9 dB spectral) is less than half
of what the 93-control schema already loses from the original presets
(11.4 dB). Ordering across presets mostly holds (brightness and loudness
0.86, release 0.93); attack doesn't (0.42). The biggest single problem is in
the **importer**, not the engines. Fix in this order.

**Status (2026-10-21):** 1–4 done, the noise half of 5 done, wavetable and
the effects done, high-resonance filters done. Open: attack (5), category
test (7), and the remaining S&H/FM/wavetable oscillator detail. Full-run
median engine gap is now **2.28 dB** (p90 5.37 dB), loudness **0.84 dB**;
see "Re-run after effects and filters". The implementation notes and the
per-component sweep are in `docs/LITE-SURGE-PARITY-PLAN.md`.

## 1. Importer: fold modulation into osc level and amp sustain (35% of the data) — done

- **Where:** `tools/surge_preset_to_vector.py`, `Converter.oscillators()` (~L172-176) and `Converter.envelopes()` (~L242-247).
- **Problem:** 1,242 presets set an osc level, or the amp sustain, to 0 and drive it with an LFO, aftertouch or a controller. The importer keeps the static 0, so the vector is silent. This explains 212 of the 218 "Lite silent, Surge not" presets, plus 2 of the 4 "Surge silent" ones.
- **Fix:** if `level_oN` (or `envN_sustain`) has incoming `modrouting`, use static + max(positive depth), clamped to 0..1, and add a `lossy` note. `Patch.routes[name]` already holds `(source, depth)`.
- **Proper fix:** add osc level as a modulation destination. That's part of the student's routing decision.
- **Check:** importer stats; the silence mismatch count in the compare report should drop from 222 to around 10.

## 2. Compare report: realistic silence threshold — done

- **Where:** `tools/compare/Features.cpp:85` (`silent = peak < 1e-4`).
- **Problem:** Surge's inaudible residue (median −71 dB RMS) counts as sound, while Lite's exact zero counts as silent.
- **Fix:** `silent = levelDb < -60.0f`. Also exclude presets whose *original* is silent (Basses/Fat Growl, Brass/Batbrass) from the comparisons.

## 3. Lite: sine is ~6 dB quieter than Surge — done

- **Where:** `source/engine/lite/LiteDsp.h` ~L267 (`Wave::Sine`).
- **Evidence:** Init Sine renders at −27.7 dB in Lite vs −21.7 dB in Surge.
- **Fix:** scale the sine output (≈ ×2) so its level matches Surge; check Init Saw stays matched (RMS 0.080 on both).

## 4. Lite: filter self-oscillation — done (clean room)

- **Where:** `source/engine/lite/LiteDsp.h:203`: `k = 2 * (1 - 0.985 * res)`.
- **Problem:** the damping never reaches 0, so Lite's filter can't ring on its own. Surge rings at −26 to −44 dB, which matters for percussion presets (Synth Tom 1-3, Ghost Notes, Tuned Resonance Harmonics).
- **Fix:** let `k` reach ~0 near res = 1; add `tanh` soft saturation in the feedback path, plus a tiny noise excitation so it can start ringing.
- **Test:** HP24 at res 0.92 with oscs near zero should give audible output, and stay finite.

## 5. Lite: noise character and attack timing — noise done, attack open

- **Noise:** the worst presets are noise-based (Noise Crash 1/2, Fireworks, Crackling). Lite's noise is much flatter (more broadband) than Surge's S&H noise.
  - Fix: match Surge's S&H noise behaviour: it holds per sample period, with correlation, plus the sync +48 st the Surge backend uses.
- **Attack:** rank agreement for attack time is only 0.42 (e.g. Gliss Pluck Distorted: Surge 0.91 s, Lite 0.01 s).
  - Compare `ShapedEnvelope` attack (curve powers 0.5/1/2) with Surge's attack shapes. Check that LFO fade-in and glide aren't feeding into it.

## 6. Lite: wavetable and effects (largest per-component gaps) — done

The old *per-preset* grouping peaked at 6–9 dB; isolated component probes
showed the root causes were mostly parameter laws. After the fixes the
isolated gaps are (median over the component sweep, `sl_frame_mel_db`):

| component | old isolated gap | now |
|---|---:|---:|
| distortion | 5.0 (up to 13.8) | **0.43** |
| delay | 8.7–10.8 | **1.04** |
| chorus | 6.6 | **1.53** |
| phaser | 2.8–3.8 | **1.66** |
| flanger | 3.2–3.6 | **1.04** |
| EQ | 4.0–4.9 | **0.84** |
| reverb | 3.1–5.6 | **2.17** |
| wavetable osc | 1.6 | 1.6 (not changed) |
| S&H noise | 2.6 | 2.6 (not changed) |

What changed:

- **Delay:** feedback `x³`, 110 Hz/2.49 kHz 2-pole feedback filters, soft clip.
- **Chorus:** 4 voices, 1/√4 gain, 15.625 ms, triangle phases, 55 Hz/3.52 kHz
  filters, 0.0625 feedback, hard clip, equal-power pans.
- **EQ:** three Orfanidis peaks at 77.8 Hz / mid / 13.4 kHz, BW 2 octaves.
- **Distortion:** 4× oversampled path, 440 Hz pre-LP, post-LP at the tone
  frequency, 4× rate.
- **Phaser/flanger:** second-order all-pass stages at 440 Hz with `2/(i+1)`
  octave spans; 4-comb flanger with additive mix, damping and gain
  compensation.
- **Reverb:** Reverb 2-style four-line FDN with pre-delay, diffusion, dampers,
  modulated taps and the Surge decay law.
- **Filters:** two-step SVF at 2× rate with Surge's state-scaling limiter and
  gain law, plus a low-cutoff resonance makeup.

## 7. Category test is inconclusive — open

The kNN category classifier is too weak to judge: 25.6% on Surge itself vs 16% chance; Lite 18.0%, originals 20.4%. Replace the features with CLAP embeddings, or use the student's tags, before reading anything into it.

## Re-run after fixes

```sh
python3 tools/surge_preset_to_vector.py <surge>/resources/data/patches_factory <surge>/resources/data/patches_3rdparty -o vectors.jsonl
build-surge/tools/compare/vae_compare_artefacts/Release/vae_compare vectors.jsonl -o compare.csv --jobs 8
python3 tools/compare_report.py compare.csv vectors.jsonl -o report.md --wav-list worst.txt
```

Medians. "Before" is the run this list came from; "after" has fixes 1–4, the
noise part of 5, the wavetable part of 6 and the extras below.

| measure | engine gap before | after 1–4+osc | after effects+filters | schema loss after |
|---|---|---|---|---|
| spectrum over time (median) | 4.9 dB | 3.5 dB | **2.28 dB** | 11.48 dB |
| spectrum over time (p90) | — | — | **5.37 dB** | 19.90 dB |
| brightness | 0.31 oct | 0.15 oct | — | 0.79 oct |
| loudness | 4.0 dB | 1.2 dB | **0.84 dB** | 6.7 dB |
| envelope correlation | 0.95 | 0.96 | **0.979** | 0.80 |
| svec/lite silence mismatch | 222 | ~36 | **21** | |

Re-run after effects and filters (3,561 presets, 2026-10-21):

```sh
python3 tools/surge_preset_to_vector.py <surge>/resources/data/patches_factory <surge>/resources/data/patches_3rdparty -o vectors.jsonl
build-surge/tools/compare/vae_compare_artefacts/Release/vae_compare vectors.jsonl -o compare.csv --jobs 8
python3 tools/compare_report.py compare.csv vectors.jsonl -o report.md --wav-list worst.txt
```

The isolated component sweep lives in `build-surge/component-sweep/` and is
generated by the script in `docs/LITE-SURGE-PARITY-PLAN.md` (Appendix A).

What changed, beyond the items as written:

- **Importer:** the noise mixer channel goes into the first muted osc slot as
  `noise` (about 100 presets mute every osc and only play that channel).
- **Osc levels:** Surge's Sine, Wavetable and FM2 oscs run ~6 dB above its
  Classic saw, not only the sine (Lite `kHotOscGain` = 2).
- **Phase:** Lite osc start phases are random per note (seeded), like Surge
  without osc retrigger, so stacked oscs sum at +3 dB on average, not +6.
- **FM:** Lite's index is 0.68 rad, matching the backend's FM2 M1 amount 0.3.
- **Noise:** sample-and-hold whose rate tracks the note (24 values per cycle),
  level and brightness fitted to Surge renders.
- **Wavetable:** both engines play one shared one-frame table
  (`dsp::defaultTableHarmonic`); the wavetable gap fell from 9.2 to 4.5 dB.
- **Filter:** clean-room resonance fit from `docs/specs/lite-filter-resonance.md`
  (see `docs/LICENSING.md`).

Still open: attack rank (0.46; with delay or distortion tails the "time to 90 %
of peak" can land on an echo), effects (distortion 7.6 dB, mod FX 5.5, delay
4.8, chorus 4.7, reverb 4.4) and item 7.

Checked: unit tests under ASan and TSan; pluginval strictness 5 on the ASan
VST3, the release AU (with auval) and the combined Surge VST3.
