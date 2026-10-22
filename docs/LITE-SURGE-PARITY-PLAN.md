# Lite ↔ Surge XT parity plan

**Status:** effects and high-res filters implemented, 2026-10-21
**Result so far:** full-preset median engine gap **2.28 dB** (was ~3.5),
p90 **5.37 dB**, loudness **0.84 dB** (was ~1.2). Isolated component medians:
delay 1.04, chorus 1.53, flanger 1.04, distortion 0.43, EQ 0.84, phaser 1.66,
reverb 2.17, filters 0.87. Remaining: the S&H/FM/wavetable oscillator detail,
the attack measurement, and the category test.
**Scope:** make the Lite engine's oscillators, filters and effects behave like the
Surge backend for the same schema vector.
**Reference:** Surge XT at `45568c7d60be52aa2a4f27dd7b7a074f56bd00ba`
(`build-surge/_deps/surge-src`, GPL-3.0-or-later, read only — see
[Licensing](#2-licensing-and-the-clean-room-rule)).

**TL;DR:** the Lite engine is already close on its own oscillators
(sine/saw/triangle/square are within about 1 dB of Surge in isolation) and on
low-resonance filters. The remaining gap is concentrated in four places:

1. **Effects.** Delay (8–11 dB), chorus (6.6 dB), distortion (5–14 dB), EQ
   (3–5 dB), reverb (3–5.6 dB), phaser/flanger (2.8–3.8 dB). These are mostly
   *parameter-law* mismatches, not subtle DSP differences.
2. **High-resonance filters.** LP12/HP12/BP are 5–10 dB off and 4–10 dB too
   loud at high cutoff and resonance ≥ 0.9. Lite's damping/limiter topology is
   different from Surge's two-sub-step SVF at 2× rate.
3. **S&H noise, FM and wavetable oscillators.** 2.6 / 1.5 / 1.6 dB in
   isolation, from correlated noise, FM2 detail and interpolation differences.
4. **Small voice/global details.** Keytrack during glide, pan law, LFO
   smooth-random shape, block-rate conventions.

The plan below fixes them in impact order, with a measurement harness first and
a concrete acceptance threshold per component.

---

## 1. What "closer" is measured against

The existing three-way comparison is the headline metric and must keep
working: original `.fxp` → schema vector on Surge → the same vector on Lite.
See [COMPARE.md](COMPARE.md) and [FIXES-lite-vs-surge.md](FIXES-lite-vs-surge.md).

The current documented position (after the earlier fixes) is a **median engine
gap of ~3.5 dB** (`svec` → `lite`, `frame_mel_db`), against a **schema loss of
~11.5 dB** (original → `svec`). That matters: engine parity has diminishing
returns for *factory preset* fidelity until the importer/schema is addressed.
This plan deliberately does **not** change the schema or the importer. Its goal
is that, for the same vector, Lite and Surge sound the same.

To see where the gap actually lives, run a component sweep instead of only
factory presets. Current results from a component sweep (defaults, one part
changed at a time, plus high-resonance/cutoff stress vectors; `sl_frame_mel_db`,
lower is better):

| probe | worst / representative gap | notes |
|---|---:|---|
| baseline: saw, filter off, no FX | **0.66–0.78 dB** | engine calibration is good |
| sine / triangle / saw / square | 0.12 / 0.23 / 0.81 / 0.50 dB | keep as regression targets |
| S&H noise | **2.64 dB** | flatness differs by 0.036 |
| wavetable | **1.64 dB** | linear vs windowed-sinc interpolation |
| FM | **1.0–1.5 dB**, level varies with start phase | compare several renders |
| LP12 at cutoff ≥0.55, res 0.9–0.99 | **4.0–8.8 dB**, 3.4–9.6 dB level (Lite louder) | resonance too strong |
| LP24 at cutoff ≥0.7, res 0.9–0.99 | **1.5–6.3 dB**, up to 5.7 dB level (Lite louder) | limiter/topology |
| HP12 at cutoff ≥0.55, res 0.9–0.99 | **4.2–9.7 dB**, 4.4–9.9 dB level (Lite louder) | resonance too strong |
| HP24 at cutoff ≥0.7, res 0.9–0.99 | **1.5–7.0 dB**, up to 6.3 dB level (Lite louder) | limiter/topology |
| BP at cutoff ≥0.55, res 0.9–0.99 | **4.2–10.2 dB**, 4.1–10.1 dB level (Lite louder) | output/gain law |
| notch | **≤0.9 dB** | already close |
| distortion, default drive/tone | **5.0 dB**, 1.0 dB level (Lite quieter) | post-LP law |
| distortion, tone 0.7 / 0.9 | **9.3 / 13.8 dB** | post-LP law, 1.5–2 dB level (Lite louder) |
| EQ low boost / high boost | **4.9 / 4.0 dB**, 7.3 / 3.2 dB level (Lite quieter) | shelves vs peaking EQ |
| chorus | **6.6 dB**, 5.3 dB level (Lite quieter) | 2 voices vs 4, gain law |
| phaser | **2.8–3.8 dB** even at depth 0 | stage frequencies/feedback |
| flanger | **3.2–3.6 dB**, 0.3–0.5 oct centroid | one comb vs four |
| delay, feedback 0.5 / 0.85 | **8.7 / 9.3 dB** | feedback ×³ and filters |
| delay, mix 0.8 | **10.8 dB** | as above, plus mix law |
| reverb | **3.1–5.6 dB** | Freeverb vs Reverb 2 |
| combined pad / lead | **10.8 / 7.2 dB** | effects dominate |

How to reproduce: [Appendix A](#appendix-a--component-sweep) generates the
vectors, and the command below renders them.

```sh
build-surge/tools/compare/vae_compare_artefacts/Release/vae_compare \
    build-surge/component-sweep/vectors.jsonl \
    -o build-surge/component-sweep/compare.csv --jobs 8
```

The comparison binary needs a `VAE_WITH_SURGE=ON` build:

```sh
cmake -S . -B build-surge -DVAE_WITH_SURGE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-surge --target vae_compare vae_engine_tests -j
```

The full factory-preset run stays the top-level gate:

```sh
python3 tools/surge_preset_to_vector.py \
    build-surge/_deps/surge-src/resources/data/patches_factory \
    build-surge/_deps/surge-src/resources/data/patches_3rdparty \
    -o build-surge/vectors.jsonl
build-surge/tools/compare/vae_compare_artefacts/Release/vae_compare \
    build-surge/vectors.jsonl -o build-surge/compare.csv --jobs 8
python3 tools/compare_report.py build-surge/compare.csv build-surge/vectors.jsonl \
    -o build-surge/report.md --wav-list build-surge/worst.txt --worst 20
```

---

## 2. Licensing and the clean-room rule

The Lite build is currently permissive project code; Surge XT and its
`sst-filters` / `sst-effects` libraries are GPL-3.0-or-later. Copying the
reference implementation into `source/engine/lite/` would relicense the Lite
engine.

Rules for this work:

- **Do not copy** code, coefficient tables or comments from
  `build-surge/_deps/surge-src` into `source/engine/lite/`.
- Read the Surge source only for **behaviour facts** (parameter defaults,
  signal flow, what a control does). Implement from public, independently
  documented algorithms: RBJ cookbook / Orfanidis HPEQ / Zavalishin TPT SVF /
  Stilson–Smith BLIT / standard FDN reverb designs. Fit the remaining
  constants to rendered audio.
- Record each fitted block in `docs/specs/lite-<component>-behaviour.md`:
  measured target, independent derivation, final constants, and the test
  that guards it. This is the same pattern already used for
  `docs/specs/lite-filter-resonance.md`.
- Keep the Surge build as a **test oracle only**; do not ship it in the
  default installer.

If the team decides exact parity is worth more than the permissive licence,
there is a much cheaper alternative: link `sst-filters`, `sst-effects` and the
Surge waveshapers into the Lite build and make that build GPL. That is a
product decision, not a DSP task. This plan assumes we keep the clean-room
route.

---

## 3. Root causes and fixes

### Where the code lives

| Area | File | Symbols to touch |
|---|---|---|
| Oscillators, envelopes, LFOs, filter unit | `source/engine/lite/LiteDsp.h` | `Oscillator`, `ShapedEnvelope`, `Lfo`, `FilterUnit` |
| Oscillator/filter building blocks | `source/engine/lite/dsp/VoiceDsp.h` | `AnalogOsc`, `Svf`, `SvfCoefs`, `Noise` |
| Voice routing, keytrack, pan, gains | `source/engine/lite/LiteVoice.h` | `Voice::render`, `Mods` |
| Effects chain | `source/engine/lite/LiteFx.h` | `FxRack::processDist/Eq/Chorus/ModFx/Delay/Reverb` |
| Effect building blocks | `source/engine/lite/dsp/FxBlocks.h` | `Biquad`, `DelayLine` |
| Control → physical mapping | `source/engine/lite/LitePatch.h/.cpp` | `FxSettings`, `Patch::set`, `setOsc`, `setFilter`, `setLfo` |
| Engine block/span handling | `source/engine/lite/LiteEngine.cpp` | `renderVoices`, `finishSpan`, `process` |
| Tests | `tests/engine/LiteEngineTests.cpp` | existing level/routing/pitch tests |
| New FX tests | `tests/engine/LiteFxTests.cpp` (new) | impulse/noise/level tests per effect |

### 3.1 Effects (highest impact)

All effects live in `source/engine/lite/LiteFx.h`; building blocks are in
`source/engine/lite/dsp/FxBlocks.h`. The schema exposes only a subset of each
Surge effect's parameters. The Surge backend leaves every unexposed parameter
at its Surge default. Lite must reproduce those defaults exactly, even though
they are not controls.

| Effect | Exposed controls | Surge behaviour to match | Lite today | Change |
|---|---|---|---|---|
| **Distortion** | on, drive (dB), tone (post high cut) | 4× oversampled FX path, rational-tanh shaper (model 0), pre-EQ peak 0 dB @ 440 Hz/BW 2, pre high cut 2-pole @ **440 Hz** (4×-effective), post high cut 2-pole @ **toneHz** (4×-effective), post-EQ peak 0 dB, feedback 0, output gain 0 dB, half-band decimation | `std::tanh(x·drive)` then **one-pole** low-pass at `toneHz`, no pre-LP, no oversampling, no EQ | Add the 2-pole pre-LP at 440 Hz and post-LP at `toneHz`; use the rational tanh and keep the drive law. The measured spectrum already matches at ~0.4 dB without 4× oversampling, so oversampling is a later refinement. The pre/post peaking EQ are 0 dB at defaults and can wait until exposed. |
| **EQ** | low gain, mid freq, mid gain, high gain | three **Orfanidis peaking** bands, all BW = 2 octaves; band 1 fixed at −30 st (77.8 Hz), band 2 at the mid control (default +6 st = 622 Hz), band 3 fixed at +54 st (13.4 kHz); output gain 0 dB, mix 1 | low shelf @ 77.8 Hz, RBJ peak, high shelf @ 13.4 kHz | Replace the shelf/peak mix with three Orfanidis peaking filters and the same bandwidth mapping. This is the main reason a +24 dB low boost is 7.3 dB too quiet. |
| **Chorus** | on, rate, depth, mix | **4 voices**, `gain = 1/√4 = 0.5` per voice, equal-power pans at positions −1, −⅓, +⅓, +1, triangle LFO per voice (phases 0, ⅓, ⅔, 1), delay = 15.625 ms × (1 + depth·triangle), feedback = 0.5 × x³ (0.0625 at x=1), 2-pole HP 55 Hz (`−36 st`) and LP 3.52 kHz (`+36 st`) in the feedback path, hard clip in the loop, linear mix | 2 delay lines, sine LFO 90° apart, 12 ms base ± 8 ms, no feedback, no filters, two full-gain voices | Rewrite `processChorus` as a 4-voice chorus with the above laws. This alone should remove the ~5.3 dB level error (Lite is too quiet). |
| **Phaser** | on, type, rate, depth, feedback, mix | sine LFO; **4 second-order all-pass biquads** (default count, Q = 1), all centered at 440 Hz, stage *i* sweep = ±(2/(i+1)) octaves × depth; feedback = 0.95 × control, clamped ±32; linear mix; stereo 0 means **no L/R LFO offset**; tone/width/spread at defaults | 4 first-order all-passes, center 800 Hz, ±2.5 octaves, and a 90° L/R sine offset; feedback 0.9 × control; no clamp | Use four second-order all-pass biquads at 440 Hz with `2/(i+1)` spans, feedback gain/clamp, same LFO phase in both channels at stereo 0. `ph_tone` is deactivated at the default, so no tone filter. |
| **Flanger** | on, type, rate, depth, feedback, mix | **4 combs**, base pitch 60 (≈3.82 ms), spacing 0, LFO phases `(i + 0.5·c)/4`, sine; **additive mix** `out = dry + mix·wet` (schema 0..1 maps to 0..1; 0.5 is dry + half wet, not a 50/50 crossfade); feedback gain ≈ `sqrt(fb)` times a voice/depth scale, cubic soft clip, HF damping 0.1, output gain compensation (`−1/√(7−voices) − 0.07·mix`), final cubic soft clip | one short modulated delay per channel, 0.8–4.8 ms, 0..1 crossfade, no damping, no gain compensation | Rewrite as a 4-comb flanger with the additive mix and the damping/soft-clip/gain laws. |
| **Delay** | on, time, feedback, mix | stereo delay, both times = control, tape read, **feedback = control³** (0.5 → 0.125), soft clip in the feedback loop, 2-pole HP at 110 Hz (`−24 st`) and 2-pole LP at 2.49 kHz (`30 st`), crossfeed 0, width 0, linear mix, time glide | one time for both, feedback = control (0.5 → 0.5), one-pole LP at 9 kHz, no HP, no soft clip | Change the feedback law to x³, add the 2-pole HP/LP pair and the soft clip. This is the single largest effect error (8–11 dB). |
| **Reverb** | on, size, decay, damping, mix | **Reverb 2**: mono input `(L+R)/2`; 62.5 ms pre-delay; 4 input all-passes (4.76/6.81/10.13/16.72 ms × room scale); 4 delay blocks, each with 2 all-passes, HF+LF one-pole dampers and 2 modulated delay taps; `loopTime = 0.5508·2^size`, `decay = 0.001^(loopTime/(4·2^decayParam))`; tap gains 1.5/1.2/1.0/0.8 ÷4; linear mix | `juce::Reverb` (Freeverb), stereo, room size from a different formula, wet by mix/3 and dry by (1−mix)/2, no pre-delay/modulation/diffusion network | Replace `juce::Reverb` with a clean-room 4-line FDN implementing the same topology. Fit delay times, decay and damping to measured Surge impulse responses. |

**Implementation order inside effects:** delay → chorus → EQ → distortion →
phaser → flanger → reverb. The first three are mostly parameter-law fixes and
should move the median quickly; distortion adds the 4× oversampled path;
flanger needs the 4-comb rewrite; phaser needs the stage-frequency/feedback
alignment; reverb is the largest single block and gets its own phase.

**Fixed defaults** that must be encoded in `LitePatch`/`FxSettings` (not
exposed as controls):

| Effect | Fixed value |
|---|---|
| Distortion | pre-EQ gain 0 dB, freq 440 Hz, BW 2 oct; pre high cut 0 st → 110 Hz base rate, 440 Hz at the 4× rate; model 0 (rational tanh); feedback 0; post-EQ gain 0 dB, freq 440 Hz, BW 2; output gain 0 dB; 4× OS |
| EQ | band 1 freq −30 st, band 3 freq +54 st, all BW 2 oct; output gain 0 dB; mix 1 |
| Chorus | time −6 → 15.625 ms; feedback 0.5 → 0.0625; low cut −36 st → 55 Hz; high cut +36 st → 3.52 kHz; width 0 dB; 4 voices |
| Phaser | waveform sine; stereo 0; stages 4 (non-legacy); spread 0; center 0; sharpness 0; width 0 dB; tone (verify deactivation) |
| Flanger | mode dry+combs; waveform sine; voices 4; base pitch 60; spacing 0; damping 0.1; width 0 dB |
| Delay | left = right; tape line; crossfeed 0; low cut −24 st → 110 Hz; high cut 30 st → 2.49 kHz; mod depth 0; input channel 0; width 0 dB; soft-clip feedback |
| Reverb | pre-delay −4 → 62.5 ms; diffusion 1; buildup 1; modulation 0.5; LF damping 0.2; width default; mono input; linear mix |

### 3.2 Filters

`FilterUnit` in `source/engine/lite/LiteDsp.h` is a good fit at low resonance
but diverges at high resonance. Surge's voice runs at 2× the base rate
(`OSC_OVERSAMPLING = 2`), and the standard `lp12/lp24/hp12/hp24/bp` types use a
**two-sub-step SVF at that 2× rate**; notch is a separate biquad and is already
close. The coefficient/limiter laws are:

- `F1 = 2·sin(π·min(0.11, f / (2·fs_os)))` where `fs_os = 2·base`; the cutoff
  clamp is `f < 0.22·fs_os = 0.44·base` (21.1 kHz at 48 kHz).
- `Q1 = 2 − √reso·(2 + overshoot) + F1²·overshoot·0.9`, clipped to
  `min(2, 2 − 1.52·F1)`; `overshoot` is 0.15 for 2-pole, 0.10 for 4-pole.
- `Gain = 1 − 0.65·√reso` on the filter output.
- A **state-scaling limiter** `R = max(0.1, 1 − 0.1·√reso·F1·B²)` applied to the
  integrator states, using the band signal. This is what bounds self-oscillation
  in Surge. Lite instead adds `limit·|band|³` to the damping, which produces a
  much louder resonance.
- LP24/HP24 are two cascaded 2-pole SVF sections sharing the same coefficients
  and one limiter; LP12/BP are one section (two sub-steps).

Before this change, Lite used a single-rate TPT SVF with `g = tan(πf/sr)`, a
fitted damping curve, `k ≥ 0.3` on the first 24 dB stage, and a `|B|³`
damping limiter. That is why LP12/HP12/BP at high cutoff and res 0.9–0.99
were up to 10 dB too loud and too bright. Notch was already within 1 dB.

Change:

1. Add a `dsp::Svf2x` in `source/engine/lite/dsp/VoiceDsp.h` that runs the
   two-sub-step SVF with the coefficient/limiter laws above (public TPT SVF
   math, independently implemented).
2. Rebuild `FilterUnit::configure/process` on top of it: 12 dB = one section,
   24 dB = two sections, BP = band output, notch = the existing low+high
   (re-verify).
3. Implement the 2× filter at base rate: each base sample performs two OS
   sub-steps with a zero-order-hold input and a box-average output, using the
   `fs_os` coefficients. This matched the high-cutoff cases without the cost
   of a full 2× voice. A future full 2× voice (distinct OS oscillator samples)
   would remove the small low-cutoff resonance makeup.
4. Use `current + pitchBend` for keytrack, not the target `note`
   (`LiteVoice.h`: `keyOffset = float(note) - 60`). Surge keytracks the
   *gliding* pitch; this is a real bug for glide + keytrack patches.
5. Keep coefficient updates per control block, but ramp F1/Q1 across the block
   to avoid zipper noise during fast filter modulation.

### 3.3 Oscillators

Isolated tests say the basic waves are already good. Do not rewrite them
without evidence.

- **S&H noise (2.64 dB).** Surge's `ot_shnoise` is a correlated-noise S&H
  oscillator with BLIT-rendered (band-limited) steps, a width control, low/high
  cut, and a sync rate; the backend sets correlation 0 and sync +48 st. Lite
  is naive sample-and-hold at 24 values per cycle with no correlation or
  band-limiting. Implement correlated S&H with band-limited transitions and
  fit the hold rate/spectrum to Surge. Keep the current level (0.21 dB is fine).
- **FM (~1–1.5 dB).** The backend sets M1 amount 0.3, M1 ratio 2, and leaves
  M2 amount/ratio/feedback at 0. Surge uses a quadrature modulator (`RM1.r`), a
  phase offset, 2× oversampling, and a full-scale sine output. Lite uses a sine
  modulator, fixed 2:1, 0.68 rad, and a fitted ×2 oscillator gain. Check the
  modulator branch/phase (quadrature vs sine), the index law (`d³·8π`), and the
  output gain; then re-fit the 0.68 rad and `kHotOscGain` for FM only. Surge's
  FM2 randomises its start phase, which moves its RMS by up to ~2 dB between
  renders, so compare the median of several renders. Do not change the
  sine/wavetable gain to fix FM.
- **Wavetable (1.64 dB).** Surge reads the table with a windowed-sinc FIR at
  2× OS and mip levels, and morphs between frames; Lite reads one frame with
  linear interpolation at base rate. With the shared one-frame table, the
  remaining difference is interpolation/mip selection. Move the read to
  cubic Hermite (or a short windowed-sinc) and align the mip selection with
  Surge's `mipmap` rule. Check that the table normalisation and phase match.
- **Saw/square/triangle/sine.** Keep as regression probes. If Phase 3 stays at
  base rate (the CPU fallback), a high-note aliasing test may justify 2× OS for
  the classic waves; if Phase 3 runs the voice at 2×, they are already there.
  This is a stretch goal, not the main gap.
- **Unison/drift/character.** The schema does not expose them; Surge's default
  patch has one voice, no drift and neutral character, so Lite is right to
  omit them. Do not add them in this plan.

### 3.4 Voice, modulation and global details

These are small but cheap:

- **Pan law.** Surge uses `L = 1 − 0.75p − 0.25p²`, `R = 1 + 0.75p − 0.25p²`
  (both 1.0 at centre, 1.5 at full side). Lite uses equal-power
  `cos/sin·√2`. Match Surge's law, or document the deviation. The mono compare
  is only ~0.5 dB off at the extremes, so this is low priority.
- **Velocity.** Lite ignores MIDI velocity. Surge's default VCA velocity
  sensitivity is 0 dB (no effect) in the default patch, so this is currently
  correct; verify it once and add a test so a future default change is caught.
- **Envelope timing.** Surge updates its envelopes once per 32-sample block;
  Lite updates per sample. Shapes match (linear in the shaped domain, then
  x^power). The reported attack rank (0.46) is probably a *measurement* issue:
  "time to 90% of peak" can land on a filter sweep, distortion tail or reverb
  build-up. Add an FX-off, filter-open attack sweep before changing the
  envelope; only match the block rate if the sweep shows a real difference.
- **LFO smooth random.** Surge's `lt_noise` is filtered/correlated noise, not
  cosine-interpolated S&H. Replace `LfoShape::SmoothRandom` with a one-pole
  filtered noise at the LFO rate, and align the sine/triangle phase
  conventions (`sine` unipolar starts +0.75 cycle; triangle non-unipolar
  starts +0.25) and the keytrigger/freerun phase handling. Verify the LFO
  envelope defaults (delay/hold/decay/sustain/release): only fade-in is
  exposed, but the unexposed stages still run in Surge.
- **DC / output.** Add a 20 Hz DC blocker at the voice or master if the new
  filter/oversampler introduces offset. Surge's oscillators have their own
  integrator high-pass; Lite's PolyBLEP waves do not need it.
- **Output hard clip.** Surge's default is `HARDCLIP_TO_18DBFS` at both the
  scene and global outputs (±7.94), not the current Lite `kOutputLimit = 4.0`
  (±12 dBFS). For loud patches this changes the waveform and the measured
  level. Match the default mode (and keep the existing NaN recovery).
- **Calibration.** `kVoiceGain = 0.25` and `kSceneVolume = 0.890899` were fitted
  to the old chain. Re-fit once at the end against the same Init Saw probe;
  don't tune per effect.

### 3.5 Out of scope for this plan

- The schema, the preset importer and the retrieval/VAE side. They account for
  the larger ~11.5 dB "schema loss"; schedule that separately.
- Changing the Surge backend or its effect defaults. It is the oracle.
- Adding new schema controls (unison, drift, character, extra FX parameters).
  Replicate Surge's defaults instead.
- Rewriting sine/saw/square/sine-level code that already measures within 1 dB.
  Revisit only if the high-note aliasing test fails.

---

## 4. Implementation phases

Each phase ends with the component sweep plus the full-preset compare, so the
next phase starts from a measured baseline. Suggested order is impact-first.
The phase checklists below are the original plan; Phases 1–3 (effects and
filters) are implemented and the results are in the status line above.

### Phase 0 — Make the gap visible (½ day)

- [ ] Add `tools/component_sweep.py` (Appendix A) that emits defaults + one
      component override per line; include filter resonance/cutoff sweeps,
      every oscillator wave, every FX on/off and the unexposed defaults.
- [ ] Add `tools/parity_report.py` that reads the compare CSV and prints a
      per-component table (or extend `tools/compare_report.py` with a
      `--by-component` mode). Keep the existing full-preset report untouched.
- [ ] Add a short "component sweep" section to `docs/COMPARE.md`.
- [ ] Record the current table from section 1 in the PR description.
- [ ] For stochastic probes (noise, FM, any random phase), render three times
      and use the median; a single render can move by 1–2 dB.

**Gate:** the sweep is reproducible from a clean build; the current numbers
match section 1 within noise.

### Phase 1 — Effect parameter laws (2–3 days)

Files: `source/engine/lite/LiteFx.h`, `source/engine/lite/dsp/FxBlocks.h`,
`source/engine/lite/LitePatch.h/.cpp`.

- [ ] Shared blocks: 2-pole LP/HP, Orfanidis peaking EQ, equal-power pan,
      soft/hard clip, 4× oversampler, mono-sum helper. Allocate all state in
      `prepare()`, never on the audio thread. Every block must accept arbitrary
      span lengths: the engine splits spans at MIDI events, so `n` is not
      always the control block or 256.
- [ ] Keep the existing 20 ms control smoothing and the discrete-change duck.
      Add per-block ramps for FX parameters that can move (drive, tone, mix,
      delay time, feedback) so dragging a control does not click. Reset the
      mod-FX state when phaser/flanger switches type.
- [ ] Delay: feedback ×³, 2-pole HP 110 Hz + LP 2.49 kHz, soft clip, linear
      mix, time glide. Add a test that measures echo 2 / echo 1 at feedback
      0.5 ≈ 0.125 and the filter −3 dB points.
- [ ] Chorus: 4 voices, 15.625 ms, triangle phases, 1/√4 gain, 55 Hz/3.52 kHz
      filters, feedback 0.0625, equal-power pans, linear mix. Linear delay
      interpolation is fine here; upgrade only if the residual is above target.
- [ ] EQ: three Orfanidis peaks, BW 2, fixed outer frequencies. Test the
      magnitude response at +12/−12 dB.
- [ ] Distortion: pre-LP 440 Hz, post-LP at `toneHz`, DC offset, rational
      tanh. Test harmonics at 1 kHz. (The 0 dB pre/post peaking EQ can wait
      until exposed; 4× oversampling is optional, the measured gap is ~0.4 dB.)

**Gate:** delay ≤2 dB and level ≤1 dB; chorus ≤2 dB and level ≤1 dB;
EQ ≤1.5 dB; distortion ≤2 dB and level ≤1 dB. Full median target ≤2.8 dB.

### Phase 2 — Phaser, flanger and Reverb 2 (2–3 days)

- [ ] Phaser: 4 stages at 440 Hz with `2/(i+1)` octave spans, feedback 0.95,
      clamp ±32, linear mix, same LFO phase in both channels at stereo 0;
      verify tone/width defaults.
- [ ] Flanger: 4 combs at pitch 60, spacing 0, additive mix (`dry + mix·wet`),
      damping, feedback scale, soft clip, output gain compensation. Short combs
      need cubic (or windowed-sinc) interpolation; the existing linear read
      will be too lossy at 3–4 ms.
- [ ] Reverb: new `source/engine/lite/dsp/Reverb2.h` implementing the clean-room
      FDN (pre-delay, input diffusion, 4 blocks, dampers, modulated taps,
      decay). Fit to Surge impulse responses: RT60 per octave, pre-delay and
      wet level. Allocate in `prepare()` and use an O(1) restart like
      `DelayLine::restart`; do not clear large buffers on the audio thread.
      Note that `FxSettings::reverbDecay` currently stores the normalized
      0..1 control; convert it to Surge's log2-seconds (`sc::toUnits`) before
      applying the FDN formula.
- [ ] Add an impulse-response test: render a 1-sample impulse through each
      reverb, compare RT60 in octave bands (target ±10 %), pre-delay
      (target ±2 ms), and wet RMS (target ±1 dB).

**Gate:** phaser/flanger ≤2 dB; reverb probe ≤2 dB and the IR tests pass; full
median target ≤2.4 dB.

### Phase 3 — Filter high-resonance parity (2–3 days)

- [ ] Add `dsp::Svf2x` and rebuild `FilterUnit` (12/24 dB, BP/notch).
- [ ] Implement Surge's state-scaling limiter and gain law; remove the old
      `|B|³` damping fit.
- [ ] Keytrack from `current + pitchBend`, not target note.
- [ ] Ramp coefficients across control blocks.
- [ ] Add white-noise filter tests at res 0, 0.5, 0.9, 0.99 and cutoffs
      0.3/0.55/0.75/0.95; compare magnitude and level.

**Gate:** every filter type within 1.5 dB and 1.5 dB level at res ≤0.99;
no self-oscillation divergence; full median target ≤2.1 dB. Because the FX
chain is post-filter, re-run the effect sweep after this phase and re-check
the effect levels; only re-fit if the change is above 0.5 dB.

### Phase 4 — Oscillators and voice polish (2 days)

- [ ] S&H noise: correlated noise, band-limited steps, sync/hold rate; fit
      spectrum and flatness.
- [ ] FM: modulator phase, index law, output gain (2× OS if Phase 3 stayed at
      base rate); fit index/level.
- [ ] Wavetable: cubic/windowed-sinc read, mip selection, phase/normalisation.
- [ ] LFO smooth-random + phase conventions.
- [ ] Pan law; velocity test; DC blocker if needed.
- [ ] Output hard clip: match Surge's default ±18 dBFS (Lite currently ±12).
- [ ] Re-fit `kVoiceGain` and `masterGain` once.

**Gate:** noise/FM/wavetable ≤1.2 dB; sine/saw/square/triangle no regression
(≤0.9 dB); full median ≤2.0 dB.

### Phase 5 — Verification and documentation (1 day)

- [ ] Full 3,561-preset compare: median `frame_mel_db` ≤2.0 dB, p90 ≤5 dB,
      median `d_level_db` ≤0.5 dB, no silence mismatches.
- [ ] Listening pass over the 20 worst presets and a fixed 20-patch set
      (bass, pad, lead, pluck, keys, noise/FX).
- [ ] `vae_engine_tests` under ASan and TSan; `pluginval --strictness-level 5`
      on the release VST3; `auval` on the AU.
- [ ] CPU benchmark: record the current Lite cost (16 voices, all FX on) and
      keep the new build within 2×; if it exceeds that, optimise before
      merging.
- [ ] Update `docs/FIXES-lite-vs-surge.md`, `docs/COMPARE.md` and
      `docs/LICENSING.md` (clean-room record and any new fitted constants).
- [ ] Update the plugin's "Lite" description if its known-differences list
      changes.

---

## 5. Acceptance criteria

| Metric | Today | Target |
|---|---:|---:|
| Full-preset median `frame_mel_db` (svec→lite) | ~3.5 dB | **≤2.0 dB** |
| Full-preset p90 `frame_mel_db` | — | **≤5 dB** |
| Median `d_level_db` | ~1.2 dB | **≤0.5 dB** |
| Component baseline (saw, no FX, filter off) | 0.66–0.78 dB | **≤0.8 dB, no regression** |
| Sine/triangle/saw/square | 0.12–0.81 dB | **≤0.9 dB** |
| S&H noise / FM / wavetable | 1.5–2.6 dB | **≤1.2 dB** |
| Filter, all types, res ≤0.99 | up to 10.2 dB | **≤1.5 dB, level ≤1.5 dB** |
| Each effect, isolated | 2.8–13.8 dB | **≤2 dB, level ≤1 dB** |
| Reverb RT60 per octave | — | **±10 %**, pre-delay ±2 ms |
| CPU (16 voices, all FX) | measure | **≤2× current** |
| Silence mismatches | 21 | **no regression** |

The target is deliberately not "0 dB". Random oscillator start phase, the
schema's own loss and the mel metric's resolution put a floor around 1–2 dB.
The important thing is that every component is in the same "kind of sound"
band and no component is an outlier.

---

## 6. Risks and open decisions

1. **Clean-room vs GPL.** If exact parity is required, linking
   `sst-filters`/`sst-effects` is cheaper but relicenses the Lite build.
   Decide before Phase 1, because the implementation strategy differs.
2. **CPU.** A 2× filter emulation + 4× distortion + FDN reverb + 4-voice
   chorus is a real cost. Measure early (Phase 1) and keep the "only when on"
   rule: distortion oversamples only when enabled, reverb only when enabled.
   The implemented 2× filter uses two OS sub-steps per base sample (no full
   2× voice); a future full 2× voice would remove the low-cutoff makeup.
3. **Random phase.** Lite and Surge randomise oscillator start phase
   independently, so sample-level null tests are meaningless. Compare spectra,
   envelopes and statistics, not waveforms.
4. **The mel metric.** It can reward a matching average spectrum while
   missing transients and phase. The reverb IR and envelope tests are there to
   cover that.
5. **The compare is mono.** `vae_compare` sums L/R, so chorus/reverb width is
   invisible. Add L/R correlation and side-energy checks for the stereo
   effects so a mono match does not hide a width error.
6. **The schema ceiling.** The schema already loses ~11.5 dB on factory
   presets. Do not promise "factory presets sound identical" from engine work
   alone; schedule the importer/routing work as a separate plan.
7. **Effect defaults are part of the contract.** If someone later exposes a
   currently fixed parameter (e.g. chorus time or delay high cut), the Lite
   defaults must change with the schema, and the component sweep must be
   updated.

---

## Appendix A — Component sweep

The sweep builds vectors from the schema defaults, changes one part at a time,
and writes one JSONL line per probe. `vae_compare` still renders all three
backends; the `orig` render fails harmlessly because the `.fxp` paths don't
exist, so only `svec` and `lite` matter.

```python
#!/usr/bin/env python3
"""Emit one JSONL control vector per component probe."""
import json
from pathlib import Path

schema = json.load(open("schema/controls.json"))
controls = schema["controls"]


def defaults():
    out = []
    for c in controls:
        d = c["default"]
        out.append(float(c["options"].index(d)) if c.get("options") else float(d))
    return out


BASE = defaults()
ROWS = []


def add(name, **overrides):
    v = BASE[:]
    for key, value in overrides.items():
        i = next(i for i, c in enumerate(controls) if c["name"] == key)
        if controls[i].get("options") and isinstance(value, str):
            value = controls[i]["options"].index(value)
        v[i] = float(value)
    ROWS.append({"file": f"component/{name}.fxp", "values": v, "lossy": []})


# Oscillators
add("baseline")
for wave in ["sine", "triangle", "saw", "square", "noise", "wavetable", "fm"]:
    add(f"osc-{wave}", **{"Osc1 Wave": wave})

# Filters: type x cutoff x resonance
for ft in ["lp12", "lp24", "hp12", "hp24", "bp", "notch"]:
    for cut in [0.35, 0.55, 0.70, 0.92]:
        for res in [0.0, 0.75, 0.9, 0.99]:
            add(f"filt-{ft}-c{cut}-r{res}",
                **{"F1 Type": ft, "F1 Cutoff": cut, "F1 Reso": res})

# Distortion, EQ, chorus, mod FX, delay, reverb
for drive in [0.25, 0.583333, 0.9]:
    add(f"dist-drive{drive}", **{"Dist On": 1, "Dist Drive": drive})
for tone in [0.2, 0.46, 0.7, 0.9]:
    add(f"dist-tone{tone}", **{"Dist On": 1, "Dist Tone": tone})
for band, val in [("EQ Low", 0.75), ("EQ Low", 0.25), ("EQ Mid", 0.75),
                  ("EQ High", 0.75), ("EQ High", 0.25)]:
    add(f"eq-{band.replace(' ', '')}-{val}", **{"EQ On": 1, band: val})
for depth in [0.3, 0.8]:
    add(f"chorus-d{depth}", **{"Chorus On": 1, "Chorus Depth": depth})
for mix in [0.5, 1.0]:
    add(f"chorus-m{mix}", **{"Chorus On": 1, "Chorus Mix": mix})
for depth in [0.0, 0.5, 1.0]:
    add(f"phaser-d{depth}",
        **{"ModFX On": 1, "ModFX Type": "phaser", "ModFX Depth": depth})
for mix in [0.25, 0.5, 1.0]:
    add(f"phaser-m{mix}",
        **{"ModFX On": 1, "ModFX Type": "phaser", "ModFX Mix": mix})
for mix in [0.5, 0.75, 1.0]:
    add(f"flanger-m{mix}",
        **{"ModFX On": 1, "ModFX Type": "flanger", "ModFX Mix": mix})
for fb in [0.0, 0.5, 0.85]:
    add(f"delay-fb{fb}", **{"Delay On": 1, "Delay Fdbk": fb})
for mix in [0.25, 0.5, 0.8]:
    add(f"delay-m{mix}", **{"Delay On": 1, "Delay Mix": mix})
for size in [0.0, 0.5, 1.0]:
    add(f"reverb-size{size}", **{"Reverb On": 1, "Reverb Size": size})
for dec in [0.2, 0.475, 0.8]:
    add(f"reverb-dec{dec}", **{"Reverb On": 1, "Reverb Decay": dec})
for mix in [0.15, 0.33, 0.6]:
    add(f"reverb-m{mix}", **{"Reverb On": 1, "Reverb Mix": mix})

out = Path("build-surge/component-sweep/vectors.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w") as f:
    for row in ROWS:
        f.write(json.dumps(row) + "\n")
```

Quick analysis:

```sh
python3 - <<'PY'
import csv, os, statistics
rows = list(csv.DictReader(open("build-surge/component-sweep/compare.csv")))
def f(r, k):
    v = r.get(k, "")
    return float(v) if v not in ("", None) else float("nan")
for r in rows:
    print(f"{os.path.basename(r['file'])[:-4]:32} "
          f"{f(r,'sl_frame_mel_db'):6.2f} {f(r,'sl_mel_db'):6.2f} "
          f"{f(r,'sl_d_level_db'):6.2f} {f(r,'sl_d_centroid_oct'):6.2f}")
print("median", statistics.median(f(r, "sl_frame_mel_db") for r in rows))
PY
```

Note: the sweep deliberately uses default vectors with one change, so it
measures the *engine* difference. For the factory-preset metric, keep using
`tools/surge_preset_to_vector.py`; the two views answer different questions.
