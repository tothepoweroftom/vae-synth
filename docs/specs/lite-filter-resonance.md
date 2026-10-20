# Lite filter resonance — functional specification

Status: clean-room spec, implemented. This document describes **required behaviour only**.
The implementation in `source/engine/lite/LiteDsp.h` (`FilterUnit`) must be
written from this document and from measurements of rendered audio, without
reading Surge XT source code (`build-surge/_deps/surge-src/` or any copy of
it). Lite's filter core is the TPT state-variable filter in
`source/engine/lite/dsp/VoiceDsp.h` (Zavalishin, "The Art of VA Filter
Design"; Simper's SVF notes); the resonance mapping, limiting and gain are
this project's own and are fitted by measurement.

## 1. Goal

For the same control vector, Lite's filter should sound like the Surge
backend's filter (`VAE_WITH_SURGE=ON`, rendered by `vae_compare`), so the
engine gap in `docs/COMPARE.md` shrinks for the filter types. Surge is
treated as a black box: only its rendered output is used.

## 2. Required behaviour

Applies to lp12, lp24, hp12, hp24 and bp. Notch keeps its current mapping
unless measurement shows a clear win.

1. **Self-oscillation.** Near full resonance the filter rings on its own: a
   near-silent input (osc level ~-100 dB) through a high-resonance filter
   must produce audible, sustained output at the cutoff frequency. The
   resonance value where ringing starts should match the Surge backend's,
   measured per type (2-pole and 4-pole may differ).
2. **Bounded.** Ringing settles at a finite level and never produces NaN or
   Inf, at any cutoff, sample rate (44.1-96 kHz) or block size, including
   while resonance and cutoff are modulated.
3. **Level vs resonance.** The filter's overall output level as resonance
   rises should follow the Surge backend's (it is allowed to drop as
   resonance rises if Surge's does).
4. **24 dB types.** Choose the stage arrangement (which stages resonate) that
   best matches Surge's measured response.
5. **Cost.** Allocation-free, per sample. Coefficients may be set per control
   block (as now).

## 3. Acceptance

- `vae_engine_tests` passes, including "high resonance self-oscillates from a
  near-silent input and stays finite".
- On the full corpus (`docs/COMPARE.md`, "Run it"), the median engine gap
  (spectrum over time) does not get worse than 3.5 dB, and the per-filter-type
  rows in "Where Lite diverges most" improve or hold.
- Constants are documented as measured fits ("fitted to Surge renders"), not
  as Surge values.

## 4. Result

Damping falls with resonance as a fitted power curve and goes below zero
near full resonance (later for the 4-pole types); extra damping proportional
to |band|^3 bounds the ringing; output gain falls with resonance. In the
24 dB types only the second stage self-oscillates (the first keeps damping
>= 0.3). Full corpus, median engine gap: spectrum over time 3.52 dB (3.82
with the previous Lite filter), loudness 1.23 dB. All tests pass.
