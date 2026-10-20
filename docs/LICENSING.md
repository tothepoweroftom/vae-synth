# Licensing notes

Not legal advice. Get a review before distributing binaries.

| Component | License | Where |
|---|---|---|
| JUCE 8 | AGPLv3 or commercial JUCE licence | fetched |
| ONNX Runtime | MIT | fetched; shared library shipped next to the plug-in |
| Surge XT (`surge-common` and its libs) | GPL-3.0-or-later | fetched only with `VAE_WITH_SURGE=ON` |
| Lite DSP (`source/engine/lite/dsp/`) | project code | oscillator, filter, noise, wavetable builder, biquad, delay line |
| Inter font | SIL OFL 1.1 (`assets/fonts/Inter-LICENSE.txt`) | vendored |
| Parameter store (`source/params/`) | this project | clean-room implementation from `docs/specs/parameter-manager.md` |
| Lite filter resonance (`FilterUnit` in `source/engine/lite/LiteDsp.h`) | project code | clean-room fit from `docs/specs/lite-filter-resonance.md` |

## Build variants

- **Default (`VAE_WITH_SURGE=OFF`)**: contains no Surge code. The schema
  stores Surge parameter *names* and `SurgeCurves.h` re-implements Surge's
  documented value ranges; no Surge source is copied.
- **`VAE_WITH_SURGE=ON`**: the plug-in links Surge XT, so a distributed binary
  is a combined work under the GPL-3.0-or-later. Use it for research and
  dataset rendering, or ship it as GPL with corresponding source.

## Parameter store clean room

`docs/specs/parameter-manager.md` was written from a feature review of an
existing proprietary parameter manager. It describes behaviour only. The
implementation was written by a separate agent given only that spec and the
JUCE sources, with no access to the original code.

## Lite filter clean room

Lite's filter core is the TPT state-variable filter from the published
literature (Zavalishin, Simper). Its resonance mapping, self-oscillation
limiting and gain-vs-resonance were fitted by an agent working only from
`docs/specs/lite-filter-resonance.md` and measurements of rendered audio,
told not to read Surge source. An earlier version that transcribed Surge's
filter coefficients was deleted before it was ever committed. The agent was
stopped before its final report, so there is no written confirmation from it;
its final constants were chosen by a sweep over the same measured renders.

Other Lite constants that match Surge's behaviour (the ~6 dB higher level of
the Sine, Wavetable and FM2 oscs, the S&H noise hold rate, random osc start
phase, the FM index for the backend's FM2 settings) are behaviour facts,
measured from renders and in some cases checked against the Surge source; no
Surge code is copied. Like `SurgeCurves.h`, they describe what Surge does,
not how it is written.
