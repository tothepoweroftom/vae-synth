# Control schema v2

`schema/controls.json` is the single source of truth. CMake generates the C++
table from it (`cmake/VaeSchema.cmake`), and `tools/vae_schema.py` reads it for
the Python side. v2 follows the student's component list:

| Section | Controls |
|---|---|
| Voice | `play_mode` (poly / mono / legato), `glide_time` |
| Osc 1–3 | `on`, `wave`, `level`, `octave`, `pitch`, `filter` (which filter) |
| Filters | `filter_routing` (serial / parallel); per filter: `type`, `cutoff`, `resonance`, `env`, `keytrack` |
| Envelopes | amp and filter EG: A/D/S/R + a curve per stage |
| LFO 1–2 | `shape`, `rate`, `amount`, `unipolar`, `retrigger`, `phase`, `fade_in` |
| FX (fixed chain) | distortion → EQ → chorus → phaser/flanger → delay → reverb |
| Global | `master_volume` |
| *Provisional* | `feg/lfo1/lfo2_destination` + `_depth` (see [Modulation routing](#modulation-routing-provisional)) |

93 controls, 191 model features.

## Values

| kind | raw value (plug-in, engines, importer) | model features |
|---|---|---|
| continuous | 0..1, linear in its curve's Surge units | 1, in 0..1 |
| toggle | 0 or 1 | 1, in 0..1 (≥ 0.5 = on) |
| categorical | option index | one per option (one-hot in, argmax out) |

A continuous value is exactly Surge's normalised value of the mapped Surge
parameter (`Parameter::get_value_f01`). The `curves` table gives each curve's
range in Surge units, e.g. `cutoff` = −60..70 semitones from A440,
`envtime` = log2 seconds −8..5.

Features are concatenated in schema order; each control's offset is
`firstFeature` (C++) / `first_feature` (Python). `encodeFeatures` /
`decodeFeatures` and their Python twins are exact inverses.

### Decoder contract (ONNX)

- input: one float tensor `[batch, latent_dimensions]` (8)
- output: first output float `[batch, 191]` in feature order. Continuous and
  toggle features in 0..1. Categorical features are scores (probabilities
  or logits); the plug-in takes the argmax per control.

`models/make_stub_decoder.py` writes a placeholder with exactly this contract.

## Defaults

The defaults are Surge's *Init Saw* template, which equals Surge's built-in
patch defaults. That means one saw osc and both filters **off**.
`tests/python/test_schema_tools.py` checks that importing Init Saw gives
exactly the defaults.

## Oscillator waves

| schema | Surge render (engine) | Surge import (sampler) |
|---|---|---|
| sine | Sine osc | Sine osc |
| triangle | see `source/engine/surge/SurgeEngine.h` | Modern osc where triangle dominates |
| saw | Classic, shape 0 | Classic with shape > −0.5; String / Twist / Alias (lossy) |
| square | Classic, shape −1, width 0.5 | Classic with shape ≤ −0.5; Modern where pulse dominates |
| noise | see `SurgeEngine.h` | S&H Noise osc |
| wavetable | Wavetable osc (default table) | Wavetable, Window |
| fm | FM2 osc | FM2, FM3 |

Classic's shape morphs square (−1) → saw (0). Surge's *Init Square*
template uses shape −1.

## Filters

Types `off, lp12, lp24, hp12, hp24, bp, notch` render as Surge filter types
0, 1, 2, 4, 5, 6, 7. On import, the other Surge types fold by family:
ladders, OB-Xd, K35, diode and warp lowpasses → `lp24`; their highpass, band
and notch variants likewise; multi-mode filters → their usual LP setting;
effect types (comb, S&H, allpass) → `off`. Each fold is reported as lossy.

`filter_routing`: Surge Serial 1–3 → `serial`, every other configuration →
`parallel`. The exact Surge configuration used for rendering is documented in
`SurgeEngine.h`.

## Effects

One unit of each type in a fixed order. On render they go into fixed Surge
slots: A1 Distortion, A2 EQ, A3 Chorus, A4 Phaser/Flanger, Global 1 Delay,
Global 2 Reverb 2. On import, the importer takes the first effect of each type
along the chain (scene inserts → sends → globals). A send effect's mix is
scaled by the scene's send level. Reverb 1 maps onto the Reverb 2 controls
(lossy). Other effect types are dropped (lossy).

## Modulation routing (provisional)

**Pending the student's decision.** For now each modulator gets one route:
`*_destination` ∈ {none, pitch, filter1_cutoff, filter2_cutoff,
filter1_resonance, volume, pan} plus `*_depth` (bipolar, d = 2v − 1). At full
modulator output (LFO ±1, or 0..1 when unipolar, × amount; filter EG 0..1) the
destination moves by **d × the full range of its Surge parameter**, in that
parameter's units. That is the same as Surge's stored mod depth divided by
the parameter range. On import, the strongest supported route per source wins
and the rest are reported as dropped.

The filter EG also always drives both cutoffs through `filter1_env` /
`filter2_env` (Surge's FEG Mod Amount), which is where most Surge presets put
it.

What the Surge factory library says (641 presets, `--stats`):

- LFO 1 → pitch is the most common route (199 presets), then LFO 1 →
  filter 1 cutoff (34). LFO 2 → pitch: 71.
- An extra filter-EG route beyond the two built-in amounts is rare (5
  presets).
- Many presets have more than one route per LFO: 179 presets drop at least
  one LFO 1 route.

Options for the final design: (a) keep one route per modulator, (b) a fixed
set of depth-only routes (e.g. LFO 1/2 → pitch, cutoff 1, cutoff 2, volume,
pan), or (c) N slots of (source, destination, depth).

## Import tool

```sh
python3 tools/surge_preset_to_vector.py <preset or dir>... -o vectors.jsonl --stats
```

Each output line has `values` (raw), `features` (model) and `lossy`, the list
of everything that did not map exactly. It uses the active scene (scene A
unless scene B is active in single-scene mode). Split and dual scene modes use
scene A only and are flagged. The noise mixer channel and ring modulators are
not in the schema and are flagged when used.
