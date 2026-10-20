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
| triangle | Modern osc, triangle mix only | Modern osc where triangle dominates |
| saw | Classic, shape 0 | Classic with shape > −0.5; String / Twist / Alias (lossy) |
| square | Classic, shape −1, width 0.5 | Classic with shape ≤ −0.5; Modern where pulse dominates |
| noise | S&H Noise, correlation 0, sync +48 st (hold rate tracks the note) | S&H Noise osc |
| wavetable | Wavetable osc on a fixed one-frame additive table shared with Lite (headless Surge has no factory tables) | Wavetable, Window |
| fm | FM2 osc, M1 amount 0.3, ratio 2 (its defaults are a pure sine) | FM2, FM3 |

Classic's shape morphs square (−1) → saw (0). Surge's *Init Square*
template uses shape −1.

## Filters

Types `off, lp12, lp24, hp12, hp24, bp, notch` render as Surge filter types
0, 1, 2, 4, 5, 6, 7. On import, the other Surge types fold by family:
ladders, OB-Xd, K35, diode and warp lowpasses → `lp24`; their highpass, band
and notch variants likewise; multi-mode filters → their usual LP setting;
effect types (comb, S&H, allpass) → `off`. Each fold is reported as lossy.

`filter_routing` renders as Surge **Serial 1** or **Dual 1** (measured):

- serial: oscs routed to `filter1` / `both` go through F1 then F2; `filter2` skips F1.
- parallel: each filter path carries the oscs routed to it at −6 dB, and the
  paths are summed.
- In both, a filter set to `off` passes its oscs through unfiltered.

Surge's template default *Wide* ignores per-osc routing, so it is not used
for rendering. On import, Serial 1–3 → `serial` and every other configuration
→ `parallel` (lossy unless Serial 1 or Dual 1). The Lite engine copies this
behaviour.

## Effects

One unit of each type in a fixed order. On render they go into fixed Surge
slots: A1 Distortion, A2 EQ, A3 Chorus, A4 Phaser/Flanger, Global 1 Delay,
Global 2 Reverb 2.

On import, the importer takes the first *audible* effect of each type along
the chain (scene inserts → sends → globals):

- Slots switched off in Surge's `fx_disable` mask are skipped.
- A send effect's mix is scaled by how much reaches it: scene send level³ ×
  send return level³. Surge applies both as amplitude cubed. A send with
  nothing sent to it is skipped.
- An effect whose resulting mix is 0 is inaudible (Surge loads it that way
  too), so it is labelled off with its controls at their defaults. That
  avoids two labels, "on but silent" and "off", for one sound.
- Reverb 1 maps onto the Reverb 2 controls (lossy). Other effect types are
  dropped (lossy).

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
of everything that did not map exactly. A file that cannot be read is
reported and skipped; the batch continues. Modulation sources in patches older
than Surge revision 9 are renumbered the way Surge does on load. It uses the active scene (scene A
unless scene B is active in single-scene mode). Split and dual scene modes use
scene A only and are flagged. The noise mixer channel is not in the schema: it
is imported into the first muted osc slot as `noise` (flagged), or flagged and
dropped when no slot is free. Ring modulators are flagged when used. An osc
level or amp/filter EG sustain driven by modulation gets its strongest positive
depth added (clamped to 1, flagged), so patches that sit at 0 and are opened by
an LFO, aftertouch or a controller do not import as silent.
