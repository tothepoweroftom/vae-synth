# Prompt listening sheet (experimental): p0 and p1

Status: **experimental**. The installed model is now **p1** (Task 21); the p0 results are kept for comparison. This sheet records what Find returns and gives you WAVs to compare by ear. Nothing
here judges sound quality; the numbers are cheap objective measures only. The feature stays labelled "experimental"
(UI tag next to the Find toggle, and the docs) until your notes support semantic claims.

## p1 vs p0 (same 12 prompts, same renderer)

| | p0 | p1 |
|---|---|---|
| Bank rows | 1,227 (Bass 485, Lead 447, Pad 295) | 4,429 (Atmosphere 751, Synth 693, Orchestral 629, Bass 527, Lead 494, Keys 474, Pluck 468, Pad 393) |
| Categories returned for the 12 prompts | Bass, Lead, Pad only ("plucky bell" gave Bass; "warm electric piano keys" and "evolving texture" gave Pad) | Bass, Lead, Pad, Pluck ("plucky bell"), Keys ("warm electric piano keys"), Atmosphere ("evolving texture") |
| dark vs bright bass, first row | 80 and 80 (same) | 131 and 53 (differ) |
| short vs long pad, first row | 1129 and 1129 (same) | 1167 and 1001 (differ) |
| soft vs sharp lead, first row | 892 and 892 (same) | 892 and 892 (same) |
| Prompts falling back for no bank rows | none | none (all 8 categories have rows) |

Objective only: two of the three contrasting pairs now select different first rows with p1, the lead pair still does
not, and the prompts naming Pluck, Keys and Atmosphere return rows of that category. Whether the rows sound like the
prompts is for the listening notes. The sections from "How it was made" to the p0 coverage table are the earlier p0
run; the p1 run is at the end.

## How it was made

`vae_p0_eval eval` (`tools/p0_eval/P0EvalMain.cpp`, see docs/JOINT-VAE-LAB.md) for each of 12 fixed prompts builds
three starts and renders each through Lite and Surge:

- **p0 Find**: the first candidate of the learned search (a real bank row; its exact stored settings).
- **typical**: the category-only start, `Typical <p0's predicted category>` decoded by the Lab decoder.
- **word list**: `interpretPrompt` start and nudges (each nudge applied alone; the panel also holds the pad's other
  axes), decoded.

Render path: `JointVae::patch` gives the Surge patch; Lite plays `liteControls(patch)` (what the Lab plays), Surge
plays the same patch with `SurgeEngine::loadPatchData`. Both use the engine classes from `source/engine` directly (the
same pattern as `tools/compare`): note 60, velocity 100, held 1.5 s, 2.2 s total, mono 48 kHz 16-bit WAV.
Files are in `build-surge/p0-eval/` (git-ignored, regenerate with the command above):
`<prompt>__<start>__<engine>.wav`.

Columns: **Bright** and **Length** are the XY-pad `measures` of the sound decoder's prediction at the start's z and
category (the Lab's own units, not Hz or seconds). **Level** is the rendered level from
`tools/compare` `analyse` (dB) and **Centroid** its spectral centroid. **Listener notes** is yours.

## Listening sheet

| Prompt | Start | Engine | WAV | Selected row / category | Bright | Length | Level | Centroid | Listener notes |
|---|---|---|---|---|---|---|---|---|---|
| dark bass | p0 Find | Lite | `build-surge/p0-eval/dark-bass__p0-Find__Lite.wav` | row 80 Bass · 0080 | -0.01 | 2.30 | -26.8 dB | 737 Hz |  |
| dark bass | p0 Find | Surge | `build-surge/p0-eval/dark-bass__p0-Find__Surge.wav` | row 80 Bass · 0080 | -0.01 | 2.30 | -18.1 dB | 313 Hz |  |
| dark bass | typical | Lite | `build-surge/p0-eval/dark-bass__typical__Lite.wav` | typical Bass | -0.22 | 2.09 | -24.1 dB | 241 Hz |  |
| dark bass | typical | Surge | `build-surge/p0-eval/dark-bass__typical__Surge.wav` | typical Bass | -0.22 | 2.09 | -21.6 dB | 191 Hz |  |
| dark bass | word list | Lite | `build-surge/p0-eval/dark-bass__word-list__Lite.wav` | Bass, 1 nudge | -0.37 | 2.09 | -24.6 dB | 226 Hz |  |
| dark bass | word list | Surge | `build-surge/p0-eval/dark-bass__word-list__Surge.wav` | Bass, 1 nudge | -0.37 | 2.09 | -20.8 dB | 168 Hz |  |
| bright bass | p0 Find | Lite | `build-surge/p0-eval/bright-bass__p0-Find__Lite.wav` | row 80 Bass · 0080 | -0.01 | 2.30 | -26.8 dB | 737 Hz |  |
| bright bass | p0 Find | Surge | `build-surge/p0-eval/bright-bass__p0-Find__Surge.wav` | row 80 Bass · 0080 | -0.01 | 2.30 | -18.1 dB | 313 Hz |  |
| bright bass | typical | Lite | `build-surge/p0-eval/bright-bass__typical__Lite.wav` | typical Bass | -0.22 | 2.09 | -24.1 dB | 241 Hz |  |
| bright bass | typical | Surge | `build-surge/p0-eval/bright-bass__typical__Surge.wav` | typical Bass | -0.22 | 2.09 | -21.6 dB | 191 Hz |  |
| bright bass | word list | Lite | `build-surge/p0-eval/bright-bass__word-list__Lite.wav` | Bass, 1 nudge | 0.04 | 2.09 | -24.5 dB | 249 Hz |  |
| bright bass | word list | Surge | `build-surge/p0-eval/bright-bass__word-list__Surge.wav` | Bass, 1 nudge | 0.04 | 2.09 | -21.8 dB | 204 Hz |  |
| short pad | p0 Find | Lite | `build-surge/p0-eval/short-pad__p0-Find__Lite.wav` | row 1129 Pad · 1129 | 0.77 | 4.48 | -40.3 dB | 546 Hz |  |
| short pad | p0 Find | Surge | `build-surge/p0-eval/short-pad__p0-Find__Surge.wav` | row 1129 Pad · 1129 | 0.77 | 4.48 | -36.1 dB | 569 Hz |  |
| short pad | typical | Lite | `build-surge/p0-eval/short-pad__typical__Lite.wav` | typical Pad | 0.88 | 4.27 | -33.8 dB | 432 Hz |  |
| short pad | typical | Surge | `build-surge/p0-eval/short-pad__typical__Surge.wav` | typical Pad | 0.88 | 4.27 | -33.5 dB | 434 Hz |  |
| short pad | word list | Lite | `build-surge/p0-eval/short-pad__word-list__Lite.wav` | Pad, 1 nudge | 0.84 | 3.97 | -30.2 dB | 344 Hz |  |
| short pad | word list | Surge | `build-surge/p0-eval/short-pad__word-list__Surge.wav` | Pad, 1 nudge | 0.84 | 3.97 | -34.0 dB | 446 Hz |  |
| long pad | p0 Find | Lite | `build-surge/p0-eval/long-pad__p0-Find__Lite.wav` | row 1129 Pad · 1129 | 0.77 | 4.48 | -40.3 dB | 546 Hz |  |
| long pad | p0 Find | Surge | `build-surge/p0-eval/long-pad__p0-Find__Surge.wav` | row 1129 Pad · 1129 | 0.77 | 4.48 | -35.7 dB | 572 Hz |  |
| long pad | typical | Lite | `build-surge/p0-eval/long-pad__typical__Lite.wav` | typical Pad | 0.88 | 4.27 | -33.8 dB | 432 Hz |  |
| long pad | typical | Surge | `build-surge/p0-eval/long-pad__typical__Surge.wav` | typical Pad | 0.88 | 4.27 | -32.4 dB | 398 Hz |  |
| long pad | word list | Lite | `build-surge/p0-eval/long-pad__word-list__Lite.wav` | Pad, 1 nudge | 0.92 | 4.52 | -30.6 dB | 377 Hz |  |
| long pad | word list | Surge | `build-surge/p0-eval/long-pad__word-list__Surge.wav` | Pad, 1 nudge | 0.92 | 4.52 | -32.9 dB | 436 Hz |  |
| soft lead | p0 Find | Lite | `build-surge/p0-eval/soft-lead__p0-Find__Lite.wav` | row 892 Lead · 0892 | 1.61 | 4.18 | -34.4 dB | 559 Hz |  |
| soft lead | p0 Find | Surge | `build-surge/p0-eval/soft-lead__p0-Find__Surge.wav` | row 892 Lead · 0892 | 1.61 | 4.18 | -44.4 dB | 223 Hz |  |
| soft lead | typical | Lite | `build-surge/p0-eval/soft-lead__typical__Lite.wav` | typical Lead | 1.14 | 2.69 | -27.1 dB | 405 Hz |  |
| soft lead | typical | Surge | `build-surge/p0-eval/soft-lead__typical__Surge.wav` | typical Lead | 1.14 | 2.69 | -26.9 dB | 396 Hz |  |
| soft lead | word list | Lite | `build-surge/p0-eval/soft-lead__word-list__Lite.wav` | Lead, 1 nudge | 1.13 | 2.90 | -26.7 dB | 416 Hz |  |
| soft lead | word list | Surge | `build-surge/p0-eval/soft-lead__word-list__Surge.wav` | Lead, 1 nudge | 1.13 | 2.90 | -26.8 dB | 399 Hz |  |
| sharp lead | p0 Find | Lite | `build-surge/p0-eval/sharp-lead__p0-Find__Lite.wav` | row 892 Lead · 0892 | 1.61 | 4.18 | -34.4 dB | 559 Hz |  |
| sharp lead | p0 Find | Surge | `build-surge/p0-eval/sharp-lead__p0-Find__Surge.wav` | row 892 Lead · 0892 | 1.61 | 4.18 | -44.3 dB | 226 Hz |  |
| sharp lead | typical | Lite | `build-surge/p0-eval/sharp-lead__typical__Lite.wav` | typical Lead | 1.14 | 2.69 | -27.1 dB | 405 Hz |  |
| sharp lead | typical | Surge | `build-surge/p0-eval/sharp-lead__typical__Surge.wav` | typical Lead | 1.14 | 2.69 | -26.9 dB | 395 Hz |  |
| sharp lead | word list | Lite | `build-surge/p0-eval/sharp-lead__word-list__Lite.wav` | Lead, 1 nudge | 1.86 | 2.73 | -27.0 dB | 436 Hz |  |
| sharp lead | word list | Surge | `build-surge/p0-eval/sharp-lead__word-list__Surge.wav` | Lead, 1 nudge | 1.86 | 2.73 | -27.0 dB | 440 Hz |  |
| deep sub bass | p0 Find | Lite | `build-surge/p0-eval/deep-sub-bass__p0-Find__Lite.wav` | row 43 Bass · 0043 | 0.44 | 3.67 | -40.3 dB | 546 Hz |  |
| deep sub bass | p0 Find | Surge | `build-surge/p0-eval/deep-sub-bass__p0-Find__Surge.wav` | row 43 Bass · 0043 | 0.44 | 3.67 | -35.9 dB | 558 Hz |  |
| deep sub bass | typical | Lite | `build-surge/p0-eval/deep-sub-bass__typical__Lite.wav` | typical Bass | -0.22 | 2.09 | -24.1 dB | 241 Hz |  |
| deep sub bass | typical | Surge | `build-surge/p0-eval/deep-sub-bass__typical__Surge.wav` | typical Bass | -0.22 | 2.09 | -21.6 dB | 191 Hz |  |
| deep sub bass | word list | Lite | `build-surge/p0-eval/deep-sub-bass__word-list__Lite.wav` | Bass | -0.22 | 2.09 | -24.1 dB | 241 Hz |  |
| deep sub bass | word list | Surge | `build-surge/p0-eval/deep-sub-bass__word-list__Surge.wav` | Bass | -0.22 | 2.09 | -21.6 dB | 191 Hz |  |
| lush ambient pad | p0 Find | Lite | `build-surge/p0-eval/lush-ambient-pad__p0-Find__Lite.wav` | row 1129 Pad · 1129 | 0.77 | 4.48 | -40.3 dB | 546 Hz |  |
| lush ambient pad | p0 Find | Surge | `build-surge/p0-eval/lush-ambient-pad__p0-Find__Surge.wav` | row 1129 Pad · 1129 | 0.77 | 4.48 | -35.9 dB | 562 Hz |  |
| lush ambient pad | typical | Lite | `build-surge/p0-eval/lush-ambient-pad__typical__Lite.wav` | typical Pad | 0.88 | 4.27 | -33.8 dB | 432 Hz |  |
| lush ambient pad | typical | Surge | `build-surge/p0-eval/lush-ambient-pad__typical__Surge.wav` | typical Pad | 0.88 | 4.27 | -33.4 dB | 438 Hz |  |
| lush ambient pad | word list | Lite | `build-surge/p0-eval/lush-ambient-pad__word-list__Lite.wav` | Pad | 0.88 | 4.27 | -33.8 dB | 432 Hz |  |
| lush ambient pad | word list | Surge | `build-surge/p0-eval/lush-ambient-pad__word-list__Surge.wav` | Pad | 0.88 | 4.27 | -33.3 dB | 444 Hz |  |
| plucky bell | p0 Find | Lite | `build-surge/p0-eval/plucky-bell__p0-Find__Lite.wav` | row 279 Bass · 0279 | 1.53 | 3.29 | -28.7 dB | 295 Hz |  |
| plucky bell | p0 Find | Surge | `build-surge/p0-eval/plucky-bell__p0-Find__Surge.wav` | row 279 Bass · 0279 | 1.53 | 3.29 | -30.0 dB | 956 Hz |  |
| plucky bell | typical | Lite | `build-surge/p0-eval/plucky-bell__typical__Lite.wav` | typical Bass | -0.22 | 2.09 | -24.1 dB | 241 Hz |  |
| plucky bell | typical | Surge | `build-surge/p0-eval/plucky-bell__typical__Surge.wav` | typical Bass | -0.22 | 2.09 | -21.5 dB | 191 Hz |  |
| plucky bell | word list | Lite | `build-surge/p0-eval/plucky-bell__word-list__Lite.wav` | Pluck, 1 nudge | 1.15 | 2.43 | -39.2 dB | 642 Hz |  |
| plucky bell | word list | Surge | `build-surge/p0-eval/plucky-bell__word-list__Surge.wav` | Pluck, 1 nudge | 1.15 | 2.43 | -38.4 dB | 797 Hz |  |
| warm electric piano keys | p0 Find | Lite | `build-surge/p0-eval/warm-electric-piano-keys__p0-Find__Lite.wav` | row 1001 Pad · 1001 | 1.17 | 5.93 | -27.5 dB | 3384 Hz |  |
| warm electric piano keys | p0 Find | Surge | `build-surge/p0-eval/warm-electric-piano-keys__p0-Find__Surge.wav` | row 1001 Pad · 1001 | 1.17 | 5.93 | -30.0 dB | 318 Hz |  |
| warm electric piano keys | typical | Lite | `build-surge/p0-eval/warm-electric-piano-keys__typical__Lite.wav` | typical Pad | 0.88 | 4.27 | -33.8 dB | 432 Hz |  |
| warm electric piano keys | typical | Surge | `build-surge/p0-eval/warm-electric-piano-keys__typical__Surge.wav` | typical Pad | 0.88 | 4.27 | -33.1 dB | 424 Hz |  |
| warm electric piano keys | word list | Lite | `build-surge/p0-eval/warm-electric-piano-keys__word-list__Lite.wav` | Keys, 1 nudge | 0.44 | 2.65 | -27.4 dB | 537 Hz |  |
| warm electric piano keys | word list | Surge | `build-surge/p0-eval/warm-electric-piano-keys__word-list__Surge.wav` | Keys, 1 nudge | 0.44 | 2.65 | -26.4 dB | 552 Hz |  |
| aggressive saw lead | p0 Find | Lite | `build-surge/p0-eval/aggressive-saw-lead__p0-Find__Lite.wav` | row 892 Lead · 0892 | 1.61 | 4.18 | -34.4 dB | 559 Hz |  |
| aggressive saw lead | p0 Find | Surge | `build-surge/p0-eval/aggressive-saw-lead__p0-Find__Surge.wav` | row 892 Lead · 0892 | 1.61 | 4.18 | -44.0 dB | 232 Hz |  |
| aggressive saw lead | typical | Lite | `build-surge/p0-eval/aggressive-saw-lead__typical__Lite.wav` | typical Lead | 1.14 | 2.69 | -27.1 dB | 405 Hz |  |
| aggressive saw lead | typical | Surge | `build-surge/p0-eval/aggressive-saw-lead__typical__Surge.wav` | typical Lead | 1.14 | 2.69 | -26.9 dB | 395 Hz |  |
| aggressive saw lead | word list | Lite | `build-surge/p0-eval/aggressive-saw-lead__word-list__Lite.wav` | Lead | 1.14 | 2.69 | -27.1 dB | 405 Hz |  |
| aggressive saw lead | word list | Surge | `build-surge/p0-eval/aggressive-saw-lead__word-list__Surge.wav` | Lead | 1.14 | 2.69 | -26.9 dB | 398 Hz |  |
| evolving texture | p0 Find | Lite | `build-surge/p0-eval/evolving-texture__p0-Find__Lite.wav` | row 944 Pad · 0944 | 0.70 | 4.44 | -18.9 dB | 529 Hz |  |
| evolving texture | p0 Find | Surge | `build-surge/p0-eval/evolving-texture__p0-Find__Surge.wav` | row 944 Pad · 0944 | 0.70 | 4.44 | -4.0 dB | 259 Hz |  |
| evolving texture | typical | Lite | `build-surge/p0-eval/evolving-texture__typical__Lite.wav` | typical Pad | 0.88 | 4.27 | -33.8 dB | 432 Hz |  |
| evolving texture | typical | Surge | `build-surge/p0-eval/evolving-texture__typical__Surge.wav` | typical Pad | 0.88 | 4.27 | -33.7 dB | 467 Hz |  |
| evolving texture | word list | Lite | `build-surge/p0-eval/evolving-texture__word-list__Lite.wav` | Atmosphere, 1 nudge | 0.99 | 5.38 | -26.9 dB | 426 Hz |  |
| evolving texture | word list | Surge | `build-surge/p0-eval/evolving-texture__word-list__Surge.wav` | Atmosphere, 1 nudge | 0.99 | 5.38 | -25.0 dB | 582 Hz |  |

## Coverage and failures

Per prompt: p0's predicted category and the selected bank rows (`row Category · name`).

| Prompt | Predicted category | Rows |
|---|---|---|
| dark bass | Bass | 4 rows: 80 Bass · 0080, 131 Bass · 0131, 7 Bass · 0007, 9 Bass · 0009 |
| bright bass | Bass | 5 rows: 80 Bass · 0080, 22 Bass · 0022, 9 Bass · 0009, 33 Bass · 0033, 40 Bass · 0040 |
| short pad | Pad | 4 rows: 1129 Pad · 1129, 994 Pad · 0994, 974 Pad · 0974, 937 Pad · 0937 |
| long pad | Pad | 5 rows: 1129 Pad · 1129, 974 Pad · 0974, 937 Pad · 0937, 994 Pad · 0994, 970 Pad · 0970 |
| soft lead | Lead | 6 rows: 892 Lead · 0892, 490 Lead · 0490, 516 Lead · 0516, 614 Lead · 0614, 689 Lead · 0689, 507 Lead · 0507 |
| sharp lead | Lead | 6 rows: 892 Lead · 0892, 490 Lead · 0490, 516 Lead · 0516, 689 Lead · 0689, 515 Lead · 0515, 507 Lead · 0507 |
| deep sub bass | Bass | 7 rows: 43 Bass · 0043, 17 Bass · 0017, 19 Bass · 0019, 34 Bass · 0034, 40 Bass · 0040, 76 Bass · 0076, 52 Bass · 0052 |
| lush ambient pad | Pad | 4 rows: 1129 Pad · 1129, 994 Pad · 0994, 974 Pad · 0974, 937 Pad · 0937 |
| plucky bell | Bass | 5 rows: 279 Bass · 0279, 34 Bass · 0034, 7 Bass · 0007, 40 Bass · 0040, 322 Bass · 0322 |
| warm electric piano keys | Pad | 3 rows: 1001 Pad · 1001, 937 Pad · 0937, 970 Pad · 0970 |
| aggressive saw lead | Lead | 4 rows: 892 Lead · 0892, 511 Lead · 0511, 689 Lead · 0689, 516 Lead · 0516 |
| evolving texture | Pad | 5 rows: 944 Pad · 0944, 1015 Pad · 1015, 937 Pad · 0937, 1162 Pad · 1162, 970 Pad · 0970 |

Observations (objective, not about sound):

- No prompt fell back for lack of bank rows: p0 only predicts Bass, Lead and Pad and the bank holds those, so all 12
  returned rows. The fallback ("no <label> presets") only appears for other predicted categories, which this
  bundle's prior does not produce.
- Every contrasting pair got the same first candidate: dark/bright bass row 80, short/long pad row 1129, soft/sharp
  lead row 892. Only the later rows differ between the two prompts of a pair. Whether the pairs sound different is for
  the listening notes.
- Some prompts named sounds outside Bass, Lead and Pad: "plucky bell" returned Bass rows, "warm electric piano keys"
  and "evolving texture" returned Pad rows.
- Typical starts for a category are identical across prompts of that category (all Pad prompts share one typical
  start), as expected from a category-only start.

Profiling and memory numbers are in docs/JOINT-VAE-LAB.md ("Known limits").


## p1 listening sheet

WAVs are in `build-surge/p1-eval/` (untracked; regenerate with `vae_p0_eval eval build-surge/p1-eval $B $P` using `models/prompt-p1`). The start labelled "p0 Find" in the tool output is the learned search (p1 here). Columns as above.

| Prompt | Start | Engine | WAV | Selected row / category | Bright | Length | Level | Centroid | Listener notes |
|---|---|---|---|---|---|---|---|---|---|
| dark bass | p0 Find | Lite | `build-surge/p1-eval/dark-bass__p0-Find__Lite.wav` | row 131 Bass · 0131 | -0.15 | 2.22 | -34.9 dB | 427 Hz |  |
| dark bass | p0 Find | Surge | `build-surge/p1-eval/dark-bass__p0-Find__Surge.wav` | row 131 Bass · 0131 | -0.15 | 2.22 | -34.0 dB | 133 Hz |  |
| dark bass | typical | Lite | `build-surge/p1-eval/dark-bass__typical__Lite.wav` | typical Bass | -0.22 | 2.09 | -24.1 dB | 241 Hz |  |
| dark bass | typical | Surge | `build-surge/p1-eval/dark-bass__typical__Surge.wav` | typical Bass | -0.22 | 2.09 | -21.6 dB | 191 Hz |  |
| dark bass | word list | Lite | `build-surge/p1-eval/dark-bass__word-list__Lite.wav` | Bass, 1 nudge | -0.37 | 2.09 | -24.6 dB | 226 Hz |  |
| dark bass | word list | Surge | `build-surge/p1-eval/dark-bass__word-list__Surge.wav` | Bass, 1 nudge | -0.37 | 2.09 | -20.8 dB | 168 Hz |  |
| bright bass | p0 Find | Lite | `build-surge/p1-eval/bright-bass__p0-Find__Lite.wav` | row 53 Bass · 0053 | 0.52 | 5.93 | -35.6 dB | 752 Hz |  |
| bright bass | p0 Find | Surge | `build-surge/p1-eval/bright-bass__p0-Find__Surge.wav` | row 53 Bass · 0053 | 0.52 | 5.93 | -14.3 dB | 713 Hz |  |
| bright bass | typical | Lite | `build-surge/p1-eval/bright-bass__typical__Lite.wav` | typical Bass | -0.22 | 2.09 | -24.1 dB | 241 Hz |  |
| bright bass | typical | Surge | `build-surge/p1-eval/bright-bass__typical__Surge.wav` | typical Bass | -0.22 | 2.09 | -21.6 dB | 191 Hz |  |
| bright bass | word list | Lite | `build-surge/p1-eval/bright-bass__word-list__Lite.wav` | Bass, 1 nudge | 0.04 | 2.09 | -24.5 dB | 249 Hz |  |
| bright bass | word list | Surge | `build-surge/p1-eval/bright-bass__word-list__Surge.wav` | Bass, 1 nudge | 0.04 | 2.09 | -21.7 dB | 203 Hz |  |
| short pad | p0 Find | Lite | `build-surge/p1-eval/short-pad__p0-Find__Lite.wav` | row 1167 Pad · 1167 | 2.52 | 5.93 | -41.3 dB | 244 Hz |  |
| short pad | p0 Find | Surge | `build-surge/p1-eval/short-pad__p0-Find__Surge.wav` | row 1167 Pad · 1167 | 2.52 | 5.93 | -13.0 dB | 253 Hz |  |
| short pad | typical | Lite | `build-surge/p1-eval/short-pad__typical__Lite.wav` | typical Pad | 0.88 | 4.27 | -33.8 dB | 432 Hz |  |
| short pad | typical | Surge | `build-surge/p1-eval/short-pad__typical__Surge.wav` | typical Pad | 0.88 | 4.27 | -33.3 dB | 443 Hz |  |
| short pad | word list | Lite | `build-surge/p1-eval/short-pad__word-list__Lite.wav` | Pad, 1 nudge | 0.84 | 3.97 | -30.2 dB | 344 Hz |  |
| short pad | word list | Surge | `build-surge/p1-eval/short-pad__word-list__Surge.wav` | Pad, 1 nudge | 0.84 | 3.97 | -33.6 dB | 424 Hz |  |
| long pad | p0 Find | Lite | `build-surge/p1-eval/long-pad__p0-Find__Lite.wav` | row 1001 Pad · 1001 | 1.17 | 5.93 | -27.5 dB | 3384 Hz |  |
| long pad | p0 Find | Surge | `build-surge/p1-eval/long-pad__p0-Find__Surge.wav` | row 1001 Pad · 1001 | 1.17 | 5.93 | -22.7 dB | 282 Hz |  |
| long pad | typical | Lite | `build-surge/p1-eval/long-pad__typical__Lite.wav` | typical Pad | 0.88 | 4.27 | -33.8 dB | 432 Hz |  |
| long pad | typical | Surge | `build-surge/p1-eval/long-pad__typical__Surge.wav` | typical Pad | 0.88 | 4.27 | -33.2 dB | 412 Hz |  |
| long pad | word list | Lite | `build-surge/p1-eval/long-pad__word-list__Lite.wav` | Pad, 1 nudge | 0.92 | 4.52 | -30.6 dB | 377 Hz |  |
| long pad | word list | Surge | `build-surge/p1-eval/long-pad__word-list__Surge.wav` | Pad, 1 nudge | 0.92 | 4.52 | -32.6 dB | 415 Hz |  |
| soft lead | p0 Find | Lite | `build-surge/p1-eval/soft-lead__p0-Find__Lite.wav` | row 892 Lead · 0892 | 1.61 | 4.18 | -34.4 dB | 559 Hz |  |
| soft lead | p0 Find | Surge | `build-surge/p1-eval/soft-lead__p0-Find__Surge.wav` | row 892 Lead · 0892 | 1.61 | 4.18 | -44.2 dB | 226 Hz |  |
| soft lead | typical | Lite | `build-surge/p1-eval/soft-lead__typical__Lite.wav` | typical Lead | 1.14 | 2.69 | -27.1 dB | 405 Hz |  |
| soft lead | typical | Surge | `build-surge/p1-eval/soft-lead__typical__Surge.wav` | typical Lead | 1.14 | 2.69 | -27.0 dB | 400 Hz |  |
| soft lead | word list | Lite | `build-surge/p1-eval/soft-lead__word-list__Lite.wav` | Lead, 1 nudge | 1.13 | 2.90 | -26.7 dB | 416 Hz |  |
| soft lead | word list | Surge | `build-surge/p1-eval/soft-lead__word-list__Surge.wav` | Lead, 1 nudge | 1.13 | 2.90 | -26.7 dB | 397 Hz |  |
| sharp lead | p0 Find | Lite | `build-surge/p1-eval/sharp-lead__p0-Find__Lite.wav` | row 892 Lead · 0892 | 1.61 | 4.18 | -34.4 dB | 559 Hz |  |
| sharp lead | p0 Find | Surge | `build-surge/p1-eval/sharp-lead__p0-Find__Surge.wav` | row 892 Lead · 0892 | 1.61 | 4.18 | -44.6 dB | 229 Hz |  |
| sharp lead | typical | Lite | `build-surge/p1-eval/sharp-lead__typical__Lite.wav` | typical Lead | 1.14 | 2.69 | -27.1 dB | 405 Hz |  |
| sharp lead | typical | Surge | `build-surge/p1-eval/sharp-lead__typical__Surge.wav` | typical Lead | 1.14 | 2.69 | -26.9 dB | 397 Hz |  |
| sharp lead | word list | Lite | `build-surge/p1-eval/sharp-lead__word-list__Lite.wav` | Lead, 1 nudge | 1.86 | 2.73 | -27.0 dB | 436 Hz |  |
| sharp lead | word list | Surge | `build-surge/p1-eval/sharp-lead__word-list__Surge.wav` | Lead, 1 nudge | 1.86 | 2.73 | -27.0 dB | 441 Hz |  |
| deep sub bass | p0 Find | Lite | `build-surge/p1-eval/deep-sub-bass__p0-Find__Lite.wav` | row 290 Bass · 0290 | 1.03 | 3.41 | -25.1 dB | 261 Hz |  |
| deep sub bass | p0 Find | Surge | `build-surge/p1-eval/deep-sub-bass__p0-Find__Surge.wav` | row 290 Bass · 0290 | 1.03 | 3.41 | -17.2 dB | 275 Hz |  |
| deep sub bass | typical | Lite | `build-surge/p1-eval/deep-sub-bass__typical__Lite.wav` | typical Bass | -0.22 | 2.09 | -24.1 dB | 241 Hz |  |
| deep sub bass | typical | Surge | `build-surge/p1-eval/deep-sub-bass__typical__Surge.wav` | typical Bass | -0.22 | 2.09 | -21.5 dB | 190 Hz |  |
| deep sub bass | word list | Lite | `build-surge/p1-eval/deep-sub-bass__word-list__Lite.wav` | Bass | -0.22 | 2.09 | -24.1 dB | 241 Hz |  |
| deep sub bass | word list | Surge | `build-surge/p1-eval/deep-sub-bass__word-list__Surge.wav` | Bass | -0.22 | 2.09 | -21.6 dB | 191 Hz |  |
| lush ambient pad | p0 Find | Lite | `build-surge/p1-eval/lush-ambient-pad__p0-Find__Lite.wav` | row 1001 Pad · 1001 | 1.17 | 5.93 | -27.5 dB | 3384 Hz |  |
| lush ambient pad | p0 Find | Surge | `build-surge/p1-eval/lush-ambient-pad__p0-Find__Surge.wav` | row 1001 Pad · 1001 | 1.17 | 5.93 | -30.3 dB | 393 Hz |  |
| lush ambient pad | typical | Lite | `build-surge/p1-eval/lush-ambient-pad__typical__Lite.wav` | typical Pad | 0.88 | 4.27 | -33.8 dB | 432 Hz |  |
| lush ambient pad | typical | Surge | `build-surge/p1-eval/lush-ambient-pad__typical__Surge.wav` | typical Pad | 0.88 | 4.27 | -33.4 dB | 445 Hz |  |
| lush ambient pad | word list | Lite | `build-surge/p1-eval/lush-ambient-pad__word-list__Lite.wav` | Pad | 0.88 | 4.27 | -33.8 dB | 432 Hz |  |
| lush ambient pad | word list | Surge | `build-surge/p1-eval/lush-ambient-pad__word-list__Surge.wav` | Pad | 0.88 | 4.27 | -33.2 dB | 422 Hz |  |
| plucky bell | p0 Find | Lite | `build-surge/p1-eval/plucky-bell__p0-Find__Lite.wav` | row 1798 Pluck · 1798 | 0.58 | 2.43 | -34.9 dB | 427 Hz |  |
| plucky bell | p0 Find | Surge | `build-surge/p1-eval/plucky-bell__p0-Find__Surge.wav` | row 1798 Pluck · 1798 | 0.58 | 2.43 | -31.9 dB | 173 Hz |  |
| plucky bell | typical | Lite | `build-surge/p1-eval/plucky-bell__typical__Lite.wav` | typical Pluck | 1.06 | 2.77 | -40.0 dB | 1260 Hz |  |
| plucky bell | typical | Surge | `build-surge/p1-eval/plucky-bell__typical__Surge.wav` | typical Pluck | 1.06 | 2.77 | -37.6 dB | 1057 Hz |  |
| plucky bell | word list | Lite | `build-surge/p1-eval/plucky-bell__word-list__Lite.wav` | Pluck, 1 nudge | 1.15 | 2.43 | -39.2 dB | 642 Hz |  |
| plucky bell | word list | Surge | `build-surge/p1-eval/plucky-bell__word-list__Surge.wav` | Pluck, 1 nudge | 1.15 | 2.43 | -38.4 dB | 798 Hz |  |
| warm electric piano keys | p0 Find | Lite | `build-surge/p1-eval/warm-electric-piano-keys__p0-Find__Lite.wav` | row 1505 Keys · 1505 | 0.55 | 2.39 | -34.9 dB | 427 Hz |  |
| warm electric piano keys | p0 Find | Surge | `build-surge/p1-eval/warm-electric-piano-keys__p0-Find__Surge.wav` | row 1505 Keys · 1505 | 0.55 | 2.39 | -33.4 dB | 228 Hz |  |
| warm electric piano keys | typical | Lite | `build-surge/p1-eval/warm-electric-piano-keys__typical__Lite.wav` | typical Keys | 0.81 | 2.69 | -27.2 dB | 543 Hz |  |
| warm electric piano keys | typical | Surge | `build-surge/p1-eval/warm-electric-piano-keys__typical__Surge.wav` | typical Keys | 0.81 | 2.69 | -27.2 dB | 576 Hz |  |
| warm electric piano keys | word list | Lite | `build-surge/p1-eval/warm-electric-piano-keys__word-list__Lite.wav` | Keys, 1 nudge | 0.44 | 2.65 | -27.4 dB | 537 Hz |  |
| warm electric piano keys | word list | Surge | `build-surge/p1-eval/warm-electric-piano-keys__word-list__Surge.wav` | Keys, 1 nudge | 0.44 | 2.65 | -26.4 dB | 552 Hz |  |
| aggressive saw lead | p0 Find | Lite | `build-surge/p1-eval/aggressive-saw-lead__p0-Find__Lite.wav` | row 892 Lead · 0892 | 1.61 | 4.18 | -34.4 dB | 559 Hz |  |
| aggressive saw lead | p0 Find | Surge | `build-surge/p1-eval/aggressive-saw-lead__p0-Find__Surge.wav` | row 892 Lead · 0892 | 1.61 | 4.18 | -44.3 dB | 223 Hz |  |
| aggressive saw lead | typical | Lite | `build-surge/p1-eval/aggressive-saw-lead__typical__Lite.wav` | typical Lead | 1.14 | 2.69 | -27.1 dB | 405 Hz |  |
| aggressive saw lead | typical | Surge | `build-surge/p1-eval/aggressive-saw-lead__typical__Surge.wav` | typical Lead | 1.14 | 2.69 | -26.9 dB | 398 Hz |  |
| aggressive saw lead | word list | Lite | `build-surge/p1-eval/aggressive-saw-lead__word-list__Lite.wav` | Lead | 1.14 | 2.69 | -27.1 dB | 405 Hz |  |
| aggressive saw lead | word list | Surge | `build-surge/p1-eval/aggressive-saw-lead__word-list__Surge.wav` | Lead | 1.14 | 2.69 | -27.0 dB | 397 Hz |  |
| evolving texture | p0 Find | Lite | `build-surge/p1-eval/evolving-texture__p0-Find__Lite.wav` | row 3257 Atmosphere · 3257 | 2.06 | 4.35 | -34.4 dB | 559 Hz |  |
| evolving texture | p0 Find | Surge | `build-surge/p1-eval/evolving-texture__p0-Find__Surge.wav` | row 3257 Atmosphere · 3257 | 2.06 | 4.35 | -39.5 dB | 272 Hz |  |
| evolving texture | typical | Lite | `build-surge/p1-eval/evolving-texture__typical__Lite.wav` | typical Atmosphere | 0.96 | 5.29 | -27.1 dB | 397 Hz |  |
| evolving texture | typical | Surge | `build-surge/p1-eval/evolving-texture__typical__Surge.wav` | typical Atmosphere | 0.96 | 5.29 | -24.1 dB | 574 Hz |  |
| evolving texture | word list | Lite | `build-surge/p1-eval/evolving-texture__word-list__Lite.wav` | Atmosphere, 1 nudge | 0.99 | 5.38 | -26.9 dB | 426 Hz |  |
| evolving texture | word list | Surge | `build-surge/p1-eval/evolving-texture__word-list__Surge.wav` | Atmosphere, 1 nudge | 0.99 | 5.38 | -25.1 dB | 576 Hz |  |

### p1 coverage

| Prompt | Predicted category | Rows |
|---|---|---|
| dark bass | Bass | 6 rows: 131 Bass · 0131, 34 Bass · 0034, 234 Bass · 0234, 9 Bass · 0009, 19 Bass · 0019, 40 Bass · 0040 |
| bright bass | Bass | 7 rows: 53 Bass · 0053, 34 Bass · 0034, 9 Bass · 0009, 234 Bass · 0234, 19 Bass · 0019, 198 Bass · 0198, 125 Bass · 0125 |
| short pad | Pad | 3 rows: 1167 Pad · 1167, 4004 Pad · 4004, 974 Pad · 0974 |
| long pad | Pad | 3 rows: 1001 Pad · 1001, 1075 Pad · 1075, 974 Pad · 0974 |
| soft lead | Lead | 6 rows: 892 Lead · 0892, 492 Lead · 0492, 761 Lead · 0761, 507 Lead · 0507, 516 Lead · 0516, 611 Lead · 0611 |
| sharp lead | Lead | 5 rows: 892 Lead · 0892, 749 Lead · 0749, 743 Lead · 0743, 516 Lead · 0516, 575 Lead · 0575 |
| deep sub bass | Bass | 5 rows: 290 Bass · 0290, 17 Bass · 0017, 19 Bass · 0019, 40 Bass · 0040, 5 Bass · 0005 |
| lush ambient pad | Pad | 3 rows: 1001 Pad · 1001, 974 Pad · 0974, 1075 Pad · 1075 |
| plucky bell | Pluck | 4 rows: 1798 Pluck · 1798, 1649 Pluck · 1649, 1660 Pluck · 1660, 1643 Pluck · 1643 |
| warm electric piano keys | Keys | 6 rows: 1505 Keys · 1505, 1261 Keys · 1261, 1329 Keys · 1329, 1263 Keys · 1263, 1262 Keys · 1262, 1452 Keys · 1452 |
| aggressive saw lead | Lead | 4 rows: 892 Lead · 0892, 749 Lead · 0749, 516 Lead · 0516, 502 Lead · 0502 |
| evolving texture | Atmosphere | 7 rows: 3257 Atmosphere · 3257, 2653 Atmosphere · 2653, 2609 Atmosphere · 2609, 2621 Atmosphere · 2621, 2607 Atmosphere · 2607, 3266 Atmosphere · 3266, 2596 Atmosphere · 2596 |
