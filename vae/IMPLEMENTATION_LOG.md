# Pitcher implementation log

One entry per round of implementation, newest at the bottom. Each entry records **when** (local time, CEST), **what** was added or changed, **where**, and how it was **verified**. The plan itself lives in `PIPELINE_VAE_SURGE.md`; this file records what actually happened.

Conventions:
- Paths are relative to `~/Documents/Pitcher/`.
- "Upstream" = code from the Surge or surge-rack repositories; changes to it are always listed explicitly.

---

## Round 0: planning document (2026-10-14, before 16:39; exact times not recorded)

Documentation only, no code.

- **Added** `PIPELINE_VAE_SURGE.md`: the 8-step pipeline linked to the codebase, with Surge source references, adjusted steps (⚠️ markers) and sources.
- **Revised it several times** after questions:
  - Hayes et al. description corrected (fixed effect chain, but on randomly sampled parameters, not presets).
  - After `surge/` was filled from a zip: version check (`src/` = Surge `main` `d8b4a7b`), missing nested libraries noted, `SURGE_DATA_HOME`, proposed `surgepy` helpers, exact ctrltype → kind table (Step 1.3).
  - Wavetable statistics added (276 names, 82 embedded tables).
  - Appendix A added: fill-in values for parameters missing from `.fxp` files, from `SurgePatch::init_default_values` and the revision upgrade steps.

---

## Round 1: Route 2 setup and raw dataset extraction (2026-10-14, 16:39–16:59 CEST)

### Environment

| Time | What | Where |
|---|---|---|
| 16:39 | **Created** Python 3.12.12 venv with `uv` (default `python3` = pyenv 3.11.0 has no SSL, so `pip` couldn't download). Installed `numpy`, `pandas`, `pyarrow` | `.venv/` |
| 16:39–16:44 | **Replaced** the zip copy of `surge/` (empty `libs/`, no `.git`) by `git clone --recursive` of Surge `main`. Same commit `d8b4a7b`; `src/` verified byte-identical. Zip copy kept as `surge-zip-backup/` until the build worked, then deleted (16:57) | `surge-rack-main/surge/`, log in `clone.log` |
| 16:41–16:43 | **Configured** the `surgepy` build: venv Python passed explicitly, `CMAKE_POLICY_VERSION_MINIMUM=3.5` for CMake 4.2 | `surge-rack-main/surge/ignore/bpy/`, log in `cmake_config.log` |
| 16:43–16:49 | **Built** `surgepy` and the Surge engine (Release, Intel macOS) | `.../ignore/bpy/src/surge-python/surgepy.cpython-312-darwin.so`, log in `cmake_build.log` |

### Upstream change: `surge-rack-main/surge/src/surge-python/surgepy.cpp` (16:41–16:51)

Saved as `pitcher/surgepy_local.patch` (re-apply after a re-clone with `git -C surge-rack-main/surge apply ../../pitcher/surgepy_local.patch`).

| Change | Why |
|---|---|
| **Added** `getParamCtrlType(p)`, `getParamOSCName(p)`, `getWavetableName(scene, osc)`, `getParamStorageName(p)` + Python bindings | Exact parameter kinds (Step 1.3), OSC addresses (Step 7), wavetable names (Step 1.1), matching parameters to the `.fxp` XML (verification) |
| **Fixed** `getPatch()`: send levels now loop over all 4 sends (was only `[0]`, `[1]`) | Send FX 3/4 levels were missing from every dump |
| **Fixed** `getPatch()`: `return_level` only for the 4 send slots (was all 16 FX slots) | 12 entries were unassigned handles all aliasing parameter 0 ("Send FX 1 Return"). Before: 774 entries / 762 parameters; after: 766 / 766 |

### Pipeline code (new)

| File | Contents |
|---|---|
| `pitcher/surge_io.py` | The only module that talks to `surgepy`: sets `SURGE_DATA_HOME`, adds the build folder to `sys.path`, `SurgeIO` class (load + settle, read values/ranges/types/ctrltypes/flags, routings, wavetable names, parameter catalogue). Asserts 766 distinct parameters, so an unpatched `surgepy` fails loudly |
| `pitcher/s1_extract.py` | Step 1.1–1.2: loads all 3,561 patches through Surge's loader, labels from folders (merge table of Step 1.2), drops categories / MPE packs / non-single scene mode / audio-input oscillators, writes the long-format tables |
| `pitcher/s1_verify.py` | Step 1 checks: label counts, Appendix A fill-ins, XML-vs-loaded comparison, with every difference traced to a loader conversion (thresholds parsed from the oscillator/effect sources) |

### Data produced (16:53)

| File | Contents |
|---|---|
| `data/presets.parquet` | 2,122 kept presets: id, file, name, label, folder category, author, revision, active scene, wavetable names |
| `data/raw_values.parquet` | 1,625,452 rows = 2,122 × 766 (long): value, min, max, default, valtype, ctrltype, flags |
| `data/routings.parquet` | Modulation routings of the kept presets |
| `data/params.parquet` | 766 parameters: synth id, storage name, OSC address |
| `data/dropped.csv` | 1,439 dropped patches with reason |

### Verified (16:53–16:59)

- Extraction: 3,561 files in 51 s.
- Label counts equal the planned table exactly (Bass 412, Lead 396, Pad 326, Keys 254, Pluck 248, Synth 188, Atmosphere 151, Orchestral 147). 0 NaN.
- 36,497 values missing from the files got exactly the Appendix A.2 fill-in (0 mismatches).
- 1,243,019 values present in the files: 3,341 differ after loading (0.27%), all 3,341 traced to a specific loader conversion.

### Findings that changed the plan

- `surgepy`'s `getPatch()` had the two bugs above (fixed).
- The user data folder is `~/Documents/Surge Synth Team/Surge XT`, not `~/Documents/Surge XT`.
- **Exclude `volume`, `fx_bypass`, `polylimit` from the features:** Surge doesn't reset them before loading, so for files older than revision 17 `volume` keeps the previous patch's value (load-order dependent).
- Appendix A.4 was incomplete. Added: oscillator- and effect-specific conversions, polyphony limit 8 → 16, Twist +4.61 cents, Sine/ring-mod waveform renumbering, LFO magnitude 2v − 1, FX value clamping, floats stored as int bits.

### Documentation updated

`PIPELINE_VAE_SURGE.md`: Sections 0.1–0.4 (actual setup), Step 1 status block, Appendix A.4 (measured table), user folder path.

---

## Round 2: implementation log (2026-10-14, 17:13 CEST)

- **Added** this file, `IMPLEMENTATION_LOG.md`, with Rounds 0–1 reconstructed from file timestamps.

---

## Round 3: Step 1.3, encoding (2026-10-14, 17:14–17:20 CEST)

### Code (new)

| Time | File | Contents |
|---|---|---|
| 17:15 | `pitcher/schema.py` | Feature kinds (cont / ord / cat / bool) from valtype + ctrltype (ordinal and categorical ctrltype lists from the plan), exclusion lists with reasons, oscillator and FX type names, FX slot layout, flag suffixes |
| 17:17 | `pitcher/s1_encode.py` | Raw long table → canonical features, masks, `schema.json`, model-ready matrix; built-in checks |

### Design decisions (deviations from / choices within the plan)

| Decision | Why |
|---|---|
| **Per-type blocks** for oscillator knobs (`scene/osc/<o>/<type>/p<i>`) and FX knobs (`fx/<type>/p<i>`), instead of shared columns + rounding | With the ctrltype available, each column can have one meaning. PCA (Step 2) and permutation importance (Step 3) need that. Wider (1,244 features before pruning), but Step 2.1 removes rarely used blocks |
| **FX canonical per type**, not per slot: `present`, `placement` (insert/send/global), `send_level`/`return_level` (send only), knobs. First heard instance per type | The same effect sits in different slots across presets; per-slot columns would mix effects |
| **Filter type + subtype → one categorical `model`** over observed pairs | A subtype's meaning depends on the type; generation can only produce valid pairs |
| **Flags as features** (`.off`, `.sync`, `.ext`, `.abs`, `.deform`) | They change what a knob does (Appendix A.3) |
| **Masks = audible** (oscillators incl. ring-mod/FM use, routed LFOs, filter/waveshaper type ≠ none, noise, FM on, heard FX) | Values you can't hear shouldn't drive the model |
| **Only the active scene**, mapped to `scene/...` | 18 presets play scene B; the model always sees one canonical scene |
| **Excluded:** `volume`, `fx_bypass`, `polylimit`, `scene_active`, `scenemode`, `splitpoint`, `fx_disable` (used for masks), pitch-bend ranges; FX types Airwindows, Vocoder, Audio Input, Convolution | Performance/global settings, load-order dependence (Round 1), or need external input / not in the Rack build |
| **Wavetable name** as categorical per Wavetable/Window oscillator: top 32 names + "other" | Plan Step 1.1 |

### Data produced (17:17)

| File | Contents |
|---|---|
| `data/schema.json` | 1,244 features: 623 cont, 35 ord, 181 cat, 405 bool; options, natural ranges, fill values, sources, active counts, model-ready column names; exclusions; notes |
| `data/codes.parquet` | 2,122 × 1,244 compact codes (NaN where masked) |
| `data/masks.parquet` | 2,122 × 1,244 masks |
| `data/dataset.parquet` | 2,122 × 2,056 model-ready matrix (one-hot categoricals, masked entries filled) |

26 FX types kept (heard in at least one preset): delay, reverb, phaser, rotary, distortion, eq, freqshift, conditioner, chorus, reverb2, flanger, ringmod, neuron, geq11, resonator, chow, exciter, ensemble, combulator, nimbus, tape, treemonster, waveshaper, mstool, spring, bonsai.

### Verified

- All 2,056 model-ready columns in [0, 1], 0 NaN.
- Every one-hot group sums exactly to its mask (0 bad rows).
- Encode → decode round trip: 0 mismatches.
- Run time 44 s.

### Known limitations (recorded in `schema.json` notes and the plan)

- The second instance of an FX type is not represented: delay 79, EQ 66, reverb 15, reverb2 7, distortion/conditioner 4, waveshaper 3, nimbus/phaser 2, resonator/chorus 1 presets (Airwindows 59, excluded anyway).
- About 40 active values outside their parameter's range are clipped to [0, 1]. Mostly junk integers in old patches, which Surge's oscillators clamp anyway (`limit_range(..., 1, MAX_UNISON)` in `ClassicOscillator.cpp:270`, `WavetableOscillator.cpp:72`, `SineOscillator.cpp:109`). Real losses: 5 FM2 ratio values, ≈6 extended-range values.
- Categorical options are taken from all 2,122 presets (the split comes in 1.5). This uses no label information, but note it if strict separation is wanted.

### Documentation updated

`PIPELINE_VAE_SURGE.md`: "Implemented" note at the top of Step 1.3; Step 1 status block extended with the 1.3 outputs, checks and limitations.

---

## Round 4: Step 1.4, audio rendering (2026-10-14, 18:40–18:55 CEST)

### Decisions (agreed in discussion before implementing)

| Decision | Why |
|---|---|
| One note: MIDI 48 (C3), velocity 100, 48 kHz, 32-bit float WAV | User choice: one note for now; float keeps levels above 0 dBFS for the clipping check in Step 8 |
| **Dataset render: fixed 2 s hold for every preset** | Same performance for all presets, so differences come from the sound; an envelope-based hold would leak the category into the recording (Step 8 auto-tagger) |
| **Dynamic release** instead of a fixed 1 s tail: record until the 50 ms RMS stays 60 dB below the loudest window for 200 ms; min 0.5 s, max 12 s | Measured: 31% of presets have an amp release > 1 s, 15% > 2 s (Pads: median 1.9 s, 90th percentile 5.4 s), plus reverb/delay tails |
| **Natural render, listening only**: hold until steady (two consecutive 250 ms windows differ < 0.5 dB, twice) or died out, 0.5–6 s; same release rule | User asked for a per-preset "natural" key press; based on the audio (catches LFO/FX effects), and kept out of training and evaluation |

### Code

| Time | File | Change |
|---|---|---|
| 18:48 | `pitcher/surge_io.py` | **Added** rendering to `SurgeIO`: `_chunk` (50 ms processing via `createMultiBlock`/`processMultiBlock`), `flush` (all notes off + 0.5 s silence, returns residual RMS), `_release` (dynamic tail), `render(natural=False/True)` returning audio + info dict |
| 18:49 | `pitcher/s1_render.py` | **New**: renders all presets in both modes with 4 worker processes (one Surge engine each); reloads the patch before each render; writes WAVs + index |
| 18:49 | dependency | **Installed** `soundfile` 0.14.0 into `.venv` |

### Fix during testing

A 40-preset test showed small residual output (up to −75 dBFS) at the start of the natural render: the fixed render's tail (possibly cut at 12 s) leaked into it. **Fixed** by reloading the patch before each render (`loadRaw` → `stopSound` + FX reload).

### Data produced (18:49–18:52)

| File | Contents |
|---|---|
| `data/audio/fixed/<preset_id>.wav` | 2,122 dataset renders |
| `data/audio/natural/<preset_id>.wav` | 2,122 listening renders |
| `data/audio_index.parquet` | 4,244 rows (preset × mode): hold/tail/total length, hold reason, tail capped, peak dBFS, max RMS dBFS, samples > 0 dBFS, silent, idle output, CPU time; plus label and name |

Run: 194 s with 4 workers; 7.67 GB on disk; log in `render.log`.

### Results and findings

- Fixed render median length 4.0 s (Bass 2.5 s … Atmosphere 7.8 s); 46 tails capped at 12 s (Pad 18, Atmosphere 14, Lead 7, Keys 5, other 2).
- Natural render median 3.5 s; hold reason: steady 1,816, died 204, max 102.
- **2 silent presets, both "Ghost Glass Harmonium" (Bluelight)**, filed under both Keys and Pads; all oscillators and noise muted, VCA ≈ −100 dB. To be excluded in 1.5.
- **Cross-category duplicate patches exist:** to be grouped in the 1.5 split.
- 29 presets peak above 0 dBFS (max +13.3 dBFS); 7 presets produce output without a note (above −90 dBFS; "Hauntology" −56 dBFS, a drone, not leakage).

### Documentation updated

`PIPELINE_VAE_SURGE.md`: "Implemented" note at the top of Step 1.4 (new render spec); Step 1 status block with the 1.4 results.

---

## Round 5: Step 1.5, train / val / test split (2026-10-14, 19:20–19:33 CEST)

### Decisions (agreed in discussion)

| Decision | Why |
|---|---|
| **Split now, before Step 2** | Steps 2–3 make data-driven choices (pruning, PCA, feature selection), which must not see the evaluation presets |
| **Option A: 80 / 10 / 10 with a separate test set** | Validation for training choices (early stopping, β); the test set gives honest final numbers and the Step 8 references (auto-tagger ceiling, novelty calibration), and is touched once at the end |
| **Equal label percentages in all three parts** | User requirement |

### Code

| Time | File | Contents |
|---|---|---|
| 19:25 | `pitcher/s1_split.py` | **New**: excludes silent renders; groups presets (union-find over name stem per author, same patch in two folders, identical encoded parameters); stratified group split with a search over 300 seeds × all (test, val) fold pairs for the most equal label shares; writes the split and prints the checks |
| 19:20 | dependency | **Installed** `scikit-learn` 1.9.1 (+ `scipy` 1.18.1) into `.venv` |

### Fix during development

The first name-stem rule also stripped letters inside words ("Tuba" → "tu", "Bowed" → "bowe"). **Fixed:** letters and Roman numerals only count as a variant marker as a separate token; trailing numbers may be attached.

### Data produced (19:32)

| File | Contents |
|---|---|
| `data/splits.json` | `{preset_id: "train" / "val" / "test"}` for 2,120 presets. Fixed from now on |
| `data/splits.parquet` | `preset_id, split, group_id, label, name, author` |

### Results

- Excluded: "Ghost Glass Harmonium" (preset_id 120 Keys, 156 Pad), silent render → 2,120 presets.
- 1,831 groups; 152 with more than one preset (441 presets, largest 17). Unions: 288 by name stem, 1 extra by identical parameters ("Poly Lala" = "Quote").
- Chosen: seed 1, test fold 1, val fold 2. Train 1,696 / val 212 / test 212 (80.0 / 10.0 / 10.0%).
- Max deviation of any label's share from its overall share: 0.24 percentage points (table in `PIPELINE_VAE_SURGE.md`, Step 1 status).

### Verified

No group spans two splits; every preset assigned exactly once; split sizes exactly 80/10/10.

### Documentation updated

`PIPELINE_VAE_SURGE.md`: Step 1 status block with the split method, groups, table and checks.

---

## Round 6: writer + round-trip test, version A (2026-10-14, 19:38–20:28 CEST)

The writer (Step 7 code) is built and tested *before* any VAE exists, on real presets whose correct values are known.

### Code (new)

| File | Contents |
|---|---|
| `pitcher/codec.py` | `decode(codes, schema)`: feature row → write `Plan` (types, subtypes, flags, values, wavetables, FX slot assignment); everything into scene A; FX blocks → slots per placement (insert/send/global) in FX-type order; optional `slot_map` (diagnostics) |
| `pitcher/audio_metrics.py` | `mel_distance_db` (64-band mel spectrogram in dB, cells within 80 dB of the louder peak, shorter clip padded), `rms_db` |
| `pitcher/s1_roundtrip.py` | Test A1 (exactness) + A2 (sound), crash-tolerant runner (timeouts, per-worker stage logs, isolated re-runs) |
| `pitcher/surge_io.py` | **Added** `neutralise`, `write_plan`, `read_back`, `set_int`, `clear_flags`, `remove_routings_to` |

### Test design

- **A1 exactness:** load the patch, reset all parameters *including types* to defaults, write the features, read back, re-encode; every written code must come out unchanged.
- **A2 sound:** load the patch, reset only the *values* (types kept, so routings survive), write the features, render, and compare with the original render. Reference: original vs a second render of the unchanged patch (Surge is not deterministic). "Transparent" = distance ≤ max(1 dB, 1.5 × the re-render distance).

### Bugs found by the test and fixed (all in the writer path)

| # | Problem | Cause (verified in source / by experiment) | Fix |
|---|---|---|---|
| 1 | Rebuilt presets lost their effects | `fx_disable` is a bitmask with a ±2³¹ range and no ctrltype; `setParamVal`'s float normalisation turns 0 into −648 → most FX slots disabled | **Upstream:** `setFxDisable(mask)` in `surgepy.cpp`; `neutralise` skips all `ct_none` params |
| 2 | **Masks wrong in 290 presets** (Step 1.3) | A level/send at 0 can be raised by modulation (macro, LFO, mod wheel): 684 such cases in 378 presets, e.g. "Clave": Macro 1 → reverb send | `s1_encode.py`: level/send counts as audible if base > 0 **or** it is a modulation destination. **Re-ran 1.3** (19:45): 6,713 mask entries switched on, none off; 2,068 model-ready columns (was 2,056). Split unchanged |
| 3 | Values snapped to log₂(1.5) = 0.585 multiples (Tape, EQ, Ensemble) | temposync/extend/absolute belong to the parameter *slot*, survive type changes, and an FX type change restores them from `fxsync`; `bound_value` snaps whenever temposync is set, even when the new ctrltype can't sync, and `setTempoSync` then refuses | **Upstream:** `clearParamFlags(p)` in `surgepy.cpp`; writer clears flags of every written param after the type change |
| 4 | **Engine crash (segfault)** in 2 presets ("Bassoon", "Joe Has Had A Bad Day") | Modulation routings onto an FX slot survive its type change; a macro offset landed on the Resonator's integer mode knob (p9) → invalid array index (`ResonatorEffect.cpp:120`). Removing routings on an *empty* slot is not accepted | Writer removes routings onto every FX slot whose type changes, and `neutralise` removes FX routings *before* switching slots off |
| 5 | MIDI key (0–127) off by one | Surge's int conversion `0.005 + 0.99·x` (`Parameter.h:640–651`) differs from `setParamVal`'s linear normalisation for ranges > ~100 | `set_int()` passes the value whose linear normalisation equals `intScaledToFloat(target)` |
| 6 | Tempo-synced values snapped to the note grid | Writing a value while temposync is on snaps it; original patches hold off-grid values | Write values first, then `setTempoSync` (doesn't snap); `setExtend` stays *before* values (it rescales the value) |
| 7 | Runner hung after a worker crash | `multiprocessing.Pool` waits forever for a crashed worker's task | Timeout, per-worker stage logs, isolated re-runs; crashes recorded in `data/roundtrip_crashes.csv` |
| 8 | Audio metric unusable (two renders of the same patch differed by 0.27) | Log-spectrum down to 1e-7 is dominated by random phase/drift and near-silent tails | Replaced by `mel_distance_db` (perceptual bands, 80 dB floor) |

Writer order now (Step 7.2, updated): remove routings onto FX slots that change type → types (ints via `set_int`, `fx_disable` via `setFxDisable`) → settle → subtypes, `clearParamFlags`, extend/absolute/deform/deactivated → settle → values → temposync → settle.

### Upstream change: `surge-rack-main/surge/src/surge-python/surgepy.cpp`

**Added** `setFxDisable(mask)` and `clearParamFlags(p)` (+ bindings); rebuilt `surgepy`; `pitcher/surgepy_local.patch` refreshed (now 6 helpers + 2 `getPatch()` fixes).

### Results (final run, 20:23–20:28, 312 s, 4 workers)

- **A1: 499,585 written items, 0 mismatches, 0 crashes.** Decode → write → read back is exact.
- **A2: 79.8% transparent overall.**

| Group | Transparent | Presets |
|---|---|---|
| FX layout unchanged | 92.6% | 1,154 |
| FX layout changed (canonical slots/order, dropped duplicates) | 64.4% | 966 |
| No excluded FX, scene A | 87.1% | 1,856 |
| Contains Airwindows (excluded FX) | 29.3% | 249 |
| Active scene B (writer writes scene A, B's routings/wavetables not carried) | 11.1% | 18 |

Level change rebuilt vs original: median 0.00 dB (5th–95th percentile −4.6 … +5.2 dB). Some high distances are presets that differ a lot between two renders of themselves ("Bam Bo Bass": 27.1 vs 27.1).

### Open decisions (for the user)

1. Airwindows: include as FX features, drop the 249 presets, or accept the loss.
2. FX order: store each effect's position within its placement (insert 1–4 etc.) so the writer can rebuild the exact chain.
3. The 18 scene-B presets: acceptable loss vs. exclude.

### Documentation updated

`PIPELINE_VAE_SURGE.md`: Step 1 status (round-trip results), Step 7.2 (writer order and the pitfalls above), Step 1.3 mask rule.

---

## Round 7: curation (drop Airwindows + scene B), FX positions, new split (2026-10-14, 20:32–20:45 CEST)

### Decisions (user)

1. Drop presets where an Airwindows effect is heard.
2. Store each effect's chain position, so the writer rebuilds the exact FX chain.
3. Drop the presets that play scene B.

### Code changes

| Time | File | Change |
|---|---|---|
| 20:35 | `pitcher/s1_encode.py` | **Curation** in one place: excludes presets with a silent render, a heard excluded FX (Airwindows; also 1 audio-input FX), or active scene B; writes `data/excluded_presets.csv`. **New feature** `fx/<type>/position` (categorical 0–3): slot index within its group of 4 (insert A1–A4 / send 1–4 / global G1–G4) = chain order. Notes in `schema.json` |
| 20:36 | `pitcher/codec.py` | Writer places each effect at its stored position (falls back to the next free slot with a warning if taken) |
| 20:36 | `pitcher/s1_split.py` | Uses the curated set from Step 1.3 (`codes.parquet`) instead of its own silent-preset filter |

### Re-run results

**Step 1.3** (20:35): excluded 247 (heard Airwindows) + 15 (scene B; the other 3 of the 18 also had Airwindows) + 1 (heard audio-input FX) + 2 (silent) = 265 → **1,857 presets**. 1,270 features (623 cont, 35 ord, 207 cat, 405 bool), 2,091 model-ready columns. All checks pass.

**Step 1.5 split re-built** (20:42). Allowed because nothing has been fitted on the split yet. Keeping the old assignment minus the dropped presets would have given a 1.37 pp max label-share deviation; re-split: 0.25 pp. Train 1,485 / val 186 / test 186 (80.0 / 10.0 / 10.0%); 1,595 groups (137 with >1 preset); seed 3, test fold 2, val fold 5. No group spans two splits. Previous split saved for reference in the session scratchpad only (not part of the project).

**Round-trip version A** (20:42–20:45, 189 s):

| | Round 6 (2,120 presets) | **Round 7 (1,857 presets)** |
|---|---|---|
| A1 mismatches | 0 / 499,585 | **0 / 431,892** |
| Crashes | 0 | **0** |
| A2 transparent, overall | 79.8% | **89.7%** |
| A2, FX layout unchanged | 92.6% (1,154) | 92.7% (1,563) |
| A2, FX layout changed | 64.4% (966) | **73.5% (294)** |
| Level change, 5th–95th percentile | −4.6 … +5.2 dB | −2.7 … +4.1 dB |

Remaining "layout changed" cases come from FX that are not heard (removed by the writer) and dropped second instances (delay 74, EQ 40, reverb 13, reverb2 5, others ≤ 3 presets). Several of the highest distances are presets that also differ strongly between two renders of themselves ("Bam Bo Bass" 27.1 vs 27.5, "Staccato String Land" 25.6 vs 25.1): randomness, not loss. Per label: Bass 93%, Lead 92%, Synth 91%, Atmosphere 91%, Pad 90%, Orchestral 87%, Pluck 86%, Keys 83%.

### Documentation updated

`PIPELINE_VAE_SURGE.md`: Step 1 status (curation, new split table, round-trip results), Step 1.3 note (positions).

---

## Round 8: Step 2.1, dropping features that (almost) never change (2026-10-14, 20:50–21:37 CEST)

### Code

| File | Contents |
|---|---|
| `pitcher/s2_prune.py` | **New.** Train-only statistics per feature (active share, changed share); keep if changed in ≥ threshold of train presets; dependencies (flags follow their parent; an FX block follows `fx/<type>/present`; `placement` and `position` always stay with their effect); writes `feature_activity.csv`, `pruning.json`, `codes_pruned.parquet` |
| `pitcher/s2_reference.py` | **New.** The *reference* value of every feature per preset: load the template patch, write only this preset's types, read the knobs = what the writer produces for a feature it doesn't write. Template provisionally `Templates/Init Saw.fxp` until Step 3 |
| `pitcher/s1_encode.py` | **Added** output `data/defaults.parquet` (getParamDef codes; turned out not to be the right reference, kept for inspection) |
| `pitcher/s1_roundtrip.py` | **Added** `--codes` / `--out` options, so the sound test can measure the cost of pruning |

### How the rule was found (each tried and measured with the round-trip sound test)

| Attempt | Dropped feature becomes | A2 transparent | Problem |
|---|---|---|---|
| Changed vs most common value | most common value | 62.3% | **11 presets silent**: knobs frozen one by one form impossible combinations (Modern oscillator: every waveform level "off", because each preset uses only one) |
| Same statistic | Surge's `getParamDef` | 17.4% | Inconsistent pair: many knobs sit at the same *non-default* value in nearly all presets; and `getParamDef` is not what a type gets when selected (Modern saw level: 0 vs full) |
| Rare oscillator-type blocks at defaults + rest at mode | mixed | 60.4% | Still silent presets: a knob inside a *common* block (Modern saw level) frozen at "off" |
| **Changed vs reference (template + types)** | **the reference value** | 63.2% | No silent presets; but `placement`/`position` were pruned too → effects claimed the same slot (107 presets) |
| **+ placement/position kept with their effect** (final rule) | reference / mode (flags) / absent (rare FX) | **63.6%** at 3% | 0 slot collisions. Remaining cost = the dropped features themselves |

Findings: (1) Surge has two "defaults": `getParamDef` (nominal per control type) and the value an oscillator/effect gets when its type is selected (`init_default_values`); the writer produces the second. (2) Presets that pruning doesn't touch stay 91% transparent (as unpruned), so the test is consistent.

### Results (train = 1,485 presets)

| Setting | Features | Model columns | A2 transparent |
|---|---|---|---|
| No pruning (Round 7) | 1,270 | 2,091 | 89.7% |
| Threshold 1% (15 presets) | 536 | 1,154 | **80.0%** |
| Threshold 3% (45 presets), plan default | 326 | 800 | 63.6% |

At 3%, pruning touches 1,253 of 1,857 presets (long tail: about 700 features, each rare, but most presets use at least one). Losses by group (transparent if affected vs not): oscillator knobs 673 presets (36% vs 80%), LFO 522 (49% vs 69%), flags 464 (44% vs 70%), FX knobs 484 (45% vs 70%), rare FX types removed 108 (21% vs 66%: graphic EQ 33, Resonator 21, Nimbus 18, Tape 16, Waveshaper 16, Combulator 13, others ≤ 10).

**Current files reflect the 3% setting** (`feature_activity.csv`, `pruning.json`, `codes_pruned.parquet`); the 1% round-trip result is in `data/roundtrip_pruned_1pct.parquet`. Threshold decision pending (user).

### Documentation updated

`PIPELINE_VAE_SURGE.md`: Step 2.1 "Implemented" note with the rule and the threshold table.

---

## Round 9: Step 2.2, PCA/UMAP + check of the 1% pruning (2026-10-14, 21:38–21:47 CEST)

User question: before choosing the pruning threshold, are the features that 1% pruning would drop ranked high by PCA/UMAP?

### Code and environment

| File | Contents |
|---|---|
| `pitcher/s2_explore.py` | **New.** PCA (train) → per-feature importance (variance carried in the components explaining 90%); UMAP (train) all vs 1%-kept features + neighbour overlap; random-forest category check (train → val); small-multiple plots |
| dependencies | **Installed** `umap-learn` 0.5.12, `matplotlib` 3.11.2; pinned `numba` 0.61.2 / `llvmlite` < 0.45 (newer `llvmlite` has no Intel-Mac build and fails to compile) |
| data | `data/pruning_1pct.json`, `data/feature_activity_1pct.csv` (1% variant; the main files stay at 3% until decided); `data/pca_feature_importance.csv` |
| plots | `runs/step2/pca.png`, `umap_all.png`, `umap_kept1pct.png`; `runs/step2/summary.json` |

Plot design: 8 categories fail a colour-blind check as one 8-colour scatter (validator: orange/green ΔE 3.2 for protanopia, red/orange 7.1 with normal vision), so each category gets its own panel (blue against grey).

### Results

**PCA** (train, 2,091 columns): PC1 7.3%, PC2 6.4%; 151 components for 90% (101 for 80%). Top features: filter configuration, osc 1 type, filter 1 model, waveshaper type, envelope shapes, osc 2 type, play mode, LFO 1 shape, character, FM routing.

**1%-dropped features (734)**: 0 in the PCA top 25, 9 in the top 50, 21 in the top 100; together 18.7% of PCA importance. **The best-ranked ones are a masking artefact:** they never change when active (e.g. `fx/delay/p1.deform` = 0 in all 758 presets with a delay), and their column correlates +1.00 with "is this block active", which kept features already carry (e.g. `fx/delay/present`).

**UMAP neighbours (k = 15)**: all vs 1%-kept features share 66% of neighbours in feature space and 32% in the 2-D maps; two UMAP runs of the *same* data (seeds 0/1) share only 48%. Random neighbours would share 0.5%.

**Category check (random forest, balanced accuracy on val, n = 186):** all features 0.547 · 1%-kept 0.550 · 3%-kept 0.534.

**Maps:** categories overlap strongly in parameter space (Bass lower half, Lead upper right, Atmosphere partly in a separate arm); a parameter-only classifier reaches about 55% balanced accuracy.

### Conclusion

The 1%-dropped features carry no information that kept features don't already carry: high PCA ranks are a mask artefact, and category accuracy is unchanged (0.550 vs 0.547). Threshold decision pending (user), recommendation 1%.

Side note for Step 4: masked entries filled with a constant make "active or not" dominate PCA variance; the VAE uses masked losses, so this doesn't affect training, but distance-based analyses (PCA, UMAP, nearest-neighbour novelty in Step 8) should account for it.

---

## Round 10: pruning threshold fixed at 1% (2026-10-14, 21:50–21:53 CEST)

Decision (user): apply the 1% threshold, after the Round 9 check.

| Change | Where |
|---|---|
| Re-ran `s2_prune.py --threshold 0.01` → **536 features kept (339 cont, 91 cat, 85 bool, 21 ord), 1,154 model-ready columns**. Dropped 734: 544 changed in < 1% of train presets, 184 belong to dropped effect types, 6 follow a dropped parent. Freeze rules: reference 210, most common value 328, absent 196. **14 FX types kept:** chorus, conditioner, delay, distortion, EQ, flanger, frequency shifter, graphic EQ, Nimbus, phaser, Resonator, reverb, reverb 2, rotary | `data/pruning.json`, `data/feature_activity.csv`, `data/codes_pruned.parquet` |
| Default threshold of `s2_prune.py` set to 0.01, so a re-run reproduces the final setting | `pitcher/s2_prune.py` |
| `s2_explore.py` no longer needs the 3% file (that comparison is recorded in Round 9); outputs regenerated on the final setting, same numbers as Round 9 | `pitcher/s2_explore.py`, `data/pca_feature_importance.csv`, `runs/step2/*` |
| Round-trip result for the 1% setting kept as `data/roundtrip_pruned.parquet` (A2 transparent 80.0%); the temporary `*_1pct` files are removed | `data/` |

### Documentation updated

`PIPELINE_VAE_SURGE.md`: Step 2.1, decision recorded.

---

## Round 11: Step 3.1–3.2, ranking parameter groups for the core set (2026-10-14, 21:58–22:19 CEST)

### Code

| File | Contents |
|---|---|
| `pitcher/s3_rank.py` | **New.** 452 groups from the 536 kept features (a feature + its flags; an effect's present/placement/position = one `fx/<type>` group). Three criteria, all on **train only**: (1) grouped permutation importance of a random forest (300 trees, balanced), 5-fold StratifiedGroupKFold inside train, 5 permutations per fold, drop in balanced accuracy; (2) within-category variation (std of the 0–1 code, or Gini impurity for cat/bool, × active share, averaged over categories); (3) 16 structural must-haves (oscillator types, levels, mutes; filter 1 model + cutoff; amp envelope A/D/S/R; scene volume). Combined order: must-haves, then alternately the next by (1) and by (2); an effect knob pulls in its effect group |
| output | `data/core_ranking.csv`: per group features, columns, importance ± SE, variation, ranks, combined order, cumulative columns |

Run time 1,112 s (18.5 min).

### Results

- RF balanced accuracy inside train (5-fold): 0.595 ± 0.013.
- **Only 63 of 452 groups have an importance clearly above 0** (more than 2 standard errors).
- Most important for category: **play mode** (poly / mono / legato) +0.092; amp attack +0.041; amp sustain +0.015; amp release +0.014; portamento; drift; filter-EG attack; osc 3 level; osc 2 octave; osc 3 keytrack; osc 3 mute; velocity → VCA; reverb p9; filter 1 cutoff; EQ p4/p7; filter balance; LFO 2 amplitude; String oscillator p2.
- Most within-category variation: filter 1/2 model, filter configuration, osc 1 type, waveshaper type, envelope shapes, character, LFO 1 shape, osc 2 type.
- Columns per K (cumulative, combined order): K = 30 → 248, 45 → 294, 60 → 322, 90 → 445, 150 → 554. The filter models are the widest groups (71 and 70 one-hot columns), then the wavetable names (33 each).

### Finding that needs a decision (template, Step 3.3)

The provisional template (`Templates/Init Saw.fxp`) has **no modulation routings and no FX**. In train, 68% of presets route Voice LFO 1 (mostly to pitch = vibrato, osc 1 shape, filter cutoff), 47% LFO 2, about half velocity or mod wheel (mostly to filter cutoff). With a routing-free template, LFO settings in the core (5 LFO groups in the first 60) have no audible effect, and generated presets have no movement.

### Documentation updated

None yet (pending the template decision).

---

## Round 12: standard routings (template option B) + version B round-trip / K curve (2026-10-15, 11:12–11:30 CEST)

The template decision (Round 11) was still open; rather than choose, both options were built and measured.

### Code

| File | Change |
|---|---|
| `pitcher/schema.py` | **Added** `STANDARD_ROUTES` (13 routings used by ≥ 5% of train presets, audible source, fixed-meaning destination) and `route_feature()` |
| `pitcher/s1_encode.py` | **Added** 13 route features `route/<source>/<dest>` (continuous, depth −1…1, 0 = not routed, always active; source id stored in the schema). Routing tuples now carry depth and source id. Wavetable top-32 list now skips every embedded name "(Patch …)", incl. `(Patch Sample)` |
| `pitcher/codec.py` | Route features → `plan.routings` (source id, path, depth) |
| `pitcher/surge_io.py` | `write_plan` now **loads wavetables by name** after the type phase (factory folder first, then 3rd party; `wavetable_path()`), and **writes the standard routings** last (`setModDepth01`); `read_back` covers both |
| `pitcher/s1_roundtrip.py` | A1 also checks routings and wavetable names |
| `pitcher/s2_reference.py` | Reference for route features = depth 0 (the template has no routings) |
| `pitcher/s3_curve.py` | **New.** Version B round-trip: val presets rebuilt on the template from their top-K core groups (+ mode-frozen flags; everything else template / type defaults), rendered, compared with the original. Variants: static (A) and + 13 routings (B). Plot `runs/step3/core_curve.png` (palette checked: blue/orange pass all checks) |

### Re-runs

- Step 1.3: 1,283 features (13 routes added), 2,104 model columns; all checks pass. Step 2.1 (1%): **549 kept** (536 + all 13 routes), 1,167 model columns. Split unchanged.
- Writer check (round-trip A, 80 presets): 19,714 items incl. routings + wavetables, **0 mismatches**.

### Results: version B K curve (186 val presets)

| K groups | A: static, transparent | B: + routes, transparent | B median distance |
|---|---|---|---|
| 16 (must-haves) | 4.8% | 4.8% | 21.1 dB |
| 30 | 3.2% | 4.3% | 21.1 dB |
| 60 | 3.8% | 5.4% | 19.7 dB |
| 90 | 4.3% | 7.0% | 18.4 dB |
| 150 | 10.2% | 12.4% | 14.1 dB |
| 250 | 16.7% | 21.5% | 12.7 dB |
| 452 (all) | 23.7% | **34.9%** | 9.6 dB |

Silent rebuilds at K = 452: 13 (A) vs 7 (B).

### Findings

1. **No knee:** strict transparency stays low at every K; even all features on the template reach only 35%, against 80% when the original patch is the base (version A). The template itself loses most of the sound.
2. **Main cause = modulation routings outside the template.** 92% of train presets use audible routings, in a long tail: 77 routings at ≥ 1% usage still cover only 54% of routing instances, and only 25% of presets completely; routings onto oscillator knobs (type-dependent) can't be fixed at all.
3. **Silent rebuilds** all have a level that is 0 in the patch and raised by a routing the template lacks: velocity → scene volume (scene volume ≈ 0), LFO 4/5/6 → oscillator levels, macro → level.
4. Option B (13 routings) helps clearly: 35% vs 24% at all features, fewer silent presets.

Decision pending (user): how to represent modulation (see the chat for options).

### Documentation updated

None yet (pending the decision).

---

## Round 13: listening test for the "same character" threshold (2026-10-15, 11:40–11:58 CEST)

User questions answered first: K = groups, not features; the 80% (round-trip A) was rebuilt on the *original patch*, the K curve (round-trip B) on the *template*; "indistinguishable" (within Surge's own render randomness) is far too strict for a generator, so the threshold is now calibrated by listening.

### Code and outputs

| File | Contents |
|---|---|
| `pitcher/s3_listening.py` | **New.** 20 val presets (Bass, Lead, Pad, Keys ×3; Pluck, Synth, Atmosphere, Orchestral ×2). Per preset 3 rebuilt versions from existing candidates (round-trip A on the original, or round-trip B at K ∈ {16 … 452}), one per distance band: close < 6 dB (target 3.5), medium 6–12 (8.5), far ≥ 12 (16); plus 4 control pairs (original vs a fresh render of itself). Re-rendered, distances re-measured; one gain per preset (peak −3 dBFS) keeps relative loudness; AAC in `.mp4`. Blind: random order, neutral ids `pair-NN`, originals `ref-NN` |
| `runs/step3/listening/index.template.html`, `index.html` | The page (no versions or distances in it). Ratings: same sound / same character / different / broken; saved per pair in the artifact's db (`ratings/<pair id>`), with a browser fallback |
| `data/listening_key.csv` | Local key: page id → preset, version (A / B<K> / R), band, re-measured distance |
| Artifact | **https://claude.ai/artifact/C7KLEnrvwcYA3iosQJzpuX** (private, capability `db`) |

64 pairs, 84 audio files, 4.5 MB. Re-measured distances: close 0.0–7.5 dB (median 3.7), medium 0.2–16.3 (7.8), far 10.7–24.5 (16.1), controls 3.0–4.3 (Surge renders vary, so bands overlap a little; the analysis uses actual distances).

### Fixes during the round

- **Pair ids and file names first revealed the version** (`p1147-B150`): anonymised to `pair-NN` / `ref-NN`, key keeps the mapping; folded into the script.
- **`.m4a` is not a served type on the artifact host:** files renamed to `.mp4` (same MPEG-4 audio); the script now writes `.mp4` directly (`afconvert -f mp4f`). The published files were produced by the first version + renames, so they match the key; the script is not re-run (renders vary slightly).

### Verified

Artifact published; one read of the `ratings` collection: reachable, empty (as expected before listening).

### Next

When the ratings are in: fit P(same character) against distance → threshold; recompute the K curve with it.

---

## Round 14: listening results → "same character" threshold (2026-10-15, 12:00–12:10 CEST)

### Code and outputs

| File | Contents |
|---|---|
| `data/listening_ratings/ratings/*.json` | The 64 ratings, exported from the artifact's db (all pairs rated) |
| `pitcher/s3_listening_analysis.py` | **New.** Joins ratings with the key; logistic fits of P(≥ same character) and P(same sound) against distance; K curve recomputed with the threshold; plots |
| `data/listening_results.csv`, `data/listening_threshold.json` | Per pair: rating, distance, version; thresholds + K curve |
| `runs/step3/listening_fit.png`, `runs/step3/core_curve_character.png` | Ratings vs distance with the 80% / 50% thresholds; K curve at the threshold |

### Results

| Version | Same sound | Same character | Different | Broken |
|---|---|---|---|---|
| Controls (original vs re-render, 3.0–4.3 dB) | 4 | 0 | 0 | 0 |
| A: original as base, 1% pruning | **18** | 0 | 0 | 0 |
| B: template (K 16–452, + 13 routings) | 7 | 8 | 13 | 14 |

- **Thresholds:** P(same character or better) = **80% at 6.7 dB**, 50% at 9.7 dB; P(same sound) = 50% at 6.9 dB. Almost everything ≤ 6 dB is "same sound"; ratings are close to binary (only 8 "same character").
- **The 1% pruning (Step 2.1) is perceptually transparent:** all 18 round-trip-A versions were rated "same sound".
- **"Broken" (silent, glitchy, unusable) only at K ≤ 90** (6 at K = 16, 3 at 30, 2 at 45, 3 at 90), with level jumps of +10 to +15 dB or −12 to −23 dB; none of the 20 template rebuilds at K ≥ 150 was rated broken.
- **K curve at the 6.7 dB threshold** (share of 186 val presets): with 13 routings K 16: 2.2% · 90: 5.4% · 150: 11.3% · 250: 19.4% · 452: **31.7%** (static: 24.2%); at the 50% threshold (9.7 dB), K 452: 51.1% (routes) / 38.7% (static).

### Conclusions

1. The ceiling of the template approach (≈ 32% "same character" with all features) is the bottleneck, not the choice of K: it comes from what the template doesn't carry (mainly modulation routings, Round 12).
2. K must be at least ~150 groups to avoid "broken" sounds.
3. Next: raise the ceiling (routing matrix), clean presets that remain unreproducible, then choose K on the "same character" curve.

### Documentation updated

`PIPELINE_VAE_SURGE.md`: Step 3.4 note.

---

## Round 15: routing matrix + new template ceiling (2026-10-15, 12:12–12:25 CEST)

### Code

| File | Change |
|---|---|
| `pitcher/s1_routes.py` | **New.** Selects the routing matrix on **train only**: (source → destination) pairs with an audible source (LFOs, velocity, keytrack, envelopes, random …; no mod wheel / macros / aftertouch) used by ≥ 1% of train presets; routings onto oscillator knobs split per oscillator type (kept if that pair+type occurs in ≥ 5 train presets). Writes `data/routing_matrix.json` |
| `pitcher/s1_encode.py` | Uses the matrix instead of the 13 standard routings (fallback: `STANDARD_ROUTES`); oscillator-knob routing features `route/<source>/osc/<o>/<type>/p<i>` are active only when that oscillator has that type |
| `pitcher/codec.py` | Oscillator-knob routings are written only when the decoded oscillator type matches |
| `pitcher/s3_curve.py` | Adds only routing features kept in Step 2.1; options `--variants`, `--out`; old 13-routing curve saved as `data/core_curve_13routes.csv` |

### Re-runs

- Matrix: **146 routing features** (77 plain + 69 per oscillator type). Sources: Voice LFO 1 (35), velocity (28), Voice LFO 2 (25), keytrack (23), Voice LFO 3 (14), others ≤ 6.
- Step 1.3: 1,416 features, 2,237 model columns; all checks pass. Step 2.1 (1%): **631 kept** (95 routing features: 74 plain + 21 per oscillator type), 1,249 model columns.
- Writer check (round-trip A, 80 presets): 25,587 written items, **0 mismatches**.

### New ceiling (template + all features + matrix, 186 val presets; `data/core_curve_matrix.csv`)

| K | Same character (≤ 6.7 dB), 13 routings → matrix | ≤ 9.7 dB, 13 → matrix | Median distance | Silent |
|---|---|---|---|---|
| 150 | 11.3% → 16.1% | 27.4% → 35.5% | 14.1 → 12.9 dB | 9 → 5 |
| 250 | 19.4% → 26.3% | 37.1% → 47.3% | 12.7 → 10.4 dB | 8 → 5 |
| 452 (all) | **31.7% → 40.9%** | 51.1% → 61.3% | 9.6 → 8.3 dB | 7 → 3 |

### What still breaks (val presets, K = all, share "same character" with / without the property)

| Property | Presets | With | Without |
|---|---|---|---|
| Audible routings outside the matrix (long tail) | 118 | 29% | 62% |
| **Macro routings** (macros have stored values, so they act like fixed offsets) | 51 | 20% | 49% |
| LFO in step-sequencer / MSEG / formula mode (curve data not in the features) | 22 | 9% | 45% |
| Routings onto FX parameters (global scope, not in the matrix) | 25 | 20% | 44% |
| Mod-wheel routings (0 in a plain render, so harmless) | 107 | 50% | 28% |
| Embedded wavetable | 9 | 56% | 40% |
| **None of the harmful properties** | 15 | **80%** | |

Conclusion: the template approach works (80%) when a preset's sound lives in parameters + matrix routings; the remaining gap is macro offsets, long-tail routings, FX-parameter routings and step-sequencer/MSEG/formula LFOs. Decision pending (user).

---

## Round 16: minimal cleaning + first CVAE (Steps 4–5) (2026-10-15, 12:30–13:03 CEST)

Decision (user): build the VAE now, with a minimal cleaning step first; routing/macro improvements later if the evaluation asks for them.

### Code

| File | Contents |
|---|---|
| `pitcher/s3_curve.py` | `--split all` option (template rebuild of every preset) |
| `pitcher/s4_clean.py` | **New.** Excludes presets whose template rebuild from all kept features (+ routing matrix) is **silent** (< −60 dB vs original) or has a **level jump > 12 dB**; same rule for every split → `data/vae_excluded.csv` |
| `pitcher/vae_data.py` | **New.** Inputs (model-ready columns of the 631 kept features + one mask bit per feature = 1,880), targets (455 cont/ord, 85 bool, 91 categorical with 709 logits), masks, labels; `Spec.to_codes()` turns decoder output into a full codes row (dropped features get their Step 2.1 freeze rule) |
| `pitcher/model.py` | **New.** CVAE: MLP encoder (features + masks + label → μ, log σ²), MLP decoder (z + label → sigmoid / logit / softmax heads); masked MSE + BCE + CE summed over features; KL with free bits |
| `pitcher/train.py` | **New.** AdamW, category-balanced batches, KL warm-up, early stopping on val reconstruction (selected only after warm-up), metrics per epoch; runs in `runs/vae/<run>/` |
| `pitcher/s4_eval.py` | **New.** Audio check: reconstructions (encoder mean → decoder → template → render → distance vs original, compared with the ceiling) and prior samples per label (validity: silent, clipping, NaN); a few sample WAVs per label |
| dependency | **Installed** `torch` 2.2.2 (last build for Intel macOS). It can't use numpy 2's C API, so tensors are built from Python lists (`tensors()`); the import prints a harmless warning |

### Cleaning (all 1,857 presets rebuilt on the template, 71 s)

Excluded **138 (7.4%)**: 123 level jump > 12 dB, 15 silent. Remaining: train 1,375 / val 166 / test 178; label shares still within ~1.3 pp across splits, so the split is kept.

### Training runs (z = 12; val reconstruction = summed masked MSE + BCE + CE)

| Run | Settings | Best val rec | KL | Informative dims (KL > 0.1) | Osc 1 type / filter 1 model / play mode correct |
|---|---|---|---|---|---|
| base | h256, dropout 0.1, β 0.5, warm-up 100 | 35.4 | 11.8 | 9 | 84% / 35% / 73% |
| h256_d01 | h256, dropout 0.1, β 0.5, warm-up 30 | 32.9 | 7.9 | 6 | 74% / 36% / 71% |
| h256_d03_wd | + dropout 0.3, weight decay 1e-3 | 35.1 | 5.8 | 4 | 63% / 30% / 68% |
| h128_d03_wd | h128, dropout 0.3, wd 1e-3 | 37.1 | 4.7 | 3 | 54% / 24% / 72% |
| h128_d03_wd_b1 | + β 1.0 | 38.0 | 3.0 | 2 | 52% / 26% / 68% |
| h128_d05_wd | h128, dropout 0.5, wd 1e-2 | 39.6 | 3.6 | 2 | 54% / 22% / 68% |
| h256_d01_b0.1 | β 0.1 | 31.5 | 18.3 | – | – |
| h256_d01_b0.02 | β 0.02 | 31.8 | 36.0 | – | – |

Strong overfitting (val best around epoch 25–40 while train keeps improving); more regularisation makes validation worse and squeezes the latent into 2–4 dimensions; most of the error is in the categorical switches (CE ≈ 21 of ≈ 32). The "active units" log value (variance of μ > 0.01) is not informative with free bits; use KL > 0.1 per dimension.

### Audio check (template + writer, listening threshold 6.7 dB)

| | Same character | Median distance | Samples valid (24 per label, z ~ N(0, I)) |
|---|---|---|---|
| Ceiling (preset's own codes), val | 43.4% | 7.5 dB | – |
| h256_d01 reconstruction, val | 0.0% | 18.3 dB | **100%** (0 silent, 0 clipping) |
| h256_d01_b0.1 reconstruction, val | 0.6% | 18.8 dB | 97.9% |
| h256_d01_b0.02 reconstruction, val | 0.6% | 18.9 dB | 93.8% (6% silent) |
| h256_d01 reconstruction, **train** | 0.4% | 17.2 dB (ceiling 57.4% / 5.6 dB) | – |

Check of the decode path: the *true* targets sent through the same path (`to_codes` → writer → render) give 50% same character (median 6.8 dB), so the low numbers are the model, not the pipeline.

### Interpretation

Exact per-preset reconstruction is near impossible for this data: each preset has ~176 discrete choices (91 categorical, 85 bool) and a single wrong one (filter model, oscillator type, effect present) changes the sound; more latent information (lower β) doesn't fix it. For a generator the relevant measure is whether *new samples* are valid and have the requested character. Samples are already 100% valid (h256_d01); their category and character still have to be measured (Step 8 auto-tagger, listening).

Example samples: `runs/vae/<run>/samples/<label>_00..03.wav`.

---

## Round 17 — Reorganised the `pitcher/` folder (2026-10-15, 13:40 CEST)

**Why:** the code had grown to 23 flat files with `s1_…`–`s4_…` prefixes. They are now grouped by plan step, with a shared `core/`, so it is clear which code is library and which is a pipeline step. No behaviour was changed.

### New layout (old → new)

| Old | New | Old | New |
|---|---|---|---|
| `surge_io.py` | `core/surge_io.py` | `s2_reference.py` | `step2_explore/reference.py` |
| `schema.py` | `core/schema.py` | `s2_prune.py` | `step2_explore/prune.py` |
| `codec.py` | `core/codec.py` | `s2_explore.py` | `step2_explore/explore.py` |
| `audio_metrics.py` | `core/audio_metrics.py` | `s3_rank.py` | `step3_core/rank.py` |
| `s1_extract.py` | `step1_dataset/extract.py` | `s3_curve.py` | `step3_core/curve.py` |
| `s1_verify.py` | `step1_dataset/verify.py` | `s3_listening.py` (+ page template) | `step3_core/listening.py`, `step3_core/listening_page.template.html` |
| `s1_render.py` | `step1_dataset/render.py` | `s3_listening_analysis.py` | `step3_core/listening_analysis.py` |
| `s1_routes.py` | `step1_dataset/routes.py` | `s4_clean.py` | `step4_vae/clean.py` |
| `s1_encode.py` | `step1_dataset/encode.py` | `vae_data.py` | `step4_vae/data.py` |
| `s1_split.py` | `step1_dataset/split.py` | `model.py` | `step4_vae/model.py` |
| `s1_roundtrip.py` | `step1_dataset/roundtrip.py` | `train.py` | `step4_vae/train.py` |
| `surgepy_local.patch` | `patches/surgepy_local.patch` | `s4_eval.py` | `step4_vae/eval.py` |

### Changes

| What | Change |
|---|---|
| packages | `__init__.py` in `pitcher/` and each subfolder; imports are now absolute (`from pitcher.core.surge_io import SurgeIO`) |
| how to run | from the project folder, as modules: `.venv/bin/python -m pitcher.step1_dataset.extract` (old: `.venv/bin/python pitcher/s1_extract.py`); docstring run lines updated |
| `core/surge_io.py` | project root is now `parents[2]` (one folder deeper); `data/`, `runs/` and the surgepy build path are unchanged |
| `pitcher/README.md` | **New.** Layout, run order with what each step reads and writes (incl. the encode → split → routes → encode loop on a fresh build), side checks, things to know |
| `PIPELINE_VAE_SURGE.md` | 27 script references updated to the new paths; the folder tree in Step 0 replaced by the current layout |
| `data/`, `runs/` | untouched (no outputs moved or re-generated) |

Backup of the old flat folder: scratchpad `pitcher_backup_before_reorg` (temporary).

### Verification

| Check | Result |
|---|---|
| all 23 modules import | ✅ |
| `--help` of the CLI modules (roundtrip, curve, train, eval) | ✅ |
| `step1_dataset.roundtrip --limit 8` | A1 0 mismatches (as before) |
| `step4_vae.clean` | same 138 excluded |
| `step4_vae.train --run _smoke` + `eval --run _smoke` | ran end to end (smoke run deleted) |
| `step3_core.listening_analysis` | output identical to before |

---

## Round 18 — Audio tagger (Step 8.0) and first category check of VAE samples (2026-10-15, 13:45–14:25 CEST)

**Why:** to measure whether a generated preset *sounds like* the requested category, without listening to thousands of samples. The tagger works from audio only, so it can't be fooled by parameters that look right but sound wrong.

**How it works:** a frozen CLAP model turns each render into a 512-number fingerprint (similar sounds → nearby fingerprints); a logistic regression trained on the train presets' fingerprints and folder labels draws the borders between the 8 categories.

### Implemented

| File | Change |
|---|---|
| `pitcher/step8_eval/clap.py` | **New.** `Clap` wrapper (audio + text embeddings, L2-normalised, mono mix of the fixed render, repeat-padded to 10 s by CLAP's feature extractor); caches `data/clap_<model>.parquet` for all 1,857 curated presets. Tensors cross via Python lists (torch 2.2 + numpy 2) |
| `pitcher/step8_eval/tagger.py` | **New.** Logistic regression (standardised, `class_weight="balanced"`, C by 5-fold CV on train); compares majority label, spectrum statistics (mean/std of log-mel and of its change: 256 numbers), CLAP zero-shot (text prompts) and 3 CLAP models; picks the model on the train CV score; reports on val (test untouched). `Tagger` class for later use. Writes `data/tagger.pkl`, `tagger_report.json`, `tagger_val_predictions.csv`, `runs/step8/tagger_confusion.png` |
| `pitcher/step8_eval/label_check.py` | **New.** Tags three groups: real val presets (original renders), the same presets rebuilt on the template from their own codes, and VAE samples (z ~ τ·N(0, I), n per label). Label agreement per label, balanced mean, mean P(label), confusion of samples. Writes `runs/step8/<run>/label_check.csv/.json` + audio |
| dependency | **Installed** `transformers==4.49.0` (+ tokenizers, safetensors, huggingface_hub); CLAP weights in `~/.cache/huggingface` |
| docs | `pitcher/README.md` run order rows 17–19; plan Step 8.0/8.1 📍 notes |

### Tagger (val, 186 presets; chance 12.5%)

| Method | Balanced accuracy | Top-2 | CV on train |
|---|---|---|---|
| majority label (Lead) | 12.5% | – | – |
| spectrum statistics + LR | 32.0% | 56% | 36.4% |
| CLAP zero-shot, htsat-unfused / larger_general / larger_music | 19.5% / 21.6% / 11.5% | – | – |
| **CLAP `clap-htsat-unfused` + LR (chosen, C = 0.01)** | **47.9%** | **73%** | 50.0% |
| CLAP `larger_clap_general` + LR | 43.1% | 68% | 50.0% |
| CLAP `larger_clap_music` + LR | 28.8% | 55% | 33.4% |

Embedding time: 267 s / 561 s / 485 s for the three models. Recall per label (chosen): Bass 80%, Orchestral 83%, Pad 52%, Synth 41%, Atmosphere 38%, Lead 34%, Keys 27%, Pluck 27%. Main confusions: Atmosphere → Pad 54%, Pad → Atmosphere 37%, Keys/Pluck spread across Pad/Atmosphere/Keys/Pluck, Lead → Bass 21%. These follow overlaps between the folder labels, so ~50% is close to what the labels allow from one note.

### Category check, run `h256_d01`, τ = 1, 50 samples per label (566 renders in 17 s)

| Label | Real val presets | Rebuilt on template (ceiling) | VAE samples |
|---|---|---|---|
| Bass | 80% | 87% | 40% |
| Lead | 35% | 26% | 22% |
| Pad | 52% | 57% | 76% |
| Keys | 19% | 24% | 16% |
| Pluck | 30% | 35% | 16% |
| Synth | 41% | 41% | 78% |
| Atmosphere | 50% | 30% | 24% |
| Orchestral | 91% | 55% | 18% |
| **balanced** | **49.8%** | **44.3%** | **36.2%** |
| mean P(label) | 0.39 | 0.37 | 0.27 |

Silent samples 0.2%. (Real val here = the 166 presets after VAE cleaning, so 49.8% differs slightly from 47.9% on all 186.)

### Interpretation

- The **template keeps most of the category** (44% vs 50%), so the core-parameter rebuild is not the main loss.
- The **label steers the samples**: rows of the sample confusion differ clearly (Pad → 76% Pad, Bass → 40% Bass, Atmosphere → 24% Atmosphere + 24% Pad), far above chance.
- **Samples drift towards a generic "Synth" sound**: 28–56% of Bass, Lead, Keys, Pluck, Atmosphere and Orchestral samples are tagged Synth, while real presets rarely are (Lead → Synth 11%). Likely cause: the decoder's "average" output for a label is a blurred, generic patch; prior samples from N(0, I) land in regions between the training presets. This is what the per-label GMM sampler (sampling where real presets of the label actually sit in the latent space) should address.
- Uncertainty: 50 samples per label → ±7 pp per cell; val ceilings rest on 11–38 presets per label.

---

## Round 19 — Per-label latent mixture sampler (Step 6) + reproducible tagger (2026-10-15, 14:30–15:30 CEST)

**Why:** in Round 18 many samples from the shared prior N(0, I) sounded like a generic "Synth". The shared prior puts mass *between* the regions where real presets of a label sit. The new sampler draws from where that label's train presets actually are.

### Implemented

| File | Change |
|---|---|
| `pitcher/step6_sample/latent.py` | **New.** Samplers `prior` (z ~ τ·N(0, I)) and `gmm`: per label, a Gaussian mixture on the aggregate posterior of its train presets (5 draws z ~ q(z \| x, label) per preset), full covariance, k = 1..5 by BIC, cached in `runs/vae/<run>/latent_gmm.pkl`; τ scales each component's spread around its mean. Also `load_model`, `decode_rows` |
| `pitcher/step8_eval/label_check.py` | Groups per sampler (`samples:prior`, `samples:gmm`); **novelty** = 1 − cosine of the CLAP fingerprint to the nearest *train* preset, with real val presets as the reference; near-copy = below their 5th percentile; output per τ in `runs/step8/<run>/tau<τ>/`; originals use cached fingerprints |
| `pitcher/step8_eval/clap.py` | **Fix:** clips are cut to the first 10 s. CLAP crops longer clips at a *random* offset, so the 127 renders longer than 10 s (up to 14 s) gave different fingerprints per run (seen as one-preset changes in the "original" column). All three caches recomputed |
| `pitcher/step8_eval/tagger.py` | Model selection: the smallest CLAP model is kept unless a larger one wins by > 1 pp in train CV (after the fix the two general models tied, 50.2% vs 50.3%). `Tagger.proba_from(embeddings)`, `Tagger.clap_name` |
| docs | README rows 19–20, `step6_sample/` in the layout; plan Step 8.1 📍 note |

### Tagger after the fix (val, balanced accuracy)

Chosen `clap-htsat-unfused` **51.9%** (was 47.9%), top-2 73%; larger_clap_general 45.8%, larger_clap_music 34.2%, spectrum statistics 32.0%. Per label: Bass 80%, Orchestral 75%, Pad 67%, Atmosphere 54%, Keys 41%, Synth 41%, Lead 39%, Pluck 18%.

### Latent mixtures (run h256_d01)

| Label | Train presets | Components (BIC) |
|---|---|---|
| Bass | 258 | 3 |
| Lead | 289 | 1 |
| Pad | 194 | 2 |
| Keys | 158 | 2 |
| Pluck | 166 | 1 |
| Synth | 134 | 2 |
| Atmosphere | 96 | 2 |
| Orchestral | 80 | 1 |

Several component centres sit far from the origin (|mean| up to 2.6), where N(0, I) rarely samples.

### Sampler comparison (150 samples per label per sampler; 2,566 renders in 80 s per τ)

| | τ = 1: prior | τ = 1: gmm | τ = 0.5: prior | τ = 0.5: gmm |
|---|---|---|---|---|
| Bass | 44% | 37% | 46% | 45% |
| Lead | 21% | 21% | 19% | 10% |
| Pad | 75% | 78% | 88% | 83% |
| Keys | 22% | 23% | 9% | 25% |
| Pluck | 21% | 24% | 17% | 22% |
| Synth | 61% | 63% | 83% | 79% |
| Atmosphere | 10% | 33% | 9% | 47% |
| Orchestral | 21% | 27% | 16% | 27% |
| **balanced agreement** | **34.2%** | **38.3%** | **35.8%** | **42.2%** |
| gmm − prior (bootstrap 95% CI) | | +4.2 pp [+0.8, +7.6] | | +6.3 pp [+3.1, +9.6] |
| tagged "Synth" | 40.7% | 35.6% | 53.7% | 45.0% |
| silent | 0.6% | 0.5% | 0.1% | 0.7% |
| novelty median | 0.125 | 0.123 | 0.130 | 0.121 |
| near-copies | 2.3% | 2.8% | 1.4% | 1.8% |

References: real val presets 51.6% agreement, novelty median 0.125 (by definition 5% near-copies), 10.8% tagged Synth; template rebuild 45.9–47.0%.

### Interpretation

- **The per-label mixture is reliably better** (CI above 0 at both τ). At τ = 0.5 it reaches 42.2%, which is 90% of the template ceiling (~46%).
- **No loss of novelty:** samples are as far from the train set as real unseen presets, and near-copies stay below the 5% reference. So the gain doesn't come from copying.
- **The biggest gain is for multi-cluster labels:** Atmosphere 10% → 33–47% (2 components far from the origin), Keys and Orchestral. Bass and Lead don't gain; Lead at τ = 0.5 even drops (one broad component near the origin).
- **The "Synth" drift remains** (36–45% of samples vs 11% of real presets), and it grows at lower τ for both samplers: the closer to the centre, the more generic. The decoder itself averages, so a better sampler alone doesn't remove it. This is where Metropolis–Hastings with the tagger as target (next) or a sharper decoder can help.
- **Run-to-run noise:** the template group varies between runs (e.g. Pad 52% vs 39% on 23 presets) because Surge renders have random oscillator phase and drift. That's why comparisons use 150 samples per label and a bootstrap CI.

---

## Round 20 — Text → preset track: plan, prompt generator, Stable Audio 3 script (2026-10-15, finished 18:55 CEST)

**Why:** hackathon idea: type a sound description and get editable Surge presets. Stable Audio 3 is the teacher during development (text → audio); the plugin itself carries only small models (CLAP text encoder → prior → VAE decoder). Full concept, prior work (AutoSynth, CTAG, …), license notes and the prompt-dataset plan are in `TEXT2PRESET_PLAN.md`.

### Implemented

| File | Change |
|---|---|
| `TEXT2PRESET_PLAN.md` | **New.** Concept (CLAP fingerprint as the bridge; versions A search / B prompt-able VAE with a text → audio-fingerprint prior / C reward fine-tuning), prior work, Stable Audio 3 access + license, evaluation, prompt-dataset plan (counts, shares, structure, synonyms, checks), build order |
| `pitcher/text2preset/vocabulary.yaml` | **New, editable.** 8 categories with noun synonym groups, category weights, preferred / avoided descriptors; 8 descriptor slots (brightness, body, grit, attack, length, movement, colour, style) with near-synonym forms (`adj` before the noun, `with` phrases after) and opposites; generic and out-of-range nouns; 10 held-out words; sentence templates; the Stable Audio suffix ("single note, C3, one-shot, dry, no reverb, synthesizer, no drums, no melody") |
| `pitcher/text2preset/prompts.py` | **New.** Grammar: category quotas by weight; 0–3 descriptor slots (+ style 15%) weighted by the category's preferences; unique per tag combination; per prompt `tags`, `sa3_prompt` (70% canonical forms + suffix), 6 `user_texts` (2 × tags / sentence / slang, no suffix) and `user_texts_unseen` (test only, with held-out words); split 80/10/10 by a hash of the tag combination (out-of-range always test) |
| `pitcher/text2preset/generate_sa3.py` | **New, standalone** (GPU machine, stable-audio-3 env). `--inspect` prints the real `generate()` signature and one output (the README doesn't document seed / steps / CFG or the return type); passes seed / steps / CFG only if accepted, else seeds torch; accepts tensor or array output in any layout; FLAC 16-bit + `metadata.jsonl`; resumable; `--limit` for model comparisons. **Tested with two mock models only** (different signatures and output layouts, resume, inspect) |
| docs | `pitcher/README.md` (layout + text → preset commands) |

### Pilot prompt list (`data/text2preset/prompts_pilot.jsonl` / `.csv`)

500 prompts: 415 main (Bass 49, Lead 49, Pad 58, Keys 49, Pluck 49, Synth 59, Atmosphere 63, Orchestral 39), 60 generic, 25 out of range; split 377 / 55 / 68 (train / val / test); descriptors per prompt 0: 26, 1: 117, 2: 221, 3: 119, 4: 17; 2,993 user texts, 221 distinct words; held-out words in train/val texts: 0; unseen-word test texts: 111.

Fixed during review of examples: held-out words hidden inside multi-word forms ("crunchy lo-fi"), "with w/ …" in slang, awkward phrase joins (now "with A and B", "…, which …"), suffix "single *sustained* note" contradicting plucks (now "single note"), "an unison" → "a unison", random category shares (now exact quotas).

### Next

Review the vocabulary with the DSP engineer → run `generate_sa3.py --inspect`, then the pilot on a GPU (also `small-sfx` / `small-music` on `--limit 100`) → post-processing in Pitcher (48 kHz, pitch / polyphony checks, loudness, quality filter, CLAP, category tagger).

---

## Round 21 — Vocabulary v2 review; generator supports new slots and per-noun avoid (2026-10-15, 19:11 CEST)

**Why:** the user revised the vocabulary against preset-browser tags, sample-library tags, timbre research and forum usage (`pitcher/text2preset/vocabulary_v2.yaml`: new slots `stereo` and `mood`, 81 values, 19 held-out words, more nouns and styles).

### Checks on v2 (all passed)

Parses; `"808"` is a string; every category `prefer` / `avoid` reference exists; no adjective form under two values; held-out words occur in slot forms only (none in nouns) and every value keeps ≥ 1 train form; templates valid. Minor: `colour:noisy` has opposite `clean`, which lives in `grit` (not used by the generator).

### Generator changes (`pitcher/text2preset/prompts.py`)

| Change | Why |
|---|---|
| `--vocab` option; vocabulary name stored per prompt | v1 and v2 side by side |
| `SPARSE = {style: 0.15, mood: 0.15}`: fuzzy slots added on top of the 0–3 descriptors | v2 asks for mood ≤ 20%; previously `mood` would have been a normal slot |
| word order includes `stereo`, `mood`; unknown slots go last instead of crashing | new slots |
| a noun can be `{forms: [...], avoid: [slot:value or slot]}` (plain lists still work) | nouns that contain slot words clashed with tags: "sawtooth saw lead", "synced square lead", "wide mono lead", "analog FM bass" (5 of 500) |

### Result

`data/text2preset/prompts_pilot_v2.jsonl/.csv`: 500 prompts, 372 distinct words (v1: 221), mood in 16%, stereo in 23%, held-out leaks 0, 123 unseen-word test texts. With suggested per-noun avoid lists (tested on a copy, not yet in v2): 0 real clashes in 2,000 prompts.

**Follow-up (19:16 CEST), decisions by the user:** `stereo` slot removed from v2 (CLAP fingerprints come from a mono mix, so width can't be learned; width becomes a UI control: unison spread, chorus, width parameter); notes stay at C3 for all prompts (matches the Surge renders; listen to the sub / 808 pilot clips); "chord stab" dropped; `colour:noisy` opposite → null; per-noun avoid lists added to `fm_bass`, `saw_lead`, `square_lead`, `sync_lead`; "lush" removed from the held-out list (its only form was in `stereo`); more Pad nouns: check after the pilot. Marked "v2.1" in the file header; previous v2 backed up in the scratchpad. **`vocabulary_v2.yaml` is now the generator default.** Pilot rebuilt from v2.1 (`data/text2preset/prompts_pilot.jsonl/.csv`, replaces the v1 pilot): 500 prompts, category quotas unchanged, 366 distinct words, mood 17%, style 16%, held-out leaks 0, 105 unseen-word test texts; format checks pass.

---

## Round 22 — Snellius package for the Stable Audio 3 dataset (2026-10-15, 19:38 CEST)

**Why:** generate the dataset on the Snellius HPC cluster (Stable Audio 3 can't run on the Intel Mac).

### Implemented

| File | Change |
|---|---|
| `hpc/snellius_sa3/` | **New, self-contained folder to copy to Snellius** (360 KB): `README.md` (step-by-step: HF access, copy, settings, setup, inspect, compare, generate, copy back, cost, troubleshooting), `config.sh` (all settings; site-specific values marked CHECK: `WORK=/scratch-shared/$USER/sa3`, `PARTITION=gpu_a100`, `ACCOUNT`, model, seeds, prompt file, `N_SHARDS=4`), `setup.sh` (login node: uv, clone stable-audio-3 + record commit, `uv sync`, soundfile, optional flash-attn wheel via `FLASH_ATTN_WHEEL`, HF login, `hf download` of the three gated models, `prefetch.py`), `submit.sh` (sbatch with partition / account / array from config), `jobs/` (`00_inspect` 30 min, `01_compare` array of 3 models × 100 prompts × 2 seeds, `02_generate` array of N_SHARDS, 12 h, resumable; `common.sh`), `postcheck.py` (counts, missing, silent, clipped, s per clip, GPU-hour estimate; stdlib only), `pack_results.sh` (one tar incl. logs, prompts, inspect clips), `prompts/prompts_pilot.jsonl` |
| `pitcher/text2preset/generate_sa3.py` | `--shard` / `--n-shards` (prompt i → shard i % n, metadata per shard), `inspect_<model>.flac` per model; copied into the folder (identical) |

Assumptions: no internet on compute nodes (everything downloaded on the login node, jobs use `HF_HUB_OFFLINE=1`); one GPU = `--gpus=1` with 18 cores; big files outside $HOME. Official SURF pages weren't reachable via search, so partition, storage and budget are config values to check (`sinfo -s`, `accinfo`).

### Verification (local simulation)

All shell / Slurm scripts pass `bash -n`. Full workflow simulated with a mock `stable_audio_3` module, a fake `sbatch` and Slurm variables: submit (partition, account, `--array=0-3`), inspect (3 models), compare task (200 clips + postcheck), generate with 4 shards (4 × 500 clips, 0 of 2,000 missing), resume (0 to do), pack (tar of 2,200 clips + logs + prompts). **Not tested on Snellius itself or with the real Stable Audio 3.**

---

## Round 23 — Snellius package: one GPU, environment inside the folder (2026-10-15, 21:06 CEST)

**Why (user):** "don't use 4 gpus in parallel"; the environment should live inside the folder.

### Changed

| File | Change |
|---|---|
| `hpc/snellius_sa3/config.sh` | `N_SHARDS` removed. Environment in the folder: `HERE` (folder path), `SA3_REPO=$HERE/stable-audio-3`, `UV_PROJECT_ENVIRONMENT=$HERE/.venv`, `UV_PYTHON_INSTALL_DIR=$HERE/.uv/python` + `UV_PYTHON_PREFERENCE=only-managed` (uv's own Python, never a system one), uv in `$HERE/.uv/bin`, `PY=$HERE/.venv/bin/python`, `UV_LINK_MODE=copy` (download cache stays in `$WORK`). Model weights and audio stay in `$WORK` |
| `setup.sh` | uv installed into `.uv/bin` (`UV_UNMANAGED_INSTALL`, leaves `~/.bashrc` alone); clone into the folder; `uv sync --inexact` builds `.venv/` in the folder; `uv pip install --python $PY`; `hf` and Python called from `.venv/bin` directly (no `uv run`) |
| `submit.sh`, `jobs/02_generate.slurm` | generate = one job on one GPU (no array); `bash submit.sh generate n` queues n jobs with `--dependency=singleton`, so they run strictly one after the other and each resumes where the previous stopped (runs > 12 h). Postcheck at the end of the job |
| `jobs/01_compare.slurm` | `--array=0-2%1`: the three models run one after the other |
| `jobs/common.sh` | `PY` comes from config.sh |
| `postcheck.py` | no division by zero when the median generation time rounds to 0 |
| `README.md` | folder tree incl. `.venv/ .uv/ stable-audio-3/`, how to activate the environment by hand, don't move the folder after setup, one-GPU workflow, troubleshooting rows |

`generate_sa3.py` keeps `--shard/--n-shards` (unused by the jobs; harmless).

### Verification (local simulation, copy of the folder)

Mock `stable-audio-3` repo (git, pyproject with torch 2.2.2 + huggingface_hub + mock `stable_audio_3`): setup steps 1–4 created `.uv/bin/uv`, `.uv/python/cpython-3.12.15`, `.venv/` in the folder (venv Python → `.uv/python`; no `.venv` inside the repo); re-run is idempotent; `.venv/bin/hf` works; prefetch ok. Fake `sbatch`: inspect (3 clips), compare (3 × 200 clips, postcheck ok), generate (2,000 clips, 0 missing), `generate 2` (both jobs: 0 to do), pack (no environment in the tar). Step 5 (HF login + downloads) not run (needs a real token). **Still not tested on Snellius itself or with the real Stable Audio 3.**

---

## Round 24 — Snellius package: everything inside the folder (2026-10-15, 21:18 CEST)

**Why (user):** everything created on Snellius (environment, models, data) must stay inside the folder; nothing in `/scratch-shared` or elsewhere.

### Changed

| File | Change |
|---|---|
| `hpc/snellius_sa3/config.sh` | `WORK` (/scratch-shared) removed. Inside the folder: `hf/` (model weights + HF token), `audio/`, `results/`, `.uv/cache`; `UV_NO_CONFIG=1`; `XDG_CACHE_HOME`, `XDG_CONFIG_HOME`, `XDG_DATA_HOME`, `TORCH_HOME`, `TRITON_CACHE_DIR`, `PYTHONPYCACHEPREFIX`, `MPLCONFIGDIR` → `.cache/`; `TMPDIR` → `.tmp/` |
| `setup.sh`, `jobs/common.sh` | create the new folders; `chmod 700` on the folder (only you can open it; it holds the HF token) |
| `pack_results.sh` | tar goes to `results/` |
| `README.md` | folder tree with all generated folders; "everything stays in this folder" section incl. what a folder can't contain (Slurm accounting, shell history, login records); `myquota` check (tens of GB); scp path; `rm -rf ~/snellius_sa3` removes everything |

### Verification (local simulation)

Copy of the folder in a fresh fake `$HOME` with `env -i` and a watched outer `TMPDIR`; mock `stable-audio-3` repo, fake `sbatch`. Setup steps 1–4, a real Hugging Face download (public tiny repo, landed in `hf/hub/`), prefetch, inspect, compare (3 × 200 clips), generate (2,000, 0 missing), pack. **Files written outside the folder: none** (home and outer TMPDIR empty). Sizes in the mock: .uv 817 MB, .venv 704 MB (CPU torch; the real CUDA one is bigger), audio 2.2 GB, results tar 2.2 GB. HF login with a real token and the gated downloads not tested.

---

## Round 25 — Snellius package: small-music only (2026-10-15, 21:28 CEST)

**Why (user):** the model "should be small-music".

### Changed

| File | Change |
|---|---|
| `hpc/snellius_sa3/config.sh` | `MODEL=small-music` (was medium); setup, inspect and generate all follow `MODEL` |
| `setup.sh` | downloads and prefetches only `stabilityai/stable-audio-3-$MODEL`; flash-attn step skipped unless `MODEL=medium` |
| `jobs/00_inspect.slurm` | one test clip of `MODEL` (was all three models) |
| `jobs/01_compare.slurm` | **removed** (no model comparison); `02_generate.slurm` renamed `01_generate.slurm`; `submit.sh`: `inspect | generate [n]` |
| `README.md` | only the small-music Hugging Face page to accept; compare step removed (steps now: settings, setup, inspect, generate, copy back); flash-attn reduced to a note; disk estimate ~15–20 GB |
| `pitcher/text2preset/generate_sa3.py` (+ copy in the folder, identical) | `--model` default small-music |
| `TEXT2PRESET_PLAN.md`, `pitcher/README.md` | model = small-music, no comparison |

### Verification (local simulation)

Fresh fake `$HOME`, mock repo, fake `sbatch`: setup shows "not needed for small-music", would download only `stabilityai/stable-audio-3-small-music`, prefetch small-music; inspect writes `inspect_small-music.flac`; generate 2,000 clips in `audio/prompts_pilot/small-music/`, 0 missing; `submit.sh compare` now prints the usage line; pack ok; nothing written outside the folder.

---

## Round 26 — Snellius setup fix: Python pinned to 3.12 (2026-10-15, 21:38 CEST)

**Problem (first real run on Snellius):** `uv sync` failed: "torch==2.7.1+cu126 … can't be installed … You're using CPython 3.15 (cp315), but torch only has wheels for cp310–cp313". With `UV_PYTHON_PREFERENCE=only-managed` and no pinned version, uv installed the newest Python (3.15).

| File | Change |
|---|---|
| `hpc/snellius_sa3/config.sh` | `UV_PYTHON=3.12` |
| `setup.sh` | before `uv sync`: an existing `.venv` with another Python version is removed and rebuilt |

**Verification (local):** mock repo (`requires-python >=3.10`, torch 2.2.2): without the pin the same error appears (cp315); re-running setup in the same folder with the fix prints "removing .venv (wrong Python: Python 3.15.0)" and builds Python 3.12.15 + torch.

---

## Round 27 — Snellius setup fix: Hugging Face login was skipped (2026-10-15, 21:45 CEST)

**Problem (second run on Snellius):** environment ok (.venv 5.2 GB), but the download ran unauthenticated ("Access denied. This repository requires approval"). Cause: in the huggingface_hub version that stable-audio-3 installs, `hf auth whoami` prints "Not logged in" and still exits 0 (`exit()`), so setup.sh skipped `hf auth login`. Setup then printed FAILED and continued to "setup done"; prefetch wrongly said "files cached".

| File | Change |
|---|---|
| `hpc/snellius_sa3/setup.sh` | login check via `HfApi().whoami()` in Python (raises without a valid token); checks again after login; a failed login or download now stops setup with a clear message instead of continuing |
| `prefetch.py` | 401 / 403 / gated errors reported as "NOT downloaded, no access" and exit code 1 (was "files cached") |
| `setup.sh` (local copy) | removed setup output that had been pasted into the file by accident (lines 25–43, inside the Python-version check); the copy on Snellius ran correctly, so it wasn't affected there |

**Verification:** all scripts pass `bash -n` / `py_compile`; step 5 run with a fake `hf` that mimics the Snellius behaviour (whoami exits 0 when logged out): setup now asks for the token, and stops with "login failed" if no token gets stored. A real login and the gated download not testable here.

---

## Round 28 — Post-processing of the Stable Audio pilot (2026-10-16, 08:56 CEST)

**Input:** `data/small-music/` (copied back from Snellius by the user): 2,000 FLAC clips (500 pilot prompts × 4 seeds, small-music, 6 s, 44.1 kHz stereo) + `metadata.jsonl`; 0.42 s per clip (0.23 GPU-hours in total). Prompt file identical to the one used on Snellius (md5 checked).

### Implemented

| File | What |
|---|---|
| `pitcher/text2preset/postprocess.py` (new) | per clip: resample to 48 kHz; trim + level like the Surge renders (render rule: tail ends 60 dB below the loudest 50 ms window; loudest window → −20 dBFS, the Surge median); pYIN pitch (most common note, octave errors folded, lowest strong partial as the heard note against "missing fundamental" errors), number of held notes, harmonic fit; attack, level after 1/2/4 s, end level; percussive share (HPSS) and re-attacks (≥ 6 dB rise in 50 ms; librosa's onset detector counted 33 onsets in a steady saw); spectral flatness and centroid; stereo width. Flags: silent, multi_note, rhythmic (reject); off_pitch, chord_like, cut_off, clipped (marked). Thresholds calibrated on 300 Surge renders run through the same code. CLAP fingerprint, tagger, text similarity + rank among all 500 descriptions, nearest Surge preset, seed consistency; **domain alignment** (mean shift, see below) → `tagged_aligned`, `pa_*`, `surge_nn_*_aligned` |
| `pitcher/text2preset/pilot_report.py` (new) | report.md + summary.json + pitch / confusion / reach plots |
| `.venv` | + librosa 1.0.0, tabulate 0.10.0 |
| `pitcher/README.md`, `TEXT2PRESET_PLAN.md` | run row; build order step 4 ✅ with the results |

Detector checks on synthetic signals before the run: saw C3 / F2 (pitch, +5 semitones), 3-note melody (3 notes), tritone two-note (2), C-E-G and C-Eb-G chords (pYIN says C1; corrected to C3, harmonic fit 0.41 vs 1.0 for single notes), wobble bass (1 note, re-attacks but no percussive share → not rhythmic), pluck, square, drum loop (percussive 1.0), noise. Bugs fixed during the run: duplicate `peak` column; clipping counted after resampling (overshoot) → counted in the file itself; crash when the median pitch fell between two notes (main note = most common note now).

### Results (runs/text2preset/pilot_small-music/report.md)

- **Kept 97.2%**: multi_note 2.9% (Surge renders 5.7%), rhythmic 0%, silent 0%; marked: off_pitch 4.0%, chord_like 1.1%, cut_off 2.1%, clipped 2.5%.
- **Pitch:** 96.0% within ±1 semitone of a C (Surge renders on the C3 key: 89.3%, octave switches in presets); bass at C3, all other categories at C4 (median +12): probably the "MIDI 60 = C3" naming. Only 32% at C3 itself.
- **Domain gap:** mean SA3 vs mean Surge fingerprint cosine 0.44; SA3 clip → nearest clip of another prompt 0.92, Surge → nearest Surge 0.88; trimming/levelling changes Surge fingerprints only slightly (cos 0.96), so it's not the processing. Raw tagger: 23.0% balanced agreement with the prompt category, 56% of clips tagged "Keys"; kNN-10 vote 29.8%. **Mean shift (label-free) 34.5%**, CORAL 31.6%; mean shift estimated on even prompts and tested on odd prompts 35.1%. Aligned recall: Pad 74, Bass 57, Pluck 46, Lead 36, Keys 20, Synth 19, Orchestral 16, Atmosphere 8%.
- **Text ↔ audio (CLAP):** own description's median rank 139 of 500 (random 250); Pad 43, Pluck 62, Bass 74, Atmosphere 252.
- **Descriptor words:** brightness followed (airy +0.45, bright +0.34, warm −0.16, dark −0.34 octaves vs category median); attack followed (slow: 75th pct 0.29 s, percussive 0.007 s, unmarked 0.11 s); grit partly (aggressive −37.7 dB flatness vs −48.1 unmarked; "rough" and "soft" quieter); length barely ("short" −10.9 dB after 2 s vs −13.2 unmarked; "sustained" −8.2).
- **Reach:** nearest Surge preset median 0.657 raw (Surge val → train 0.875), 0.799 aligned. Seed consistency median 0.82.

Open decisions for the main run: octave (prompt "C2" vs rendering Surge at the clip's octave); length words; weak categories (Atmosphere, Orchestral, Synth, Keys) — vocabulary or tagger limits; domain alignment for versions A/B.

---

## Round 29 — Synthetic Surge dataset: plan, deterministic renders, stages 0 and 0b (2026-10-16, 13:50–15:00 CEST)

**Decision (user):** write the synthetic-Surge-dataset plan into `TEXT2PRESET_PLAN.md` and run stages 0 (sensitivity screening) and 0b (perceptual scales) on the Mac.

### Implemented

| File | What |
|---|---|
| `TEXT2PRESET_PLAN.md` | **New §3**: the three directions (synthetic Surge data; prior snaps to reachable, category- and descriptor-constrained Surge fingerprints after alignment; measurable descriptors as within-category percentiles + PCA-16 CLAP), sampling design (50% perturbations of train presets, 30% layered exploration with Sobol through human quantiles, 20% VAE mixtures), stages 0–3, stop rules (coverage + learning curves), throughput, storage (no audio kept), Snellius containment notes, limits. Build order renumbered (§4), new step 5 |
| `surge-rack-main/surge/src/surge-python/surgepy.cpp` | **`seedRandom(seed)`**: reseeds the storage RNG, the noise RNG, `std::rand` (S&H / String / drift / unison), `stmlib::Random` (Twist), the per-note random modulators, every scene LFO's generator, and resets the song position (tempo-synced / free-running LFOs) |
| `surge-rack-main/surge/src/common/dsp/modulators/LFOModulationSource.h` | `reseedRandom(seed)` (one line; the LFO's generator is private) |
| `pitcher/patches/surgepy_local.patch` | regenerated from both files (checked: applies cleanly to upstream) |
| `pitcher/core/surge_io.py` | `render(seed=…)`; **`render_plan(plan, template, seed, **kw)`**: load template → a short priming note on the template **at the same key** → load template → write plan → seeded render |
| `pitcher/core/codec.py` | **Bug fix: `FALLBACK_WAVETABLE = "Sine Octaves"`** for a wavetable / window oscillator whose table is "other" or not given; option `wavetable_fallback=False` for writes onto the preset's own patch (`step1_dataset/roundtrip.py`, version A in `step3_core/listening.py`) |
| `pitcher/synth_data/` (new) | `common.py` (base rows, groups, donors, deterministic worker, `compare` = raw + **level-matched** mel distance), `screen.py` (stage 0), `rulers.py` (stage 0b) |
| `pitcher/README.md` | layout row, run rows, notes on deterministic renders and the wavetable fallback |

### Finding 1: renders were not reproducible, for two reasons

- **Surge's randomness** (oscillator phases, drift, noise; time-seeded). With `seedRandom`, 14 of 20 original presets became bit-identical.
- **State carried over between renders**, which is worse because it isn't random:
  - **Stale wavetable (bug).** A wavetable oscillator whose table is "other" got nothing written and kept the table of whatever the engine rendered last. This affects **21.6% of presets**, so every template render of them (K curves, cleaning, VAE evaluation, tagger checks) depended on the previous preset in that worker, by up to 22 dB.
  - **Hidden DSP state** that survives patch loads (up to 7 dB with identical parameters, routings and wavetables). A fresh engine is deterministic but costs 0.12 s; a short template note first makes the result independent of the previous preset.
- **Result:** in a second engine, rendering in reverse order, **98.3% of 120 template rebuilds are bit-identical (max 0.04 dB)**. Different seeds differ by a median 2.4 dB (p90 4.1 dB): Surge's real per-note variation. 0.13 s per `render_plan`.
- **Effect of the wavetable fix on earlier numbers:** template ceiling (val, K = all + matrix) 40.9% → **39.2%** same character (median 8.26 → 8.17 dB); the 32 affected presets 43.8% → 34.4% (the stale table sometimes happened to fit), the rest unchanged (40.3%). Earlier conclusions stand; the cleaning list (`vae_excluded.csv`) and VAE/tagger checks were computed with the bug and are not re-run (`data/core_curve_matrix_wtfix.csv`).
- **Side effect found and fixed:** the first screening primed at C3, so at C4 portamento glided from the priming note (1.4 dB "effect"); priming now uses the render's key (portamento: 0.0 dB at both notes).

### Stage 0: sensitivity screening (`data/synth/screening*.{parquet,csv}`, `runs/synth/screening.png`, 14,388 renders, 14 min)

32 base presets (4 per category, cleaned train), 224 groups per base (452 core-ranking groups + 95 routing features, active ones only), each group set to a random **donor**'s value (another train preset, active and different; switched-in types bring the donor's knobs), at C3 and C4 with the base's seed. **Level-matched** distance (loudest 50 ms window matched): the synthetic set is level-normalised, and the raw distance ranked pure gain knobs (scene volume, VCA level, velocity → volume) as top "sound" changes; a pure gain scores 0.0, scene volume still changes the sound when a distortion follows it.

- **514 groups tested; 322 on ≥ 5 bases (reliable)**, 192 only on 1–4 (knobs of rare oscillator / effect types: tier is a guess, treated as tier 2).
- **Tiers (reliable):** **36 tier 1** (change the character in ≥ 25% of bases: oscillator mutes and types, filter models / cutoffs / balance, octave, sustain, release, resonator, reverb, delay, wavetable choice and position, the main filter routings), **182 tier 2**, **104 tier 3** (< 1 dB on average at both notes). 40 tier-3 groups are **gated**: > 3 dB in a few presets (noise / osc 3 levels, envelope-shape switches), i.e. they act only when their source is on → explored when the structure turns it on.
- **Not audible in a single fixed-velocity note:** play mode 0.0 dB, portamento 0.0 dB (as predicted); `character` 0.5 dB. **Velocity routings are audible** (fixed velocity = a fixed offset), but they are confounded with the destination's own value (only the sum matters in one note).
- **Effects depend strongly on the patch:** the spread across bases (sigma) is about as large as the mean for most groups → perturbations must start from many different presets, not a few.
- **One step changes the character in 10% of variants**; 54% change the sound by less than 1 dB.
- **C3 vs C4:** per-variant distances correlate 0.95; the main difference is keytracking, which is *weaker* at C4 (key-track root is C4). A second note therefore adds little new sensitivity; its value is matching Stable Audio's octave and making key-tracking identifiable from the pair.

### Stage 0b: perceptual scales (`data/synth/rulers{_raw.parquet,.json}`, `runs/synth/rulers.png`, 13,200 renders, 18 min)

40 most sensitive knobs (level-matched, tested on ≥ 10 bases), 11 equal code steps from 30 presets each, level-matched distance between neighbouring steps; scale = mean of the per-preset cumulative curves (the median jumped for filter balance, which acts only in some filter configurations).

- **Surge's 0–1 codes are mostly perceptually even already:** median unevenness 0.12 (max deviation of the scale from the diagonal).
- **8 knobs need re-mapping (> 0.2):**
  - amp release (half the change in the top 18%: short releases sound alike) and velocity → release;
  - reverb p3 (decay);
  - amp sustain (half the change in the bottom 19%);
  - waveshaper drive and pre-filter gain (level-matched: saturation sets in high in the range);
  - filter 2 resonance;
  - osc 1 level.
- Octave knobs are stepped as expected.

### Next

Descriptors (incl. movement and a better grit measure), the stage-1 sampler using these tiers and scales, the 50k pilot on the Mac.

**Decision (user, 15:10):** no second note at C4; the synthetic set renders one note (C3).

**Check (15:10): label importance (Round 11) vs sound effect (stage 0)**, 237 groups in both: Spearman 0.03 (within-category variation vs sound: 0.26). Of the 51 label-relevant groups tested here, 7 are sound tier 1, 26 tier 2, 18 tier 3 (play mode, portamento, VCA velocity sensitivity: they tell the category from the settings, not from one note's sound). 21 of 28 sound tier-1 groups have no clear label importance (filter models, oscillator mutes, wavetables: they change the sound within every category). Caveat: the mel distance averages over the whole render, so short events weigh little (amp attack: 2.2 dB on average although it separates categories); the attack descriptor must be measured separately.

## Round 30 — Plan revised to the inverter pipeline (2026-10-16, 19:40 CEST)

**Decision (user):** the main pipeline is diverse synthetic Surge parameter vectors → render at C3 → a model that predicts parameters from the waveform (generative inverter) → invert the cleaned Stable Audio clips, refine with a short search, keep only good matches (threshold calibrated on held-out Surge presets, checked with the second CLAP model) → VAE on human + matched presets → text → z prior. Measurable descriptors are no longer a conditioning signal (the user wasn't confident in them). Licence cleared (the Stable Audio creators confirmed this use). "yes first update the plan".

### Changed

| File | What |
|---|---|
| `TEXT2PRESET_PLAN.md` | §1: new **"Main pipeline"** (diagram offline / plugin; the inverter replaces the CLAP-fingerprint bridge); earlier options: A = demo fallback, B (fingerprint / descriptor conditioning, constrained snap) **superseded**, C later. §2.5: clips on C3 or C4 kept (octave handled at matching), 6 s mono, loudness rule of `compare`. **§3** rewritten as training data for the inverter: sampling sources with tiers / scales, diversity measured in sound (coverage > 6.7 dB), stages with 0 / 0b ✅ and new 1b (fill to 100k), adaptive top-up driven by the inverter's error / disagreement; what is stored per row (codes + 64-band mel, float16, 6 s, ~70 KB + CLAP; ~7 GB per 100k; no audio). **New §4** inverter (mel → CNN → VAE decoder heads, same masked loss; generative = conditional VAE over parameters given the audio code, flow matching if it averages; augmentation; scored in sound on human val presets vs template ceiling 39% and retrieval). **New §5** matching (clean → octave +1 trick for C4 clips, reset after → best of 16 → ~50-render search for near misses → threshold from held-out nearest-neighbour reference → second-CLAP check; cost ~1M + ≤ 3M renders, bulk on Snellius). **New §6** VAE on human + matched (z probe before and after; reconstruction check; re-run wavetable-affected checks). **New §7** text → z prior (small conditional flow / diffusion over z, MDN fallback; text encoder comparison; risks; propose–render–select + retrieval prior as safeguards; evaluation on held-out prompts, word swaps, spread, baselines). **§8** build order: 6 z probe, 7 stage-1 sampler + 100k, 8 inverter, 9 main Stable Audio run (parallel), 10 matching, 11 VAE, 12 prior, 13 plugin |

The previous version is backed up in the session scratchpad (`TEXT2PRESET_PLAN_before_round30.md`). Code references to the plan (§3, §3.2, section 2, 2.5) still match.

### Next

z probe on the current VAE; stage-1 sampler.

## Round 31 — z probe on the current VAE (2026-10-16, 20:20 CEST)

**Request (user):** "yes first start with step 6" (TEXT2PRESET_PLAN.md §8 step 6: does the VAE's z hold the category and simple sound measures? A text → z prior can only select what z encodes).

### Built

| File | What |
|---|---|
| `pitcher/step4_vae/zprobe.py` (new) | Renders every cleaned preset's template rebuild (train / val / test, 1,719) at C3 with a fixed seed (deterministic, Round 29); val + test also with a second seed (noise ceiling) and through the VAE (encoder mean → decoder → render). Measures with the Stable Audio post-processing's own code (`text2preset.postprocess.normalise` + `analyse`): brightness (log2 centroid / detected fundamental), attack (log10 s), decay (level at 1 s, clipped at −60 dB), length (log2 s), noisiness (log10 flatness), percussive share. Probes fitted on train, scored on val + test (344, never seen by the VAE): category one-hot alone / z / z + category / the full encoder input + category; ridge and gradient boosting. `--reuse <run>` takes the original renders from another run (only the decoded presets are rendered, ~2 min). Writes `runs/vae/<run>/zprobe/` (measures.parquet, probe.csv, summary.json, zprobe.png) |
| `pitcher/README.md` | Run-order row 21 |
| `TEXT2PRESET_PLAN.md` | §8 step 6 ✅ with the result and its consequence |

Runtime: 2,407 renders + measurements in 14 min (4 workers; pYIN is most of it).

### Results (R² on val + test, best of ridge / boosted; category: accuracy)

| Measure | category alone | z | z + category | full parameters + category | decoder round trip | seed-to-seed (ceiling) |
|---|---|---|---|---|---|---|
| brightness | 0.06 | 0.04 | 0.09 | 0.21 | −0.29 (corr 0.16) | 0.93 |
| attack | 0.39 | 0.15 | 0.44 | 0.59 | 0.02 (corr 0.46) | 0.83 |
| decay | 0.01 | 0.01 | 0.06 | 0.71 | −0.13 (corr 0.16) | 0.91 |
| length | 0.16 | 0.34 | 0.40 | 0.87 | 0.36 (corr 0.64) | 1.00 |
| noisiness | 0.05 | 0.05 | 0.10 | 0.34 | −0.42 (corr 0.18) | 0.97 |
| percussive | −0.04 | 0.00 | −0.04 | 0.29 | −0.82 | 0.79 |
| category (accuracy) | – | 0.33 | – | 0.51 (params alone) | – | – |

- **z holds almost none of what prompts talk about** beyond what the category word already gives: brightness, noisiness, decay and percussiveness are at ~0.1 or below; only length (+0.24 over the category) and a little attack.
- **The information is in the parameters but lost in z:** the full parameters predict decay 0.71 and length 0.87 with the same simple probes. Brightness and noisiness are hard even from the parameters (0.21 / 0.34): with ~1,400 presets the parameter → sound map is too non-linear for a simple model, which is why the inverter needs the large synthetic set.
- **Through the decoder** the measures mostly don't survive (brightness corr 0.16, decoded spread 0.99 vs 1.45 octaves: decoded presets drift to the middle).
- **Lower β doesn't fix it:** β 0.1 (11 active dims) brightness 0.14, noisiness 0.15; β 0.02 (all 12 active, KL 2–4 per dim) brightness 0.12, noisiness 0.10; decoder round trip no better. The bottleneck is a latent learnt from parameter reconstruction on ~1,400 presets with no sound in the loss, not the KL weight.
- Category from z: 33% (chance 12.5%): low by design, since the CVAE's encoder gets the label.
- Silent renders: 0% of rebuilds, 0.9% of decoded presets.

### Consequence

The text → z prior can't work on this z. Before step 12 the latent must become sound-aware. Two routes, chosen once the inverter exists (step 8), then re-probed:
1. VAE trained on the synthetic set too (100k+ presets instead of ~1,400) with a sound term in the loss (predict the render's mel code / measures from z).
2. The prior targets the inverter's audio code c (learnt from mel spectrograms, so sound-aware by construction): text → c → inverter → preset.

### Next

Stage-1 sampler (§3) → 50k pilot → 100k.

## Round 32 — Stage-1 sampler and the 50k pilot (2026-10-16, 22:45 CEST)

**Request (user):** "explain how the stage-1 sampler will work", then "yes start building it"; later "where are these new sampled sounds stored? I would like to check them".

### Built

| File | What |
|---|---|
| `pitcher/synth_data/sampler.py` (new) | `Space` (extends `common.Base`): tiers (stage 0; untested / unreliable groups and gated tier 3 count as tier 2), perceptual rulers (stage 0b), human value distributions, structure frequencies, a Gaussian copula over the 455 knobs + routings (pairwise normal-score correlation over the train presets, shrunk 0.7, eigenvalue floor). **Activity rules** for canonical rows (encode's logic: oscillator audible via mute / level / ring / FM, osc-type blocks, filter type ≠ off, LFO used by a routing, waveshaper, noise, FM depth, FX present / send). `complete()` fills features that become active from one donor per block; `finish()` then masks inactive features with NaN (= the inverter's target mask). Three sources: **perturb** (0–3 switch-group donor swaps; knobs moved on the ruler by tier; routing depths moved, Poisson-count additions / removals; strengths small / medium / large), **layered** (human structure with weight frequency^-0.5 per preset, 0–3 switch mutations, every active knob from scrambled Sobol → copula → human quantile; filter knobs from presets with the same filter type; gain-only knobs keep the structure preset's value; 5% of values uniform on the ruler), **vae** (per-category latent GMM → decoder; routing depths |d| < 0.04 snapped to 0). A fast `row()` from arrays (0.2 ms instead of 22 ms; identical output). `python -m pitcher.synth_data.sampler` checks the activity rules |
| `pitcher/synth_data/stage1.py` (new) | Chunked runner (chunk 0 = the 1,375 human train presets; then chunks of 5,000 with fixed shares, shuffled; resumable). Rows are built in the workers; render at C3 with seed = id + 7, at most 6 s (2 s hold + 4 s release); invalid = NaN or loudest 50 ms window < −90 dBFS; flags hot (> 0 dBFS raw) and cut off at 6 s. Stores codes (kept features, NaN = inactive), a 64-band mel of the level-normalised first 6 s (559 × 64, float16, floored 80 dB below its peak) and ~1% audio as FLAC. Coverage: PCA of time-pooled mels for candidates (k per render + earlier rows of the same 2,000 block), exact `mel_distance_db` rule on the stored mels; recall check against 200 candidates. `--listen` builds a listening folder |
| `pitcher/synth_data/common.py` | `render(..., **kw)` passes render options; `run_pool(..., init=)` takes another worker initialiser |
| `pitcher/README.md`, `TEXT2PRESET_PLAN.md` | Module rows / run order; §3.2 stage 1 ✅ with results, 1b proposal; §8 step 7 updated |

### Checks before the run

- **Activity rules vs encode's masks** (train presets): 98.3% of (preset, feature) cells agree. Every difference is "mask active, rule not": LFOs or oscillator levels modulated by routings outside the 95-routing matrix (macros, ...), which the template rebuild doesn't write, so the rule is right for rebuilt rows.
- **Doing nothing changes nothing:** a perturbation with all steps at zero (but `complete` + `finish`) renders identical to the base (0.00 dB).
- **Strength calibration** (40 rows each, level-matched distance to the base preset): first version too strong (small: median 8.5 dB; each of ~85 unused routings had its own 2% chance → ~1.7 routings added per row). By component: one added routing 0.5 dB, one switch 2.2 dB (30% over 6.7), tier-1 knobs at sd 0.03 4.4 dB, 20% of tier-2 knobs at sd 0.02 2.9 dB. Final: small sd 0.015 / 0.010 (median 4.4 dB, 72% under 6.7), medium 0.05 / 0.035 (9.6 dB), large 0.15 / 0.10 (17.7 dB); routing changes as Poisson counts (0.1 / 0.3 / 0.6 added).
- **Layered silence:** first test 18% silent → causes: master volume drawn low (median 0.50 vs 0.86) and filter knobs drawn from all filter types pooled (type 3 over-represented among silent rows). Fixed (gain-only knobs kept, filter-type-specific distributions, widening 15% → 5%; with ~100 active knobs, 15% per knob meant ~15 random values per row). Many remaining "silent" rows were just quiet (−61 to −75 dBFS): Surge renders in float, so the threshold went to −90 dBFS (output level is set after matching).
- **Speed:** 7–9 renders/s at first. Profile: render 90 ms, base rebuild 22 ms, row build 25 ms, mel 10 ms per row; scaling stopped at 2 workers (1 → 5.3/s, 2 → 10.5/s, 4 → 8.7/s; load average 14 from other apps, thermal level 121). Fixes: one maths thread per worker, fast `row()`, 6 s render cap, `.metadata_never_index` against Spotlight. The pilot then ran at 17.5–20.8/s.

### Pilot results (50,000 synthetic + 1,375 human, 47 min; 3.6 GB; coverage with k = 40: recall of the nearest 93%, new / not-new call 100% vs 200 candidates)

| Source | Rows | Invalid | Hot (> 0 dBFS raw) | Share new (> 6.7 dB from all earlier) | Median distance to nearest earlier | Curve: first → last 2,000 |
|---|---|---|---|---|---|---|
| human (reference) | 1,375 | 0% | 1% | 75% | 8.1 dB | – |
| layered | 14,364 | 4.2% | 15% | **53%** | 6.8 dB | 74% → 49% |
| perturb | 24,684 | 1.3% | 5% | 28% | 5.5 dB | 48% → 20% |
| vae | 9,998 | 0.0% | 1% | 19% | 5.4 dB | 43% → 13% |

- Perturb by strength: small 9% new (median 3.9 dB: the intended near neighbours), medium 29%, large 47%.
- Same picture in every category (layered 48–58% new, perturb 21–35%, VAE 8–30%; VAE weakest for Synth, Bass, Lead).
- Layered is far from saturated after 50k; perturbations and VAE samples fall off (VAE: consistent with the z probe, Round 31: decoded presets drift to the middle).
- First pass with k = 10 (recall 86%) gave nearly the same shares (layered 55%, perturb 29%, VAE 19%); kept as `coverage_k10.parquet` / `report_k10.json`.

Listening folder: `data/synth/stage1/pilot/listen/` (478 saved renders under descriptive names + 242 base-preset renders for the perturbed ones + `index.csv`).

### Next

Stage 1b (fill to 100k) with shares from these curves (proposal ~60% layered / ~35% perturb with fewer small steps / ~5% VAE); then the first inverter (§4).

## Round 33 — Notes for the next VAE (2026-10-17)

**Request (user):** discussion of Le Vaillant et al., "Improving Synthesizer Programming From VAE Latent Space" (DAFx 2021, IEEE 9768218); a joint audio + parameter VAE ("cast the audio into the mel and append this to the feature vector"); how categorical one-hots work; "keep the β warmup in mind".

| File | What |
|---|---|
| `TEXT2PRESET_PLAN.md` §6 | New "Training notes for the next VAE": β warm-up from 50% (paper: from 0% hurt validation; our `h256_d01` warms up 0 → 0.5 over 30 epochs and its best epoch is 33), ord knobs as categories (NumCat++), the joint audio + parameter VAE with modality dropout, and the v1 / v2 / v3 comparison protocol |

No code changed.

## Round 34 — Inverter v0, clip matching, text → preset command (2026-10-17, 13:35 CEST)

**Request (user):** "the complete pipeline up and running. So that a prompt "dark bass sound" is able to output parameters" → "okay then start working on the inverter".

| File | What |
|---|---|
| `pitcher/inverter/__init__.py`, `data.py` (new) | Targets in the VAE `Spec` layout; `MelSet` (all valid stage-1 rows in memory, 2% synthetic val); `full_row` (kept features + frozen modes → full codes row; 0.0 dB vs the stored renders on 40 val presets); `build_eval` / `load_eval`: the 344 human val + test presets rendered like the training rows → `data/inverter/eval/` |
| `pitcher/inverter/model.py` (new) | Mel (peak-relative, 4-frame time pooling → 140 × 64) → 4 conv blocks 16–32–64–128 → 512 → audio code c (256) → the VAE decoder's heads; mel-domain augmentation (EQ tilt, raised floor, onset delay, noise); 2.84M parameters |
| `pitcher/inverter/train.py` (new) | AdamW 2e-3, OneCycle, batch 128, masked recon loss of the VAE; model selected by human-val loss → `runs/inverter/<run>/{model.pt, config.json, metrics.csv}` |
| `pitcher/inverter/evaluate.py` (new) | Sound score: 16 candidates (0 = most likely, others with sampled switches; routing snap), oracle, retrieval (PCA k = 20 + exact distance), combined; other seed for every render; switch accuracy; listening examples → `runs/inverter/<run>/eval/` |
| `pitcher/text2preset/match.py` (new) | Pilot clips → presets: cleaned clip mel, 16 inverter candidates + retrieval at C3, best kept, octave reset (scene octave, remainder on the oscillators), check at the clip's own note, keep ≤ 6.7 dB → `data/text2preset/<name>/matched*.parquet`, `matched_summary.json` |
| `pitcher/text2preset/make.py` (new) | Sentence → presets (retrieval prior): CLAP text index of the kept clips' paraphrases (`text_index.npz`), top 12 rendered, score = text sim + CLAP audio–text sim, best 4 → `runs/text2preset/make/<sentence>/` (.fxp, .wav, info.json, fxp reload check). Category-free prompts are named `open` |
| `pitcher/synth_data/stage1.py` | `render_mel(row, seed, note)` and `mel_view(x)` split out for reuse |
| `pitcher/core/surge_io.py` | `SurgeIO.save_patch(path)` (surgepy `savePatch`) |
| `pitcher/README.md`, `TEXT2PRESET_PLAN.md` | `inverter/` row and commands; §8 steps 8, 10, 12 with results |

### Training (`runs/inverter/v0`, 49,460 rows of which 1,375 human; ~7.5 min per epoch on the Mac)

| Epoch | Train | Synthetic val | Human val | Osc type acc |
|---|---|---|---|---|
| 1 | 47.95 | 43.26 | 48.55 | 0.38 |
| 3 | 38.69 | 37.86 | 44.49 | 0.49 |
| **7** | 31.85 | 33.65 | **44.42** | 0.58 |
| 11 | 27.07 | 31.49 | 46.28 | 0.56 |
| 15 | 23.67 | 31.21 | 47.82 | 0.55 |

Stopped after epoch 15 of 20: human-val loss rose eight epochs in a row; the kept model is epoch 7.

### Sound score (344 human val + test presets; same character ≤ 6.7 dB)

| | Same character | Median |
|---|---|---|
| oracle (true preset, other seed) | 96% | 2.3 dB |
| inverter, most likely values | 4% | 14.5 dB |
| inverter, best of 16 | 16% | 9.4 dB |
| **retrieval (nearest of the 50k)** | **65%** | 6.1 dB |
| combined | 66% | 6.0 dB |

- Retrieval by category (test): Orchestral 83%, Synth 75%, Bass 73%, Lead 71%, Pad 67%, Keys 64%, Atmosphere 58%, Pluck 24%.
- Switch accuracy: osc type 0.64, filter model 0.24, filter configuration 0.51, waveshaper 0.59.
- **Swap test** (80 presets: true preset with one group replaced by the prediction): knobs 15.0 dB, categoricals 12.3, routings 6.2, on/offs 5.4 (true 2.3). The predicted routings switch on ~11 per preset (true 3.5).
- **Median knob correlation** (predicted vs true): human train 0.68, synthetic train 0.47, synthetic val 0.42 (perturb 0.55, VAE 0.56, layered 0.08), human val / test 0.26. Weak everywhere; parameter-only regression doesn't learn what matters for the sound.

### Matching the 2,000 pilot clips (1,943 kept by post-processing; 41 min)

- **14.6% kept** (284 clips), median distance 7.7 dB (inverter's best 11.7, retrieval 7.7); retrieval wins 1,928 of 1,943.
- By category: Bass 44%, Synth 13%, Orchestral 12%, Atmosphere 12%, Keys 12%, Lead 12%, Pluck 6.5%, Pad 5%. By octave: C2 43%, C3 24%, C4 8%, C5 6%.
- Tried: pitch-shifting the clip to C3 before retrieval, checking at the clip's own note (60 clips per octave): C2 43% → 8%, C4 3% → 0%, C5 7% → 2%. Rejected; the octave-reset path stays.
- The 4-clip smoke-test `matched.parquet` from earlier is overwritten.

### Text → preset (`make.py`, first runs)

- "dark bass sound": bass, sub bass, 808, FM bass (matches 5.8–6.1 dB).
- "long faded dark pad": 2 dark textures (category-free prompts), string pad, choir pad.
- "quick aggressive guitar-like pluck": harp and music-box plucks, but also a screech lead and a brass (few plucks were matched).
- **.fxp reload check:** the saved .fxp, reloaded and rendered, is 0–1.3 dB from the row for most presets, 2.3 and 2.6 dB for two. Values survive the save; cause not found yet. Acceptable for v0.
- Surge's "database is locked" SQL stack traces during these runs are harmless noise.

### Next

Inverter v1 / short search to raise the matched share (the limit of the text side now); main Stable Audio run once the vocabulary is final.

## Round 35 — Listening test for the matching threshold (2026-10-17, 20:57 CEST)

**Request (user):** "build the page so I can listen and possible set a new threshold". Background: matching keeps 15% of the pilot clips at 6.7 dB (calibrated Surge vs Surge, Round 14) but 60% at 8 dB and 98% at 10 dB; half the clips sit at 7.1–8.4 dB.

| File | What |
|---|---|
| `pitcher/text2preset/listen_threshold.py` (new) | 30 blind pairs: cleaned Stable Audio clip vs its matched, octave-reset Surge preset rendered at the clip's own note (seed 7; re-measured distances equal the stored ones). Bands (max of match / after-reset distance): ≤ 6.7 dB 4 (control), 6.7–7.3 6, 7.3–7.9 6, 7.9–8.5 6, 8.5–9.5 6, > 9.5 2; round-robin over categories. Page from the Round 13 template (same scale: same sound / same character / different / broken; db collection `ratings`) |
| `runs/text2preset/listen_threshold/` | `index.html`, `pairs.json`, `audio/*.mp4` (AAC, 6.0 MB) |
| `data/text2preset/pilot_small-music/listen_key.csv` | Key: pair → clip, category, band, distances |

Published: https://claude.ai/artifact/ULZw3eqkaniM4DQo2eenqJ (private; db read once: reachable, empty). Results and the new threshold follow after rating.

### Results (all 30 pairs rated, 2026-10-17)

Ratings: 21 different, 8 same character, 1 same sound, 0 broken. All 30 matches came from retrieval.

| Band (dB) | Pairs | Same character or better |
|---|---|---|
| ≤ 6.7 (kept before) | 4 | 2 |
| 6.7–7.3 | 6 | 4 |
| 7.3–7.9 | 6 | 0 |
| 7.9–8.5 | 6 | 1 |
| 8.5–9.5 | 6 | 1 |
| > 9.5 | 2 | 1 |

- ≤ 7.3 dB: 6 of 10; 7.3–8.5 dB: 1 of 12 (Fisher p = 0.02). Above 7.3 dB the matches are mostly a different sound, so 8 dB is not acceptable.
- **New threshold: 7.3 dB** (`match.py` `MATCH_DB`; same quality as the 6.7 dB set on these ratings). Kept: 284 → **653 clips** (33.6%), prompts with a kept clip 167 → **309**. By category: Bass 70%, Synth / Orchestral / Keys 34%, Lead 29%, Atmosphere 28%, Pluck 25%, Pad 18%; by octave C3 50%, C4 23%.
- Even accepted matches are only ~55–60% "same character": the mel distance is a noisy judge for Stable Audio vs Surge. CLAP audio similarity (clip vs match, clap-htsat-unfused) separates the ratings a little better (AUC 0.70 vs 0.66 for mel distance; no "OK" pair below 0.38); `larger_clap_music` gave near-identical embeddings (0.99+) for all pairs, unusable here. A CLAP check is worth adding once more ratings exist (30 is too few to fit a two-signal rule).
- Files: `data/text2preset/pilot_small-music/listen_ratings/` (db export), `listen_results.csv` (key + rating); `match.py --rekeep` re-applies a threshold without renders.
- Text index rebuilt; the three test prompts again: "dark bass sound" 4 basses; "long faded dark pad" texture, soundscape, 2 pads; "quick aggressive guitar-like pluck" 3 plucks + 1 open sound (before: a lead and a brass).

## Round 36 — Joint VAE (parameters + sound, k = 48) (2026-10-17, 21:31 CEST)

**Request (user):** build the joint VAE as a separate model (keep the current VAE); "are you sure that the mels stored in 64 numbers is the most efficient?" → measured → "start building but with k=48".

### Sound code size (measured; 600 held-out stage-1 rows, reconstruction distance on the threshold's dB scale)

| k | 8 | 16 | 32 | 48 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|---|---|
| median dB | 4.36 | 3.53 | 2.85 | 2.64 | 2.43 | 2.10 | 1.85 | 1.67 |
| 90th pct dB | 6.95 | 5.67 | 4.71 | 4.14 | 3.88 | 3.56 | 3.27 | 2.95 |

Time pooling alone (4 frames → 1) 0.95 dB; Surge's render-to-render noise 2.3 dB; "same character" 6.7 / 7.3 dB. No paper fixes k for this setup; 64 had been a rule of thumb. User chose 48.

| File | What |
|---|---|
| `pitcher/joint_vae/__init__.py` (new) | Package; the parameter-only CVAE (`pitcher/step4_vae`, `runs/vae/h256_d01`) is unchanged |
| `pitcher/joint_vae/sound_code.py` (new) | Mel → 4-frame pooling, peak-relative, floor −80 dB → PCA 48 (fit on 12k train rows), one global scale (mean code variance 1, no whitening) → `data/joint_vae/sound_pca.npz`, `codes_pilot.npy` (50,421 rows, MelSet order), `codes_eval.npy` (344); check: 96.0% variance, reconstruction median 2.64 dB / 90th 4.61 dB on synthetic val |
| `pitcher/joint_vae/data.py` (new) | `JSpec`: 435 num, 85 bool, 111 cat incl. 20 stepped knobs with ≤ 32 levels as categories (NumCat++; keytrack root stays continuous); `to_row` (verified lossless: 0.00 dB on 10 presets); labels 8 + unknown; `JointSet` (stage-1 rows without mels, same order / val split as MelSet) |
| `pitcher/joint_vae/model.py` (new) | Params expert + sound expert → product of experts with a N(0,1) prior expert → z (32) → params decoder (VAE heads) + sound decoder (48); loss = masked param terms + 0.5·SSE(sound) + β·KL (free bits 0.05); 2-layer MLPs, hidden 512 |
| `pitcher/joint_vae/train.py` (new) | Modality dropout (both 50 / params 25 / sound 25%), β 0.25 warmed up from 50% over 20 epochs, AdamW 1e-3 cosine, batch 256; human-val score = invert params + hear sound; early stop 25 → `runs/joint_vae/<run>/` |
| `pitcher/joint_vae/evaluate.py` (new) | Inverter role (16 candidates: mean, 7 posterior draws, 8 sampled switches; other seed) vs oracle / retrieval / inverter v0 (from `runs/inverter/v0/eval`); z probe with the zprobe measures |
| `pitcher/text2preset/match.py` | `--joint <run>`: candidates from the joint VAE; results to `matched_<run>.parquet` etc. (the 7.3 dB matches stay) |

### Training (`runs/joint_vae/j48`, 49,460 rows, ~13 s / epoch, 2.0M parameters)

Best epoch 13 of 38 (early stop). Human val: invert params 53.0, hear sound 14.2, both 38.8 / 2.9. Synthetic val kept improving (invert 49 → 42, hear 21 → 12.5) while human val stalled after ~epoch 13: same synthetic → human gap as inverter v0.

### Results (344 human val + test presets)

**z probe** (R², best of ridge / boosted; fit on the 1,375 train presets):

| Features | brightness | attack | decay | length | noisiness | percussive | category acc |
|---|---|---|---|---|---|---|---|
| old VAE: z + label | 0.09 | 0.44 | 0.06 | 0.39 | 0.10 | −0.04 | – |
| old VAE: raw params | 0.21 | 0.59 | 0.71 | 0.87 | 0.34 | 0.29 | 0.51 |
| **joint: z from params** | **0.26** | 0.25 | **0.27** | **0.66** | **0.35** | 0.04 | 0.33 |
| joint: z from sound | 0.72 | 0.38 | 0.78 | 0.89 | 0.79 | 0.09 | 0.44 |
| joint: z from both | 0.69 | 0.34 | 0.66 | 0.88 | 0.81 | 0.07 | 0.35 |

z encoded from parameters alone is now sound-aware: brightness 3×, decay 4.5×, noisiness 3.5×, length 1.7× the old VAE; brightness and noisiness even above the raw parameters. Attack and percussive stay weak (attack comes mostly from the label).

**Inverter role** (same character ≤ 6.7 dB, other seed): oracle 97%, inverter v0 best-of-16 16% (9.4 dB), joint top 3% (12.9 dB), joint posterior draws 11%, **joint best-of-16 15% (9.1 dB)**, retrieval 65% (6.1 dB), joint or retrieval 65%. Best candidate: 7 mean, 142 posterior draws, 195 sampled switches. Key switches from sound: osc type 0.53, filter model 0.18, filter config 0.44, waveshaper 0.57.

**Conclusion:** the joint VAE fixes the latent (z is organised by sound, which the text → z prior needs) but not inversion: decoding parameters from a sound still averages over the many presets that make it. Re-matching the pilot clips with it would not beat retrieval (not run). Next for matching: a render-based local search from the best retrieved preset (§5 step 4).

## Round 37 — Round-trip check of the decoders in sound (2026-10-17, 21:41 CEST)

**Request (user):** "yes do the check" (does a point in z decode into a preset that sounds like it? = the ceiling of text → z → decoder generation).

| File | What |
|---|---|
| `pitcher/joint_vae/roundtrip.py` (new) | Human val + test presets → encoder → z → decoder → preset → render (other seed) vs the original's render; joint VAE from params / from both / best of 8 (mean + 7 posterior draws), the old VAE h256_d01, oracle → `runs/joint_vae/<run>/roundtrip/` |

### Results (344 presets; same character ≤ 6.7 dB)

| Decoded from | Same character | Median | 90th pct |
|---|---|---|---|
| oracle (true preset, other seed) | 97% | 2.3 dB | 4.4 dB |
| old VAE, params → mean | 0% | 18.0 dB | 28.5 dB |
| joint, params → mean | 2% | 17.1 dB | 27.6 dB |
| joint, params + sound → mean | 2% | 14.7 dB | 23.7 dB |
| joint, params, best of 8 (render-select) | 5% | 13.5 dB | 22.6 dB |

Old VAE consistent with Round 16 (0%, 18.3 dB).

**Swap test** (86 presets; true preset with one group from the joint decoder's round trip): all 17.4 dB; continuous knobs 15.1, categoricals 12.5, routings 6.8 (16.3 active vs 2.7 true; round trip without the routing snap), stepped knobs 4.7, on/offs 3.5. Knobs by family: filter cutoff (2 knobs) 6.9, other 6.6, envelopes 6.3, effects 6.1, oscillator / filter parameters 5.1, levels 4.8, pitch 3.5, resonance 3.5, LFOs 2.9. No single family: many small errors over ~40–60 active knobs add up. Accuracy on active switches: on/off 94%, categoricals 75% (filter model 36%, LFO shapes 37–55%, oscillator octaves 44–46%).

**Conclusion:** neither decoder can be used as-is for generation: a preset decoded from z, even from the exact parameters, rarely keeps the character. The z is sound-aware (Round 36) but the parameter decoder is too imprecise in sound: per-feature MSE / CE through a 32-dim bottleneck gets most values roughly right and the audible ones not right enough.

## Round 38 — Prompt file for the main Stable Audio run (2026-10-17)

**Request (user):** run the main Stable Audio run (~12k prompts × 4 seeds) on Snellius; the vocabulary stays as it is (no extra Pad nouns, sub / 808 nouns kept: 308 prompts).

- The Snellius folder from the pilot lives at `~/runs/snellius_sa3` (not `~/snellius_sa3`; environment, weights and the 2,000 pilot clips intact, so no new setup). A first generate job (28245769) still had the pilot prompt file: 0 clips to generate. On Snellius only `PROMPTS` in that folder's config.sh is switched (sed), so partition / budget stay; copy back from `runs/snellius_sa3/audio/prompts_main/small-music/`.
- `data/text2preset/prompts_main.jsonl` (+ .csv): 12,000 unique prompts, vocabulary v2.1, seed 0. main 9,960 / generic 1,440 / out-of-range 600; train 9,086 / val 1,210 / test 1,704; md5 `2e03cf0c1d40a3d72f3dcb0f286269dc`.
- Copied to `hpc/snellius_sa3/prompts/`; `hpc/snellius_sa3/config.sh`: `PROMPTS="prompts/prompts_main.jsonl"` (was the pilot file).
- Expected: 48,000 clips, ~5.6 GPU-h at the pilot's 0.42 s per clip (one 12 h job), ~13 GB of FLAC (the pilot's 2,000 clips were 547 MB).

## Round 39 — Sound-weighted parameter loss for the joint VAE (2026-10-17)

**Request (user):** "yes start with option 1" (sound-weighted loss retrain, while the main Stable Audio run is on Snellius).

### Changed

| File | What |
|---|---|
| `pitcher/joint_vae/weights.py` (new) | Per-feature audibility weights from the stage-0 screening (`mu_norm` per group, all 631 features mapped via `Base.groups()`; groups on 1–4 bases shrunk to the kind's median with 5 pseudo-bases). Weight = (dB / median dB of the kind)^power, clipped [0.1, 10], mean 1 within num / bool / cat. Writes `data/joint_vae/loss_weights.csv`. Heaviest: amp sustain, filter cutoffs / balance / model, octave, osc type, osc 1 mute; lightest: portamento, LFO→LFO routings, LFO delay flags. (Stage-0b rulers only cover 40 knobs.) |
| `pitcher/joint_vae/model.py` | `recon(params, b, w)`: weighted masked terms; `loss_fn(..., w=None)` |
| `pitcher/joint_vae/train.py` | `--weights none|screening`, `--w-power`; human-val score uses the same weights |

### Results (344 human val + test presets; round trip = params → z mean → decoder → render, other seed; swap test on 86 presets)

| | j48 (β 0.25) | j48w (weighted) | j48w_b05 (weighted, β 0.05) |
|---|---|---|---|
| KL (nats) | ~30 | ~30 | ~63 |
| round trip params → mean | 1.7% (17.1 dB) | 0.9% (16.7) | 1.2% (16.5) |
| both → mean | 1.7% (14.7) | 2.6% (14.7) | 3.2% (15.1) |
| best of 8 | 4.7% (13.5) | 2.9% (13.0) | 4.1% (14.3) |
| swap: knobs / cats / stepped / routes / bools (dB) | 15.1 / 12.5 / 4.7 / 6.8 / 3.5 | 15.2 / 9.3 / 4.2 / 6.4 / 3.1 | 16.0 / 7.2 / 3.1 / 7.3 / 3.0 |
| cat / bool accuracy | 75% / 94% | 74% / 95% | 77% / 96% |
| decoded active routes (true 2.7) | 16.3 | 17.0 | 19.9 |

- Weighting + a wider bottleneck help the **switches** (cats 12.5 → 7.2 dB, stepped 4.7 → 3.1) but **not the continuous knobs** (~15–16 dB) nor the spurious routes. The round trip as a whole stays at 1–4%.
- So the knob error is not the loss weighting and not the information bottleneck: per-knob independent regression (averaged values summed over hundreds of knobs) + spurious weak routings remain → render-based search and / or a joint (flow-matching) decoder.
- j48w_b01 (β 0.01, KL ~105): trained (best epoch 9), **round trip not run** (stopped by the user for the day).
- Swap test script: scratchpad `rt_ablate.py` (`RUN=<run> PYTHONPATH=.`). Logs: `runs/joint_vae/j48w*_{train,rt}.log`.

### Next

- Round trip of j48w_b01: `.venv/bin/python -m pitcher.joint_vae.roundtrip --run j48w_b01 --old ''`.
- Main Stable Audio run on Snellius (job started 17 Oct evening, `~/runs/snellius_sa3`) → rsync back to **`data/prompts_main/small-music/`** (user, 18 Oct) → postprocess → match (~17 h on the Mac at pilot speed).
- Decide: render-based local search after decoding (also for matching) vs a flow-matching decoder; route sparsity (decoded 17–20 active routes vs 2.7).

## Round 40 — Main Stable Audio run copied back and verified (2026-10-18)

- Snellius job 28246111 (`~/runs/snellius_sa3`): 48,000 clips (12,000 prompts × 4 seeds, small-music), missing 0, silent 2, clipped 2,046 (4.3%; pilot 4.0%).
- Copied to **`data/prompts_main/small-music/`** (user's choice). The first copies stopped at 12,641 files: macOS `/usr/bin/rsync` is openrsync (protocol 29), which re-checked the existing files (the first copy hadn't kept the timestamps) and quit mid-way, leaving one empty file. Fix: Homebrew rsync 3.x with `--size-only --partial`.
- Check (Mac): 48,000 FLAC + `metadata.jsonl` with 48,000 lines, no missing / extra / duplicate / empty files, 13.7 GB; every clip decodes at 44.1 kHz, exactly 6 s, peak equal to the metadata; every metadata prompt text equals `data/text2preset/prompts_main.jsonl`. (Scratchpad `flac_check.py`.)
- Next: `postprocess --audio data/prompts_main/small-music --name main_small-music --prompts data/text2preset/prompts_main.jsonl`, then `match` (first `--limit 500` for speed).

## Round 41 — Main clips post-processed; XY-pad concept page (2026-10-18)

### Post-processing of the main Stable Audio run
`postprocess --audio data/prompts_main/small-music --name main_small-music --prompts data/text2preset/prompts_main.jsonl` (Mac, 4 h 40 min: analysis 3 h 10 min, CLAP ~2.5 h). **Kept 96.3% (46,220 of 48,000)**; multi-note 3.7%, silent / rhythmic 0%; marked: off-pitch 5.0%, chord-like 1.6%, cut-off 2.2%, clipped 2.6%. Pitch: 95.0% within ±1 semitone of a C, median +12 semitones from C3 (bass at C3, the rest at C4, as in the pilot). Tagger agreement after alignment 32.7% (pilot 34.5%). Output `data/text2preset/main_small-music/`, report `runs/text2preset/main_small-music/report.md`.

### XY-pad concept (user idea: 2 or 3 continuous control groups on a pad, a list of groups ranked by label impact)
| File | What |
|---|---|
| `pitcher/xypad/analyse.py` (new) | 123 continuous knobs (no switches, choices, stepped knobs or routing depths; active in >= 25% of the 1,719 cleaned presets). Correlation: only 3.1% of knob pairs have abs(Spearman) >= 0.3, 8 small clusters at 0.5 (several chance pairs, e.g. LFO hold + delay return) → **61 functional groups** (same job on several units). Label importance: random forest (balanced accuracy 0.577, shuffled 5-fold CV), grouped permutation importance. Exports per-preset knob values + a 60-tree depth-10 forest (0.38 on unshuffled folds = unseen sound packs) for the page. Writes `data/xypad/{knobs.json, importance.csv}` |
| `pitcher/xypad/page.template.html` + `build_page.py` (new) | Page: XY pad + optional Z slider, category map (forest on a 40×40 grid, other knobs at a category's typical values), real presets as dots, category probabilities, knob readout in real units, group list ranked with X / Y / Z buttons, rough Web Audio preview synth (30 groups audible) with a C3–C4 keyboard. `runs/xypad/xypad.html`, published https://claude.ai/artifact/UPQ59qySKLzYr1ffqBv7FB |

Ranking (top): amp attack +6.3 pts, glide +3.4, amp release +3.3, amp sustain +3.0, osc 2/3 detune +1.3, cutoff +0.9, velocity → volume +0.8, reverb tone, filter env attack / sustain. About half the groups are within the noise.

### Next
- Matching the 46,220 kept clips (`match --audio data/prompts_main/small-music --name main_small-music`, first `--limit 500` for speed).
- Round trip of j48w_b01; render-based search vs flow-matching decoder.

---

## Round 42 — Code pushed to GitHub (2026-10-21)

Private repo **https://github.com/mvork/pitcher** (branch `main`, 75 files, ~0.9 MB): `pitcher/`, `.gitignore`, the plans and this log (docs unchanged, user decision).

`.gitignore` keeps out: `hpc/` (cluster package, never in a repo), credentials (`hf/`, `.env`, `*token*`), `data/`, `runs/`, `logs/`, `surge-rack-main/`, `ignore/` (surgepy build), model / array / audio / preset files (`*.pt`, `*.onnx`, `*.npz`, `*.parquet`, `*.wav`, `*.flac`, `*.fxp`, …), `.venv/`, `__pycache__/`, `.DS_Store`. Checked before pushing: no tokens in the tracked files, nothing from `hpc/`. Repo-local setting only: `http.postBuffer` (first push timed out with HTTP 408).

Same day, user request: `runs/` and `logs/` added to the repo too (272 files, 19 MB: logs, reports, CSV / JSON results, plots, the XY-pad page, the listening-test mp4s, `latent_gmm.pkl`); models / arrays / audio inside them stay out via the file-type rules. Scanned first: no tokens, no cluster usernames or paths (some logs contain the local Mac path).

### Next
- Invite the developer as a collaborator if they need the code; add a `requirements.txt` / top-level README when convenient.
- Matching the 46,220 kept clips; round trip of j48w_b01.
