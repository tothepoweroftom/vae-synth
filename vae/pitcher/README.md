# pitcher

Code for **Pitcher**, a category-conditional VAE that generates Surge XT presets. The plan is in `../PIPELINE_VAE_SURGE.md`, and the record of what was built and measured is in `../IMPLEMENTATION_LOG.md`.

Run everything from the project folder (`~/Documents/Pitcher`) as modules, with the project's environment:

```bash
.venv/bin/python -m pitcher.step1_dataset.extract
```

## Layout

| Folder | What it holds |
|---|---|
| `core/` | Shared library, no pipeline logic: `surge_io` (the **only** module that talks to `surgepy`: load, read, write, render), `schema` (feature kinds, exclusions, FX/oscillator tables, standard routings), `codec` (feature row → write plan for Surge), `audio_metrics` (mel distance, RMS) |
| `step1_dataset/` | Plan Step 1: from `.fxp` patches to an encoded, split dataset with audio |
| `step2_explore/` | Plan Step 2: reference values, pruning of rarely changed features, PCA/UMAP |
| `step3_core/` | Plan Step 3: ranking of parameter groups, template rebuild curves, listening test |
| `step4_vae/` | Plan Steps 4–5: cleaning, data tensors, the CVAE, training, audio evaluation, z probe |
| `step6_sample/` | Plan Step 6: where to draw latent vectors (`prior` = N(0, I), `gmm` = per-label mixture fitted on the train presets' latents) |
| `step8_eval/` | Plan Step 8: CLAP fingerprints, the audio tagger, category check of generated presets |
| `text2preset/` | Hackathon track (`../TEXT2PRESET_PLAN.md`): prompt vocabulary (`vocabulary_v2.yaml`, the default, editable; v1 `vocabulary.yaml` kept), prompt generator, standalone Stable Audio 3 script for a GPU machine, post-processing + pilot report (`postprocess.py`, `pilot_report.py`; needs librosa) |
| `synth_data/` | Synthetic Surge dataset (`../TEXT2PRESET_PLAN.md` §3): `common` (base presets, parameter groups, donors, deterministic render worker), `screen` (stage 0: sensitivity screening), `rulers` (stage 0b: perceptual scales per knob), `sampler` (stage 1 row sources: perturb / layered / vae, activity rules), `stage1` (render + store parameters and mel spectrograms, coverage in sound) |
| `inverter/` | Sound → parameters (`../TEXT2PRESET_PLAN.md` §4): `data` (stage-1 mels + targets in the VAE decoder's layout; human val / test eval renders), `model` (mel CNN → VAE decoder heads, mel augmentation), `train`, `evaluate` (render → invert → render on human val / test vs oracle and retrieval) |
| `patches/` | `surgepy_local.patch`: local changes to `surge/src/surge-python/surgepy.cpp` (helpers, `getPatch()` fixes, `seedRandom`) and one line in `src/common/dsp/modulators/LFOModulationSource.h` (`reseedRandom`). Re-apply after a re-clone: `git -C surge-rack-main/surge apply ../../pitcher/patches/surgepy_local.patch` |

## Run order

Each step reads the outputs of the steps before it (all under `../data/`, plots under `../runs/`). Steps 4–6 form a loop on a fresh build: `encode` without a routing matrix falls back to the 13 standard routings, `split` uses those codes, `routes` picks routings on the train split, and `encode` then runs again with the matrix.

| # | Command (`-m pitcher.…`) | Reads | Writes |
|---|---|---|---|
| 1 | `step1_dataset.extract` | `.fxp` patches | `presets`, `raw_values`, `routings`, `params` (.parquet), `dropped.csv` |
| 2 | `step1_dataset.verify` | step 1 + `.fxp` XML | (checks only) |
| 3 | `step1_dataset.render` | `presets` | `audio/fixed/*.wav` (dataset), `audio/natural/*.wav` (listening), `audio_index.parquet` |
| 4 | `step1_dataset.encode` | 1, 3, `routing_matrix.json` if present | `schema.json`, `codes`, `masks`, `defaults`, `dataset` (.parquet), `excluded_presets.csv` |
| 5 | `step1_dataset.split` | 4 | `splits.json`, `splits.parquet` (fixed; only re-run when the preset set changes) |
| 6 | `step1_dataset.routes`, then `encode` again | `routings`, `splits` | `routing_matrix.json` (which routings become features; train only) |
| 7 | `step2_explore.reference` | 6 | `reference_codes.parquet` (template + each preset's types) |
| 8 | `step2_explore.prune` | 5–7 | `pruning.json`, `feature_activity.csv`, `codes_pruned.parquet` (threshold 1%) |
| 9 | `step2_explore.explore` | 5, 6, 8 | `pca_feature_importance.csv`, `runs/step2/*.png`, `summary.json` |
| 10 | `step3_core.rank` | 5, 6, 8 | `core_ranking.csv` (~18 min) |
| 11 | `step3_core.curve` | 5–10 | `core_curve*.csv`, `runs/step3/core_curve.png` |
| 12 | `step3_core.listening` | 11, `roundtrip_pruned.parquet` (see below) | `runs/step3/listening/` (page + audio), `listening_key.csv` |
| 13 | `step3_core.listening_analysis` | 12 + `listening_ratings/` (exported from the listening page) | `listening_results.csv`, `listening_threshold.json`, `runs/step3/listening_fit.png`, `core_curve_character.png` |
| 14 | `step4_vae.clean` | `core_curve_all_kall.csv` (`step3_core.curve --ks 452 --variants routes --split all --out core_curve_all_kall.csv`) | `vae_excluded.csv` |
| 15 | `step4_vae.train --run <name>` | 5, 6, 8, 14 | `runs/vae/<name>/` (model, config, metrics) |
| 16 | `step4_vae.eval --run <name>` | 15 | `runs/vae/<name>/eval_*.csv`, `samples/*.wav` |
| 17 | `step8_eval.clap --model <m>` (×3 models) | `audio/fixed` | `clap_<model>.parquet` (512-number fingerprint per preset) |
| 18 | `step8_eval.tagger` | 17, `splits` | `tagger.pkl`, `tagger_report.json`, `tagger_val_predictions.csv`, `runs/step8/tagger_confusion.png` |
| 19 | `step6_sample.latent --run <name>` | 15 | `runs/vae/<name>/latent_gmm.pkl` (per-label mixture in the latent space; also fitted on demand by 20) |
| 20 | `step8_eval.label_check --run <name> [--tau 1 --n 150 --samplers prior,gmm]` | 15, 18, 19 | `runs/step8/<name>/tau<tau>/label_check.csv/.json`, `audio/{template,prior,gmm}/*.wav` |
| 21 | `step4_vae.zprobe --run <name>` (~15 min; `--analyse-only` re-reads the renders, ~3 min; `--reuse <run>` renders only the decoded presets) | 15 | `runs/vae/<name>/zprobe/` (`measures.parquet`, `probe.csv`, `summary.json`, `zprobe.png`): can z predict category and simple sound measures (`../TEXT2PRESET_PLAN.md` §6) |

Text → preset track (`../TEXT2PRESET_PLAN.md`):

| Command | Reads | Writes |
|---|---|---|
| `.venv/bin/python -m pitcher.text2preset.prompts --name pilot --n 500` | `text2preset/vocabulary_v2.yaml` (or `--vocab`) | `data/text2preset/prompts_<name>.jsonl` + `.csv` (for review) |
| **on Snellius:** the self-contained folder `../hpc/snellius_sa3/` (README, config, setup, Slurm jobs); or on any GPU machine, in the stable-audio-3 env: `uv run python generate_sa3.py --inspect`, then `--prompts prompts_pilot.jsonl --out audio_pilot --model small-music --seeds 4` | the prompts `.jsonl` | `<out>/<model>/<id>_s<seed>.flac` + `metadata.jsonl` (copy back to `data/text2preset/`) |
| `.venv/bin/python -m pitcher.text2preset.postprocess --audio data/small-music --name pilot_small-music` (~13 min; then `pilot_report` runs automatically, or alone: `-m pitcher.text2preset.pilot_report --name …`) | the clips + `metadata.jsonl`, the prompts `.jsonl`, `data/tagger.pkl`, Surge CLAP cache | `data/text2preset/<name>/clips.parquet` (metrics, flags, `keep`, tagger raw + domain-aligned, CLAP scores), `clap.parquet`, `surge_controls.parquet`; `runs/text2preset/<name>/report.md` + plots |

Synthetic Surge dataset (`../TEXT2PRESET_PLAN.md` §3):

| Command | Reads | Writes |
|---|---|---|
| `.venv/bin/python -m pitcher.synth_data.screen` (`--analyse-only` re-reads the renders) | `codes`, `codes_pruned`, `pruning.json`, `core_ranking.csv`, `splits`, `vae_excluded.csv` | `data/synth/screening.parquet`, `screening_groups.csv` (per group: effect size, spread, tier); `runs/synth/screening.png`, `screening_summary.json` |
| `.venv/bin/python -m pitcher.synth_data.rulers` (`--analyse-only`) | the above + `screening_groups.csv` | `data/synth/rulers_raw.parquet`, `rulers.json` (perceptual scale per knob); `runs/synth/rulers.png`, `rulers_summary.json` |
| `.venv/bin/python -m pitcher.synth_data.sampler` (check only: activity rules vs `masks.parquet`) | the above + `rulers.json` | (prints agreement) |
| `.venv/bin/python -m pitcher.synth_data.stage1 --name pilot --n 50000` (~1.5 h on the Mac; resumes per chunk; `--analyse-only` re-runs the coverage) | the above + `runs/vae/<run>/` (VAE source) | `data/synth/stage1/<name>/chunk_XXX/` (`meta.parquet`, `codes.parquet` = kept features with NaN = inactive, `mel.npy` = 64-band mel of the level-normalised first 6 s, float16, `audio/*.flac` ~1%), `coverage.parquet`; `runs/synth/stage1_<name>/report.json`, `coverage.png` |

Inverter and the text → preset chain (`../TEXT2PRESET_PLAN.md` §4, §5, §7; v0 = simplest working version of each step):

| Command | Reads | Writes |
|---|---|---|
| `.venv/bin/python -m pitcher.inverter.data` (once) | splits, codes | `data/inverter/eval/` (human val / test renders, same view as stage 1) |
| `.venv/bin/python -m pitcher.inverter.train --run v0 --data pilot` (~2.5 h on the Mac CPU) | stage-1 set + eval | `runs/inverter/<run>/` (`model.pt`, `config.json`, `metrics.csv`) |
| `.venv/bin/python -m pitcher.inverter.evaluate --run v0` | the above | `runs/inverter/<run>/eval/` (`per_preset.csv`, `summary.json`, `examples/*.flac`) |
| `.venv/bin/python -m pitcher.text2preset.match --name pilot_small-music --audio data/small-music --run v0` | clips + inverter + stage-1 set (retrieval) | `data/text2preset/<name>/matched.parquet`, `matched_codes.parquet`, `matched_summary.json` |
| `.venv/bin/python -m pitcher.text2preset.make "dark bass sound"` | matched set + prompts (CLAP text index cached as `text_index.npz`) | `runs/text2preset/make/<sentence>/` (`.fxp` presets for Surge XT, `.wav` previews, `info.json`) |

Checks that are not part of the main line:
- `step1_dataset.roundtrip`: writer round-trip on the original patch (A1 exactness, A2 sound), with `--codes`, `--out`, `--limit`. Run with `--codes codes_pruned.parquet --out roundtrip_pruned.parquet` for the file step 12 uses.

## Things to know

- **`core/surge_io.py` sets `SURGE_DATA_HOME`** to `surge-rack-main/surge/resources/data` and adds the local `surgepy` build to `sys.path`; no Surge XT app is needed.
- **Deterministic renders** (Round 29): `SurgeIO.render_plan(plan, template, seed=…)` gives the same audio for the same plan and seed, whatever the engine rendered before (Surge's random sources reseeded via the patched `seedRandom`; a short template note first resets DSP state that survives patch loads). `render(seed=None)` keeps Surge's own randomness. After changing the patch: `make surgepy -j8` in `surge-rack-main/surge/ignore/bpy` (~1 min).
- **Wavetable fallback** (Round 29): a wavetable / window oscillator whose table is "other" (not in the top-32 list) gets `Sine Octaves` (`codec.FALLBACK_WAVETABLE`); before, it kept the table of whatever the engine had rendered last.
- **The writer order matters** (`SurgeIO.write_plan`): routings onto FX slots that change type are removed, then types, then subtypes + flags, then values, then temposync and routings. Each rule exists because the round-trip test found it; see the docstrings and `IMPLEMENTATION_LOG.md`, Rounds 6 and 12.
- **torch 2.2.2** (last build for Intel macOS) cannot use numpy 2's C API: tensors are built from Python lists (`step4_vae.train.tensors`), and importing torch prints a harmless warning.
- **transformers 4.49.0** is pinned: later versions need a newer torch. CLAP weights are downloaded to `~/.cache/huggingface` on first use (~600 MB–770 MB per model).
- **librosa 1.0.0** (pYIN pitch tracking, onsets, harmonic-percussive split) and **tabulate** (markdown tables) were added for `text2preset/postprocess.py` / `pilot_report.py` (16 Oct 2026).
- **Splits are fixed:** `step1_dataset.split` is only re-run when the curated preset set changes; everything that fits something uses train only.
