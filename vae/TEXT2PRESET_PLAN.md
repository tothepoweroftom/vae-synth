# Text → Surge XT preset (hackathon plan)

*Written 15 Oct 2026, pipeline revised 16 Oct 2026 (§1 "Main pipeline", §3–§7). Builds on the VAE pipeline in `PIPELINE_VAE_SURGE.md`; log of what was built in `IMPLEMENTATION_LOG.md`.*

**Goal:** a plugin where you type a sound description ("dark long pad with slow attack") and get several real, editable presets back (parameter settings, not audio). Surge XT now; our own synth later (only the adapter `pitcher/core/surge_io.py` changes).

---

## 1. Concept

### Prior work (checked 15 Oct 2026)

| Work | What it does | Difference from ours |
|---|---|---|
| **AutoSynth** (Wu et al., arXiv 2610.10774, 7 Oct 2026) | Text → **Stable Audio 3 Medium at runtime** → audio → Transformer writes a Helm program token by token. Trained on ~2 M perturbed-preset (audio, program) pairs, then reward fine-tuning (GRPO; CLAP + pitch + spectral reward) on 500 k Stable Audio clips | Helm, not Surge; needs a GPU for every text prompt; no plugin; one program per sample, no latent space |
| CTAG (Cherep et al., ICML 2024) | Searches 78 parameters of a small modular synth until LAION-CLAP matches the text | Random start per prompt, no learned prior |
| Synth Permutations (Hayes et al., 2025) | Audio → Surge XT parameters (flow matching) | Audio only; FX and routing fixed |
| Bruford et al. 2024; Combes et al. 2025; preset-gen-vae (Le Vaillant et al. 2021) | Audio → parameters (Massive / black-box synths / Dexed VAE) | No text |
| Commercial (Prompt2Patch, PresetLab, Serum generators) | Text → Vital / Serum / Diva, often an LLM writing values | Surge XT "coming soon" at Prompt2Patch; claims unverified |

**What's ours:** full Surge XT presets (oscillator types, filters, FX, routing) from text; a browsable latent space (variations, morphing, sliders); runs offline on a laptop.

### Main pipeline (decided 16 Oct)

The VAE models **presets**, not audio, so it can't learn from Stable Audio clips directly. The bridge is an **inverter**: a model that turns a sound into Surge parameters. It is trained on Surge's own sounds (we can render as many as we like), then applied to the Stable Audio clips; the matched presets inherit the clip's prompt.

```
offline (development)
  diverse Surge parameter vectors ──render C3──► (sound, parameters) pairs ──► train INVERTER  (§3, §4)
  prompt ──Stable Audio 3──► clean clip ──INVERTER + short search──► preset  (keep good matches only, §5)
  human presets + matched presets ──► train VAE  (§6)
  (prompt text, z of matched preset) pairs ──► train text → z PRIOR  (§7)

in the plugin (no Stable Audio, no GPU)
  text ──text encoder──► prior p(z | text) ──sample z──► VAE decoder ──► presets
       (optional: render ~50 candidates in Surge, keep the 4 that fit the text best)
```

Stable Audio is only the teacher during development (same structure as DALL·E 2 / unCLIP), and the inverter matches **audio to audio** inside Surge's own sound space, so the CLAP "modality gap" (Round 18: text matching 20% vs audio–audio 52% on our 8 categories) and the Stable Audio / Surge offset in CLAP space (pilot: cosine 0.44 between the means) don't sit in the main path.

### Earlier options

- **A. Search (no new training):** prompt → Stable Audio → 4–8 clips → mean CLAP fingerprint = target; draw ~500 z from the per-label mixture (Round 19), render, fingerprint, keep the best with sampling–importance–resampling, refine with a short Metropolis–Hastings walk (`PIPELINE_VAE_SURGE.md` §6.1). Needs Stable Audio at runtime. Kept as a **demo fallback** if the main pipeline isn't ready in time.
- **B. Fingerprint-conditioned VAE + text → fingerprint / descriptor prior** (15–16 Oct versions). **Superseded** by the main pipeline (user, 16 Oct): the inverter replaces the CLAP-fingerprint bridge, and measurable descriptors are no longer a conditioning signal (the user wasn't confident in them). Simple measures (brightness, attack, length) are still used for **testing** (§6, §7).
- **C. AutoSynth-style reward fine-tuning:** later, not for the hackathon.

### Stable Audio 3: access, compute, license

- Small (433 M, `small-sfx` / `small-music`) and medium (1.4 B, CUDA + FlashAttention 2, torch 2.7.1) are open weights; large is API-only (paid). Hugging Face weights are **gated**: free account, accept conditions, `hf auth login` with a free token. Not runnable on the Intel Mac (torch 2.2.2): generate on a GPU machine, copy the audio over.
- Output: stereo 44.1 kHz; medium peaks at ~5 GB VRAM for 5 s; < 2 s per clip on an H200.
- **License: cleared.** The user is in contact with the Stable Audio creators, who confirmed (16 Oct) that their model may be used for this purpose, incl. training on its outputs. (Background: Stability AI Community License; the text encoder T5Gemma falls under Gemma terms.)

### Evaluation (for the jury)

- Independent judge: not the CLAP model used anywhere in the pipeline (use the other CLAP model or LanguageBind) + a short listening test.
- Baselines: **retrieval** (nearest existing preset to the prompt's clips; must beat it), category-only sampling (the per-label mixture), CTAG-style search from random parameters, the Stable Audio clip itself as an upper reference.
- Demo: prompt → 4 Surge presets → morph between two prompts → download the preset.

---

## 2. Prompt dataset plan

### 2.1 How many clips

- **Pilot:** ~500 prompts × 4 seeds = 2,000 clips (~1 h on a GPU). Checks whether Stable Audio gives single, dry notes and how many clips get discarded. **Model: `small-music`** (user decision, 15 Oct; no model comparison).
- **Main:** ~10,000–20,000 unique prompts × 4 seeds = **40,000–80,000 clips** of ~5–6 s. Unique prompts matter more than seeds; 3–4 seeds per prompt show how wide a description is (→ 4 variations in the UI). A large share will be dropped at matching (§5), so err on the high side.
- **Stop rule:** learning curve — train the prior on 1 k / 2 k / 5 k / 10 k prompts; stop when held-out error stops dropping.
- Keep the audio (FLAC 16-bit, ~0.5 MB per clip, 25–40 GB), not only fingerprints.

### 2.2 Category shares

- ~80% balanced across the 8 categories, somewhat more for wide ones (Atmosphere, Synth, Pad).
- ~10–15% **no category** (character only: "warm, hollow, slowly evolving").
- ~5% **out of range** ("nylon guitar", "choir"): test only, for graceful failure.
- After generation: run our category tagger over the clips; rebalance or filter where Stable Audio ignored the category.

### 2.3 Prompt structure

Slots, generated from a grammar; **structured tags stored with every prompt** (for balancing, per-word evaluation, auxiliary targets for the prior).

| Slot | Values (opposites ↔) |
|---|---|
| category / noun | bass, sub bass, reese, lead, pad, keys, e-piano, pluck, bell, drone, strings, brass, … |
| brightness | dark, muffled, warm ↔ bright, airy, glassy |
| body | thin, hollow, nasal ↔ fat, thick, wide |
| grit | clean, smooth ↔ gritty, distorted, aggressive |
| attack | slow attack, swelling ↔ snappy, punchy, percussive |
| length | short, staccato ↔ sustained, long release |
| movement | static ↔ evolving, opening filter, wobble, vibrato, pulsing |
| colour / tech | detuned, supersaw, FM, metallic, chorus, analog, digital, lo-fi |
| style (sparingly) | 80s, trance, dubstep, ambient, techno |

Three surface styles: **tags** ("dark reese bass, detuned, gritty"), **sentences** ("a warm pad that slowly opens up"), **producer slang** ("fat trance supersaw lead").

**Two texts per clip:**
- `sa3_prompt` (sent to Stable Audio) = description + fixed suffix: *"single note, C3, one-shot, dry, no reverb, synthesizer, no drums, no melody"*.
- `user_texts` (train the prior) = the description without the suffix, in several paraphrases.

### 2.4 Synonyms

- On the **user-text side**: 5–10 paraphrases per clip cost only a text embedding, not a generation.
- **Near-synonym groups per slot value**, curated (the DSP engineer is the right reviewer): "warm" ≠ "soft" ≠ "mellow".
- Include **opposites**, so the prior learns directions.
- LLM paraphrasing only constrained by the vocabulary list.
- **Hold out some words entirely** (never in train) to test generalisation.

### 2.5 Other

- **Pitch and polyphony:** detect pitch (pYIN / CREPE); keep clips on a C (C3 or C4: the octave is handled at matching, §5), reject others; reject chords, melodies, drums (onsets, polyphony).
- **Same conditions as the Surge renders:** resample 44.1 → 48 kHz, mono, loudness-normalised (loudest 50 ms window, the rule of `synth_data.common.compare`), same length (6 s).
- **Quality filter:** low CLAP text–audio score, silence, artefacts, wrong category per our tagger; expect 10–30% discarded; log the reason.
- **Reachability:** the matching step (§5) shows per category and per word what Surge + template can make (template keeps ~39–41% of real presets' character, Rounds 15 / 29).
- **Splits:** by prompt combination, never by clip (paraphrases and seeds stay together); held-out sets for unseen combinations and unseen words.
- **Honest test set:** 5–10 people write ~50 prompts freely, without seeing the vocabulary.
- **Log per clip:** prompt, tags, user texts, seed, model, steps, CFG, filter results.

---

## 3. Synthetic Surge dataset (decided 16 Oct)

Training data for the inverter (§4): (sound, parameters) pairs, rendered ourselves, no Stable Audio involved. The pilot showed why: only ~1,400 cleaned human train presets, and Stable Audio clips sit apart from Surge in CLAP space. The goal is a **diverse** set — many parameter vectors that sound different from each other — drawn efficiently; how to draw it is what stages 0 and 0b measured.

### 3.1 How the synthetic set is sampled

The full space can't be covered (176 discrete choices: 91 categorical + 85 on/off, plus 95 routings and ~450 knobs), so the goal is to cover the part that **sounds different**.

| Source | Share | How |
|---|---|---|
| Perturbations of human presets | ~50% | Template rebuilds of **train** presets only (val / test stay clean for evaluation). Structure kept or 1–2 switches changed; knobs moved on their perceptual scale (stage 0b), step size by screening tier (stage 0): tier 1 moved most, tier 3 left at the preset's value unless gated on |
| Layered exploration | ~30% | Structure first (human structures with mutations; rare ones upweighted with frequency^0.5), then the knobs of that structure with scrambled Sobol points mapped through the human value distribution (slightly widened), so correlations and realistic ranges are kept |
| VAE per-category mixtures | ~20% | Where the generator will actually go |

Rules:
- Each row records its source.
- **Diversity is measured in sound, not in parameters:** coverage = share of new renders more than 6.7 dB (level-matched mel distance; the "same character" threshold, Round 14) from everything so far.
- Near-duplicates in sound are kept (downweighted in training, not deleted): settings that sound alike teach the inverter that several settings can give one sound.
- Validity by rules (silent, clipping, level jumps), not by a classifier (a render costs ~0.1 core-second).
- Parameters that can't be heard in a single fixed-velocity note (play mode, portamento; measured 0.0 dB) are taken from human frequencies, not explored; velocity routings are audible but act as a fixed offset, confounded with the destination's value.

**One note only (C3, MIDI 48): no second note at C4** (user decision, 16 Oct, after the screening found C3 and C4 sensitivities nearly the same, r = 0.95). The Stable Audio octave mismatch (non-bass clips at C4) is handled at matching (§5).

### 3.2 Stages

| Stage | Renders | Where | Output |
|---|---|---|---|
| 0. Sensitivity screening ✅ | 14k | Mac, 14 min | Per parameter group: how much the sound changes when the group takes another human preset's value (from 32 human base presets); spread across bases; tiers (36 tier 1, 182 tier 2, 104 tier 3 of which 40 gated) |
| 0b. Perceptual scales ✅ | 13k | Mac, 18 min | 11-step sweeps of the 40 most sensitive knobs from 30 base presets each; 8 knobs need re-mapping; all sampling uses these scales |
| 1. Pilot ✅ | 50k (+1,375 human) | Mac, 47 min (~20 renders/s) + 20 min coverage | Invalid: layered 4.2%, perturb 1.3%, VAE 0%. Share new (> 6.7 dB from everything earlier): **layered 53%** (curve 74% → 49%, not saturated), perturb 28% (small 9% / medium 29% / large 47%; 48% → 20%), VAE 19% (43% → 13%); same pattern in every category (Round 32) |
| 1b. Fill to 100k | +50k | Mac, ~50 min | Shares adjusted by the pilot's coverage per source (proposal: ~60% layered, ~35% perturb with fewer small steps, ~5% VAE); first inverter trained on this |
| 2. Main | chunks of 250k, up to ~1M | Snellius, one GPU node | Next chunk only if the curves below say so |
| 3. Adaptive top-up | 10–20% | Snellius | Where the inverter's re-render error is high, or where 2–3 inverters trained on different subsets disagree; regions near the Stable Audio clips that failed matching |

**How many: measured, not guessed.** (a) Coverage curve (above): when the share of new sounds falls to a few percent, the sampler adds little. (b) Learning curve: inverter trained on 50k / 100k / 200k / 400k, scored in sound on held-out human val presets (render → inverter → preset → render → "same character" rate, §4).

**What is stored per row:** parameters (codes row + source + seed), the **64-band mel spectrogram** of the level-normalised C3 render (the metric's own: 2048 / 512 at 48 kHz, first 6 s, float16, ~70 KB; ~7 GB per 100k), the CLAP audio embedding (512 floats, 2 KB), level and length. Raw audio is **not stored** (1.5 MB per render) except ~1% for listening; the renders are deterministic (below), so any row can be re-rendered.

**Throughput (measured):** rendering ~33 per second on the Mac (4 workers), CLAP ~7 per second on the Mac CPU (the bottleneck: 100k ≈ 4 h; can be skipped on the Mac and computed on the Snellius GPU from re-renders). Snellius from ~250k (estimate: 120–150 renders per second on 18 cores, ~0.4M per hour). For containment on Snellius: surgepy compiled inside the folder (with our patch), factory data via `SURGE_DATA_HOME`, `HOME` pointed inside the folder for the jobs (Surge creates a user-data folder in the home directory). The compile is the main risk: test it first.

**Deterministic renders (Round 29):** the same plan and seed give the same audio (`SurgeIO.render_plan`; 98.3% bit-identical across engines and render order); different seeds differ by a median 2.4 dB (Surge's random phase / drift).

**Limit:** this covers the template's feature space (Round 15 ceiling: 39–41% of val presets keep their character with all features). Macros, MSEGs and long-tail routings stay out of reach however much is sampled; the matching step's drop rate per category shows how much that matters for real prompts.

---

## 4. Inverter: sound → parameters

**Model.** Input: the 64-band mel spectrogram of a level-normalised 6 s note at C3 → small CNN encoder → audio code c (optionally concatenated with the CLAP audio embedding). Output: the **same heads as the VAE decoder** (continuous / ordinal values, categorical one-hots, activity masks) with the **same masked loss**, so the inverter's output is directly a codes row that `codec.decode()` writes.

**Generative, because many presets give one sound.** A plain regressor averages between settings that sound alike (e.g. two filter models at different cutoffs) and lands on neither. So the inverter is a conditional generator, decoder(z, c), in the spirit of Hayes et al. (flow matching for Surge): start as a **conditional VAE** over parameters with the audio code as the condition (reuses the step 4 code; z captures what the sound doesn't determine), switch to flow matching if the samples still average. Sampling several z for one sound gives several candidate presets. This is the same model as the earlier "version B" decoder, conditioned on audio instead of descriptors.

**Training data:** the synthetic set (§3) + the cleaned human train presets' renders. Val / test: the human val / test presets (never used for sampling), so the score measures real presets, not the sampler's own distribution.

**Augmentation for the Stable Audio domain:** light reverb, EQ tilt, a little noise, small pitch shifts (±30 cents), level — the clips are close to dry single notes (pilot: 97% kept) but not identical to Surge renders.

**Score (in sound):** val preset → render → inverter (best of 16 samples) → preset → render → level-matched mel distance; "same character" rate (≤ 6.7 dB). References: the template ceiling (39%: the best any parameter-space method can do within the template) and the **retrieval baseline** (nearest training render's preset). Training on a Snellius GPU; a small pilot can run on the Mac CPU.

---

## 5. Matching the Stable Audio clips

1. **Clean** every clip (§2.5): single note, mono, 48 kHz, 6 s, loudness-normalised.
2. **Pitch:** a clip at C3 is matched as is. A clip at C4 (pilot: everything except bass) is matched with the oscillator **octave parameter +1** (rendered at C3, so it sounds at C4); after matching, the octave is reset, and the stored preset plays the same sound one octave lower at C3. (Caveat: keytracked filters change slightly with the octave; checked by re-rendering after the reset.)
3. **Invert all clips in one go:** 16 candidate presets per clip from the inverter, all rendered, the closest kept.
4. **Short search** from that candidate: ~50 renders (moves on tier-1/2 knobs on their perceptual scales, a few switch flips; Metropolis-style acceptance), instead of the ~500 a search from scratch needs. Only for clips whose best candidate is within ~1.5× the threshold (the others won't make it).
5. **Threshold, calibrated on a reference:** how close a real held-out Surge preset gets to its nearest neighbour (val render vs the inverter's / the training set's best render). A clip is kept when its match is at least as close as, e.g., the median of that reference.
6. **Independent check:** survivors are checked with the **second CLAP model** (not used anywhere else): the matched render must be closer to its own clip than to other clips of the same category.
7. Expect a **large share to be dropped** (sounds Surge + template can't make). Log the reason; the drop rate per category and per word is the reachability map.

**Cost:** 60k clips × 16 candidates ≈ 1M renders (~9 h on the Mac, ~2 h on Snellius), plus up to 50 search renders for the near misses (worst case ~3M more): the bulk runs on Snellius.

Output: (clip, prompt, user texts, tags, matched preset, distance, octave) per surviving clip.

---

## 6. VAE on human + matched presets

- Train the conditional VAE (step 4 of `PIPELINE_VAE_SURGE.md`) on the cleaned human presets + the matched presets. Category condition: the prompt's category, confirmed by our tagger on the matched render; no-category prompts get the tagger's label or are left out of the conditional part. Human presets keep a higher weight; splits by prompt combination (§2.5).
- **Before** (on the current VAE) and **after**, a **z probe**: can a simple model predict, from z (encoder mean), the category and simple measures of the preset's render (brightness = centroid relative to the note, attack time, length, noisiness)? If z can't encode what prompts talk about, no text prior can select it, however good it is; then the VAE changes first (larger z, a loss term in sound, training on synthetic presets too).
- Re-run the reconstruction check in sound (Round 16: 0% "same character" on val; the 12-number z carries coarse information). Also re-run the checks computed with the stale-wavetable bug (cleaning list `vae_excluded.csv`, VAE / tagger checks; Round 29).
- **Training notes for the next VAE (17 Oct, from Le Vaillant et al., DAFx 2021):**
  - **β warm-up from 50%, not 0%:** they found warm-up starting at 0 clearly worse on validation. Ours goes 0 → β over `--warmup` epochs (run `h256_d01`: 0 → 0.5 over 30 epochs; best epoch 33, i.e. picked right after the warm-up). Add a `--warmup-start` option (default 0.5 × β) and compare.
  - **Discrete knobs as categories ("NumCat++"):** the 21 kept ord knobs (octaves, semitone steps, small integer ranges, ≤ 32 values) as softmax heads instead of one MSE number; categorical losses scaled (they used 0.2), losses normalised per modality.
  - **Joint audio + parameter VAE** (user idea, 17 Oct): encoder input = parameters + an audio code of the render (first step: 64 PCA coefficients of the time-pooled mel; later a small CNN on the mel), decoder reconstructs both. Modality dropout (sometimes only audio, sometimes only parameters as input) so the same model encodes a Stable Audio clip from audio alone: then it is inverter and generator in one, with a sound-aware z.
  - **Comparison protocol** v1 (human) / v2 (human + synthetic, same model) / v3 (joint audio + parameters): on held-out human presets, in sound: active units + KL + latent decorrelation; z probe; reconstruction (same-character rate vs 39% ceiling); ~1,000 samples per model: Vendi score, coverage / density against held-out human renders, tagger category accuracy, invalid rate; smoothness along z interpolations.

---

## 7. Text → z prior

**Data:** every surviving clip gives (user text paraphrases, z = VAE encoder mean of its matched preset); 10–20k prompts × ≤ 4 seeds after drops ≈ 20–40k pairs, ample for a 12-dim target (~6 informative dims in run `h256_d01`).

**Model:** text → text encoder → **small conditional flow or diffusion model over z** (a few hundred thousand parameters, a few MB). Multimodal on purpose: a single Gaussian per prompt averages the options and lands on a generic sound (the "Synth drift" seen when sampling N(0, I)). Fallback: a mixture density network with ~8 components. Structured slot tags as an auxiliary target.

**Text encoder:** start with CLAP's text tower (already in the plugin plan, ~125 M parameters, ONNX); compare with a sentence encoder. CLAP's zero-shot text matching was weak (20% on categories; pilot: a prompt's own clip ranked median 139 of 500), but a trained head on its embeddings can still use word identity.

**Risks (16 Oct):** the prior's size is not the hard part (small target, enough pairs). The real risks are (1) the text encoder not separating the words that matter, (2) z not holding text-level detail (VAE, §6), (3) samples landing in latent regions that decode to odd presets, (4) z neighbourhoods built from parameters, not sound (nearby z can sound different).

**Safeguards:**
- **Propose, render, select** at sampling time: the prior proposes ~50 z, the decoder makes presets, Surge renders them (~0.13 s each), a scorer keeps the best 4 (sampling–importance–resampling, `PIPELINE_VAE_SURGE.md` §6.1). The prior then only needs to be roughly right with good coverage. Scorer: a small text–audio head trained on our (user text, matched render) pairs; it must beat raw CLAP text–audio first.
- **Retrieval prior** as baseline and fallback: nearest training prompts in text space → their z's → a small mixture.

**Evaluation (held-out prompts: unseen combinations and unseen words):**
- Negative log-likelihood of the held-out z's.
- Sound-level rank: rendered samples closer to their own prompt's Stable Audio clips than to other prompts' (as in the pilot).
- Word swaps ("dark" → "bright", "short" → "long") move the rendered sound in the right direction (simple measures from §6).
- Spread of the samples ≈ spread of the 4 seeds' matched presets.
- Must beat category-only sampling (the per-label mixture) and the retrieval prior.

**In the plugin:** text encoder (ONNX) → prior → VAE decoder → presets → (optional) render and select → load into Surge. Offline, no GPU.

---

## 8. Build order

1. ✅ This plan (revised 16 Oct to the inverter pipeline).
2. ✅ Prompt grammar + vocabulary (`pitcher/text2preset/vocabulary_v2.yaml`, revised by the user on 15 Oct: research-based terms, mood slot, no stereo slot) + generator (`pitcher/text2preset/prompts.py`) → pilot list of 500 prompts (`data/text2preset/prompts_pilot.csv`).
3. ✅ GPU generation script (`pitcher/text2preset/generate_sa3.py`). Snellius package: `hpc/snellius_sa3/` (small-music only; everything inside the folder, inspect / generate jobs on one GPU at a time, postcheck, packing). ✅ Pilot run on Snellius.
4. ✅ Post-processing (`pitcher/text2preset/postprocess.py` + `pilot_report.py`; report in `runs/text2preset/pilot_small-music/report.md`). Pilot results (16 Oct, small-music, 2,000 clips, 0.42 s per clip on an A100):
   - **97.2% kept** (single held note; 2.9% multi-note, 0% drums/rhythm, 0% silent); thresholds calibrated on 300 Surge renders.
   - **Pitch:** 96% on a C, but the octave is Stable Audio's own: bass at C3 (MIDI 48), everything else at C4 (MIDI 60). Probably the DAW naming where MIDI 60 is called "C3" (Ableton, Yamaha). **Decided:** match at the clip's octave via the octave parameter, then reset (§5).
   - **Domain gap in CLAP space:** the mean Stable Audio and mean Surge fingerprints have cosine 0.44. Raw tagger agreement with the prompt's category 23% (mostly "Keys"); after a label-free **mean shift** 34.5% (Surge val reference 51.9%). Per category (aligned): Pad 74%, Bass 57%, Pluck 46%, Lead 36%, Keys 20%, Synth 19%, Orchestral 16%, Atmosphere 8%. The inverter matches on mel spectrograms in Surge's own sound space, which sidesteps most of this; the CLAP check of survivors (§5) still needs it.
   - **Descriptor words work for brightness** (airy +0.45, bright +0.34, dark −0.34, warm −0.16 octaves vs category median), **attack** (slow: 75th percentile 0.29 s vs 0.11 s without; percussive 0.007 s) and partly **grit** (aggressive noisier); **length words barely** ("short" isn't shorter than unmarked).
   - Raw CLAP similarity to the nearest Surge preset: median 0.66 (Surge val → train 0.875); 0.80 after alignment.
5. Synthetic Surge dataset (§3): ✅ deterministic renders (seeded Surge + priming; stale-wavetable bug fixed). ✅ Stage 0 screening (Round 29): of 322 reliably tested groups, 36 tier 1, 182 tier 2, 104 tier 3 (40 of them gated); play mode / portamento inaudible in one note; effects depend strongly on the patch (spread ≈ mean); C3 and C4 agree (r = 0.95). ✅ Stage 0b scales: Surge's codes are mostly even (median unevenness 0.12); 8 knobs need re-mapping. One note only (C3; user, 16 Oct).
6. ✅ **z probe** on the current VAE (§6; `pitcher/step4_vae/zprobe.py`, Round 31, 16 Oct). **Answer: no.** R² on val + test presets (best of ridge / boosted; noise ceiling from a second seed 0.79–1.0): brightness 0.09 from z + category (category alone 0.06; the preset's full parameters 0.21), noisiness 0.10 (params 0.34), decay 0.06 (params 0.71), percussive ~0 (params 0.29); only length (0.40 vs category 0.16) and attack (0.44 vs 0.39) carry something. Through the decoder (mu → preset → render) brightness correlates 0.16 with the original, noisiness 0.18, decay 0.16, length 0.64. Lower β doesn't fix it (β 0.02 with all 12 dims active: brightness 0.12, noisiness 0.10). So the bottleneck is not β but a latent learnt from parameter reconstruction on ~1,400 presets, without any sound in the loss. Consequence (§6): before the prior, the latent must become sound-aware: the VAE trained on the synthetic set too (many more presets) with a sound term (e.g. predict the render's mel code / measures from z), or the prior targets the inverter's audio code c directly (text → c → inverter → preset). Decided after the inverter exists (step 8); re-probe then.
7. ⏳ **Stage-1 sampler** ✅ (`pitcher/synth_data/sampler.py` + `stage1.py`, Round 32) → 50k pilot ✅ (layered 53% new, perturb 28%, VAE 19%; listening folder `data/synth/stage1/pilot/listen/`) → ⏳ 100k on the Mac with adjusted shares (parameters + mel; CLAP later on a GPU).
**Priority (user, 17 Oct): the complete pipeline first** ("dark bass sound" → parameters), with simple versions of each step; diversity of the synthetic set (stage 1b, joint VAE, v1/v2/v3 comparison) is improved later. The 50k pilot set is the inverter's training data for now. Fastest path: inverter v0 → match the **pilot clips** (2,000 already post-processed) → a retrieval prior (text → nearest training prompts → their matched presets) → a command that writes a preset; the main Stable Audio run, the VAE retrain and the learned prior then replace the simple parts.

8. ⏳ **Inverter** (§4): conditional VAE over parameters given the audio code; trained on a Snellius GPU; scored on human val presets vs the template ceiling and retrieval.
   - ✅ **v0** (Round 34, 17 Oct; `pitcher/inverter/`): mel CNN → VAE-decoder heads, trained on the 50k pilot on the Mac (best epoch 7 of 15 by human-val loss; human-val loss rises after it while synthetic val keeps falling). **Sound score on 344 human val + test presets** (same character ≤ 6.7 dB, other seed): oracle 96% (2.3 dB), inverter top 4% (14.5 dB), best of 16 candidates 16% (9.4 dB), **retrieval 65% (6.1 dB)**, combined 66%. Swap test: the predicted continuous knobs alone ruin the sound (15 dB); it predicts ~11 active routings vs 3.5 true. Median knob correlation: human val 0.26, synthetic val 0.42 (layered rows 0.08), human train 0.68: weak everywhere, not only a domain gap. Parameter regression without a sound term is ill-posed (Le Vaillant 2021).
   - Next (v1): render-based short search from the best candidate (§5 step 4), more weight on human / perturbed rows and less on layered knobs, a sound-aware loss or a generative decoder (several presets per sound).
9. **Main Stable Audio run** (parallel to 7–8): finish the vocabulary first (more Pad nouns, check the sub / 808 clips); ~12k prompts × 4 seeds ≈ 5.6 GPU-h; copy back with rsync.
10. ⏳ **Match the clips** (§5): clean → best of 16 → short search → threshold + second-CLAP check; surgepy compiled on Snellius for the bulk.
   - ✅ **v0 on the pilot clips** (Round 34; `pitcher/text2preset/match.py`, no short search / CLAP check yet): 1,943 clips, **14.6% kept** (≤ 6.7 dB at C3 and after the octave reset), median 7.7 dB; retrieval wins 99% of the time. By category: Bass 44%, Atmosphere / Orchestral / Synth / Keys / Lead 12–13%, Pluck 6.5%, Pad 5%; by octave: C2 43%, C3 24%, C4 8%, C5 6%. Tested and rejected: pitch-shifting C4 / C5 clips to C3 before retrieval (C4 kept 3% → 0%: the retrieved presets don't transpose back to the clip's note).
   - ✅ **Threshold by listening** (Round 35, 30 rated pairs): ≤ 7.3 dB 6 of 10 same character, 7.3–8.5 dB 1 of 12 → **7.3 dB** (`MATCH_DB`): 653 clips kept (33.6%), 309 prompts. Mel distance is a noisy judge (accepted matches ~55–60% OK); CLAP clip-vs-match similarity adds a little (AUC 0.70 vs 0.66).
11. **VAE on human + matched presets** (§6); re-run the z probe, the reconstruction check and the wavetable-affected checks.
   - ✅ **Joint VAE j48** (Round 36, 17 Oct; `pitcher/joint_vae/`, separate from the old VAE): parameters + 48-number sound code (PCA, k measured: 2.6 dB median reconstruction vs 2.3 dB render noise), product of experts, NumCat++, β warm-up from 50%, trained on the 50k pilot + human (Mac, 13 s / epoch). **z is now sound-aware** (z from parameters only: brightness R² 0.26 vs 0.09 old, decay 0.27 vs 0.06, noisiness 0.35 vs 0.10, length 0.66 vs 0.39). **Inversion not improved** (best of 16: 15% same character vs v0 16%, retrieval 65%). Not yet: human/matched rows weighted up, matched presets added.
   - ✅ **Decoder round trip in sound** (Round 37): preset → z → decoder → render keeps the character for only 2% (joint) / 0% (old VAE); best of 8 decoded proposals 5% (median 13.5 dB). Errors are spread over all knob families (cutoff, envelopes, effects, levels) plus switches. **The decoder is the blocker for generation**, not z. Options: sound-weighted parameter loss (weights from the stage-0b rulers: dB per unit change), render-based local search after decoding (same tool as the matching search), or a decoder that models the parameters jointly (flow matching).
   - ✅ **Sound-weighted loss** (Round 39, `pitcher/joint_vae/weights.py`, `--weights screening`) and smaller β (0.05): switches improve (swap cats 12.5 → 7.2 dB), continuous knobs and spurious routes don't (~16 dB); round trip still 1–4%. Remaining options: render-based search, flow-matching decoder.
12. **Text → z prior** (§7) with retrieval and category baselines; propose–render–select.
   - ✅ **Retrieval prior v0** (Round 34; `pitcher/text2preset/make.py`): sentence → CLAP text embedding → nearest user-text paraphrases of the kept matched clips (one per prompt) → render top 12 → score = text similarity + CLAP text–audio similarity of the render → best 4 written as .fxp + .wav + info.json in `runs/text2preset/make/<sentence>/`. Tried: "dark bass sound" (4 basses, matches 5.8–6.1 dB), "long faded dark pad" (2 pads + 2 dark textures), "quick aggressive guitar-like pluck" (harp / music-box plucks but also a screech lead and a brass: few plucks were kept). Coverage is the limit: 284 kept clips.
13. Plugin export (ONNX text encoder, prior, decoder) with the DSP engineer. Version A as demo fallback if 8–12 aren't ready.
