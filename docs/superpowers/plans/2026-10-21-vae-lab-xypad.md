# VAE Lab v2 (XY pad) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the VAE Lab screen with the student's XY-pad demo (prompt → preset → explore the latent space), played live on a smoothed Lite engine.

**Architecture:** The pipeline is pad / prompt / slider / dot → z → `decoder.onnx` → `StudentCodec` plan → `SurgePatchBuilder` patch → a C++ port of `tools/surge_preset_to_vector.py` → 93 Lite controls. All of it runs on the existing single `JointVaeWorker` thread, where the latest request wins. Per-model pad data (axes, preset dots, knob groups, category forest) comes from the student's page and ships in the bundle as `xypad.json`. Pure math (axes, constrain, mel measures, forest, knob groups) lives in engine-free files checked against Python reference values.

**Tech Stack:** C++20, JUCE 8 (UnitTest, Components), ONNX Runtime (`ml::OnnxModel`), Surge XT headless (`VAE_WITH_SURGE=ON`), Python 3 + torch/onnx/onnxruntime/numpy for export.

**Spec:** `docs/superpowers/specs/2026-10-21-vae-lab-xypad-design.md`. The reference behaviour is the student's page: `/Users/thomaspower/Downloads/new_xy/pitcher/xypad/page_v2.template.html`, copied into the repo in Task 1.

## Global Constraints

- Model `j48w_b05`: checkpoint `/Users/thomaspower/Downloads/model.pt` (sha1 starts `61f0a8d6`), config `vae/runs/joint_vae/j48w_b05/config.json`, and `schema.json` / `pruning.json` / `sound_pca.npz` from `/Users/thomaspower/Downloads/missing_files/`. The bundle goes to `models/joint-vae-j48w_b05/` (git-ignored).
- The Lab requires `VAE_WITH_SURGE=ON`. A non-Surge build shows "VAE Lab needs a Surge build" and nothing else.
- No inference, model loading, patch building or UI logic on the audio thread.
- The Synth and Prompt tabs behave as before. Lite's smoothing is the only change.
- Lite smoothing must not change any render that starts from `prepare()`/`reset()`: values snap there.
- Pad range `RANGE = 5`, so the pad spans ±2.5 sd. The map grid is 24×24 and recomputes 150 ms after movement stops. Dimensions under 2 dB are "minor". Arrow keys move ±0.02, Shift ±0.1. A dot click is a tap under 3 px.
- Labels: Bass, Lead, Pad, Keys, Pluck, Synth, Atmosphere, Orchestral, Unknown (index 8).
- Editor width 1240 px. Two columns: pad column about 2/3, axis list about 1/3.
- Build/test: `cmake --build build-surge && VAE_JOINT_BUNDLE="$PWD/models/joint-vae-j48w_b05/bundle.json" ctest --test-dir build-surge --output-on-failure`.

## Review Focus

1. **An old j48 bundle (no `xypad.json`) is installed or picked.** The Lab says "This bundle has no XY-pad data (re-export with tools/joint_vae/export_xypad.py)" and keeps the previous bundle. Test in Task 6.
2. **Two assigned axes nearly parallel**, for example a sound axis and a latent dimension along the same direction, or the same axis on X and Z. `constrain` returns finite z and never NaN. Test in Task 3.
3. **A prompt with nothing usable** (empty, only unknown words, only stop words) changes nothing and replies with what it ignored. Test in Task 5.
4. **Send to Surge, then drag again.** Playback returns to Lite with no stale Surge patch mode. A held note keeps sounding without a click. Test in Task 6.
5. **The editor closes while a decode or map pass is running.** Late callbacks don't touch a deleted panel, and the map pass stops. Test in Task 8.

---

## File Structure

| File | Responsibility |
|---|---|
| `vae/pitcher/xypad/{latent.py,page_v2.template.html,build_page.py}` | The student's files, copied as-is (reference) |
| `tools/joint_vae/export_bundle.py` | + `sound_decoder.onnx` export and parity |
| `tools/joint_vae/export_xypad.py` (new) | `xypad_v2.html` → bundle `xypad.json` + `xypad-parity.json` |
| `source/engine/lite/LiteEngine.{h,cpp}` | Control smoothing + duck on discrete changes |
| `source/experiments/joint_vae/LiteTranslation.{h,cpp}` (new) | Surge patch data → `schema::Snapshot` (port of `surge_preset_to_vector.py`) |
| `source/experiments/joint_vae/XyPad.{h,cpp}` (new) | `xypad.json` loading; axes, `constrain`, measures, forest, knob groups (no JUCE GUI) |
| `source/experiments/joint_vae/PromptWords.{h,cpp}` (new) | Word-list prompt → start and nudges |
| `source/experiments/joint_vae/JointVae.{h,cpp}` | + sound decoder, lock, Lite output, `render(codes)` |
| `source/experiments/joint_vae/JointVaePanel.{h,cpp}` | Rewritten: the v2 screen |
| `source/experiments/joint_vae/LabPad.{h,cpp}` (new) | Pad component: map image, dots, dot, keyboard/mouse |
| `source/plugin/PluginProcessor.{h,cpp}` | + `playLiteControls` |
| `source/plugin/PluginEditor.cpp` | Wire `onLite` / `onPatch` |
| `tests/ml/XyPadTests.cpp`, `tests/ml/LiteTranslationTests.cpp`, `tests/ml/PromptWordsTests.cpp` (new) | Unit and parity tests |
| `tests/engine/LiteEngineTests.cpp`, `tests/ml/JointVaeTests.cpp`, `tests/plugin/PluginTests.cpp` | Extended |
| `scripts/install-models.sh`, `docs/JOINT-VAE-LAB.md` | Install the new bundle; document v2 |

---

### Task 1: Bundle v2 (`j48w_b05`, sound decoder, `xypad.json`)

**Files:**
- Create: `vae/pitcher/xypad/latent.py`, `vae/pitcher/xypad/page_v2.template.html` (copied from `~/Downloads/new_xy/pitcher/xypad/`). Overwrite `vae/pitcher/xypad/build_page.py` with the student's version.
- Create: `tools/joint_vae/export_xypad.py`
- Modify: `tools/joint_vae/export_bundle.py`

**Interfaces:**
- Produces: bundle files `sound_decoder.onnx` (inputs `latent[1,32]`, `category[1,9]`; output `sound[1,48]`, the model's `decode(z, y)[1]`), `xypad.json`, `xypad-parity.json`, plus `bundle.json` keys `"sound_decoder"` and `"xypad"`. The bundle id hashes them too.
- `xypad.json` = the page's `#latent` object **without** `dec_p`, `dec_a`, `activity`, `pca` (the PCA is already `sound_pca.npz`), plus `"knobs"` = the page's `#data` object (`knobs`, `groups`, `presets`, `forest`, `labels`). The preset z (`presets.z`) stays float16 base64, as in the page.
- `xypad-parity.json`: `{"constrain":[{"z0":[32],"axes":["s:brightness","z:26"],"targets":[2],"z":[32]}…], "measures":[{"sound":[48],"category":c,"latent":[32],"mel_max":f,"measures":{5 keys}}…], "forest":[{"knobs":[123 or null],"p":[8]}…], "groups":[{"group":name,"vals":[123],"pos":f}…]}`, 12 cases each. The reference implementations are numpy: `constrain` solves `G a = r` with `G[i][j] = u_i·u_j + 1e-6·δij`. `measures` is copied from `latent.py`. The forest walks as the page does, with a missing value treated as `-1`.

- [ ] **Step 1:** Copy the student's three files into `vae/pitcher/xypad/` and diff `build_page.py` against the old one: only the `--v2` additions differ.
- [ ] **Step 2:** In `export_bundle.py`, add a `SoundDecoder(torch.nn.Module)` wrapper returning `model.decode(latent, category)[1]`. Export it like the decoder and add it to the 36-case parity loop. Expected: max |PyTorch − ONNX| < 1e-4.
- [ ] **Step 3:** Write `export_xypad.py --html PATH --bundle DIR`. It reads the two `<script type="application/json">` blocks with `json`/`re` only, checks `latent["run"] == bundle config run` and fails otherwise, then writes `xypad.json` and `xypad-parity.json`.
- [ ] **Step 4:** Assemble a source dir and export. Run:
  ```sh
  mkdir -p /tmp/j48w_b05 && cp ~/Downloads/missing_files/{schema.json,pruning.json,sound_pca.npz} /tmp/j48w_b05/ \
    && cp ~/Downloads/model.pt /tmp/j48w_b05/ && cp vae/runs/joint_vae/j48w_b05/config.json /tmp/j48w_b05/
  python3 tools/joint_vae/export_bundle.py --source /tmp/j48w_b05 --out models/joint-vae-j48w_b05
  python3 tools/joint_vae/export_xypad.py --html ~/Downloads/new_xy/runs/xypad/xypad_v2.html --bundle models/joint-vae-j48w_b05
  python3 tools/joint_vae/codec_parity.py --bundle models/joint-vae-j48w_b05
  ```
  Expected: parity passes for both decoders, and `xypad.json` holds 5 directions, 31 dims and 1,375 presets.
- [ ] **Step 5:** Run the existing native tests against the new bundle (`VAE_JOINT_BUNDLE=…/joint-vae-j48w_b05/bundle.json ctest --test-dir build-surge -R vae_ml`). Expected: PASS. The decoder/codec parity sections now cover j48w_b05.
- [ ] **Step 6:** Commit `vae/pitcher/xypad/*`, `tools/joint_vae/*`.

---

### Task 2: Smoothed Lite controls

**Files:**
- Modify: `source/engine/lite/LiteEngine.h`, `source/engine/lite/LiteEngine.cpp`
- Test: `tests/engine/LiteEngineTests.cpp`

**Interfaces:**
- Produces: unchanged `SynthEngine` API. `setControl` now sets a target. Continuous controls (`schema::Kind::Continuous`) move toward it with a one-pole of **τ = 20 ms**, evaluated once per `kControlBlock`, calling `patch.set(i, controls)` only for controls still moving. They snap when |Δ| < 1e-4. A Toggle or Categorical change starts a **duck**: the output gain ramps to 0 over **5 ms**, all pending discrete targets are applied, then the gain ramps back over 5 ms. `prepare()`, `reset()` and changes made with no voice sounding and no FX tail apply at once. The builder may treat "no FX tail" as "no voice active".

- [ ] **Step 1: Write failing tests** in `LiteEngineTests.cpp`:
  - `"cutoff change glides"`: hold a note, set `filter1_cutoff` 0.2 → 0.8. 5 ms later the effective cutoff is strictly between the two. After 150 ms it equals 0.8 within 1e-3. Read it through a test accessor `float currentControl (int index) const noexcept`.
  - `"discrete change does not click"`: hold a saw note at the default patch and record 200 ms. Change `osc1_wave` to Square mid-block and record another 200 ms. Expected: the max |x[n] − x[n−1]| within ±10 ms of the change is ≤ 1.5 × the max over the steady 100 ms before it.
  - `"fresh render unchanged"`: `prepare` → `setControl` ×93 from a fixed snapshot → render 1 s. The result is bit-identical to `prepare` → write `controls` directly. The builder may make this an equivalence against a stored pre-change render.
- [ ] **Step 2:** Run `ctest --test-dir build-surge -R vae_engine --output-on-failure`. Expected: the new tests FAIL.
- [ ] **Step 3:** Implement as described in Interfaces. Keep `controls` as the current values and add `schema::Snapshot targets`. Compute the coefficient from `sampleRate` in `prepare()`.
- [ ] **Step 4:** Run again. Expected: PASS, and all existing Lite tests still pass.
- [ ] **Step 5:** Re-run the Lite-vs-Surge comparison from `docs/FIXES-lite-vs-surge.md` ("Re-run after fixes"). Expected: the medians equal the "engine gap after" column (3.5 dB / 0.15 oct / 1.2 dB / 0.96).
- [ ] **Step 6:** Commit.

---

### Task 3: Pad math (`XyPad`): data, axes, `constrain`

**Files:**
- Create: `source/experiments/joint_vae/XyPad.h`, `XyPad.cpp`
- Test: `tests/ml/XyPadTests.cpp` (add to `tests/ml/CMakeLists.txt`; add `XyPad.cpp` to `vae_joint_experiment` in `CMakeLists.txt`)

**Interfaces:**
- Consumes: `JointLatent` from `JointVae.h`.
- Produces (namespace `vae::experimental`):
  ```cpp
  struct LatentAxis { juce::String id, kind /* "sound"|"dim" */, name, low, high, desc, key; JointLatent u {};
                      float mean = 0, sd = 1, impact = 0; bool minor = false;
                      std::vector<std::pair<juce::String, float>> sections; };
  struct XyPadData {
      std::vector<LatentAxis> axes;                      // sound axes first, then dims by impact (page order)
      std::map<juce::String, JointLatent> typical;       // label name -> mean_z
      std::vector<JointLatent> presetZ; std::vector<int> presetLabel; juce::StringArray presetName;
      std::vector<float> bandHz; float dt = 0, f0 = 0; int holdFrames = 0;
      static juce::Result load (const juce::File& xypadJson, XyPadData&);   // fails without touching out
      const LatentAxis* find (const juce::String& id) const;
  };
  float axisTarget (const LatentAxis&, float pos);                 // tOf
  float axisPosition (const LatentAxis&, const JointLatent&);      // posOf
  JointLatent constrain (const JointLatent& z0, std::span<const LatentAxis* const>, std::span<const float> targets);
  ```
  Axis names and descriptions are built as the page builds them (`fmtEff`, `EFF_MIN`, `pname`, the "Up: …" and section-name rules) when loading.

- [ ] **Step 1: Write failing tests**:
  - `"load"`: 5 sound axes + 31 dims, `axes[0].id == "s:brightness"`, 1,375 presets, 8 `typical` entries.
  - `"constrain parity"`: for each `xypad-parity.json` `constrain` case, |z − ref| < 2e-4 per element. Skip when `VAE_JOINT_BUNDLE` is unset, as existing tests do.
  - `"constrain hits targets and keeps the rest"`: after `constrain`, `axisTarget`/`axisPosition` round-trip within 1e-4, and `z − z0` is a linear combination of the axes' `u` (its residual after projecting onto them is < 1e-5).
  - `"degenerate axes stay finite"` (Review Focus 2): the same axis passed twice with different targets, and two axes with cosine 0.9999, give all-finite z.
  - `"bad file"`: missing keys → `failed()`, output unchanged.
- [ ] **Step 2:** Run `ctest --test-dir build-surge -R vae_ml`. Expected: FAIL (missing symbols).
- [ ] **Step 3:** Implement. `constrain` is the page's Gaussian elimination with partial pivoting on `G + 1e-6 I`. If a pivot is |·| < 1e-9, drop that axis rather than divide.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Commit.

---

### Task 4: Expected sound and category guess (`XyPad`, `JointVae`)

**Files:**
- Modify: `XyPad.h/.cpp`, `JointVae.h/.cpp`
- Test: `tests/ml/XyPadTests.cpp`

**Interfaces:**
- Produces:
  ```cpp
  struct SoundMeasures { float brightness, attack, length, noise, movement; };
  // XyPad.h
  struct SoundPca { std::vector<float> components /* 48 x F */, mean; float scale = 1; int frames = 0; };
  juce::Result loadPca (const juce::File& soundPcaNpz, SoundPca&);   // npz = zip of .npy (float32)
  std::vector<float> melFromSound (const SoundPca&, const SoundCode&);  // frames x 64, peak-relative, floor -80
  SoundMeasures measures (const XyPadData&, std::span<const float> mel);
  struct CategoryForest { /* trees from xypad.json "knobs.forest" */
      static juce::Result load (const juce::var&, CategoryForest&);
      std::array<float, 8> predict (std::span<const float> knobs123) const; };  // NaN = missing -> -1
  struct KnobGroups { /* "knobs.knobs" + "knobs.groups" */
      float position (int group, std::span<const float> knobs123) const;      // groupPos, NaN if none set
      void set (int group, float pos, std::span<float> knobs123) const; };    // setGroup
  // JointVae.h
  juce::Result decodeSound (const JointLatent&, int category, SoundCode&);   // sound_decoder.onnx
  ```
- [ ] **Step 1: Write failing tests**: `"measures parity"` (sound → mel → measures within 1e-3 of `xypad-parity.json`, and `mel_max` = 0), `"sound decoder parity"` (latent → sound within 1e-4), `"forest parity"` (p within 1e-6, including cases with nulls), and `"group parity"` (position within 1e-6; `set` then `position` returns the pos).
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement. Read `.npz` with `juce::ZipFile` plus a minimal `.npy` header parser (float32/float64, C order). `JointVae::load` also loads `sound_decoder.onnx`.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Commit.

---

### Task 5: Prompt words

**Files:**
- Create: `source/experiments/joint_vae/PromptWords.h`, `PromptWords.cpp`
- Test: `tests/ml/PromptWordsTests.cpp`

**Interfaces:**
- Consumes: `JointVae::labelFor` (existing), `XyPadData`.
- Produces:
  ```cpp
  struct PromptReply {
      std::optional<std::pair<JointLatent, int>> start;   // set when this message picks a new start
      std::vector<std::pair<juce::String, float>> nudges; // axis id ("s:brightness"), pad-position delta
      juce::StringArray ignored; juce::String message;    // one line shown in the chat log
  };
  // The one function the text -> z encoder replaces later.
  PromptReply interpretPrompt (const juce::String& text, bool hasStart, const XyPadData&);
  ```
- Rules (exact):
  - **Start:**
    - A noun (via `labelFor`) gives `typical[label]` and that category.
    - No noun with `!hasStart` gives the mean of all `presetZ` with category 8, and the message includes "no sound type found".
    - No noun with `hasStart` gives no start.
  - **Adjective → (axis, sign):**
    - `s:brightness`: dark darker dull muted warm −, bright brighter crisp sharp airy +
    - `s:length`: short shorter tight staccato −, long longer sustained tail +
    - `s:attack`: snappy punchy fast plucky −, slow soft swelling +
    - `s:noise`: tonal clean pure −, noisy breathy gritty +
    - `s:movement`: static still steady −, moving evolving wobbly +
  - **Step size:** 0.1 pad units. "slightly" or "a bit" ×0.5 and "much", "a lot", "very" or "way" ×2 for the next adjective. "less X" flips the sign and "more X" keeps it.
  - **Ignored words:** stop words (a an the it make sound more less with and some bit lot) are not reported. Any other word that is neither noun nor adjective is listed.
  - **Message:**
    - It reads `"<Label> · Dark ↔ Bright −1 step · didn't understand 'x', 'y'"`, with the parts that apply joined by " · ".
    - For `s:noise` and `s:movement` it appends " (small effect)".
    - Nothing applied gives "Nothing to change. Try a sound type (bass, pad…) or brighter / longer / snappier."
- [ ] **Step 1: Write failing tests**:
  - "dark reese bass" with no start: start = typical Bass, category 0, nudges `{s:brightness, −0.1}`, message starts "Bass".
  - "a lot brighter" with a start: no start, `{s:brightness, +0.2}`.
  - "now make it a pad" with a start: start = typical Pad.
  - "something dark" with no start: category 8, "no sound type found" in the message.
  - "" and "zorp blarg": no start, no nudges, ignored = {zorp, blarg}, message "Nothing to change…" (Review Focus 3).
  - "noisier": the word isn't on the list, so it's ignored. Document this in a test so the list is visibly closed.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement. Tokenise into lowercase whole words, as `labelFor` does.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Commit.

---

### Task 6: Lite translation and the playback path

**Files:**
- Create: `source/experiments/joint_vae/LiteTranslation.h`, `LiteTranslation.cpp`
- Modify: `JointVae.h/.cpp`, `source/plugin/PluginProcessor.h/.cpp`
- Test: `tests/ml/LiteTranslationTests.cpp`, `tests/ml/JointVaeTests.cpp`, `tests/plugin/PluginTests.cpp`
- Fixture: `tests/ml/fixtures/surge-factory-vectors.jsonl`, from `python3 tools/surge_preset_to_vector.py "$VAE_SURGE_DATA_DIR/patches_factory" -o …`. Keep ≤ 1 MB; take a fixed subset if larger.

**Interfaces:**
- Produces:
  ```cpp
  // LiteTranslation.h: port of tools/surge_preset_to_vector.py (Patch + Converter + convert), same rules and order.
  juce::Result liteControls (const juce::MemoryBlock& surgePatchData, schema::Snapshot& out);  // finds "<?xml"
  // JointVae.h
  struct LabRequest { JointLatent z {}; int category = 8; std::optional<std::vector<float>> lockedTail; // codes[435..]
                      std::optional<std::vector<float>> codes; /* knob pad: skip the decoder */ };
  // JointVaeWorker::Output gains: schema::Snapshot lite; SoundCode sound; std::vector<float> mel;
  void JointVaeWorker::lab (LabRequest, Callback);   // decode (unless codes) -> apply lockedTail -> plan -> patch -> lite + sound
  // PluginProcessor.h
  void playLiteControls (const schema::Snapshot&);   // selects the Lite engine, leaves Surge patch mode, params.set without gestures
  ```
  - **Lock:** `lockedTail` replaces every decoder output from index `numNumeric` (435) on, which covers the bools, categoricals and stepped values. Read `numNumeric` from the bundle layout as the index of the first `kind == "bool"` entry, not a literal. Numeric entries can be `cont` or non-stepped `ord`; stepped `ord` entries come last (`export_bundle.py:layout`).
  - **Knob values for the forest and the knob pad:** for each of the 123 `knobs.knobs[].name`, the code of that feature, or NaN when the codec plan writes no `Value` for its path.
- [ ] **Step 1: Write failing tests**:
  - `"factory parity"`: every fixture line's `.fxp` → `liteControls` equals `values` within 1e-5. This needs `VAE_SURGE_DATA_DIR`.
  - `"decoded point plays on Lite"`: for parity case 0, `lab` gives `lite` with every value passing `schema::sanitiseValue` unchanged, and a non-empty `patch`.
  - `"lock keeps structure"`: with `lockedTail` from point A, decoding point B gives codes[435..] == A's.
  - `"old bundle rejected"` (Review Focus 1): a bundle dir without `xypad.json` fails `load` with a message containing "export_xypad.py", and the previous bundle still decodes.
  - Plugin `"Lite after Surge"` (Review Focus 4): `loadSurgePatch(p)` → `playLiteControls(s)` → the engine param is Lite, `SurgeEngine::hasPatchData()` is false, and a held note keeps sounding: RMS of the next 50 ms > 0.
  - `"lab timing"`: log the mean `lab()` time over 50 calls. Not an assertion, but report it in the task summary. If it is over 30 ms, flag it to the user. The latest-request-wins rule already caps the rate.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement `liteControls` as a line-by-line port. Implement `lab` on the worker, then `playLiteControls`: set `kEngineParamId` to the Lite index and call `useSchemaControls()` on the Surge engine as `applyControls` does, but without begin/end gesture.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Commit.

---

### Task 7: The v2 screen: layout, pad, axis list, toolbar (latent mode)

**Files:**
- Create: `source/experiments/joint_vae/LabPad.h`, `LabPad.cpp`
- Rewrite: `source/experiments/joint_vae/JointVaePanel.h`, `JointVaePanel.cpp`
- Modify: `source/plugin/PluginEditor.cpp` (wire `onLite` → `playLiteControls`, `onPatch` → `loadSurgePatch`)
- Test: `tests/plugin/PluginTests.cpp`

**Interfaces:**
- Consumes: Tasks 3–6.
- Produces:
  - `JointVaePanel` public API: `std::function<void (const schema::Snapshot&)> onLite; std::function<juce::Result (const juce::MemoryBlock&)> onPatch;` The bundle auto-loads from the installed path as now.
  - `LabPad`: `std::function<void (juce::Point<float>)> moved;` (0..1, y up), `std::function<void (juce::Point<float>)> tapped;`, `void setDot (juce::Point<float>)`, `void setMap (juce::Image)`, `void setPresetDots (std::vector<std::tuple<float, float, int>>, int activeLabel)`.
- **Layout** (spec mockup `lab-layout-v3.html`):
  - The toolbar row is on top.
  - Left column, 2/3 of the width: the pad (rectangle, fills the remaining height), the X label row, the Z slider row (hidden when no Z), the prompt box (Task 9) and the readout strip, about 120 px tall (Task 8).
  - Right column: the axis list in a `juce::Viewport`, with section headers "Sound axes" and "Latent dimensions", and minor dims inside a "N more dimensions (< 2 dB)" toggle. Each row has its name, X/Y/Z buttons, description, tags and a slider.
- **Behaviour:**
  - Every pad move, slider move, Z move or arrow key runs `constrain` and then `worker.lab`.
  - A result calls `onLite(out.lite)`.
  - Assigning works like the page's `assign()`: swap if the axis is already used, and clicking Z again clears Z.
  - Defaults are X `s:brightness`, Y `s:length`, Z the first dim.
  - Toolbar:
    - Start from: Typical ⟨label⟩ ×8 and Random real preset.
    - Category: 9 labels.
    - Lock structure: captures `codes[435..]` of the current output.
    - Reset: reload the base.
    - Preset dots: a toggle.
    - Send to Surge: `onPatch(current.patch)`.
    - Latent / Knob pad.
- [ ] **Step 1: Write failing tests**: `"VAE Lab v2 layout"`, at the editor's size with the Lab shown. The pad's width is ≥ 0.6 × the panel width minus margins, and its height is ≥ 0.55 × the panel height. The axis list is right of the pad and does not overlap it. A Viewport is present. Update or remove the v1 layout assertions that reference the old controls.
- [ ] **Step 2:** Run `ctest --test-dir build-surge -R vae_plugin`. Expected: FAIL.
- [ ] **Step 3:** Implement. Keep the existing `VAE_LAB_SCREENSHOT` export working.
- [ ] **Step 4:** Run. Expected: PASS. Export a screenshot (`VAE_LAB_SCREENSHOT=/tmp/vae-lab-v2.png`) and look at it against the mockup.
- [ ] **Step 5:** Commit.

---

### Task 8: Map, preset dots, readout strip

**Files:**
- Modify: `JointVaePanel.h/.cpp`, `LabPad.cpp`, `JointVae.h/.cpp`
- Test: `tests/ml/JointVaeTests.cpp`, `tests/plugin/PluginTests.cpp`

**Interfaces:**
- Produces: `void JointVaeWorker::map (MapRequest, std::function<void (juce::Result, MapResult)>)`.
  - `MapRequest { JointLatent z; int category; std::vector<juce::String> axisIds; float zPos; juce::String kind /* cat|brightness|length|attack|noise|none */; std::optional<std::vector<float>> lockedTail; }`
  - `MapResult { int n = 24; std::vector<int> cells; std::vector<float> conf, val; float lo, hi; }`
  - It runs on the same worker and is cancelled by any newer `lab` or `map` request. Category cells take the forest's argmax over the codes; measure cells come from the sound decoder and `measures`. `lo` and `hi` are the 2nd and 98th percentiles.
- **Readout strip:**
  - The spectrogram is drawn as the page's `specCanvas` draws it: `((mel + 70) / 70)^0.8`, accent colour over the background, frames × 64, low bands at the bottom, captions "0 s · note off 2 s · 6 s".
  - The measures are worded as the page words them.
  - The decoded preset lists the osc 1–3 types, with "(muted)", the effects that are on, cutoff, resonance and amp A/D/S/R. The values come from the codec plan, formatted as the page's `fmt` formats them.
  - The category guess shows the 8 bars.
- **Dots:** `posOf` of each preset z on X and Y. The current label is drawn bright and the others dim. A tap on a dot (< 7 px) jumps to that preset's z and label, with the chat line "Jumped to the real preset “name” (Label)."
- **Map colours:** the category colours are the page's `--c0..--c8` dark values. The measure map uses accent alpha 0.03–0.75.
- [ ] **Step 1: Write failing tests**:
  - `"map cancels"`: a `map` followed at once by a `lab` → the map callback reports cancelled (or never fires) and the `lab` result arrives.
  - `"map shape"`: kind "brightness" gives 576 finite `val` values with lo < hi.
  - Plugin `"editor closes mid-map"` (Review Focus 5): start a map, delete the editor at once, pump the message loop for 500 ms → no crash, and the leak detector stays quiet.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement. Schedule a map 150 ms after the last move with `juce::Timer`, and recompute it when the axes, category, base or lock change.
- [ ] **Step 4:** Run. Expected: PASS. Take a screenshot and compare it with the page at the same axes.
- [ ] **Step 5:** Commit.

---

### Task 9: Prompt box

**Files:**
- Modify: `JointVaePanel.h/.cpp`
- Test: `tests/plugin/PluginTests.cpp`

**Interfaces:**
- Consumes: `interpretPrompt` (Task 5).
- Behaviour:
  - A single-line `TextEditor` under the pad, with a read-only log of the last 4 lines above it (newest at the bottom). Return submits.
  - The user's line is shown as "› text", then the `PromptReply.message`.
  - A `start` loads that z and category like a base, without changing Lock. Then each nudge moves the named axis position by its delta, clamped to 0..1. Each nudge runs `constrain` with that axis plus the pad's assigned axes held, as the page's list slider does. Last, one `lab` decode runs.
- [ ] **Step 1: Write failing test** `"prompt starts and nudges"`: with the bundle loaded, submit "dark bass". The category combo shows Bass, and `axisPosition(brightness)` is the typical Bass position −0.1 (±1e-3). Submit "brighter": it is back at the typical position (±1e-3). Skip without a bundle.
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Commit.

---

### Task 10: Knob pad

**Files:**
- Modify: `JointVaePanel.h/.cpp`
- Test: `tests/plugin/PluginTests.cpp`

**Interfaces:**
- Consumes: `KnobGroups` (Task 4), `LabRequest::codes` (Task 6).
- Behaviour:
  - The mode switch keeps the current decoded codes.
  - The axis list shows the 61 groups ranked as `knobs.groups` orders them: name, section, description, importance meter (grey when importance ≤ 2·se, value as "+x.x pts"), and tags (knob count, |ρ|, "in N% of presets"). Rows have X/Y/Z buttons; defaults are groups 0, 1, 2.
  - A pad move calls `KnobGroups::set` for each assigned group on a copy of the 123 knob values. It writes them back into the numeric codes by feature name, then runs `lab({codes})`.
  - The map shows the category guess only: 24×24 forest predictions, with the other knobs at their current values.
  - Switching back to the latent pad keeps the last z. The knob edits are discarded, as on the page.
- [ ] **Step 1: Write failing test** `"knob pad moves only its group"`: assign group 0 to X and move to x = 1. Codes of knobs outside groups X/Y are unchanged, and the group's knobs equal `p95` (sign +) or `p5` (sign −).
- [ ] **Step 2:** Run. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run. Expected: PASS.
- [ ] **Step 5:** Commit.

---

### Task 11: Install, docs, end-to-end check

**Files:**
- Modify: `scripts/install-models.sh` (install `models/joint-vae-j48w_b05` as the Lab bundle), `docs/JOINT-VAE-LAB.md` (v2 section: workflow, bundle contents, `export_xypad.py`, limits; the v1 description points to tag `vae-lab-v1`)

- [ ] **Step 1:** Run the full suite: `cmake --build build-surge && VAE_JOINT_BUNDLE=… ctest --test-dir build-surge --output-on-failure`. Expected: all pass. Also build the default (non-Surge) configuration: the Lab shows "VAE Lab needs a Surge build", and `ctest --test-dir build` passes.
- [ ] **Step 2:** Run `scripts/install-models.sh`, open the standalone, and walk through the workflow:
  - Type "dark reese bass" → hear Lite.
  - Drag on the pad with a held note → no clicks.
  - Type "longer" → the dot moves.
  - Lock, drag → the osc types stay.
  - Click a dot → jump to that preset.
  - Knob pad → fine-tune.
  - Send to Surge → the Surge patch plays.
  - Drag again → back on Lite.

  Note anything that differs from the page.
- [ ] **Step 3:** Commit.
