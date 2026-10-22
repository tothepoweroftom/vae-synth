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
  - A single-line `TextEditor` under the pad with a compact submit arrow, and a one-line inline status beside or under it showing the latest `PromptReply.message`. There is no transcript or log (user decision 21 Oct 2026, Task 17 answer 4). Return submits.
  - The text stays editable after submit.
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

---

## Follow-up: p0 prompt-to-preset (Tasks 12–15)

**Spec:** `docs/superpowers/specs/2026-10-21-p0-prompt-integration.md`. That spec is binding for these tasks. Start only after Tasks 1–11 pass.

**Source assets:** `/Users/thomaspower/Downloads/p0/`. These are import-only, never read at runtime.
- `text_encoder.onnx` is 501 MB.
- `tokenizer/` holds a RoBERTa / CLAP tokenizer.
- `prior.onnx` has outputs `category_logits`, `mix_logits`, `mu`, `logsd`.
- `layout.json` gives prior p0, vae j48w_b05, z 32, k 8, 9 labels, and 435/85/111 features.
- `bank.npz` holds `z[1227,32]`, `codes[1227,631]`, `category[1227]` and `feats[631]`.

Facts checked by the controller:
- The bank's `feats` are the same **set** as our bundle's features, but in a **different order**. The bank uses schema order and starts with `character`; the bundle starts with `fx/chorus/p0`. Rows must be reordered by name.
- The bank's numeric and categorical codes contain **NaN**.
- The bank's bool codes are **not** 0/1. Values like 3.7e-5 appear.

So bank rows are not in the decoder-output format that `StudentCodec::plan` takes. Task 12 pins their meaning.

### Task 12: p0 bundle: import, bank semantics, reference fixtures

**Files:**
- Create: `tools/joint_vae/export_prompt.py`
- Create: `tools/joint_vae/prompt_reference.py`. It is the offline reference runner, using `tokenizers`, `onnxruntime` and `numpy`.
- Output (git-ignored): `models/prompt-p0/`, containing `prompt.json`, `text_encoder.onnx`, `tokenizer.json`, `prior.onnx`, `bank.bin`, `bank.json` and `prompt-parity.json`.

**Interfaces:**
- Produces `prompt.json`:
  - `id`, a hash of every file.
  - `lab_bundle_id`, the installed Lab bundle id it is valid for.
  - `prior: "p0"` and `vae: "j48w_b05"`.
  - `labels`.
  - `tokenizer: {max_length, pad_id: 1, bos: 0, eos: 2, padding, truncation}`, with the values actually used, as found in step 1.
  - `embedding: {dim: 512, l2_normalise: true}`.
  - `bank: {rows, feats_in_bundle_order: true}`.
- `bank.bin` holds little-endian float32 values: `z[rows,32]` followed by `codes[rows,631]`, with codes in **bundle feature order** and in **`StudentCodec::plan` input format**.
- `bank.json` holds `category[rows]` as label indices, and `name[rows]`, written as `"<Label> · <row:04d>"`.
- `prompt-parity.json` has ≥ 12 cases. Each gives text, token ids and mask, normalised embedding, category logits, mix weights, `mu`, the selected row ids (selection rules as in the spec) and the selected codes. Include an empty string, one longer than max_length, Unicode, Bass/Lead/Pad prompts, an unsupported category (Keys), the swaps dark/bright and short/long, and one near-tie case.

- [ ] **Step 1: Pin preprocessing.**
  - Read `tokenizer_config.json`, `special_tokens_map.json` and `tokenizer.json` to establish max length, padding and truncation, then write them into `prompt.json`.
  - Record in the task report that this reproduces **our** interpretation, not the student's golden outputs. If the folder has golden outputs, use them.
- [ ] **Step 2: Pin bank-code semantics.**
  - Find how the student builds bank `codes`, in `vae/pitcher/` (`joint_vae`, `inverter/data.py:full_row`, `core/codec.py`).
  - Write the conversion to `StudentCodec::plan` input: what NaN means (unset, so the frozen value applies, or inactive), how bools threshold, and how categorical and stepped values map.
  - Verify it: for 20 bank rows, the plan from our converted codes run through the C++ codec must equal the student's Python codec applied to the original row. Extend `tools/joint_vae/codec_parity.py` with a `--bank` mode.
  - If the semantics can't be pinned, stop and report BLOCKED. Don't guess.
- [ ] **Step 3: Check compatibility.**
  - `export_prompt.py --source ~/Downloads/p0 --lab models/joint-vae-j48w_b05 --out models/prompt-p0` fails unless the feature sets are equal, the counts are 435/85/111, the labels match in order, and `layout.vae == lab config run`.
  - It also runs the supplied `decoder.onnx` against the Lab's `decoder.onnx` on 8 fixed (z, label) inputs, comparing after argmax and threshold. Expected: identical outputs. Report the max difference on numeric values.
  - It never copies the supplied decoder into the Lab bundle.
- [ ] **Step 4:** Run export and reference. Expected: `prompt-parity.json` is written, and the selection is deterministic over two runs (byte-identical).
- [ ] **Step 5:** Commit the tools only.

### Task 13: Native prompt model (`PromptModel`)

**Files:**
- Create: `source/experiments/joint_vae/PromptModel.h`, `PromptModel.cpp`
- Modify: `source/ml/tokenizer/src/lib.rs`. Add a configurable entry point and keep the existing 256 / pad 0 / float path and its callers unchanged.
- Test: `tests/ml/PromptModelTests.cpp`

**Interfaces:**
```cpp
struct PromptCandidate { juce::String bundleId; int row; int category; JointLatent z; std::vector<float> codes; juce::String name; };
class PromptModel {   // worker thread only
public:
    juce::Result load (const juce::File& promptJson, const JointVae& lab);   // validates against lab.layout().id; keeps previous on failure
    juce::Result find (const juce::String& text, std::vector<PromptCandidate>& out, int& category);
};
```
- Behaviour:
  - Tokenise with int64 ids and mask.
  - Run the text encoder and L2-normalise. Fail on a non-finite or zero-norm embedding.
  - Run the prior.
  - The category is the argmax over logits 0–7.
  - Softmax the mix logits and keep weights ≥ 0.02, highest first; equal weights go by component index.
  - For each kept mean, pick the nearest unused bank row in that category by squared Euclidean distance on the raw z; equal distances go by row index.
  - Return at most 8 rows.
- No bank row for the category returns a failed result with the message `This bundle has no <Label> presets`, and `out` stays empty.
- Its thread count is bounded at 2. It gets its own worker, separate from `JointVaeWorker`, so a slow prompt never blocks pad decoding.

- [ ] **Step 1: Write failing tests.**
  - `"prompt parity"`: for each `prompt-parity.json` case, the token ids and mask are equal, the embedding, logits, weights and mu are within 1e-4, and the rows and codes are exactly equal. Skip without `VAE_PROMPT_BUNDLE`.
  - `"no bank category"`: the Keys prompt fails with the message above.
  - `"bad bundle keeps previous"`: a wrong `lab_bundle_id` fails and the previous model still works.
  - `"existing tokenizer unchanged"`: the existing Prompt-tab tokenizer test vectors still pass.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run them. Expected: PASS. Log cold and warm `find` latency and peak RSS in the report.
- [ ] **Step 5:** Commit.

### Task 14: Wire prompt search into the Lab

**Files:**
- Modify: `JointVaePanel.{h,cpp}` and the worker or controller from Tasks 6–9
- Test: `tests/plugin/PluginTests.cpp`

**Behaviour:** as in the spec's "User flow", "Exploration after selecting a bank preset" and "Follow-up words".
- **Two actions in the existing prompt box.**
  - **Find sounds** runs `PromptModel::find` on the whole text, with no adjective nudges. Mixed or ambiguous text uses Find sounds.
  - **Nudge** uses `interpretPrompt`.
  - Return triggers Find sounds when a p0 bundle is loaded, and Nudge otherwise.
- **The candidate list** shows up to 8 entries, labelled `Bass · 0142`. The first one is selected.
- **Selecting a candidate:**
  - The Lab enters bank mode: z = bank z, the category is set, `codes` = the bank row.
  - It runs `lab({codes})` → Lite.
  - Send to Surge uses those codes (no decoder call).
  - The status reads `Selected a bank preset · Lite preview`.
- **In bank mode:**
  - Pad and axis-slider sound edits are disabled; inspection still works.
  - The knob pad edits a copy of the bank codes.
  - **Explore generated variations** decodes the bank z once and returns to normal v2 dragging.
  - Re-selecting the candidate restores its exact codes.
- **Without a p0 bundle,** the prompt box works as in Task 9. It is labelled "word list", and no Find sounds button is shown.
- **Stale results:**
  - Each prompt request carries a token.
  - A newer prompt, a manual edit, a candidate selection, Reset, a bundle change or the panel being destroyed invalidates older tokens.
  - A late result with an invalid token is dropped.

- [ ] **Step 1: Write failing tests.**
  - `"find selects fixture row"`: the Bass parity prompt gives the expected row, its codes reach `lab()`, and the dot sits at the bank z.
  - `"alternative restores exact codes"`.
  - `"bank mode blocks latent edits until explore"`.
  - `"stale prompt result dropped"`: send prompt A, then prompt B, and let A finish last. The state is B's.
  - `"no-bank category keeps sound"`.
  - `"editor closes mid-prompt"`.
  - Each test skips without `VAE_PROMPT_BUNDLE`.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run them. Expected: PASS, and every v2 test still passes.
- [ ] **Step 5:** Commit.

### Task 15: Install, profile, musical comparison, docs

**Files:**
- Modify: `scripts/install-models.sh`, which installs `models/prompt-p0` next to the Lab bundle when it is present.
- Modify: `docs/JOINT-VAE-LAB.md`.
- Create: `docs/P0-PROMPT-EVAL.md`.

- [ ] **Step 1: Profile.** Hold a note in the standalone and run 20 prompts, half of them while dragging. Record cold and warm latency, peak memory and audio underruns.
- [ ] **Step 2: Compare by ear.** Use a fixed prompt set: the contrasting pairs dark/bright bass, short/long pad and soft/sharp lead, plus 6 more. Compare p0 Find sounds against the category-only and word-list starts. Render each through Lite and Surge with the existing render path. Record coverage and failures in `docs/P0-PROMPT-EVAL.md`. If the results don't support semantic claims, label the feature "experimental" in the UI and the docs.
- [ ] **Step 3:** Update the docs: bundle contents, export order, the two prompt actions, bank mode and its limits. Then commit.

---

### Task 16: One sound across the Lab and Synth views; remove the Prompt tab

*User decisions (21 Oct 2026). They supersede the spec's "Synth and Prompt tabs behave as before" and the p0 spec's "no changes to the Prompt tab". Runs after Task 11 and before Task 14.*

**Files:**
- Modify: `source/plugin/PluginEditor.{h,cpp}`, `JointVaePanel.{h,cpp}`, and `PluginProcessor.{h,cpp}` if the edited-state flag lives there.
- Test: `tests/plugin/PluginTests.cpp`

**Behaviour:**
- **Shared views (option b).** Only two views remain: **Lab** and **Synth**. They show the same sound.
  - The Lab already plays by writing the 93 Lite parameters, so the Synth view's knobs show the Lab's preset. Tweaking there works normally.
- **Synth edits carry back.** When any of the 93 parameters changes from a source other than the Lab, through a Synth-view knob or host automation, the Lab marks the sound **edited**:
  - The status line reads "Edited in Synth view".
  - The next Lab sound edit (pad, slider, Z, prompt nudge or base change) asks once: "Replace your Synth edits?" The two answers are Replace and Keep editing.
  - Keep editing leaves the sound and the Lab state unchanged.
  - Choosing a dot or a base also asks.
  - Lab-originated writes never set the edited flag. Tag them (a flag set around `playLiteControls`, or a comparison against the last Lab snapshot) so the flag can't be raised by the Lab itself.
- **After Send to Surge** the Synth view's knobs don't control Surge's full patch, which is the existing behaviour. The Synth view shows a one-line banner, "Playing the full Surge patch from the Lab. Knobs return to Lite", and the first knob move returns to Lite (the existing `applyControls` path).
- **Prompt tab removed.**
  - Remove its button and panel from the editor.
  - Wrap the Prompt-tab UI and its wiring (button, panel member, `generateFromPrompt` callers in the editor) in `#if VAE_PROMPT_TAB` (default 0) rather than deleting it. Leave a one-line comment pointing at this task.
  - The processor-side Stable Audio / SAME code keeps compiling and stays unused by the UI.
  - Existing Prompt-tab tests are guarded the same way.
  - *My call:* an `#if` guard instead of commented-out code, so the code keeps compiling-checkable and can be re-enabled with one define.

- [ ] **Step 1: Write failing tests:**
  - `"two views only"`: the editor has Lab and Synth buttons and no Prompt button.
  - `"lab preset shows in synth view"`: after a Lab pad move, the Synth view's control values equal the Lab's `lite` snapshot.
  - `"synth edit marks lab edited"`: change one parameter through the store as a user would, and the Lab reports edited. A Lab pad move then asks; Keep editing leaves the parameters unchanged, Replace applies the new point.
  - `"lab writes do not mark edited"`.
  - `"prompt tab compiled out"`: with `VAE_PROMPT_TAB` 0, the build has no Prompt view.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run them, plus the full ctest with and without Surge. Expected: PASS.
- [ ] **Step 5:** Commit.

---

### Task 17: Brutalist re-skin from the JUCE design study (last; after the final review of Tasks 1–16)

*User request (21 Oct 2026). It replaces the deferred "re-skin to the design study" item.*

**Source:**
- Handoff: `docs/design/PITCHER-BRUTALIST-JUCE.md`, which is binding for visuals and interaction.
- Study: `design-study/juce/`. Reuse `design-study/juce/Source/ui/`, the `pitcher_study_ui` target: `PitcherLookAndFeel`, `PitcherKeyboard`, `AxisRow`, `OrbitView`/`OrbitMath`, `Spectrum`, `PitcherLayout`.
- Do NOT copy `StudyView`, `StudyData`, the embedded study data or the study's approximate category matching, prompt lookup or message-thread inference. The completed Lab's worker, forest, prompt and data are authoritative.

- **User decisions (21 Oct 2026):**
  1. **Orbit is the only view of the latent pad.** The 2D category/measure map from Task 8 is not shown. The map worker request may stay for later use or be removed. *My call:* the **knob pad** keeps its flat 2D pad, because it moves knob groups, not latent axes, so it has no 3D cloud.
  2. **Toolbar controls go in the left sidebar, under the category view.** That covers Latent/Knob pad, the start point, category, Lock structure, Reset and Send to Surge.
  3. **No Presets tab and no favourites** ("Shape my space" is skipped too). The right sidebar shows only the axis list.
  4. **Single-line prompt field** with inline status, and no transcript. This is already applied to Task 9.
  5. **The study's 3-octave centre keyboard replaces the editor keyboard** while the Lab is shown.
- [ ] **Step 1 (done): the user's decisions are above.** These are the conflicts with the built Lab:
  - **Orbit only, or Orbit + flat XY?** The handoff says Orbit is the only view. That removes Task 8's 2D category/measure map and click-to-jump on the map.
  - **Toolbar:** the study has no toolbar. Where do Latent/Knob pad, the start point, category, Lock structure, Reset and Send to Surge go? Hidden, a compact row, or a menu?
  - **Presets tab and favourites:** do they persist in plug-in state, and does "Shape my space" stay a concept-only preview?
  - **Prompt:** the single-line field (no transcript) versus Task 9's 4-line log, and how Task 14's Find sounds candidates are shown.
  - **Window size:** the layout targets ≥ 1000 px; the plug-in editor is 1240 px wide above its own keyboard. Does the study's 3-octave centre keyboard replace the editor keyboard while the Lab is shown?
- [ ] **Step 2:** Write the agreed answers into this task, then build it test-first like the other UI tasks. Link `pitcher_study_ui` into the plug-in, re-lay out `JointVaePanel::resized()`, apply the tokens and fonts, and keep every v2 behaviour and test green.
- [ ] **Step 3:** Screenshot against `docs/design/pitcher-native-study.png`, then run the full ctest with and without Surge, then commit.

---

### Task 18: Find and fix the crackle when dragging (runs next, after Task 16)

*User report (21 Oct 2026): dragging on the latent pad, or moving individual X/Y/Z axis sliders, crackles a lot in the standalone. That breaks the spec's core goal ("drags never click or zipper"). Use superpowers:systematic-debugging. Reproduce and measure before changing anything.*

**Files:** the cause decides them. Likely candidates are `source/engine/lite/LiteEngine.{h,cpp}`, `PluginProcessor::playLiteControls`, `JointVaePanel` (request rate) and `LiteFx.h`.

- [ ] **Step 1: Reproduce in a test.**
  - Write a headless repro: prepare the processor with Lite, hold a note, and drive a sweep of 60 `lab` results per second through `playLiteControls` along a real pad path with the bundle, latent, Lock off.
  - Render 2 s and measure:
    - the number of discrete-control changes per second;
    - the number of ducks started;
    - the number of samples with the duck gain below 1;
    - the max sample-to-sample step;
    - the process() time per block (mean and max) against the block budget.
  - Repeat the measurements with Lock on, and with an axis slider moving a single z coordinate.
- [ ] **Step 2: Rank the hypotheses with the numbers.**
  - (a) Discrete changes flip often enough that repeated 5 ms ducks modulate the sound.
  - (b) Unchanged discrete values written again still start a duck.
  - (c) FX on/off or type changes reset or reallocate effect state.
  - (d) Continuous smoothing (τ 20 ms) is too short for the request rate, or isn't applied on some path (e.g. `setControls` / prepare snapping, or `playLiteControls` bypassing smoothing).
  - (e) CPU: process() exceeds its budget or allocates.
  - (f) The message thread writes 93 parameters per move, flooding the parameter store or host.
- [ ] **Step 3: Fix the root cause(s) only.**
  - Each fix gets a regression test built from the Step 1 measurement: zero ducks for unchanged discrete values, a bounded duck rate during a drag, a bounded max step, and process() within budget.
  - If discrete flips during a drag are the cause, choose the smallest musical fix and record the choice. Options:
    - rate-limit discrete changes while dragging, applying them on release;
    - a hysteresis/hold on discrete outputs;
    - a longer crossfade.
  - The Lite-vs-Surge comparison must still match (fresh renders unchanged).
- [ ] **Step 4:** Ask the user to drag again in the standalone. Commit with the before and after measurements in the commit message.

---

### Task 19: Orbit LFO, slow movement around the current preset (final sprint, after Task 17)

*User request (21 Oct 2026): "an orbit LFO so the synth moves smoothly around its current preset", in the spirit of Animoog's orbit.*

**Idea:**
- The current point becomes the orbit **centre**: the anchor z plus the pad positions of the assigned axes.
- The orbit moves the dot along a closed path around that centre in the assigned axes: X/Y, or X/Y/Z in the Orbit view.
- Each step runs `constrain` from the centre, so the other latent directions stay at the anchor. It is decoded through the normal Lab path to Lite, so the sound drifts continuously around the preset and returns to it.

**Defaults (my calls; confirm in Step 1):**
- **Shape:** a circle in the X/Y axes. A Z wobble at a 0.7× rate is used when Z is assigned.
- **Rate:** 0.02–2 Hz on a log scale, default 0.1 Hz. It is free-running and not tempo-synced.
- **Depth:** 0–0.25 pad units of radius, default 0.08.
- **Lock structure is forced on while the orbit runs.** Only continuous knobs move, so a switch flip never ducks or crackles. This builds on the Task 18 findings.
- **Update rate:** a message-thread timer at 30 Hz submits `lab` requests (latest wins). Lite's 20 ms smoothing fills the gaps between them.
- **Ownership:** a drag, a prompt nudge or a dot click re-centres the orbit at the new point. The orbit stops on Reset, a base or category change, and in knob-pad mode.
- **Display:** the Orbit view shows the centre (hollow ring), the path (dashed) and the moving handle. Axis sliders follow the handle.
- **R14:** turning the orbit on is a user action.

**Files:**
- Create: `source/experiments/joint_vae/LatentOrbit.{h,cpp}`, a pure function with no GUI.
- Modify: `JointVaePanel.{h,cpp}` (controls in the left sidebar under the category view, per Task 17, plus a timer) and the Orbit view.
- Test: `tests/ml/XyPadTests.cpp` or a new `LatentOrbitTests.cpp`, plus `tests/plugin/PluginTests.cpp`.

**Interface:**
```cpp
struct OrbitSettings { bool on = false; float rateHz = 0.1f, depth = 0.08f; };
// Pad positions (x, y, z) on the orbit at phase t (seconds); centre + depth*(cos, sin, 0.5*sin(0.7 wt)), clamped to 0..1.
std::array<float, 3> orbitPosition (std::array<float, 3> centre, const OrbitSettings&, double seconds, bool hasZ);
```

- [ ] **Step 1: Confirm the defaults with the user.** Cover the shape (circle or figure-8), the rate range, tempo sync, the depth range, and whether forcing Lock on during the orbit is acceptable.
- [ ] **Step 2: Write failing tests:**
  - `orbitPosition` at t = 0 and one full period returns the centre plus (depth, 0); it stays within depth of the centre and inside 0..1.
  - A panel test with the bundle: orbit on, pump for 1 s → at least 20 `onLite` calls, the codes from index 435 on are constant (Lock), and the dot moves.
  - Orbit off returns the sound to the centre point.
  - A user drag re-centres the orbit.
  - Reset stops it.
  - Reuse the Task 18 drag-crackle measurement: run the orbit for 2 s with a held note and assert zero ducks and a bounded max sample step.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the tests, then ask the user to listen with a held note. Commit.

---

### Task 20: Find highlights its matches on the pad; exploring continues from the selected one (runs next, after Task 15)

*User decision (21 Oct 2026), after trying Task 14. It replaces the p0 spec's "explicit Explore generated variations transition": bank mode must not lock the pad.*

**Behaviour:**
- **Find marks its candidates on the pad.**
  - Each candidate's bank z is projected with `axisPosition` on the assigned axes and drawn as a highlighted dot: an accent ring and its number 1–8.
  - The highlights sit above the ordinary preset dots.
  - They reproject when the axes change.
  - They stay until the next Find, Reset, a base change or a bundle change.
  - The chips remain and stay in sync with the dots: the selected chip and the selected dot match.
- **Clicking a highlighted dot (7 px) or its chip** selects that candidate. It goes through `edit()`, as in Task 14, and plays the bank row's exact codes. The handle moves to that point.
- **Exploring is never locked.**
  - The first pad, Z, axis-slider or prompt-nudge move after selecting a candidate leaves exact-preset mode.
  - The move is decoded from the bank z with the normal v2 path, and dragging continues live.
  - The status reads "Exploring from <name>".
  - Re-selecting the dot or chip restores the exact codes.
- **Removed:** the "Explore generated variations" button and every bank-mode guard that disabled pad, slider, Z, category or nudge.
- **Kept:**
  - Knob pad on a copy of the bank codes.
  - Send to Surge sends the exact preset's codes (no decoder) while still on the exact preset, and the decoded patch after exploring.
  - Stale-result tokens.
  - R14.
  - The edit gate.
- **Docs:** update the p0 spec ("Exploration after selecting a bank preset" now points to this user decision) and docs/JOINT-VAE-LAB.md.

- [ ] **Step 1: Write failing tests:**
  - `"find highlights candidates"`: after a fixture Find, the pad has N highlighted dots at `axisPosition`(bank z), within 1e-4.
  - `"dot click selects candidate"`: clicking highlighted dot 2 plays row 2's exact codes and moves the handle.
  - `"drag explores from candidate"`: after selecting, an arrow-key move calls `lab` without `codes` (decoded), starting from the bank z. Nothing is blocked.
  - `"reselect restores exact codes"`.
  - `"send to surge exact then decoded"`.
  - Update or remove the Task 14 bank-mode-blocks tests.
- [ ] **Step 2:** Run them. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run them, plus the full ctest in both builds. Expected: PASS. Take a screenshot.
- [ ] **Step 5:** Commit.

---

### Task 21: Switch prompt search to the student's p1 model

*User request (21 Oct 2026). Source: `/Users/thomaspower/Downloads/p1/`. This is import-only data and must never be read at runtime.*

**Checked by the controller:**
- p1 has the same `text_encoder.onnx`, tokenizer and `decoder.onnx` as p0 (identical sha1).
- `prior.onnx` is **retrained**: layout `prior: "p1"`, max diff vs PyTorch 3.8e-6.
- `bank.npz` has **4,429 rows covering all 8 categories**: Atmosphere 751, Synth 693, Orchestral 629, Bass 527, Lead 494, Keys 474, Pluck 468, Pad 393.
- `feats` are in the same order as p0's. The VAE is still j48w_b05 with 435/85/111 features.

**Files:**
- Modify:
  - `tools/joint_vae/export_prompt.py` and `tools/joint_vae/prompt_reference.py`: accept any prior id, recorded in prompt.json, instead of requiring "p0".
  - `scripts/install-models.sh`: install `models/prompt-p1` as the prompt bundle, falling back to p0 when it's absent.
  - Any native code that hard-codes "p0" or the p0 bank size or path.
  - `docs/JOINT-VAE-LAB.md`.
  - `docs/P0-PROMPT-EVAL.md`: rename it to a prompt-eval sheet covering both models, or add a p1 section.
- Output (git-ignored): `models/prompt-p1/`.

- [ ] **Step 1:** Run `export_prompt.py --source ~/Downloads/p1 --lab models/joint-vae-j48w_b05 --out models/prompt-p1`. Expected: all compatibility checks pass, including the decoder comparison and the feature order. Then run `prompt_reference.py` for p1 parity fixtures, including a Keys prompt that now selects real Keys rows, and `codec_parity.py --bank` on 20 p1 rows.
- [ ] **Step 2:** Run the native `PromptModelTests` and the plugin prompt tests with `VAE_PROMPT_BUNDLE=models/prompt-p1/prompt.json`. Expected: parity passes, with no code changes beyond any hard-coded p0 assumptions. Keep the p0 tests passing too.
- [ ] **Step 3:** Re-render the listening sheet with p1, using the same prompts and `vae_p0_eval`. Record next to p0: the categories returned, whether the contrasting pairs (dark/bright, short/long, soft/sharp) now select different first rows, and coverage. Make no quality claims; listening is the user's.
- [ ] **Step 4:** Switch the installer and docs to p1, then run the full ctest in both builds and commit.

---

### Task 22: No header row; brutalist theme for the Synth view too (after Task 19)

*User request (22 Oct 2026): "avoid the empty space up top, move the Synth / VAE Lab buttons to the first column under the logo name. Also theme the Synth view the same way as the Lab."*

**Behaviour (my calls are marked):**
- **Remove the editor's top header row** in both views. The workspace starts at the top margin.
- **Left column, shared by both views, from the top:**
  - the "pitcher" logo;
  - the **Synth / VAE Lab** view tabs, styled as the design doc's tabs;
  - *my call:* **Init** and the **engine selector** (Lite / Surge XT) directly under the tabs;
  - *my call:* the Surge banner ("Playing the full Surge patch from the Lab…") moves here from the old header.
- **Lab view:** the rest of the left column stays as Task 17/19 built it: Category match, the controls, the Orbit LFO controls, and the Expected sound at the bottom. The column now starts with the logo and tabs, so recheck that everything still fits without clipping at the editor's size, and compact it where needed.
- **Synth view:** the same left column, then the existing knob sections in the remaining width.
  - Theme it with the brutalist tokens: graphite `#252525`, green `#70ff63` accents, square controls, Space Mono labels and hairline section rules. Use `PitcherLookAndFeel` plus `LabTheme.h`, so every colour is still defined once.
  - *My call:* use the same 3-octave `PitcherKeyboard` along the bottom of the Synth view's right area, so both views share one keyboard style.
  - Section layout and every control's behaviour stay unchanged.
- The non-Surge build still works: the Lab shows its label, and the Synth view is fully themed.

**Files:** `source/plugin/PluginEditor.{h,cpp}` (header removal, shared left column, Synth theming), possibly the `JointVaePanel` left-column layout, `tests/plugin/PluginTests.cpp`, and `docs/JOINT-VAE-LAB.md`.

- [ ] **Step 1: Write failing layout tests:**
  - No component sits in a top header band in either view.
  - The view tabs, Init and the engine selector sit in the left column under the logo in both views.
  - The Synth view's sections are right of the left column and don't overlap it.
  - The keyboard is inside the right/centre area.
  - Lab layout: no left-column component is clipped.
- [ ] **Step 2:** Implement.
- [ ] **Step 3:** Take screenshots of both views and compare them with `docs/design/pitcher-native-study.png`. Run the full ctest in both builds. vae_format_check failures on the user's uncommitted Lite files are expected. Commit.
