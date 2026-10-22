# Pitcher native design study

A **separate JUCE GUI application**, translating the approved graphite/green Brutalist browser study. No changes to the plug-in editor, engine, root build or installation paths are required.

Design handoff: [PITCHER-BRUTALIST-JUCE.md](../../docs/design/PITCHER-BRUTALIST-JUCE.md).

## Run

From the repository root:

```sh
./design-study/juce/run.sh
```

Requires CMake, a C++20 compiler and an existing JUCE 8 checkout. By default it uses the repository's `build/_deps/juce-src`. To choose another:

```sh
./design-study/juce/run.sh -DPITCHER_JUCE_DIR=/absolute/path/to/JUCE
```

On macOS this builds and opens `build/PitcherStudy_artefacts/Release/Pitcher Design Study.app` under this directory. Build output is ignored by git. Run GUI applications from a normal desktop session: restricted automation can cause macOS `_RegisterApplication` to abort before the program's code executes.

## What works

- Equal-width, equal-height sidebars; compact flat axis list; centre-only 56 px three-octave keyboard; prompt; floppy-disk save; graphite surfaces and green controls.
- **1,375 real student preset embeddings**, musical directions and latent dimensions, extracted from the browser's `j48w_b05` payload.
- Interactive orbit camera, wheel/button zoom, reset; camera-plane handle dragging; Shift-depth dragging; XYZ constraint gizmo. Arrows rotate the camera; Page Up/Down edit depth; Home resets the camera.
- Category clicks highlight visual groups without changing the current latent vector or conditioning category.
- X/Y/Z assignments reproject the real cloud. Assigned sliders and handle editing solve the minimum latent change needed to satisfy the chosen coordinates.
- **Actual student sound decoder and PCA reconstruction**, generating the spectrogram and brightness/attack/length/noise/movement tooltip. This predicts a sound representation, not rendered audio.
- Named, session-only favourites capture latent vector, category and axis assignments and can be recalled.
- Prompt category words select a real preset in that category. No text encoder, remote service or LLM.

## Deliberate study boundaries

- **No audio output**, no device opening, no Surge or Lite engine, no parameter decoder and no ONNX runtime. Keyboard keys demonstrate visual press feedback only.
- Category bars use weighted proximity to the 32 nearest latent presets. The UI labels this approximation. This differs from the browser's random forest on decoded parameters; production should use the existing Lab model/controller.
- The spectrum tooltip includes measured model-spectrum properties only. The full browser tooltip's oscillator/FX/filter/ADSR values need the parameter decoder and must be supplied by the native implementor.
- Favourites last for the session; the browser's rename/removal/storage behavior remains the full handoff target.
- “Shape my space” is a concept notice in this native study. The browser remains the reference for its illustrative selection/strength flow. No model training is performed anywhere in this study.
- Sound inference is coalesced at 20 Hz on the message thread for this isolated experiment. Production should move it onto the existing worker, returning immutable results to the UI. Never copy this onto an audio callback.

## Renderer decision

Use JUCE `Graphics` with orthographic projection first. The cloud consists of small coloured dots and a wireframe cube, not meshes, lighting or volumetric surfaces. `OrbitMath.h` owns projection/inverse dragging; `OrbitView` owns painting and hit testing, independently of model state. It repaints on interaction, not on an idle animation loop.

For this dataset, a Release software-image benchmark at 700 × 500 measured **3.071 ms median / 3.302 ms p95**, 60 paints after 5 warmups, with 1,355 visible points from 1,375 total (21 October 2026, this development Mac). This is CPU paint callback time, **not presented frame rate**, not a GPU benchmark, and not a guarantee under audio load or at Retina scale. Re-run on target hardware and under the real plug-in workload before setting a frame-rate target.

OpenGL is not needed to reproduce the current visual design. Reconsider it if profiling larger clouds or high-DPI interaction identifies drawing as the bottleneck. Preserve the component's data/camera API if replacing its painter. JUCE's OpenGL context introduces a rendering thread and explicit lifetime management; do not add it merely to obtain a 3D-looking projection. See [JUCE OpenGLContext documentation](https://docs.juce.com/master/classjuce_1_1OpenGLContext.html).

## Checks and source map

```sh
# After building, on macOS:
'design-study/juce/build/PitcherStudy_artefacts/Release/Pitcher Design Study.app/Contents/MacOS/Pitcher Design Study' --checks
# Capture a native component image to /tmp/pitcher-native.png and exit:
'design-study/juce/build/PitcherStudy_artefacts/Release/Pitcher Design Study.app/Contents/MacOS/Pitcher Design Study' --snapshot
```

- `Source/StudyView.*`: standalone composition, axis rows, categories, favourites, keyboard styling.
- `Source/OrbitView.*`: replaceable JUCE 2D renderer and gestures.
- `Source/OrbitMath.h`: camera projection, inverse plane/depth movement and constrained movement.
- `Source/StudyData.*`: study-only data loading, axes and sound inference.
- `Source/StudyChecks.h`: real-asset smoke checks, latent constraints, 72 camera orientations, bounds and software-paint measurements. Failures return a nonzero process status even in Release.
- `export_data.py`: reproducibly extracts `Assets/study.json` from `../xy-pad/index.html` using only Python's standard library. Retains the student's data, not generated dummy points.

The implementation handoff should reuse the visual components and interaction math, connect them to the completed Lab's state/worker, and use the Markdown specification for differences listed above. Do not copy this application's `JUCEApplication`, session-only storage or message-thread model inference into the plug-in.

## Reuse in the plug-in

The UI now has an asset- and model-independent CMake target:

```cmake
# JUCE targets must already exist. Add once, alongside the existing plug-in build.
add_subdirectory(design-study/juce/Source/ui pitcher-ui)
target_link_libraries(YourPlugin PRIVATE pitcher_study_ui)
```

This does **not** compile `Main`, `StudyView`, `StudyData`, embedded model data or test checks. It adds only the reusable controls and orbit painter. The interface target follows this repository's pattern so JUCE modules are compiled in the final target rather than a second library.

- `PitcherLookAndFeel`: fonts injected by the host (optional; uses JUCE fallback without them). Own it in the editor before its controls; detach it before destruction. Linear horizontal/vertical sliders have a 10 px square green thumb. Other slider types fall back to JUCE.
- `PitcherKeyboard`: constructed with the processor's existing `MidiKeyboardState&`; it owns no synth, audio device or MIDI state. Naturals are charcoal, accidentals pale grey, both use solid primary green when held. The study still opens no audio device.
- `AxisRow`: callbacks emit axis id/slot and normalized values. `update()` is notification-free. `onGestureStart` and `onGestureEnd` let the controller bracket host gestures if appropriate; latent-coordinate edits are not automatically APVTS parameter changes.
- `OrbitView`: callbacks emit normalized XYZ only; camera rotation/highlighting have no sound side effects. Set point clouds and positions from the message thread after the worker completes. Keep inference/codec work outside callbacks and use the existing latest-request-wins queue.
- `Spectrum`: receives a JUCE image and tooltip string; it knows nothing about decoders.
- `PitcherLayout::calculate`: shared column, pad, prompt and keyboard geometry, independent of model/controller. Designed for a minimum 1000 px workspace; scale or specify a compact layout separately for narrower plug-in editors.

`StudyView` is the sample wiring/controller, not a production editor to copy wholesale. Replace its approximate category matching, prompt lookup, session-only saves and coalesced message-thread inference with existing Lab services. All reusable component calls are message-thread operations. No part of the painting/model code belongs in `processBlock`.

## Micro-interaction experiment

The standalone study now uses `StudyOrbit` and `StudyAxisRow`, isolated experiment
versions of the shared controls. **The plugin still uses the unchanged `OrbitView`
and `AxisRow`.** These copies are intentionally temporary: review the interaction
feel here before consolidating accepted behavior into the reusable controls.

Try this sequence:

1. Hover the central handle, then drag it. Its halo and movement plane strengthen;
   assigned axis rows briefly highlight and retain their names alongside percentage values.
2. Assign another X/Y/Z axis. Its row highlights; the cloud reprojects as before.
3. Click **Pin A**, explore, then **Swap A/B** to compare two model predictions.
   **Back** restores the previous gesture, including axes and category (64 states maximum).
4. Rotate/zoom the camera, then **Reset view** for a 220 ms eased reset. The sound
   position does not move. Category bars settle with a short easing transition.
5. Toggle **Motion: off** for immediate camera/bar updates and no decorative fades.

The explorer identifies pending versus completed **predictions**, never audio playback.
The study still produces no audio. Percentages on axis rows are normalized positions,
not physical synth parameter units. Direct numerical entry, selectable preset-dot
previews, Orbit LFO, and engine-transfer feedback are not part of this first experiment.
Animations run on message-thread timers which settle and stop; inference remains
coalesced at 20 Hz at most. This is a visual prototype, not a production threading design.

`--checks` now exercises the experimental painter plus pinned comparison, gesture
history and camera-reset state preservation. The shared production UI sources are
unchanged by this experiment.
