# Pitcher UI motion pass — implementor handoff

Date: 22 October 2026

## Intent and decision authority

Elevate the plugin's feel through precise interaction feedback, readable state changes,
and restrained animation. Preserve the graphite surfaces, fine rules, typography, and
signal-green identity established in the Lab.

**The main implementor has final say on design and implementation decisions.** This
is a set of proposals and prototype observations, not a mandatory feature checklist
or a pixel-perfect specification. Adapt, simplify, defer, or reject suggestions based
on the current code, musical workflow, performance, accessibility, and your judgment.
Routine design tradeoffs do not require another approval round. Briefly record material
deviations and their rationale in the implementation summary.

Review the current branch before changing anything: another session has been fixing
the production plugin. Preserve that work and reconcile against its final behavior.

## Reference and scope

- [Native study instructions](../../design-study/juce/README.md)
- [Existing design handoff](PITCHER-BRUTALIST-JUCE.md)
- Study-only experiments: `design-study/juce/Source/StudyOrbit.*`, `StudyAxis.h`,
  and `StudyView.*`.
- Production integration: `source/experiments/joint_vae/JointVaePanel.*`,
  `LabOrbit.*`, `LabPad.*`, `LabAxisList.*`, `LabTheme.h`, and
  `source/plugin/PluginEditor.*` / `ui/VaeLookAndFeel.*`.
- Shared controls currently live in `design-study/juce/Source/OrbitView.*` and
  `Source/ui/PitcherControls.*`; these also serve the plugin.

The experiments intentionally use separate study-only controls. They are temporary
copies for evaluating the feel, not an architecture to reproduce in production.
Consolidate accepted behavior into the appropriate reusable components rather than
maintaining parallel implementations. Do not overwrite newer production behavior
with the prototype's older controller assumptions.

This pass should primarily improve existing interactions. Undo/A/B, new preset
selection behavior, and direct value entry are separate functional work if the
production architecture does not already support them. Scope them deliberately.

## What you can try now

Run `./design-study/juce/run.sh` from the repository root.

1. Hover and drag the central handle: its halo and movement plane strengthen;
   assigned axis rows briefly highlight as their values change.
2. Change X/Y/Z assignments: the assigned row highlights and the cloud reprojects.
3. Read assigned sliders: parameter names remain visible alongside normalized
   percentage values.
4. Pin A, explore, and Swap A/B; use Back to restore a previous exploration gesture.
5. Rotate or zoom the camera, then Reset view to see its eased return.
6. Toggle Motion off to compare immediate updates with animated feedback.

Category bars ease toward new proximity results. The explorer distinguishes pending
and completed predictions. **The study has no audio output:** it cannot validate
playback latency, sound continuity, production worker completion, or host automation.
Its prediction state is not evidence that a sound has reached the audio engine.

Not prototyped here: preset-dot hover previews/selection, Orbit LFO polish, engine
transfer feedback, numeric entry, and a general button/tab transition system.

## Suggested order of work

| Priority | Interaction | Proposed behavior | Judgment call |
| --- | --- | --- | --- |
| First | Handle hover, focus, and drag | Reveal a restrained halo and movement guides; strengthen the active plane/axis while dragging; release gracefully. | Tune visibility against the dense cloud. Keep hit testing and visible affordances consistent. |
| First | Assigned axes | Link the active axis row to the corresponding pad guide through a brief highlight and a persistent assignment indicator. | Avoid flashing many rows continuously during Orbit or automation. |
| First | Values | Preserve names while displaying values with appropriate units; make focus as informative as hover. | Study percentages are normalized coordinates, not synth units. Choose a layout that remains readable at minimum size. |
| First | Requested versus applied state | If decoding lags, acknowledge the request without implying that it is already audible. Resolve pending state only at the appropriate accepted-result/application boundary. | Use the real worker/engine lifecycle; determine whether a separate applied-position marker is useful. |
| Next | Category results | Ease bar lengths between completed results; keep category order stable and numbers readable. | Coalesce rapid updates. Avoid implying statistical confidence beyond what the underlying model provides. |
| Next | Camera reset and navigation | Ease camera reset without changing the sound; keep direct manipulation immediate. | Camera animation must be interruptible by a new gesture. Decide whether view/tab fades add value. |
| Next | Prompt workflow | Keep the current sound available during work; communicate actual loading/search/result/error states and visibly link a selected candidate to its cloud marker. | Do not fabricate progress or add an artificial delay. Preserve the existing latest-request-wins behavior. |
| Next | Orbit and structure lock | Keep centre, path, and playhead distinguishable. Explain an automatically enabled lock locally. | Represent actual Orbit start/stop behavior; decorative motion must not affect modulation. |
| Next | Engine handoff | Confirm successful transfer only after success; retain a clear indication of the active playback engine. | Reconcile with current Lite/Surge transition and edited-sound behavior. |
| Optional | Exploration recovery | A dependable previous-state action and pinned comparison point make exploration safer. | Integrate with existing history/state ownership; do not introduce a second conflicting undo stack. |
| Optional | Preset discovery | Give selectable dots a useful hover/focus affordance and compact identity information. | Do not make nonselectable points appear clickable; preserve camera-drag and handle-drag disambiguation. |

## Motion vocabulary to evaluate

These are starting values, not acceptance thresholds:

| Transition | Starting treatment |
| --- | --- |
| Hover / focus | 80–120 ms colour or outline change |
| Press / drag start | Immediate acknowledgement |
| Release | 100–160 ms return to resting emphasis |
| Selection / result appearance | 120–180 ms fade |
| Category bar update | Roughly 150–220 ms settle |
| Camera reset | Roughly 180–250 ms ease-out |
| Successful action | Brief acknowledgement followed by a persistent, accurate state |

Keep pointer-controlled handles and parameter values synchronous with input. Apply
animation to supporting visuals, not to the user's control signal. Avoid bounce,
overshoot, gratuitous pulsing, and perpetual decorative motion. A new interaction
should retarget or interrupt an animation rather than queue behind it.

Use the current theme tokens; do not scatter new hard-coded colours across components.
Check contrast and hit targets at the actual plugin size, not only in a large screenshot.

## Production constraints

- All visual animation belongs on the message/UI thread. No allocation, file I/O,
  locks, or dynamic container growth in the audio callback for this work.
- Keep model inference on the existing worker path. The study performs coalesced
  inference on the message thread and must not be copied as a production design.
- Use existing bounded, lock-free communication where audio/UI communication is
  needed. Do not add per-frame callbacks or UI object access to DSP processing.
- Keep sound-state ownership in the controller/processor. Visual interpolation must
  not write parameters or generate additional host automation gestures.
- Prefer shared animation scheduling, elapsed-time interpolation, and repainting
  only affected regions. Stop decorative timers when settled or hidden. Resume
  cleanly when the editor becomes visible again.
- Preserve focus, scrolling, keyboard operation, and accessible control names.
  Modifier meanings already differ between contexts: verify actual bindings and
  describe them accurately rather than imposing a global Shift convention.
- Support reduced motion through the appropriate platform preference and/or a
  plugin setting. Essential position/state information must remain visible when
  decorative transitions are disabled. The study toggle is only a demonstration.
- Do not weaken the existing edited-sound replacement gate or change engine-switch
  semantics incidentally while polishing their presentation.

## Verification guidance

Choose checks proportional to the implementation. Important behaviors to exercise:

- Rapid drag/release/re-grab, pointer leaving the component, and interruption of a
  camera reset. No stuck hover/drag state or delayed control response.
- Keyboard focus and editing, minimum supported editor size, high-DPI rendering,
  reduced motion, closing/reopening the editor, and hiding/showing the relevant view.
- Rapid prompt changes, cancelled/superseded work, failures, and delayed results.
  Stale completions must not clear a newer pending state or overwrite a newer sound.
- Host automation and external parameter changes. Animation must not send writes
  back to the host or multiply undo/gesture entries.
- Orbit running while changing views or interacting with controls; truthful centre,
  playhead, lock, and playback feedback.
- If recovery/comparison is adopted, restore all relevant state together, including
  axes and engine/patch context where applicable. Bracket one continuous gesture
  as one undoable action and avoid recording animation frames.
- Profile under real audio load and at the target display scale; check idle CPU as
  well as active animation. A smooth standalone demo is insufficient evidence.

The study's Release build and `--checks` passed during this experiment. Checks cover
pinned comparison round trips, history restoration, camera reset preserving sound
position, model assets/geometry, and the experimental painter. A native snapshot was
also inspected. The measured software paint callback was about 3.0 ms median and
3.1 ms p95 at 700 × 500 with 1,355 visible points. This is not presented frame rate,
Retina performance, or a measurement under audio load.

## Expected handoff back

Summarize which interactions were adopted, adapted, or deferred; explain substantial
tradeoffs; list relevant validation and remaining limitations. The implementor owns
the final coherent result, not completion of every suggestion in this document.
