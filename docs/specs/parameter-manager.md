# Parameter store — functional specification

Status: clean-room spec. This document describes **required behaviour only**.
It was written from a feature review of an existing proprietary component;
the implementation in `source/params/` must be written from this document
alone, without reference to that component's source. Names, structure and
algorithms below are this project's own.

Namespace: `vae::params`. Depends only on JUCE (`juce_audio_processors`,
`juce_gui_basics`) and the C++20 standard library.

## 1. Goals

A single owner of a plug-in's parameter state that:

1. Declares parameters from plain descriptors and builds the JUCE
   `AudioProcessorValueTreeState` (APVTS) from them.
2. Delivers parameter changes to DSP **on the audio thread**, at a point the
   processor chooses, never from the thread that set the value.
3. Supports parameters the host never sees ("hidden" parameters) that still
   persist with the plug-in state and drive UI controls the same way.
4. Saves/loads full state and loads presets that skip selected parameters.
5. Offers UI attachments (slider, button, combo box) for both host-visible
   and hidden parameters.
6. Optionally links pairs of parameters (section 7).

## 2. Parameter descriptor — `ParamSpec`

A copyable value type. Fields (all have sensible defaults):

| Field | Meaning |
|---|---|
| `id` | Unique string id; also the APVTS id. Required. |
| `name` | Host-visible display name. |
| `kind` | `Float`, `Int`, `Bool`, `Choice`. |
| `range` | `juce::NormalisableRange<float>` in raw units. For `Choice`, if left unset, derived as `[0, choices.size()-1]` with interval 1. For `Bool`, `[0,1]` interval 1. |
| `defaultValue` | Raw (not normalised) default. |
| `unit` | Suffix such as `"Hz"`, used by the default text formatter. |
| `choices` | `juce::StringArray` for `Choice`. |
| `hostVisible` | Default `true`. `false` = hidden parameter (section 5). |
| `restoreFromPreset` | Default `true`. `false` = keep current value when loading a preset (section 6). |
| `automatable`, `discrete`, `meta` | Host hints passed through to the JUCE parameter attributes. `meta` is forced `true` for parameters that take part in a link. |
| `versionHint` | Integer, default 1, passed to the JUCE `ParameterID`. |
| `category` | `juce::AudioProcessorParameter::Category`, default generic. |
| `toText` / `fromText` | Optional `std::function` formatters. When unset, defaults apply: Float → value with 2 decimals + unit; Int → integer + unit; Bool → "On"/"Off" (parsing accepts "on", "yes", "y", "true", "1", any case); Choice → the choice name (parsing matches a name case-insensitively, else parses an integer index). |
| `onAudioChange` | Optional `std::function<void(float raw)>`, invoked on the audio thread (section 4). |

Helper: `static NormalisableRange<float> frequencyRange(float lo, float hi)`
returning a logarithmic (skewed so the geometric centre sits at 0.5) range
suitable for frequencies.

## 3. Construction — `ParamStore`

```text
ParamStore store;
store.build(std::vector<ParamSpec>, juce::AudioProcessor&, juce::Identifier treeType,
            juce::UndoManager* = nullptr);
```

- `build` is called exactly once, at the end of the processor's constructor.
  Before it, every other call is a no-op returning defaults (`0.0f`,
  `nullptr`); this must not crash.
- Host-visible specs become JUCE parameters in declaration order (that order
  is the host's parameter order). Use `AudioParameterFloat/Int/Bool/Choice`
  per `kind`, wiring the formatters and attributes.
- Hidden specs are not given to the APVTS; their values live in the store
  (lock-free atomics) and are persisted in a child tree of the APVTS state
  (section 6).
- Duplicate ids are a programming error: `jassertfalse` and ignore the
  duplicate.
- After `build`, every parameter is marked "changed" so the first `dispatch()`
  delivers every value once (initial sync of DSP).

## 4. Audio-thread delivery

- `void dispatch()` — call at the top of `processBlock`. Delivers every
  parameter change that happened since the previous call, in arrival order,
  coalescing repeats of the same id within one call (deliver latest value
  once). For each delivered change, invoke the spec's `onAudioChange(raw)` if
  set.
- The path from setter thread to `dispatch()` must be **lock-free and
  allocation-free on the audio side**: use a fixed-capacity SPSC or MPSC queue
  of parameter *indices* (not strings) plus a per-parameter "dirty" flag, so
  overflow cannot lose a change (dirty flag is authoritative; the queue is an
  ordering hint). Capacity >= number of parameters.
- The DSP never reads a value that changed mid-`dispatch` inconsistently:
  `onAudioChange` receives the value current at delivery time.
- Values set from the audio thread itself (e.g. by DSP) are also delivered on
  the next `dispatch()`; they do not re-enter synchronously.

## 5. Values, setters, getters

All by string id; unknown ids `jassertfalse` and no-op / return `0.0f`.
Provide an id → index lookup and index-based overloads (`ParamIndex`) for hot
paths.

| Call | Behaviour |
|---|---|
| `float get(id)` | Raw value. Thread-safe, lock-free. |
| `float getNormalised(id)` | `range.convertTo0to1(get(id))`. |
| `void set(id, raw, Notify)` | Snap to the range's legal value, store, mark dirty, notify listeners. `Notify::host` (default) goes through the JUCE parameter's `setValueNotifyingHost`; `Notify::listenersOnly` updates internal state and listeners without a host notification. Hidden params ignore the host part. |
| `void setNormalised(id, norm, Notify)` | Same via normalised value (clamped to 0..1). |
| `void resetToDefault(id)` / `resetAllToDefaults()` | Set to `defaultValue`, notifying host. |
| `void beginGesture(id)` / `endGesture(id)` | Forward to the JUCE parameter's change-gesture calls. No-op for hidden params. Calls must be balanced (assert in debug). |
| `const ParamSpec& spec(id)` | Descriptor. Throws `std::out_of_range` for unknown ids. |
| `juce::RangedAudioParameter* hostParam(id)` | `nullptr` for hidden/unknown. |
| `juce::AudioProcessorValueTreeState* apvts()` | `nullptr` before `build`. |
| `size_t size()`, `ids()` | Enumeration in declaration order. |

### Listeners

`struct ParamStore::Listener { virtual void paramChanged(const juce::String& id, float raw) = 0; }`

- `addListener(id, Listener*)`, `removeListener(id, Listener*)`; per-id lists.
- Called synchronously **on whatever thread set the value**. Listener lists
  are guarded so add/remove may happen from the message thread while a
  callback runs on another thread.
- Host-visible and hidden parameters notify identically.

## 6. State and presets

- `void save(juce::MemoryBlock&)` — serialise APVTS state as XML (binary XML
  via `AudioProcessor::copyXmlToBinary` is fine), including a child element
  holding every hidden parameter as `(id, value)` pairs.
- `void load(const void*, int)` — restore everything: host-visible params via
  the APVTS (`replaceState`), hidden params from the child element. Unknown
  ids are ignored; ids missing from the data keep their **default** value
  (forward compatibility when parameters are added). Afterwards every
  parameter is marked dirty so the next `dispatch()` resyncs DSP.
- `void loadPreset(const void*, int)` — as `load`, except parameters whose
  `restoreFromPreset == false` keep their current value.
- Wrong tree type or malformed data: no change, return `false` (both loaders
  return `bool`).
- While loading, link rules (section 7) are suspended and recomputed after.

## 7. Links (optional pairs)

`store.addLink(LinkRule)` after `build`. A rule names `first`, `second`, an
optional `enabledBy` parameter id (0 = off, 1 = on; empty = always on), a
`Mode`, a `Space` (`Raw` or `Normalised` — which value space offsets are
measured in) and a `minGap` (for `Ordered`).

| Mode | Behaviour when either member moves while the rule is enabled |
|---|---|
| `Mirror` | The other member takes the same value. On enable, `second` snaps to `first`. |
| `Offset` | The offset `first - second` captured at enable time is preserved. |
| `Opposite` | The other member moves by the negated amount the moved member changed. |
| `Ordered` | Keep `first <= second - minGap`. If the moved member would violate it, push the other member away (clamped at its range end); if the other member is at its end, clamp the moved member. |

- The follower update must not recurse into the same rule (per-rule re-entry
  guard). Followers are set with host notification so automation records.
- The captured offset for `Offset` is persisted in state (hidden value named
  `<first>__<second>__offset`) so it survives reload.
- `float constrain(id, candidateRaw)` returns the nearest value for `id` that
  satisfies all enabled `Ordered` rules; attachments use it to limit sliders.

## 8. UI attachments (`source/params/Attachments.h`)

Message-thread only. Each works for host-visible and hidden parameters, and
mirrors JUCE's APVTS attachments in usage.

- `ValueAttachment(store, id, std::function<void(float raw)> onChange)` —
  the generic building block. Registers as listener; marshals changes to the
  message thread (`AsyncUpdater`, latest value wins). Offers
  `sendInitialUpdate()`, `setAsCompleteGesture(raw)`, `beginGesture()`,
  `setDuringGesture(raw)`, `endGesture()`.
- `SliderBinding(store, id, juce::Slider&)` — sets slider range/interval
  from the spec range (keeping the skew), text formatting from
  `toText/fromText`, double-click resets to default, drag start/end map to
  gestures, and if the id is in an `Ordered` link the slider's legal value is
  limited via `constrain`.
- `ButtonBinding(store, id, juce::Button&)` — toggle state ↔ bool value; click
  is a complete gesture.
- `ComboBinding(store, id, juce::ComboBox&)` — populates items from `choices`
  (item id = index + 1) when empty; selection ↔ choice index.
- No feedback loops: a binding never re-sends a value it is currently
  applying from the store.
- Destructors unregister; destroying a binding during a pending async update
  is safe.

## 9. Required tests (`tests/params/`)

Unit tests (JUCE `UnitTest` or Catch2) covering at least:

1. Build with mixed kinds; host sees only visible params, in order.
2. First `dispatch()` delivers every param once; later only changed ones,
   coalesced.
3. `set` from a non-audio thread is delivered only at `dispatch()`.
4. Hidden params: not in host list, persist through `save`/`load`.
5. `loadPreset` keeps `restoreFromPreset == false` params; `load` restores all.
6. Missing ids on load keep defaults; unknown ids ignored; bad data returns false.
7. Each link mode, including re-entry guard and `constrain`.
8. Text formatters round-trip for each kind.
9. Calls before `build` do not crash.

## Implementation notes

Choices made where this spec is ambiguous (implementation in `source/params/`):

- **Range "unset" detection.** `ParamSpec::range` is a plain `NormalisableRange<float>`;
  it counts as unset when it equals the default-constructed value (0..1, no interval,
  skew 1). A `Choice` with an unset range gets `[0, n-1]` step 1. `Int` ranges get an
  interval of at least 1. Defaults are snapped into the range. An empty `name` falls
  back to `id`. For host-visible `Int`/`Bool`/`Choice` params the JUCE parameter's own
  range is copied back into the stored spec (so `spec(id).range` always matches what the
  host sees; `AudioParameterInt` is linear, so a skew on a visible Int is dropped).
- **Formatter signatures.** `toText` is `String(float raw)`, `fromText` is
  `float(const String&)`. `ParamSpec::valueToText()` / `textToValue()` apply either the
  custom or the default formatter and are wired into the JUCE parameters (host text
  is truncated to the host's max length). Default Float/Int text puts a space before
  the unit (`"440.00 Hz"`, `"-7 st"`). The unit is not also passed as the JUCE label,
  so hosts don't show it twice. An unmatched Choice string that is not a number parses
  as index 0, then gets clamped.
- **`frequencyRange`** uses JUCE's skew (`setSkewForCentre(sqrt(lo*hi))`), so the
  geometric centre maps to 0.5 exactly. The curve is a power curve, not a pure log.
- **`meta` after build.** Links are added after the JUCE parameters exist, so the JUCE
  parameters are thin subclasses whose `isMetaParameter()` also returns true once the
  parameter joins a link. `discrete` reaches JUCE for Float params; Int/Bool/Choice are
  discrete already.
- **Precision.** For host-visible params, `get()` returns the exact raw value passed to
  `set()` whenever it maps to the same normalised host value (no float round-trip drift).
- **`Notify::listenersOnly` on visible params** calls `setValue()` on the JUCE parameter
  (no listeners, no host). This leaves the APVTS's cached raw value
  (`getRawParameterValue`) stale, so `save()` writes the store's own values into the
  state tree.
- **Gestures** nest. Only the outermost begin/end reaches JUCE. An unmatched
  `endGesture` asserts and is ignored.
- **Dispatch.** The bounded lock-free MPMC (used as MPSC) index queue only gets an index
  on a clean-to-dirty transition, so it can't overflow. If a push ever fails, a flag
  makes the next `dispatch()` scan every dirty flag. `dispatch()` takes a snapshot of
  what is queued, so a value set while it is delivering (for example from
  `onAudioChange`) waits for the next call. The one exception: if that value belongs to a
  parameter later in the same snapshot, it is delivered in this call, at its latest
  value, without recursion.
- **Listeners** are called with a per-parameter `CriticalSection` held. `removeListener`
  therefore waits for a callback running on another thread to finish, which is what
  makes destroying attachments safe. Calling the listeners is the only locking on the
  `set()` path. JUCE's own parameter notification also locks. Neither lock is taken by
  `dispatch()`.
- **Before `build()`** every call is a silent no-op (no assertion). Getters return 0 /
  nullptr / empty, loaders return false, `save()` produces an empty block, and
  `constrain()` returns its input. `spec()` still throws.
- **State format.** This is the APVTS tree (its `PARAM id/value` children) plus one child
  `VAE_HIDDEN_PARAMS` that holds `PARAM id/value` entries for every hidden parameter and
  every Offset link (`<first>__<second>__offset`). These link offsets are internal: they
  are not listed in `ids()` and can't be used with `get()`. `load()` writes the target
  value into the incoming tree for every host-visible id before calling `replaceState`:
  the data's value, or the default when the id is missing, or the current value for
  preset-protected params. A value missing from the data therefore ends up at its default,
  not wherever the APVTS happened to be. Values are clamped and snapped. A null pointer,
  zero size, non-JUCE-binary data or the wrong root tag makes `load()` return false.
- **Links.** `addLink` is setup-time (after `build`, before audio starts) and isn't
  thread-safe with concurrent setters. A rule with unknown ids or `first == second`
  asserts and is ignored. The enabler counts as "on" at `>= 0.5`. On enable: Mirror snaps
  second to first, Offset captures the offset, Ordered pushes second if the order is
  violated, Opposite has nothing to capture. A rule without `enabledBy` is enabled
  (and these enable actions run) inside `addLink`. Followers are clamped to their range.
  An Offset that gets clamped keeps its captured offset. Followers that already hold the
  target value (within 1e-6 relative) aren't set again.
- **Ordered and `constrain()`.** Since the other member can be pushed, `constrain()`
  bounds a member by the other member's *range end*, not its current value. `first` is
  capped at `second.max - minGap` and `second` is floored at `first.min + minGap`. This
  is the set of values that the push/clamp rules can satisfy. If no value satisfies every
  rule, the candidate is just clamped to the parameter's own range. Chains of Ordered
  rules aren't solved transitively.
- **Link propagation and locking.** For `set()` calls, links are applied after the JUCE
  parameter's own notification returns, so two linked parameters never hold each other's
  JUCE listener locks in opposite orders. One consequence: the host sees the moved
  member's value first and then any correction. Changes made by the host directly
  (automation) are propagated inside JUCE's callback.
- **After load**, rule enable states are recomputed from the loaded values. Offsets come
  from the data, or are re-captured if missing. Mirror and Ordered invariants are
  re-applied, which changes nothing for consistent data.
- **Attachments.** `ValueAttachment` applies store changes straight away when they come in
  on the message thread, and through `AsyncUpdater` (latest value wins) from any other
  thread. The slider's legal values come from a `NormalisableRange<double>` that wraps
  the spec range's own mapping functions (which keeps skew or custom lambdas) and applies
  `constrain()` in its snap function. A slider value change outside a drag is a complete
  gesture. Bindings, and any slider they configured, must not outlive the store.
- **CMake.** `JUCE_STANDALONE_APPLICATION=0` is a PRIVATE definition of `vae_params`.
  `juce_add_console_app`, `juce_add_gui_app` and `juce_add_plugin` define their own value,
  so a PUBLIC one would trigger macro-redefinition warnings in consumers.
  `JUCE_GLOBAL_MODULE_SETTINGS_INCLUDED=1`, `JUCE_WEB_BROWSER=0` and `JUCE_USE_CURL=0`
  are PUBLIC. The JUCE module targets are INTERFACE libraries that carry their sources,
  so linking them PUBLIC compiles the module code into `vae_params` and also into each
  final binary. The binary's own copies resolve the symbols first, so the archive copies
  are never linked in. The test target needs `JUCE_MODAL_LOOPS_PERMITTED=1` to pump the
  message loop.
