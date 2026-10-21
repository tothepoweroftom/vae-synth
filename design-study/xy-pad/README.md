# Precision Yellow / XY Lab

Open http://127.0.0.1:8765/design-study/xy-pad/ while serving the repository root, or open `index.html` directly. No install or build step is needed. All fonts and data are local.

This is a working visual adaptation of the supplied Pitcher XY v2 prototype, not a JUCE implementation. Source:

- `/Users/thomaspower/Downloads/new_xy/pitcher/xypad/page_v2.template.html`: markup, original layout rules and browser runtime.
- `/Users/thomaspower/Downloads/new_xy/runs/xypad/xypad_v2.html`: embedded `data` and `latent` JSON (copied unchanged).

The supplied Downloads files remain untouched. `index.html` embeds the source model and preset payloads, `app.js` contains the browser logic, and `precision.css` contains the original structural rules followed by the Precision Yellow skin. Fonts are shared with the repository assets.

## Design

- #121212 instrument background, #ffff07 primary accent, neutral supporting surfaces.
- Original two-column XY field / axis-assignment workflow, with Latent and Knob modes.
- Yellow selected axes, current-point marker, sound heat map and predicted spectrogram.
- Muted secondary category colours preserve the meaning of the preset dots and category map.
- Piano audition strip below the workspace; research notes collapsed beneath it.
- Local preference keys are namespaced separately from the original prototype.

The original decoders, measured sound directions, forest, preset navigation, axis assignments, lock structure and rough Web Audio preview remain. This browser preview is not Surge; the expected spectrogram is a model prediction. Sound has not been checked for fidelity. Small runtime changes reduce heat-map opacity, stop sustained notes on browser blur, and avoid note shortcuts when a button has focus.

## Landscape layout

At widths of 1000px and above, the pad uses a wide aspect ratio, category/spectrogram readouts sit beside the pad, the axis list scrolls in a bounded right panel, and the header/keyboard are compact. Narrower screens retain the stacked layout. This is a layout change only; data and decoder behavior are unchanged.
