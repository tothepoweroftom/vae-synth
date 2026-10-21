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

## Same-layout style comparison

The study strip switches between Precision, Brutalist, Swiss Modern, Industrial and Editorial. All use #121212 / #ffff07 and the same minimal workspace. The alternate skins live in `styles.css`; `styles.js` selects them without reloading or resetting the sound, axes, mode or filter. Selection persists locally and can be linked using `?style=precision`, `?style=brutalist`, `?style=swiss`, `?style=industrial` or `?style=editorial`.

- Brutalist: yellow header, heavy borders, monospace labels and square controls.
- Swiss Modern: sans-serif hierarchy, flat surfaces, disciplined rules and spacing.
- Industrial: recessed display, panel seams and tactile control caps.
- Editorial: serif display type, fine rules and circular axis buttons.

Syntax checks passed; Swiss Modern and the shared controls were confirmed loaded in Safari. The subsequent screenshot/style-switch inspection timed out.

## Chat column

The desktop workspace now has three columns: Chat, XY, Axes. Chat has a scrollable session history, prompt suggestions, a multiline composer (Enter sends, Shift+Enter adds a line), and a clear-conversation action. Style switching preserves the conversation and applies matching chat controls in all five skins. At narrower widths the axes sit below the pad; on phones the columns stack.

`chat.js` implements a local category-matching demo, not an LLM or text-to-latent model. Recognised nouns select the corresponding category at the current latent point; sending from Knob mode switches to Latent mode. Descriptive words and follow-up edits are not applied, and responses state that limitation. No prompts are transmitted or persisted. Suggestions populate the composer for review before sending.

## XYZ pad variant

Use the XY / XYZ switch above the field, or open `?view=xyz` (can be combined with `&style=swiss`). `xyz.js` provides an oblique 3D projection with exact pointer inversion on the active XY plane. The yellow plane shows current depth; real presets are projected using all three assigned axes. This is a local coordinate view, not a global semantic sound map.

Drag for X/Y, Shift-drag vertically for Z, or use the existing Z slider. Arrow keys move X/Y; Page Up / Page Down move Z (Shift increases the keyboard step). A missing Z assignment is populated when entering XYZ. If Z is subsequently cleared, the view asks for an assignment and depth is inactive. XY keeps its existing heat maps and click-to-preset action; XYZ shows the volume and preset dots, without a heat map or click-to-preset snapping. The model and browser synth use the same parameter update path as the original pad.

Validation: JavaScript syntax, 48 projection/inverse round trips including bounds, and a renderer smoke check passed.

## Orbit camera variant

Select **Orbit** beside XY / XYZ, or open `?view=orbit`. Drag to orbit freely through a full rotation on both camera axes, scroll or use +/− to zoom, and use Front / Top / Side / Reset view for orientation. Focused-canvas arrow keys rotate; +/− zoom; Home resets. Touch drag rotates, with on-screen buttons for zoom. The projection is orthographic, preserving relative spatial distances without perspective distortion. Dots are sorted by camera depth.

Camera gestures redraw only the cloud and never invoke model decoding or parameter updates. Sound editing is available through direct handle manipulation (below), the axis sliders and the Z control. All three views and five visual styles remain available.

Checks: 72 camera orientations preserve 3D distance; projected coordinates stay finite, including poles. Renderer smoke check and camera-only drag isolation pass. Visual browser verification is still pending for this variant.


### Direct handle manipulation in Orbit

- Drag the yellow handle: move across a camera-facing plane.
- Shift-drag the handle vertically: move toward/away from the camera. The modifier is selected when the drag starts, avoiding jumps mid-gesture.
- Click the handle: reveal X/Y/Z arrows. Drag an arrow to constrain movement to that coordinate.
- Drag empty space: rotate the camera and dismiss the arrows, without decoding or changing the point.
- Movements stay within the displayed [0,1] volume. Nearly end-on axes have no usable screen direction; rotate the camera or use that axis's slider.
- Touch supports handle/arrow dragging and camera rotation; use the Z slider for depth without a keyboard.

Run `node design-study/xy-pad/xyz.test.cjs` from the repository root. Checks cover 72 camera orientations for screen-plane and depth movement, single-axis constraints, end-on stability, bounds, handle/arrow hit tests and isolation between camera drags and sound edits. Visual browser verification is pending.
