# VAE interface design study

Six deliberately different visual directions for the VAE Lab, with matching synth views. Open `index.html` directly in a browser, or serve the repository root:

```sh
python3 -m http.server 8765 --bind 127.0.0.1
```

Then visit http://127.0.0.1:8765/design-study/.

No dependencies, external font requests, build step, audio or model inference. Fonts come from the repository's existing assets. The study changes no plugin code.

## Directions

- **Precision:** graphite, acid green, contour field, compact technical controls.
- **Field notes:** light graph paper, drafting blue, notebook typography, reversed sidebars.
- **Studio 76:** warm metal, wood edges, phosphor display and tactile switches.
- **Playground:** lilac, orange, organic forms and a friendly oversized headline.
- **Quiet form:** editorial serif, negative space, vermilion and restrained linework.
- **Orbital:** dark spatial canvas, luminous orbit lines and a floating point.

Click a direction, switch between Synth and VAE Lab, or enter focus view. Direction hashes are linkable. Shortlisting persists locally when browser storage is available. The pad supports pointer input and arrow keys (Shift for larger movements; Home to centre). Axes remain distinct. Category selection is a simplified noun-matching demonstration, not the complete plugin vocabulary. Export saves a clearly marked design-study JSON file, not a usable synth preset. Keyboard and knobs are visual interactions only.

The field drawings are decorative: they do not encode measured model properties or imply a global sound map. Synth controls represent the separate schema workflow, not editing of full VAE patches. Future semantic axes and full text conditioning have deliberately not been presented as available capabilities.

To narrow the study, choose a visual direction, a field treatment and a preferred density independently. For example, the Studio 76 shell could house the Precision field, or Quiet form typography could accompany Field notes controls.

## Carbon / yellow palette studies

Two additional selectable versions preserve Precision and Orbital's layouts while applying **#121212** backgrounds and **#ffff07** accents. Supporting surfaces, borders and text use neutral greys. The original directions remain available for comparison.

- `#precision-yellow`: Precision / Yellow
- `#orbital-yellow`: Orbital / Yellow

The palette carries through both the Lab and Synth views, including field art, cursor glow, controls, waveforms, keyboard highlights and meters.
