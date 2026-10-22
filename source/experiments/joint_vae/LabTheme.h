#pragma once
#include <juce_graphics/juce_graphics.h>

namespace vae::experimental::lab
{
// The Lab's colours, defined once: the brutalist tokens of docs/design/PITCHER-BRUTALIST-JUCE.md
inline const juce::Colour workspace { 0xff252525 }; // workspace and viewer
inline const juce::Colour ink { 0xfff0f0ec };       // primary text
inline const juce::Colour muted { 0xffa5aaa5 };     // secondary text
inline const juce::Colour line { 0xff4b504b };      // hairlines and separators
inline const juce::Colour green { 0xff70ff63 };     // highlight, active controls, the handle, the Replace banner
inline const juce::Colour onGreen { 0xff182018 };   // text on green
inline const juce::Colour dialog { 0xff2c2c2c };    // menus and popovers
inline const juce::Colour inset { 0xff202020 };     // inset input, the knob pad's well, the spectrogram floor

/** Category colours in jointLabels order; the ninth is Unknown. */
inline const juce::Colour categories[9] = { juce::Colour (0xff70bcff), juce::Colour (0xffffad66),
                                            juce::Colour (0xff86eeac), juce::Colour (0xffffe078),
                                            juce::Colour (0xffff8fc7), juce::Colour (0xff63e0d5),
                                            juce::Colour (0xffb6a0ff), juce::Colour (0xffff918c),
                                            juce::Colour (0xffb4b9b4) };
/** The knob pad's category map alpha (page --map-alpha). */
inline constexpr float mapAlpha = 0.36f;

/** The identity at the top of the left column, shared by the Lab and the Synth view: the green "spinning ball
    stripes" mark (a 100 x 100 design drawn in a 40 px square, centred on the 36 px row) and the lowercase wordmark in `bold`
    (25 px Inter Bold), 36 px tall from (x, y). */
inline void drawWordmark (juce::Graphics& g, float x, float y, juce::Font bold)
{
    const auto toMark = juce::AffineTransform::scale (0.4f).translated (x, y - 3); // stripes span y + 7 to y + 27
    juce::Path clip;                                                               // a ball and the rect trailing it
    clip.addEllipse (36, 24, 52, 52);
    clip.addRectangle (0, 24, 62, 52);
    juce::Path stripes;
    for (const auto& r : { juce::Rectangle<float> (44, 24, 50, 5),
                           { 26, 33, 70, 5 },
                           { 12, 42, 84, 5 },
                           { 4, 51, 92, 5 },
                           { 18, 60, 78, 5 },
                           { 36, 69, 60, 5 } })
        stripes.addRoundedRectangle (r, 2.5f);
    {
        const juce::Graphics::ScopedSaveState saved (g);
        g.reduceClipRegion (clip, toMark);
        g.setColour (green);
        g.fillPath (stripes, toMark);
    }
    g.setColour (ink);
    g.setFont (bold);
    g.drawText ("pitcher", (int) x + 47, (int) y - 2, 180, 38, juce::Justification::centredLeft);
}
} // namespace vae::experimental::lab
