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
} // namespace vae::experimental::lab
