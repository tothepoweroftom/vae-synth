/*
  Look and feel of the Synth view's knobs, boxes and toggles: the Lab's graphite workspace, ink and green
  (LabTheme.h) for values and "on" states. Square, hairline rules, no fills or gradients, Space Mono
  throughout. The view tabs and Init use the Lab's PitcherLookAndFeel.
*/

#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_audio_utils/juce_audio_utils.h>

#include "experiments/joint_vae/LabTheme.h"

namespace vae
{

class VaeLookAndFeel final : public juce::LookAndFeel_V4
{
public:
    VaeLookAndFeel();

    // The Lab's brutalist tokens (Task 22): every colour is defined once, in LabTheme.h
    static inline const juce::Colour& paper = experimental::lab::workspace;
    static inline const juce::Colour& ink = experimental::lab::ink;
    static inline const juce::Colour& accent = experimental::lab::green;
    static inline const juce::Colour& faint = experimental::lab::line; // knob tracks, guide lines, section rules
    static juce::Colour quiet() { return experimental::lab::muted; }   // secondary text

    /** Label font (callers uppercase the text). */
    static juce::FontOptions labelFont (float size, bool bold = false);

    juce::Typeface::Ptr getTypefaceForFont (const juce::Font&) override;

    void drawRotarySlider (juce::Graphics&, int x, int y, int w, int h, float pos, float startAngle, float endAngle,
                           juce::Slider&) override;
    void drawLinearSlider (juce::Graphics&, int x, int y, int w, int h, float pos, float minPos, float maxPos,
                           juce::Slider::SliderStyle, juce::Slider&) override;
    void drawToggleButton (juce::Graphics&, juce::ToggleButton&, bool highlighted, bool down) override;
    void drawComboBox (juce::Graphics&, int w, int h, bool down, int bx, int by, int bw, int bh,
                       juce::ComboBox&) override;
    juce::Font getComboBoxFont (juce::ComboBox&) override;
    void positionComboBoxText (juce::ComboBox&, juce::Label&) override;
    void drawButtonBackground (juce::Graphics&, juce::Button&, const juce::Colour&, bool highlighted,
                               bool down) override;
    juce::Font getTextButtonFont (juce::TextButton&, int buttonHeight) override;
    void drawButtonText (juce::Graphics&, juce::TextButton&, bool highlighted, bool down) override;
    void fillTextEditorBackground (juce::Graphics&, int w, int h, juce::TextEditor&) override;
    void drawTextEditorOutline (juce::Graphics&, int w, int h, juce::TextEditor&) override;
    void drawPopupMenuBackground (juce::Graphics&, int w, int h) override;
    juce::Font getPopupMenuFont() override;

private:
    juce::Typeface::Ptr regular, bold;
};

} // namespace vae
