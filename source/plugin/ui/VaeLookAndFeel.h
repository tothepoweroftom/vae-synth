/*
  Look and feel: dark minimal: near-black paper, white ink, one signal yellow for values
  and "on" states. Hairline rules, no fills or gradients, Space Mono throughout.
*/

#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_audio_utils/juce_audio_utils.h>

namespace vae
{

class VaeLookAndFeel final : public juce::LookAndFeel_V4
{
public:
    VaeLookAndFeel();

    static inline const juce::Colour paper { 0xff151515 };
    static inline const juce::Colour ink { 0xffffffff };
    static inline const juce::Colour accent { 0xffffff07 };
    static inline const juce::Colour faint { 0xff333333 };        // knob tracks, guide lines
    static juce::Colour quiet() { return ink.withAlpha (0.45f); } // secondary text

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

class FlatKeyboard final : public juce::MidiKeyboardComponent
{
public:
    using MidiKeyboardComponent::MidiKeyboardComponent;

    void drawBlackNote (int, juce::Graphics& g, juce::Rectangle<float> area, bool isDown, bool isOver,
                        juce::Colour) override
    {
        g.setColour (isDown ? VaeLookAndFeel::accent : VaeLookAndFeel::ink);
        g.fillRect (area);
        if (isOver && ! isDown)
        {
            g.setColour (VaeLookAndFeel::accent.withAlpha (0.5f));
            g.fillRect (area);
        }
    }
};

} // namespace vae
