#pragma once
#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_audio_utils/juce_audio_utils.h>
#include <array>
#include <functional>

// Reusable message-thread UI. No model, processor, assets or application dependencies.
class PitcherLookAndFeel final : public juce::LookAndFeel_V4
{
public:
    explicit PitcherLookAndFeel(juce::Typeface::Ptr bodyFont = {}, juce::Typeface::Ptr boldFont = {}, juce::Typeface::Ptr monoFont = {});
    void drawLinearSlider(juce::Graphics&, int, int, int, int, float, float, float, juce::Slider::SliderStyle, juce::Slider&) override;
    juce::Font getTextButtonFont(juce::TextButton&, int) override;
    void drawButtonText(juce::Graphics&, juce::TextButton&, bool, bool) override;
    void drawButtonBackground(juce::Graphics&, juce::Button&, const juce::Colour&, bool, bool) override;
    juce::Rectangle<int> getTooltipBounds(const juce::String&, juce::Point<int>, juce::Rectangle<int>) override;
    void drawTooltip(juce::Graphics&, const juce::String&, int, int) override;
    juce::Font monoFont(float size) const;
    static juce::Font monoFontFor(juce::Component&, float size);
    juce::Typeface::Ptr body, bold, mono;
};

class Spectrum final : public juce::Component, public juce::SettableTooltipClient
{
public:
    juce::Image image;
    void paint(juce::Graphics&) override;
};

class AxisRow final : public juce::Component
{
public:
    AxisRow(juce::String name, int index, std::function<void(int,int)> assign, std::function<void(int,float)> move);
    void update(const std::array<int,3>& axes, float value);
    void lookAndFeelChanged() override;
    std::function<void(int)> onGestureStart, onGestureEnd;
    void resized() override;
    void paint(juce::Graphics&) override;
private:
    juce::Label label;
    std::array<juce::TextButton,3> buttons;
    juce::Slider slider;
    int index;
    bool assigned=false;
};

class PitcherKeyboard final : public juce::MidiKeyboardComponent
{
public:
    explicit PitcherKeyboard(juce::MidiKeyboardState& state):MidiKeyboardComponent(state,horizontalKeyboard){}
    void drawWhiteNote(int midi,juce::Graphics& g,juce::Rectangle<float> r,bool down,bool over,juce::Colour,juce::Colour) override
    {
        g.setColour(juce::Colour(down?0xff70ff63:over?0xff414741:0xff252825));g.fillRect(r);
        g.setColour(juce::Colour(0xff737773));g.drawVerticalLine(juce::roundToInt(r.getRight()-1),r.getY(),r.getBottom());
        if(midi%12==0){g.setFont(juce::Font(juce::FontOptions(9)));g.setColour(juce::Colour(down?0xff182018:0xffa5aaa5));g.drawText("C"+juce::String(midi/12-1),r.removeFromBottom(13),juce::Justification::centred);}
    }
    void drawBlackNote(int,juce::Graphics& g,juce::Rectangle<float> r,bool down,bool over,juce::Colour) override
    {g.setColour(juce::Colour(down?0xff70ff63:over?0xffc7ccc6:0xffb7bcb6));g.fillRect(r);}
};

