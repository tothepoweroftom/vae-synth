/*
  Editor: a prompt bar (prompt -> preset, random latent, model loading, engine
  choice) above the schema controls grouped by their schema "section", and an
  on-screen keyboard. Every control binds through params::SliderBinding /
  ComboBinding, so hidden and host parameters behave alike.
*/

#pragma once

#include "PluginProcessor.h"
#include "params/Attachments.h"
#include "ui/VaeLookAndFeel.h"

namespace vae
{

/** One schema control: rotary (continuous), combo box (categorical) or
    toggle, plus its name; a knob shows its value while hovered/dragged. */
class ControlCell final : public juce::Component
{
public:
    ControlCell (params::ParamStore&, int controlIndex);
    void resized() override;
    void mouseEnter (const juce::MouseEvent&) override { showValue (true); }
    void mouseExit (const juce::MouseEvent&) override { showValue (slider.isMouseButtonDown()); }

private:
    void showValue (bool);

    schema::Kind kind;
    juce::String name;
    juce::Label label;
    juce::Slider slider;
    juce::ComboBox combo;
    juce::ToggleButton toggle;
    std::unique_ptr<params::SliderBinding> sliderBinding;
    std::unique_ptr<params::ComboBinding> comboBinding;
    std::unique_ptr<params::ButtonBinding> buttonBinding;
    bool showing = false;
};

/** A titled group of cells (one schema section), laid out in kRows rows. */
class ControlSection final : public juce::Component
{
public:
    ControlSection (params::ParamStore&, const juce::String& title, const juce::Array<int>& controls, int number);
    void paint (juce::Graphics&) override;
    void resized() override;
    int getPreferredWidth() const;

    static constexpr int kCellW = 64, kCellH = 66, kHeaderH = 28, kRows = 2;
    static constexpr int kHeight = kHeaderH + kRows * kCellH;

private:
    int columns() const { return juce::jmax (1, (cells.size() + kRows - 1) / kRows); }

    juce::String title;
    int number;
    juce::OwnedArray<ControlCell> cells;
};

class PluginEditor final : public juce::AudioProcessorEditor
{
public:
    explicit PluginEditor (PluginProcessor&);
    ~PluginEditor() override;

    void paint (juce::Graphics&) override;
    void resized() override;

private:
    void generate();
    void chooseModel();
    void setStatus (const juce::String&);

    PluginProcessor& processor;
    VaeLookAndFeel lookAndFeel;

    juce::TextEditor promptEditor;
    juce::TextButton generateButton { "Generate" }, randomButton { "Random" },
                     initButton { "Init" }, modelButton { "Model..." }, stubButton { "Stub" };
    juce::ComboBox engineBox;
    std::unique_ptr<params::ComboBinding> engineBinding;
    juce::Label status;
    std::unique_ptr<juce::FileChooser> chooser;

    juce::OwnedArray<ControlSection> sections;
    FlatKeyboard keyboard;
    juce::TooltipWindow tooltips { this };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (PluginEditor)
};

} // namespace vae
