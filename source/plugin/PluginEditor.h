/*
  Editor: two toggleable views over one header (engine choice, init, status) and
  an on-screen keyboard. Synth view: the schema controls as knobs grouped by
  their schema "section". Prompt view: prompt and model tools, a latent XY pad,
  and a sidebar of every control as a linear bar. Every control binds through params::SliderBinding /
  ComboBinding, so hidden and host parameters behave alike.
*/

#pragma once

#include "PluginProcessor.h"
#include "params/Attachments.h"
#include "ui/VaeLookAndFeel.h"

namespace vae
{

/** One schema control: rotary or linear bar (continuous), combo box (categorical)
    or toggle, plus its name; a slider shows its value while hovered/dragged. */
class ControlCell final : public juce::Component
{
public:
    ControlCell (params::ParamStore&, int controlIndex, bool bar = false);
    void resized() override;
    void mouseEnter (const juce::MouseEvent&) override { showValue (true); }
    void mouseExit (const juce::MouseEvent&) override { showValue (slider.isMouseButtonDown()); }

private:
    void showValue (bool);

    schema::Kind kind;
    bool bar;
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
    ControlSection (params::ParamStore&, const juce::String& title, const juce::Array<int>& controls);
    void paint (juce::Graphics&) override;
    void resized() override;
    int getPreferredWidth() const;
    int getNumCells() const { return cells.size(); }

    static constexpr int kCellW = 64, kCellH = 66, kHeaderH = 28, kRows = 2;
    static constexpr int kHeight = kHeaderH + kRows * kCellH;

private:
    int columns() const { return juce::jmax (1, (cells.size() + kRows - 1) / kRows); }

    juce::String title;
    juce::OwnedArray<ControlCell> cells;
};

/** Every control as a labelled bar row, grouped under section headers. */
class ParamList final : public juce::Component
{
public:
    explicit ParamList (params::ParamStore&);
    void paint (juce::Graphics&) override;
    void resized() override;

    static constexpr int kRowH = 22, kHeaderH = 26;

private:
    juce::OwnedArray<ControlCell> cells;
    std::vector<std::pair<int, juce::String>> headers; // index of first cell, title
};

/** Square pad; reports the handle position as x, y in [-1, 1] while dragged. */
class XYPad final : public juce::Component
{
public:
    std::function<void (float x, float y)> onMove;
    void paint (juce::Graphics&) override;
    void mouseDown (const juce::MouseEvent& e) override { mouseDrag (e); }
    void mouseDrag (const juce::MouseEvent&) override;

private:
    juce::Point<float> pos { 0.5f, 0.5f }; // normalised, y down
};

/** Rotating arc while `busy` returns true; draws nothing otherwise. */
class Spinner final
    : public juce::Component
    , private juce::Timer
{
public:
    explicit Spinner (std::function<bool()> isBusy)
        : busy (std::move (isBusy))
    {
        startTimerHz (30);
    }
    void paint (juce::Graphics&) override;

private:
    void timerCallback() override;
    std::function<bool()> busy;
    float angle = 0.0f;
    bool spinning = false;
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
    void chooseGenerator();
    void chooseModel();
    void choosePreset();
    void chooseSearchFile (bool bank);
    void setStatus (const juce::String&);
    void showView (bool prompt);

    PluginProcessor& processor;
    // Shared by every open editor: it is also the global default, which is where JUCE resolves fonts.
    juce::SharedResourcePointer<VaeLookAndFeel> lookAndFeel;

    juce::TextEditor promptEditor;
    juce::TextButton generatorPackageButton { "Audio model..." };
    juce::TextButton generateButton { "Generate" }, randomButton { "Random" }, initButton { "Init" },
        modelButton { "Model..." }, stubButton { "Stub" }, presetButton { "Preset..." };
    juce::TextButton bankButton { "SAME bank..." }, referenceButton { "Find from audio..." },
        cancelSearchButton { "Cancel search" };
    juce::ComboBox matchesBox;
    std::vector<ml::SameSearch::Match> referenceMatches;
    juce::ComboBox engineBox;
    std::unique_ptr<params::ComboBinding> engineBinding;
    juce::Label status;
    Spinner spinner { [this] { return processor.getReferenceSearch().isBusy(); } };
    std::unique_ptr<juce::FileChooser> chooser;

    juce::TextButton synthViewButton { "Synth" }, promptViewButton { "Prompt" };
    bool promptView = false;
    XYPad xyPad;
    schema::Latent xyLatent {};
    juce::Viewport paramViewport;

    juce::OwnedArray<ControlSection> sections;
    FlatKeyboard keyboard;
    juce::TooltipWindow tooltips { this };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (PluginEditor)
};

} // namespace vae
