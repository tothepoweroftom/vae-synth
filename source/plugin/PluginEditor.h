/*
  Editor: Synth and VAE Lab views over one header (engine choice, init, status) and an on-screen
  keyboard. Both show one sound: the Lab plays by writing the schema controls, which the Synth
  view's knobs (grouped by schema "section") show and edit; an edit there marks the Lab's sound
  edited. The Prompt view (prompt and model tools, a latent XY pad, a sidebar of every control as a
  linear bar) builds only with VAE_PROMPT_TAB=1. Every control binds through params::SliderBinding /
  ComboBinding, so hidden and host parameters behave alike.
*/

#pragma once

// Task 16 compiled the Prompt tab out of the editor; build with -DVAE_PROMPT_TAB=1 to bring it back.
#ifndef VAE_PROMPT_TAB
#define VAE_PROMPT_TAB 0
#endif

#include "PluginProcessor.h"
#include "experiments/joint_vae/JointVaePanel.h"
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

class PluginEditor final
    : public juce::AudioProcessorEditor
    , private params::ParamStore::Listener
    , private juce::AsyncUpdater
{
public:
    explicit PluginEditor (PluginProcessor&);
    ~PluginEditor() override;

    void paint (juce::Graphics&) override;
    void resized() override;

private:
#if VAE_PROMPT_TAB
    void generate();
    void chooseGenerator();
    void chooseModel();
    void choosePreset();
    void chooseSearchFile (bool bank);
#endif
    void setStatus (const juce::String&);
    // Any thread: a control changed. The message-thread check compares against the Lab's last write
    // (PluginProcessor::editedSinceLab).
    void paramChanged (const juce::String&, float) override { triggerAsyncUpdate(); }
    void handleAsyncUpdate() override;
    enum class View
    {
        synth,
#if VAE_PROMPT_TAB
        prompt,
#endif
        jointVae
    };
    void showView (View);

    PluginProcessor& processor;
    // Shared by every open editor: it is also the global default, which is where JUCE resolves fonts.
    juce::SharedResourcePointer<VaeLookAndFeel> lookAndFeel;

    juce::TextButton initButton { "Init" };
#if VAE_PROMPT_TAB
    juce::TextEditor promptEditor;
    juce::TextButton generatorPackageButton { "Audio model..." };
    juce::TextButton generateButton { "Generate" }, randomButton { "Random" }, modelButton { "Model..." },
        stubButton { "Stub" }, presetButton { "Preset..." };
    juce::TextButton bankButton { "SAME bank..." }, referenceButton { "Find from audio..." },
        cancelSearchButton { "Cancel search" };
    juce::ComboBox matchesBox;
    std::vector<ml::SameSearch::Match> referenceMatches;
    Spinner spinner { [this] { return processor.getReferenceSearch().isBusy(); } };
    std::unique_ptr<juce::FileChooser> chooser;
    juce::TextButton promptViewButton { "Prompt" };
    XYPad xyPad;
    schema::Latent xyLatent {};
    juce::Viewport paramViewport;
#endif
    juce::ComboBox engineBox;
    std::unique_ptr<params::ComboBinding> engineBinding;
    juce::Label status; // Synth view: Init and the Send to Surge banner

    juce::TextButton synthViewButton { "Synth" };
    juce::TextButton jointVaeViewButton { "VAE Lab" };
    View currentView = View::synth;
    experimental::JointVaePanel jointVaePanel; // its edited gate and Surge patch state live on the processor

    juce::OwnedArray<ControlSection> sections;
    FlatKeyboard keyboard;
    juce::TooltipWindow tooltips { this };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (PluginEditor)
};

} // namespace vae
