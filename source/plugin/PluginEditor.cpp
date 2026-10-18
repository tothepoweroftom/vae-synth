#include "PluginEditor.h"

namespace vae
{
namespace
{
using LF = VaeLookAndFeel;

struct SectionInfo
{
    const char* key;
    const char* title;
};

constexpr SectionInfo kSections[] = {
    { "osc1",   "Osc 1" },
    { "osc2",   "Osc 2" },
    { "osc3",   "Osc 3 / Noise" },
    { "filter", "Filter 1" },
    { "amp",    "Amp EG" },
    { "feg",    "Filter EG" },
    { "global", "LFO 1 / Global" },
};

constexpr int kMargin = 20, kTopBarH = 76, kKeyboardH = 64, kGap = 12;
} // namespace

//==============================================================================
ControlKnob::ControlKnob (params::ParamStore& store, int controlIndex)
    : name (schema::controls()[(size_t) controlIndex].name)
{
    slider.setSliderStyle (juce::Slider::RotaryVerticalDrag);
    slider.setTextBoxStyle (juce::Slider::NoTextBox, false, 0, 0);
    slider.setRotaryParameters (juce::degreesToRadians (225.0f), juce::degreesToRadians (495.0f), true);
    slider.setTooltip (juce::String (schema::controls()[(size_t) controlIndex].surgeParam));
    slider.onDragEnd = [this] { showValue (isMouseOver (true)); };
    slider.onValueChange = [this] { if (showing) showValue (true); };
    slider.addMouseListener (this, false);
    addAndMakeVisible (slider);

    label.setJustificationType (juce::Justification::centred);
    label.setFont (LF::labelFont (10.0f));
    label.setColour (juce::Label::textColourId, LF::quiet());
    label.setInterceptsMouseClicks (false, false);
    label.setText (name.toUpperCase(), juce::dontSendNotification);
    addAndMakeVisible (label);

    binding = std::make_unique<params::SliderBinding> (store, schema::controls()[(size_t) controlIndex].id, slider);
}

void ControlKnob::showValue (bool show)
{
    showing = show;
    label.setColour (juce::Label::textColourId, show ? LF::green : LF::quiet());
    label.setText (show ? slider.getTextFromValue (slider.getValue()) : name.toUpperCase(), juce::dontSendNotification);
}

void ControlKnob::resized()
{
    auto r = getLocalBounds();
    label.setBounds (r.removeFromBottom (16));
    slider.setBounds (r.reduced (4));
}

//==============================================================================
ControlSection::ControlSection (params::ParamStore& store, const juce::String& t, const juce::Array<int>& controls, int n)
    : title (t), number (n)
{
    for (auto index : controls)
        addAndMakeVisible (knobs.add (new ControlKnob (store, index)));
}

int ControlSection::getPreferredWidth() const
{
    const int columns = juce::jmax (1, (knobs.size() + kRows - 1) / kRows);
    return columns * kCellW;
}

void ControlSection::paint (juce::Graphics& g)
{
    g.setColour (LF::ink);
    g.fillRect (0, 0, getWidth(), 1);
    g.setFont (LF::labelFont (11.0f, true));
    g.drawText (juce::String (number).paddedLeft ('0', 2) + "  " + title.toUpperCase(),
                0, 8, getWidth(), 16, juce::Justification::centredLeft);
}

void ControlSection::resized()
{
    const int columns = juce::jmax (1, (knobs.size() + kRows - 1) / kRows);
    for (int i = 0; i < knobs.size(); ++i)
        knobs[i]->setBounds ((i % columns) * kCellW, kHeaderH + (i / columns) * kCellH, kCellW, kCellH);
}

//==============================================================================
PluginEditor::PluginEditor (PluginProcessor& p)
    : AudioProcessorEditor (p),
      processor (p),
      keyboard (p.getKeyboardState(), juce::MidiKeyboardComponent::horizontalKeyboard)
{
    setLookAndFeel (&lookAndFeel);

    promptEditor.setTextToShowWhenEmpty ("Describe a sound: warm detuned pad, plucky bass...", LF::quiet());
    promptEditor.setText (processor.getLastPrompt(), false);
    promptEditor.setFont (juce::FontOptions (15.0f));
    promptEditor.onReturnKey = [this] { generate(); };
    addAndMakeVisible (promptEditor);

    generateButton.onClick = [this] { generate(); };
    randomButton.setTooltip ("Decode a random latent vector");
    randomButton.onClick = [this]
    {
        processor.generateRandom ([safe = juce::Component::SafePointer (this)] (const juce::String& note)
                                  { if (safe != nullptr) safe->setStatus ("Random latent: " + note); });
    };
    initButton.setTooltip ("Reset every control to the schema default");
    initButton.onClick = [this] { processor.applyControls (schema::Snapshot::defaults()); setStatus ("Init"); };
    modelButton.setTooltip ("Load an ONNX latent decoder ([1, latent] -> [1, controls])");
    modelButton.onClick = [this] { chooseModel(); };
    stubButton.setTooltip ("Use the embedded stub decoder");
    stubButton.onClick = [this]
    {
        processor.loadDecoderModel ({}, [safe = juce::Component::SafePointer (this)] (juce::Result r)
        {
            if (safe != nullptr)
                safe->setStatus (r.wasOk() ? "Stub decoder loaded" : r.getErrorMessage());
        });
    };
    for (auto* b : { &generateButton, &randomButton, &initButton, &modelButton, &stubButton })
        addAndMakeVisible (b);

    engineBox.setTooltip ("Synth engine backend");
    addAndMakeVisible (engineBox);
    engineBinding = std::make_unique<params::ComboBinding> (processor.getParams(), PluginProcessor::kEngineParamId, engineBox);

    status.setFont (LF::labelFont (10.0f));
    status.setColour (juce::Label::textColourId, LF::quiet());
    addAndMakeVisible (status);
    setStatus (processor.getGenerator().describe());

    int number = 1;
    for (const auto& info : kSections)
    {
        juce::Array<int> controls;
        for (int i = 0; i < schema::kControlCount; ++i)
            if (juce::String (schema::controls()[(size_t) i].section) == info.key)
                controls.add (i);
        if (! controls.isEmpty())
            addAndMakeVisible (sections.add (new ControlSection (processor.getParams(), info.title, controls, number++)));
    }

    keyboard.setKeyWidth (16.0f);
    keyboard.setLowestVisibleKey (36);
    addAndMakeVisible (keyboard);

    // Two rows of sections; width from the widest row.
    int rowW[2] = { 0, 0 };
    for (int i = 0; i < sections.size(); ++i)
        rowW[i < 4 ? 0 : 1] += sections[i]->getPreferredWidth() + kGap;
    const int width = juce::jmax (760, juce::jmax (rowW[0], rowW[1]) - kGap + 2 * kMargin);
    const int sectionH = ControlSection::kHeaderH + ControlSection::kRows * ControlSection::kCellH;
    setSize (width, kMargin + kTopBarH + 2 * (sectionH + kGap) + kKeyboardH + kMargin);
}

PluginEditor::~PluginEditor()
{
    setLookAndFeel (nullptr);
}

void PluginEditor::generate()
{
    const auto prompt = promptEditor.getText().trim();
    if (prompt.isEmpty())
        return;
    setStatus ("Generating...");
    processor.generateFromPrompt (prompt, [safe = juce::Component::SafePointer (this)] (const juce::String& note)
                                  { if (safe != nullptr) safe->setStatus (note); });
}

void PluginEditor::chooseModel()
{
    chooser = std::make_unique<juce::FileChooser> ("Load ONNX decoder", processor.getDecoderModelFile(), "*.onnx");
    chooser->launchAsync (juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
                          [this] (const juce::FileChooser& fc)
    {
        const auto file = fc.getResult();
        if (file == juce::File())
            return;
        processor.loadDecoderModel (file, [safe = juce::Component::SafePointer (this)] (juce::Result r)
        {
            if (safe != nullptr)
                safe->setStatus (r.wasOk() ? safe->processor.getGenerator().describe() : r.getErrorMessage());
        });
    });
}

void PluginEditor::setStatus (const juce::String& text)
{
    status.setText (text, juce::dontSendNotification);
}

void PluginEditor::paint (juce::Graphics& g)
{
    g.fillAll (LF::paper);
    g.setColour (LF::ink);
    g.setFont (LF::labelFont (18.0f, true));
    g.drawText ("VAE SYNTH", kMargin, kMargin - 4, 200, 24, juce::Justification::centredLeft);
    g.setFont (LF::labelFont (10.0f));
    g.setColour (LF::quiet());
    g.drawText ("SCHEMA V" + juce::String (schema::kSchemaVersion) + "  /  "
                    + juce::String (schema::kControlCount) + " CONTROLS  /  LATENT "
                    + juce::String (schema::kLatentDimensions),
                kMargin + 140, kMargin - 4, 300, 24, juce::Justification::centredLeft);
}

void PluginEditor::resized()
{
    auto r = getLocalBounds().reduced (kMargin);

    auto top = r.removeFromTop (kTopBarH);
    auto firstLine = top.removeFromTop (24);
    engineBox.setBounds (firstLine.removeFromRight (120));

    top.removeFromTop (6);
    auto promptRow = top.removeFromTop (28);
    for (auto* b : { &stubButton, &modelButton, &initButton, &randomButton, &generateButton })
    {
        b->setBounds (promptRow.removeFromRight (b == &generateButton ? 96 : 72));
        promptRow.removeFromRight (6);
    }
    promptEditor.setBounds (promptRow);
    status.setBounds (top.removeFromTop (16));

    keyboard.setBounds (r.removeFromBottom (kKeyboardH));
    r.removeFromBottom (kGap);

    const int sectionH = ControlSection::kHeaderH + ControlSection::kRows * ControlSection::kCellH;
    for (int row = 0; row < 2; ++row)
    {
        auto line = r.removeFromTop (sectionH);
        r.removeFromTop (kGap);
        for (int i = row * 4; i < juce::jmin (sections.size(), row * 4 + 4); ++i)
        {
            sections[i]->setBounds (line.removeFromLeft (sections[i]->getPreferredWidth()));
            line.removeFromLeft (kGap);
        }
    }
}

} // namespace vae
