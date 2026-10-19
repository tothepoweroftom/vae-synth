#include "PluginEditor.h"

#include <numeric>

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
    { "voice", "Voice" },      { "osc1", "Osc 1" },       { "osc2", "Osc 2" },
    { "osc3", "Osc 3" },       { "filters", "Routing" },  { "filter1", "Filter 1" },
    { "filter2", "Filter 2" }, { "amp_env", "Amp Env" },  { "filter_env", "Filter Env" },
    { "lfo1", "LFO 1" },       { "lfo2", "LFO 2" },       { "fx_dist", "Distortion" },
    { "fx_eq", "EQ" },         { "fx_chorus", "Chorus" }, { "fx_mod", "Phaser / Flanger" },
    { "fx_delay", "Delay" },   { "fx_reverb", "Reverb" }, { "global", "Out" },
};

constexpr int kMargin = 20, kTopBarH = 76, kKeyboardH = 56, kGap = 12, kEditorW = 1240;

/** Wrapping left-to-right layout of sections; returns the height used. */
int layOutSections (juce::OwnedArray<ControlSection>& sections, juce::Rectangle<int> area, bool apply)
{
    int x = area.getX(), y = area.getY();
    for (auto* section : sections)
    {
        const int w = section->getPreferredWidth();
        if (x > area.getX() && x + w > area.getRight())
        {
            x = area.getX();
            y += ControlSection::kHeight + kGap;
        }
        if (apply)
            section->setBounds (x, y, w, ControlSection::kHeight);
        x += w + kGap;
    }
    return y + ControlSection::kHeight - area.getY();
}
} // namespace

//==============================================================================
ControlCell::ControlCell (params::ParamStore& store, int controlIndex)
    : kind (schema::control (controlIndex).kind)
    , name (schema::control (controlIndex).name)
{
    const auto& c = schema::control (controlIndex);
    juce::String tip (c.id);
    if (c.provisional)
    {
        name += "*";
        tip += " (provisional: modulation routing pending)";
    }

    switch (kind)
    {
        case schema::Kind::Continuous:
            slider.setSliderStyle (juce::Slider::RotaryVerticalDrag);
            slider.setTextBoxStyle (juce::Slider::NoTextBox, false, 0, 0);
            slider.setRotaryParameters (juce::degreesToRadians (225.0f), juce::degreesToRadians (495.0f), true);
            slider.setTooltip (tip);
            slider.onDragEnd = [this] { showValue (isMouseOver (true)); };
            slider.onValueChange = [this]
            {
                if (showing)
                    showValue (true);
            };
            slider.addMouseListener (this, false);
            addAndMakeVisible (slider);
            sliderBinding = std::make_unique<params::SliderBinding> (store, c.id, slider);
            break;
        case schema::Kind::Categorical:
            combo.setTooltip (tip);
            addAndMakeVisible (combo);
            comboBinding = std::make_unique<params::ComboBinding> (store, c.id, combo);
            break;
        case schema::Kind::Toggle:
            toggle.setTooltip (tip);
            addAndMakeVisible (toggle);
            buttonBinding = std::make_unique<params::ButtonBinding> (store, c.id, toggle);
            break;
    }

    label.setJustificationType (juce::Justification::centred);
    label.setFont (LF::labelFont (9.5f));
    label.setColour (juce::Label::textColourId, LF::quiet());
    label.setInterceptsMouseClicks (false, false);
    label.setMinimumHorizontalScale (0.7f);
    label.setText (name.toUpperCase(), juce::dontSendNotification);
    addAndMakeVisible (label);
}

void ControlCell::showValue (bool show)
{
    if (kind != schema::Kind::Continuous)
        return;
    showing = show;
    label.setColour (juce::Label::textColourId, show ? LF::green : LF::quiet());
    label.setText (show ? slider.getTextFromValue (slider.getValue()) : name.toUpperCase(), juce::dontSendNotification);
}

void ControlCell::resized()
{
    auto r = getLocalBounds();
    label.setBounds (r.removeFromBottom (16));
    switch (kind)
    {
        case schema::Kind::Continuous:  slider.setBounds (r.reduced (5)); break;
        case schema::Kind::Categorical: combo.setBounds (r.withSizeKeepingCentre (r.getWidth() - 6, 22)); break;
        case schema::Kind::Toggle:      toggle.setBounds (r.withSizeKeepingCentre (24, 24)); break;
    }
}

//==============================================================================
ControlSection::ControlSection (params::ParamStore& store, const juce::String& t, const juce::Array<int>& controls,
                                int n)
    : title (t)
    , number (n)
{
    for (auto index : controls)
        addAndMakeVisible (cells.add (new ControlCell (store, index)));
}

int ControlSection::getPreferredWidth() const { return juce::jmax (columns() * kCellW, 96); }

void ControlSection::paint (juce::Graphics& g)
{
    g.setColour (LF::ink);
    g.fillRect (0, 0, getWidth(), 1);
    g.setFont (LF::labelFont (11.0f, true));
    g.drawText (juce::String (number).paddedLeft ('0', 2) + "  " + title.toUpperCase(), 0, 8, getWidth(), 16,
                juce::Justification::centredLeft);
}

void ControlSection::resized()
{
    const int n = columns();
    for (int i = 0; i < cells.size(); ++i)
        cells[i]->setBounds ((i % n) * kCellW, kHeaderH + (i / n) * kCellH, kCellW, kCellH);
}

//==============================================================================
PluginEditor::PluginEditor (PluginProcessor& p)
    : AudioProcessorEditor (p)
    , processor (p)
    , keyboard (p.getKeyboardState(), juce::MidiKeyboardComponent::horizontalKeyboard)
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
        processor.generateRandom (
            [safe = juce::Component::SafePointer (this)] (const juce::String& note)
            {
                if (safe != nullptr)
                    safe->setStatus ("Random latent: " + note);
            });
    };
    initButton.setTooltip ("Reset every control to the schema default");
    initButton.onClick = [this]
    {
        processor.applyControls (schema::Snapshot::defaults());
        setStatus ("Init");
    };
    modelButton.setTooltip ("Load an ONNX latent decoder ([1, " + juce::String (schema::kLatentDimensions) + "] -> [1, "
                            + juce::String (schema::kFeatureCount) + "] features)");
    modelButton.onClick = [this] { chooseModel(); };
    stubButton.setTooltip ("Use the embedded stub decoder");
    stubButton.onClick = [this]
    {
        processor.loadDecoderModel ({},
                                    [safe = juce::Component::SafePointer (this)] (juce::Result r)
                                    {
                                        if (safe != nullptr)
                                            safe->setStatus (r.wasOk() ? "Stub decoder loaded" : r.getErrorMessage());
                                    });
    };
    for (auto* b : { &generateButton, &randomButton, &initButton, &modelButton, &stubButton })
        addAndMakeVisible (b);

    engineBox.setTooltip ("Synth engine backend");
    addAndMakeVisible (engineBox);
    engineBinding =
        std::make_unique<params::ComboBinding> (processor.getParams(), PluginProcessor::kEngineParamId, engineBox);

    status.setFont (LF::labelFont (10.0f));
    status.setColour (juce::Label::textColourId, LF::quiet());
    addAndMakeVisible (status);
    setStatus (processor.describeModel());

    int number = 1;
    for (const auto& info : kSections)
    {
        juce::Array<int> controls;
        for (int i = 0; i < schema::kControlCount; ++i)
            if (juce::String (schema::controls()[(size_t) i].section) == info.key)
                controls.add (i);
        if (! controls.isEmpty())
            addAndMakeVisible (
                sections.add (new ControlSection (processor.getParams(), info.title, controls, number++)));
    }
    jassert (std::accumulate (sections.begin(), sections.end(), 0,
                              [] (int n, const ControlSection* s) { return n + s->getNumCells(); })
             == schema::kControlCount); // a schema section missing from kSections hides its controls

    keyboard.setKeyWidth (16.0f);
    keyboard.setLowestVisibleKey (36);
    addAndMakeVisible (keyboard);

    const int sectionsH = layOutSections (sections, { kMargin, 0, kEditorW - 2 * kMargin, 10000 }, false);
    setSize (kEditorW, kMargin + kTopBarH + sectionsH + kGap + kKeyboardH + kMargin);
}

PluginEditor::~PluginEditor() { setLookAndFeel (nullptr); }

void PluginEditor::generate()
{
    const auto prompt = promptEditor.getText().trim();
    if (prompt.isEmpty())
        return;
    setStatus ("Generating...");
    processor.generateFromPrompt (prompt,
                                  [safe = juce::Component::SafePointer (this)] (const juce::String& note)
                                  {
                                      if (safe != nullptr)
                                          safe->setStatus (note);
                                  });
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
                              processor.loadDecoderModel (file,
                                                          [safe = juce::Component::SafePointer (this)] (juce::Result r)
                                                          {
                                                              if (safe != nullptr)
                                                                  safe->setStatus (r.wasOk()
                                                                                       ? safe->processor.describeModel()
                                                                                       : r.getErrorMessage());
                                                          });
                          });
}

void PluginEditor::setStatus (const juce::String& text) { status.setText (text, juce::dontSendNotification); }

void PluginEditor::paint (juce::Graphics& g)
{
    g.fillAll (LF::paper);
    g.setColour (LF::ink);
    g.setFont (LF::labelFont (18.0f, true));
    g.drawText ("VAE SYNTH", kMargin, kMargin - 4, 200, 24, juce::Justification::centredLeft);
    g.setFont (LF::labelFont (10.0f));
    g.setColour (LF::quiet());
    g.drawText ("SCHEMA V" + juce::String (schema::kSchemaVersion) + "  /  " + juce::String (schema::kControlCount)
                    + " CONTROLS  /  LATENT " + juce::String (schema::kLatentDimensions),
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

    layOutSections (sections, r, true);
}

} // namespace vae
