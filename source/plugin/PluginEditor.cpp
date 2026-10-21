#include "PluginEditor.h"

#include "BinaryData.h"

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

constexpr const char* kSurgePatchStatus = "Playing the full Surge patch from the Lab. Knobs return to Lite";
constexpr int kMargin = 20, kTopBarH = 36, kKeyboardH = 56, kGap = 12, kEditorW = 1240;

juce::Array<int> sectionControls (const char* key)
{
    juce::Array<int> controls;
    for (int i = 0; i < schema::kControlCount; ++i)
        if (juce::String (schema::controls()[(size_t) i].section) == key)
            controls.add (i);
    return controls;
}

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
ControlCell::ControlCell (params::ParamStore& store, int controlIndex, bool asBar)
    : kind (schema::control (controlIndex).kind)
    , bar (asBar)
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
            slider.setSliderStyle (bar ? juce::Slider::LinearBar : juce::Slider::RotaryVerticalDrag);
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

    label.setJustificationType (bar ? juce::Justification::centredLeft : juce::Justification::centred);
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
    label.setColour (juce::Label::textColourId, show ? LF::accent : LF::quiet());
    label.setText (show ? slider.getTextFromValue (slider.getValue()) : name.toUpperCase(), juce::dontSendNotification);
}

void ControlCell::resized()
{
    auto r = getLocalBounds();
    if (bar)
    {
        label.setBounds (r.removeFromLeft (r.getWidth() * 2 / 5));
        switch (kind)
        {
            case schema::Kind::Continuous:  slider.setBounds (r); break;
            case schema::Kind::Categorical: combo.setBounds (r); break;
            case schema::Kind::Toggle:      toggle.setBounds (r.removeFromLeft (r.getHeight())); break;
        }
        return;
    }
    label.setBounds (r.removeFromBottom (16));
    switch (kind)
    {
        case schema::Kind::Continuous:  slider.setBounds (r.reduced (5)); break;
        case schema::Kind::Categorical: combo.setBounds (r.withSizeKeepingCentre (r.getWidth() - 6, 22)); break;
        case schema::Kind::Toggle:      toggle.setBounds (r.withSizeKeepingCentre (24, 24)); break;
    }
}

//==============================================================================
ControlSection::ControlSection (params::ParamStore& store, const juce::String& t, const juce::Array<int>& controls)
    : title (t)
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
    g.drawText (title.toUpperCase(), 0, 8, getWidth(), 16, juce::Justification::centredLeft);
}

void ControlSection::resized()
{
    const int n = columns();
    for (int i = 0; i < cells.size(); ++i)
        cells[i]->setBounds ((i % n) * kCellW, kHeaderH + (i / n) * kCellH, kCellW, kCellH);
}

//==============================================================================
ParamList::ParamList (params::ParamStore& store)
{
    for (const auto& info : kSections)
    {
        const auto controls = sectionControls (info.key);
        if (controls.isEmpty())
            continue;
        headers.emplace_back (cells.size(), juce::String (info.title).toUpperCase());
        for (auto index : controls)
            addAndMakeVisible (cells.add (new ControlCell (store, index, true)));
    }
    setSize (280, (int) headers.size() * kHeaderH + cells.size() * kRowH);
}

void ParamList::paint (juce::Graphics& g)
{
    g.setFont (LF::labelFont (11.0f, true));
    for (size_t h = 0; h < headers.size(); ++h)
    {
        const int y = (int) h * kHeaderH + headers[h].first * kRowH;
        g.setColour (LF::ink);
        g.fillRect (0, y, getWidth(), 1);
        g.drawText (headers[h].second, 0, y + 6, getWidth(), 16, juce::Justification::centredLeft);
    }
}

void ParamList::resized()
{
    size_t h = 0;
    for (int i = 0; i < cells.size(); ++i)
    {
        while (h < headers.size() && headers[h].first <= i)
            ++h;
        cells[i]->setBounds (0, (int) h * kHeaderH + i * kRowH, getWidth(), kRowH);
    }
}

//==============================================================================
void XYPad::paint (juce::Graphics& g)
{
    const auto r = getLocalBounds().toFloat();
    g.setColour (LF::faint);
    g.drawHorizontalLine (getHeight() / 2, 0.0f, r.getRight());
    g.drawVerticalLine (getWidth() / 2, 0.0f, r.getBottom());
    g.setColour (LF::ink);
    g.drawRect (r, 1.0f);
    const juce::Point<float> handle (pos.x * r.getWidth(), pos.y * r.getHeight());
    g.setColour (LF::accent);
    g.fillEllipse (juce::Rectangle<float> (12.0f, 12.0f).withCentre (handle));
}

void XYPad::mouseDrag (const juce::MouseEvent& e)
{
    pos = { juce::jlimit (0.0f, 1.0f, e.position.x / (float) getWidth()),
            juce::jlimit (0.0f, 1.0f, e.position.y / (float) getHeight()) };
    repaint();
    if (onMove)
        onMove (pos.x * 2.0f - 1.0f, 1.0f - pos.y * 2.0f);
}

//==============================================================================
void Spinner::timerCallback()
{
    const bool now = busy();
    if (now)
        angle += 0.25f;
    if (now || now != spinning)
        repaint();
    spinning = now;
}

void Spinner::paint (juce::Graphics& g)
{
    if (! spinning)
        return;
    const auto r = getLocalBounds().toFloat().reduced (2.0f);
    juce::Path arc;
    arc.addCentredArc (r.getCentreX(), r.getCentreY(), r.getWidth() * 0.5f, r.getHeight() * 0.5f, angle, 0.0f,
                       juce::MathConstants<float>::pi * 1.5f, true);
    g.setColour (LF::accent);
    g.strokePath (arc, juce::PathStrokeType (2.0f));
}

//==============================================================================
PluginEditor::PluginEditor (PluginProcessor& p)
    : AudioProcessorEditor (p)
    , processor (p)
    , jointVaePanel (
          p.getKeyboardState(),
          juce::Typeface::createSystemTypefaceFor (BinaryData::InterRegular_ttf, BinaryData::InterRegular_ttfSize),
          juce::Typeface::createSystemTypefaceFor (BinaryData::InterBold_ttf, BinaryData::InterBold_ttfSize),
          juce::Typeface::createSystemTypefaceFor (BinaryData::SpaceMonoRegular_ttf,
                                                   BinaryData::SpaceMonoRegular_ttfSize))
    , keyboard (p.getKeyboardState(), juce::MidiKeyboardComponent::horizontalKeyboard)
{
    setLookAndFeel (&lookAndFeel.get());
    juce::LookAndFeel::setDefaultLookAndFeel (&lookAndFeel.get());

    initButton.setTooltip ("Reset every control to the schema default");
    initButton.onClick = [this]
    {
        processor.applyControls (schema::Snapshot::defaults());
        setStatus ("Init");
    };
    addAndMakeVisible (initButton);

#if VAE_PROMPT_TAB // Task 16 compiled the Prompt tab out; this is its UI and wiring
    promptEditor.setTextToShowWhenEmpty ("Describe a sound: warm detuned pad, plucky bass...", LF::quiet());
    promptEditor.setText (processor.getLastPrompt(), false);
    promptEditor.setFont (juce::FontOptions (15.0f));
    promptEditor.setMultiLine (true);
    promptEditor.setReturnKeyStartsNewLine (false); // Return still generates
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
    presetButton.setTooltip ("Load an editable preset from reference audio retrieval");
    presetButton.onClick = [this] { choosePreset(); };
    for (auto* b : { &generateButton, &randomButton, &modelButton, &stubButton, &presetButton })
        addAndMakeVisible (b);

    generatorPackageButton.onClick = [this] { chooseGenerator(); };
    addAndMakeVisible (generatorPackageButton);
    bankButton.onClick = [this] { chooseSearchFile (true); };
    referenceButton.onClick = [this] { chooseSearchFile (false); };
    cancelSearchButton.onClick = [this]
    {
        processor.getReferenceSearch().cancel();
        setStatus ("Search cancelled");
    };
    for (auto* button : { &bankButton, &referenceButton, &cancelSearchButton })
        addAndMakeVisible (button);
    matchesBox.setTextWhenNothingSelected ("SAME matches — select to play");
    matchesBox.onChange = [this]
    {
        const int i = matchesBox.getSelectedItemIndex();
        if (i < 0 || i >= int (referenceMatches.size()))
            return;
        processor.getParams().set (PluginProcessor::kEngineParamId, 0.0f); // Lite is always first
        processor.applyControls (referenceMatches[(size_t) i].controls);
        setStatus ("Loaded " + referenceMatches[(size_t) i].name + " — play with MIDI or the keyboard");
    };
    addAndMakeVisible (matchesBox);
#endif

    engineBox.setTooltip ("Synth engine backend");
    addAndMakeVisible (engineBox);
    engineBinding =
        std::make_unique<params::ComboBinding> (processor.getParams(), PluginProcessor::kEngineParamId, engineBox);

    status.setFont (LF::labelFont (10.0f));
    status.setColour (juce::Label::textColourId, LF::quiet());
    addAndMakeVisible (status);
#if VAE_PROMPT_TAB
    setStatus (processor.describeModel());
    addAndMakeVisible (spinner);
    processor.getReferenceSearch().preload (
        [safe = juce::Component::SafePointer (this)] (juce::Result r, auto)
        {
            if (safe != nullptr)
                safe->setStatus (r.wasOk() ? "Models ready — type a prompt and Generate" : r.getErrorMessage());
        },
        [safe = juce::Component::SafePointer (this)] (juce::String message)
        {
            if (safe != nullptr)
                safe->setStatus (message);
        });
    promptViewButton.setClickingTogglesState (false);
    promptViewButton.onClick = [this] { showView (View::prompt); };
    addAndMakeVisible (promptViewButton);
#endif

    for (auto* b : { &synthViewButton, &jointVaeViewButton })
    {
        b->setClickingTogglesState (false);
        addAndMakeVisible (b);
    }
    synthViewButton.onClick = [this] { showView (View::synth); };

    jointVaeViewButton.onClick = [this] { showView (View::jointVae); };
    addChildComponent (jointVaePanel);
    jointVaePanel.onPatch = [this] (const juce::MemoryBlock& patch)
    {
        auto r = processor.loadSurgePatch (patch);
        if (r.wasOk())
            setStatus (kSurgePatchStatus);
        return r;
    };
    jointVaePanel.onLite = [this] (const schema::Snapshot& lite)
    {
        // A decode that a Synth or host edit overtook: that edit stays, and the next Lab edit asks
        if (processor.editedSinceLab())
        {
            jointVaePanel.markEdited();
            return;
        }
        const bool patch = processor.isPlayingSurgePatch();
        processor.playLiteControls (lite);
        processor.markLabControls(); // the store's notifications arrive later; they compare equal
        if (patch)
            setStatus ({});
    };
    jointVaePanel.onReplace = [this] { processor.markLabControls(); }; // the user gave the sound to the Lab
    // A reopened editor: the gate and the Surge patch outlive the previous one (opening plays nothing, R14)
    if (processor.editedSinceLab())
        jointVaePanel.markEdited();
    if (processor.isPlayingSurgePatch())
        setStatus (kSurgePatchStatus);
    for (int i = 0; i < schema::kControlCount; ++i)
        processor.getParams().addListener (schema::control (i).id, this);
#if VAE_PROMPT_TAB

    // shortcut: the pad drives latent dims 0 and 1 with the rest at 0; add per-axis dim pickers if wanted.
    xyPad.onMove = [this] (float x, float y)
    {
        xyLatent[0] = x;
        xyLatent[1] = y;
        processor.getGenerator().generateFromLatent (xyLatent,
                                                     [safe = juce::Component::SafePointer (this)] (const auto& r)
                                                     {
                                                         if (safe == nullptr)
                                                             return;
                                                         safe->processor.applyControls (r.controls);
                                                         safe->setStatus ("Latent: " + r.note);
                                                     });
    };
    addChildComponent (xyPad);

    paramViewport.setViewedComponent (new ParamList (processor.getParams()), true);
    paramViewport.setScrollBarsShown (true, false);
    addChildComponent (paramViewport);
#endif

    for (const auto& info : kSections)
    {
        const auto controls = sectionControls (info.key);
        if (! controls.isEmpty())
            addAndMakeVisible (sections.add (new ControlSection (processor.getParams(), info.title, controls)));
    }
    jassert (std::accumulate (sections.begin(), sections.end(), 0,
                              [] (int n, const ControlSection* s) { return n + s->getNumCells(); })
             == schema::kControlCount); // a schema section missing from kSections hides its controls

    keyboard.setKeyWidth (16.0f);
    keyboard.setLowestVisibleKey (36);
    addAndMakeVisible (keyboard);

    const int sectionsH = layOutSections (sections, { kMargin, 0, kEditorW - 2 * kMargin, 10000 }, false);
    setSize (kEditorW, kMargin + kTopBarH + sectionsH + kGap + kKeyboardH + kMargin);
    showView (View::synth);
}

void PluginEditor::showView (View view)
{
    currentView = view;
    jointVaePanel.setVisible (view == View::jointVae);
    jointVaeViewButton.setToggleState (view == View::jointVae, juce::dontSendNotification);
#if VAE_PROMPT_TAB
    const bool prompt = view == View::prompt;
    for (auto* c : std::initializer_list<juce::Component*> { &promptEditor, &generateButton, &randomButton,
                                                             &modelButton, &stubButton, &presetButton,
                                                             &generatorPackageButton, &bankButton, &referenceButton,
                                                             &cancelSearchButton, &matchesBox, &xyPad, &paramViewport })
        c->setVisible (prompt);
    promptViewButton.setToggleState (prompt, juce::dontSendNotification);
#endif
    for (auto* s : sections)
        s->setVisible (view == View::synth);
    status.setVisible (view != View::jointVae); // the Lab has its own status line
#if VAE_WITH_SURGE
    keyboard.setVisible (view != View::jointVae); // the Lab has its own 3-octave keyboard (Task 17)
#endif
    // The Lab's tooltips (the spectrogram's details) in its own look; the panel outlives the window
    tooltips.setLookAndFeel (view == View::jointVae ? &jointVaePanel.getLookAndFeel() : nullptr);
    synthViewButton.setToggleState (view == View::synth, juce::dontSendNotification);
    resized();
}

void PluginEditor::handleAsyncUpdate()
{
    if (! processor.editedSinceLab())
        return; // the Lab's own write (or nothing new)
    jointVaePanel.markEdited();
    if (processor.isPlayingSurgePatch())
    {
        processor.playLiteControls (processor.getControls()); // the first knob move after Send to Surge returns to Lite
        setStatus ({});
    }
}

PluginEditor::~PluginEditor()
{
    for (int i = 0; i < schema::kControlCount; ++i)
        processor.getParams().removeListener (schema::control (i).id, this);
    cancelPendingUpdate();
    setLookAndFeel (nullptr);
    if (lookAndFeel.getReferenceCount() == 1) // last editor: the shared instance dies with us
        juce::LookAndFeel::setDefaultLookAndFeel (nullptr);
}

#if VAE_PROMPT_TAB
void PluginEditor::generate()
{
    const auto prompt = promptEditor.getText().trim();
    if (prompt.isEmpty())
        return;
    processor.setLastPrompt (prompt);
    setStatus ("Generating...");
    const auto dir =
        juce::File::getSpecialLocation (juce::File::userApplicationDataDirectory).getChildFile ("VAE Synth/generated");
    const auto created = dir.createDirectory();
    if (created.failed())
    {
        setStatus (created.getErrorMessage());
        return;
    }
    const auto output = dir.getChildFile (juce::Uuid().toString() + ".wav");
    processor.getReferenceSearch().generate (
        prompt, uint32_t (juce::Random::getSystemRandom().nextInt()), output,
        [safe = juce::Component::SafePointer (this)] (juce::Result result, std::vector<ml::SameSearch::Match> matches)
        {
            if (safe == nullptr)
                return;
            if (result.failed())
            {
                safe->setStatus (result.getErrorMessage());
                return;
            }
            safe->referenceMatches = std::move (matches);
            safe->matchesBox.clear (juce::dontSendNotification);
            for (size_t i = 0; i < safe->referenceMatches.size(); ++i)
                safe->matchesBox.addItem (juce::String (int (i) + 1) + ". " + safe->referenceMatches[i].name,
                                          int (i) + 1);
            safe->matchesBox.setSelectedId (1, juce::sendNotificationSync);
        },
        [safe = juce::Component::SafePointer (this)] (juce::String message)
        {
            if (safe != nullptr)
                safe->setStatus (message);
        });
}

void PluginEditor::chooseGenerator()
{
    chooser = std::make_unique<juce::FileChooser> ("Load native Stable Audio package", juce::File(), "*.json");
    chooser->launchAsync (juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
                          [safe = juce::Component::SafePointer (this)] (const juce::FileChooser& fc)
                          {
                              if (safe == nullptr || fc.getResult() == juce::File())
                                  return;
                              safe->setStatus ("Loading Stable Audio Small...");
                              safe->processor.getReferenceSearch().loadGenerator (
                                  fc.getResult(),
                                  [safe] (juce::Result result, auto)
                                  {
                                      if (safe != nullptr)
                                          safe->setStatus (result.wasOk()
                                                               ? "Stable Audio ready — type a prompt and Generate"
                                                               : result.getErrorMessage());
                                  });
                          });
}

void PluginEditor::chooseSearchFile (bool bank)
{
    chooser = std::make_unique<juce::FileChooser> (bank ? "Load native SAME bank" : "Find presets from reference audio",
                                                   juce::File(), bank ? "*.json" : "*.wav;*.aiff;*.flac");
    chooser->launchAsync (juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
                          [safe = juce::Component::SafePointer (this), bank] (const juce::FileChooser& fc)
                          {
                              if (safe == nullptr || fc.getResult() == juce::File())
                                  return;
                              safe->setStatus (bank ? "Loading SAME bank..." : "Encoding reference and searching...");
                              auto done = [safe, bank] (juce::Result result, std::vector<ml::SameSearch::Match> matches)
                              {
                                  if (safe == nullptr)
                                      return;
                                  if (result.failed())
                                  {
                                      safe->setStatus (result.getErrorMessage());
                                      return;
                                  }
                                  if (bank)
                                  {
                                      safe->setStatus ("SAME ready — choose Find from audio...");
                                      return;
                                  }
                                  safe->referenceMatches = std::move (matches);
                                  safe->matchesBox.clear (juce::dontSendNotification);
                                  for (size_t i = 0; i < safe->referenceMatches.size(); ++i)
                                      safe->matchesBox.addItem (juce::String (int (i) + 1) + ". "
                                                                    + safe->referenceMatches[i].name,
                                                                int (i) + 1);
                                  safe->setStatus ("Found " + juce::String (int (safe->referenceMatches.size()))
                                                   + " matches — choose one to play");
                              };
                              if (bank)
                                  safe->processor.getReferenceSearch().load (fc.getResult(), done);
                              else
                                  safe->processor.getReferenceSearch().search (fc.getResult(), done);
                          });
}

void PluginEditor::choosePreset()
{
    chooser = std::make_unique<juce::FileChooser> ("Load retrieved preset", juce::File(), "*.json");
    chooser->launchAsync (juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
                          [safe = juce::Component::SafePointer (this)] (const juce::FileChooser& fc)
                          {
                              if (safe == nullptr || fc.getResult() == juce::File())
                                  return;
                              const auto result = safe->processor.loadPresetFile (fc.getResult());
                              safe->setStatus (result.wasOk() ? "Loaded " + fc.getResult().getFileNameWithoutExtension()
                                                              : result.getErrorMessage());
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

#endif

void PluginEditor::setStatus (const juce::String& text) { status.setText (text, juce::dontSendNotification); }

void PluginEditor::paint (juce::Graphics& g)
{
    // The Lab's workspace runs up under the header: one continuous background
    g.fillAll (currentView == View::jointVae ? experimental::lab::workspace : LF::paper);
}

void PluginEditor::resized()
{
    auto r = getLocalBounds().reduced (kMargin);

    auto header = r.removeFromTop (kTopBarH).removeFromTop (24);
    synthViewButton.setBounds (header.removeFromLeft (80));
#if VAE_PROMPT_TAB
    promptViewButton.setBounds (header.removeFromLeft (80));
#endif
    jointVaeViewButton.setBounds (header.removeFromLeft (90));
    header.removeFromLeft (16);
    engineBox.setBounds (header.removeFromRight (120));
    header.removeFromRight (12);
    initButton.setBounds (header.removeFromRight (72));
    header.removeFromRight (16);
#if VAE_PROMPT_TAB
    spinner.setBounds (header.removeFromLeft (24));
    header.removeFromLeft (8);
#endif
    status.setBounds (header);

    if (currentView == View::jointVae)
    {
#if VAE_WITH_SURGE
        // Edge to edge below the header: the Lab keeps its own margins and keyboard
        jointVaePanel.setBounds (getLocalBounds().withTrimmedTop (kMargin + kTopBarH));
#else
        keyboard.setBounds (r.removeFromBottom (kKeyboardH));
        r.removeFromBottom (kGap);
        jointVaePanel.setBounds (r);
#endif
        return;
    }
    keyboard.setBounds (r.removeFromBottom (kKeyboardH));
    r.removeFromBottom (kGap);

    if (currentView == View::synth)
    {
        layOutSections (sections, r, true);
        return;
    }
#if VAE_PROMPT_TAB
    // 60:40 split: XY pad over the prompt tools, then the parameter sidebar.
    auto main = r.removeFromLeft (r.getWidth() * 3 / 5);
    main.removeFromRight (kGap);
    r.removeFromLeft (kGap);
    paramViewport.setBounds (r);
    paramViewport.getViewedComponent()->setSize (paramViewport.getMaximumVisibleWidth(),
                                                 paramViewport.getViewedComponent()->getHeight());

    auto row = [&main] (int h)
    {
        auto line = main.removeFromBottom (h);
        main.removeFromBottom (6);
        return line;
    };
    // Lays items left to right across a row, sharing its width.
    auto split = [] (juce::Rectangle<int> line, std::initializer_list<juce::Component*> items)
    {
        const int w = (line.getWidth() - 6 * ((int) items.size() - 1)) / (int) items.size();
        for (auto* c : items)
        {
            c->setBounds (line.removeFromLeft (w));
            line.removeFromLeft (6);
        }
    };
    matchesBox.setBounds (row (28));
    split (row (28), { &generatorPackageButton, &bankButton, &referenceButton, &cancelSearchButton });
    split (row (28), { &generateButton, &randomButton, &modelButton, &stubButton, &presetButton });
    promptEditor.setBounds (row (64));
    main.removeFromBottom (kGap - 6);
    xyPad.setBounds (main);
#endif
}

} // namespace vae
