#include "JointVaePanel.h"

namespace vae::experimental
{
#if ! VAE_WITH_SURGE
JointVaePanel::JointVaePanel (juce::MidiKeyboardState&, juce::Typeface::Ptr, juce::Typeface::Ptr, juce::Typeface::Ptr)
{
    note.setText ("VAE Lab needs a Surge build", juce::dontSendNotification);
    note.setJustificationType (juce::Justification::centred);
    addAndMakeVisible (note);
}
JointVaePanel::~JointVaePanel() = default;
void JointVaePanel::loadBundle (const juce::File&) {}
void JointVaePanel::markEdited() {}
void JointVaePanel::resized() { note.setBounds (getLocalBounds()); }
void JointVaePanel::paint (juce::Graphics&) {}
#else
namespace
{
constexpr int kRandomBase = 100, kRowH = 26, kRowGap = 6, kChipRowH = 22, kAskH = 46, kGap = 8, kSpectrumH = 124,
              kTabH = 38;
const juce::String kAsking ("Replace your Synth edits? Answer above the prompt box.");
const juce::String dotSep (juce::CharPointer_UTF8 (" \xc2\xb7 "));
juce::String quoted (const juce::String& name)
{
    return juce::String (juce::CharPointer_UTF8 ("\xe2\x80\x9c")) + name
           + juce::String (juce::CharPointer_UTF8 ("\xe2\x80\x9d"));
}
juce::Font bodyFont (juce::Component& c, float size, bool bold = false)
{
    if (auto* look = dynamic_cast<PitcherLookAndFeel*> (&c.getLookAndFeel()))
        if (auto face = bold ? look->bold : look->body)
            return juce::Font (juce::FontOptions (face)).withHeight (size);
    return juce::FontOptions (size, bold ? juce::Font::bold : juce::Font::plain);
}
/** One label: value row of the spectrogram's details, the value right-aligned (the tooltip is monospaced). */
juce::String detailRow (const juce::String& key, const juce::String& value)
{
    constexpr int width = 44;
    return key + juce::String::repeatedString (" ", juce::jmax (1, width - key.length() - value.length())) + value
           + "\n";
}
juce::String presetName (const XyPadData& xy, size_t k)
{
    return (int) k < xy.presetName.size() && xy.presetName[(int) k].isNotEmpty() ? xy.presetName[(int) k]
                                                                                 : "preset " + juce::String (k);
}
} // namespace

JointVaePanel::JointVaePanel (juce::MidiKeyboardState& keys, juce::Typeface::Ptr body, juce::Typeface::Ptr bold,
                              juce::Typeface::Ptr mono)
    : look (std::move (body), std::move (bold), std::move (mono))
    , keyboard (keys)
{
    // The study's look covers buttons, sliders and the prompt; these are the colours it leaves to JUCE
    look.setColour (juce::TextEditor::highlightColourId, lab::green.withAlpha (0.25f));
    look.setColour (juce::CaretComponent::caretColourId, lab::green);
    look.setColour (juce::ScrollBar::thumbColourId, lab::line);
    setLookAndFeel (&look);
    base.setLookAndFeel (&comboLook);
    category.setLookAndFeel (&comboLook);

    for (auto* c : std::initializer_list<juce::Component*> {
             &latentButton, &knobButton,  &base,      &category,   &lockButton,     &resetButton,    &sendButton,
             &spectrum,     &orbit,       &listView,  &status,     &promptBox,      &promptGo,       &nudgeButton,
             &keyboard,     &orbitButton, &orbitRate, &orbitDepth, &orbitRateLabel, &orbitDepthLabel })
        addAndMakeVisible (c);
    addChildComponent (pad);
    addChildComponent (findButton);
    addChildComponent (experimentalTag);
    for (size_t i = 0; i < categoryRows.size(); ++i)
    {
        auto& row = categoryRows[i];
        row.index = (int) i;
        row.setButtonText (jointLabels[i]); // its accessible name
        row.setTooltip ("Highlight these presets in the cloud; the sound stays");
        row.onClick = [this, i]
        {
            // A visual filter only: the decoder's category and the sound stay (handoff, left sidebar)
            highlighted = highlighted == (int) i ? -1 : (int) i;
            for (auto& r : categoryRows)
                r.setToggleState (r.index == highlighted, juce::dontSendNotification);
            orbit.setCategoryHighlight (highlighted);
        };
        addAndMakeVisible (row);
    }
    for (auto* c : std::initializer_list<juce::Component*> { &askNote, &askReplace, &askKeep })
        addChildComponent (c);
    askNote.setFont (bodyFont (*this, 11.5f));
    askNote.setColour (juce::Label::textColourId, lab::ink);
    askNote.setMinimumHorizontalScale (1.0f); // wraps onto a second line instead of squeezing
    askNote.setJustificationType (juce::Justification::centredLeft);
    askReplace.onClick = [this] { answerReplace (true); };
    askKeep.onClick = [this] { answerReplace (false); };
    experimentalTag.setFont (look.monoFont (10.0f));
    experimentalTag.setColour (juce::Label::textColourId, lab::muted);
    experimentalTag.setTooltip ("Learned prompt search is unproven: see docs/PROMPT-EVAL.md");
    for (size_t i = 0; i < chips.size(); ++i)
    {
        addChildComponent (chips[i]);
        chips[i].onClick = [this, i] { selectCandidate ((int) i); };
        chips[i].setTooltip (
            "A preset from the prompt bundle's bank, numbered in the cloud; selecting it plays its exact "
            "settings, and the next move explores from it");
    }
    latentButton.setToggleState (true, juce::dontSendNotification);
    knobButton.setEnabled (false);
    latentButton.setTooltip ("Move through the latent space: the orbit view");
    knobButton.setTooltip ("Move two knob groups on a flat pad");
    latentButton.onClick = [this]
    {
        if (knobMode)
            edit ([this] { setKnobMode (false); }); // back on the latent pad, the kept z plays
    };
    knobButton.onClick = [this] { setKnobMode (true); };
    for (int i = 0; i < 9; ++i)
        category.addItem (jointLabels[i], i + 1);
    category.setTooltip ("Category the decoder is conditioned on");
    base.setTooltip ("Start from");
    lockButton.setTooltip ("Keep oscillator types, effects and other switches fixed while moving");
    resetButton.setTooltip ("Back to the start point");
    // Orbit LFO (Task 19): the controls under the others in the left sidebar
    orbitButton.setTooltip ("Move slowly around the current sound; switches and effects stay put");
    orbitButton.onClick = [this] { edit ([this] { orbitSettings.on ? stopOrbit (true) : startOrbit(); }); };
    for (auto* s : { &orbitRate, &orbitDepth })
    {
        s->setSliderStyle (juce::Slider::LinearHorizontal);
        s->setTextBoxStyle (juce::Slider::NoTextBox, false, 0, 0);
    }
    orbitRate.setTitle ("Orbit rate");
    orbitRate.setTooltip ("How fast the orbit goes round, 0.02 to 2 Hz");
    orbitRate.setRange (0.02, 2.0);
    orbitRate.setSkewFactorFromMidPoint (0.2); // log scale: 0.2 Hz, the geometric middle, mid-track
    orbitRate.setValue (orbitSettings.rateHz, juce::dontSendNotification);
    orbitDepth.setTitle ("Orbit depth");
    orbitDepth.setTooltip ("How far the orbit strays from the sound, in pad units");
    orbitDepth.setRange (0.0, 0.25);
    orbitDepth.setValue (orbitSettings.depth, juce::dontSendNotification);
    for (auto* l : { &orbitRateLabel, &orbitDepthLabel })
    {
        l->setFont (look.monoFont (10.0f));
        l->setColour (juce::Label::textColourId, lab::muted);
        l->setBorderSize ({});
    }
    const auto showOrbitValues = [this]
    {
        orbitRateLabel.setText ("RATE " + juce::String (orbitSettings.rateHz, 2) + " Hz", juce::dontSendNotification);
        orbitDepthLabel.setText ("DEPTH " + juce::String (orbitSettings.depth, 2), juce::dontSendNotification);
    };
    showOrbitValues();
    orbitRate.onValueChange = [this, showOrbitValues]
    {
        // The phase carries on: a new rate never jumps the handle
        const auto now = juce::Time::getMillisecondCounterHiRes();
        const auto rate = (float) orbitRate.getValue();
        orbitStart = now - (now - orbitStart) * orbitSettings.rateHz / rate;
        orbitSettings.rateHz = rate;
        showOrbitValues();
    };
    orbitDepth.onValueChange = [this, showOrbitValues]
    {
        orbitSettings.depth = (float) orbitDepth.getValue();
        showOrbitValues();
        showOrbitPath();
    };
    listView.setViewedComponent (&list, false);
    listView.setScrollBarsShown (true, false);
    listView.setScrollBarThickness (6);
    status.setFont (look.monoFont (10.0f));
    status.setColour (juce::Label::textColourId, lab::muted);
    status.setJustificationType (juce::Justification::centredLeft);
    status.setMinimumHorizontalScale (1.0f);
    orbit.setLabelFont (look.monoFont (10.0f));
    spectrum.setTitle ("Expected sound");

    // Single line, as the study: monospaced, left-aligned, vertically centred, 12 px inset
    promptBox.setFont (look.monoFont (14.4f));
    promptBox.setMultiLine (false);
    promptBox.setJustification (juce::Justification::centredLeft);
    promptBox.setBorder (juce::BorderSize<int> (1));
    promptBox.setIndents (12, 0);
    promptBox.setSelectAllWhenFocused (true);
    promptBox.onReturnKey = promptGo.onClick = [this] { submitPrompt (promptBox.getText()); };
    for (auto* b : { &findButton, &nudgeButton })
    {
        b->setRadioGroupId (1);
        b->setClickingTogglesState (true);
        b->onClick = [this] { updatePromptUi(); };
    }
    findButton.setTooltip ("Learned prompt search: suggests real presets from the prompt bundle");
    nudgeButton.setTooltip ("Word list: picks a starting point and nudges it with words like darker or longer");
    // Installed beside the Lab bundle by scripts/install-models.sh; loads after each Lab bundle
    promptFile = ml::appDataDir().getChildFile ("models/prompt-p1/prompt.json");
    if (! promptFile.existsAsFile()) // older installs
        promptFile = ml::appDataDir().getChildFile ("models/prompt-p0/prompt.json");
    updatePromptUi();

    // The study's 3-octave keyboard, C3 (MIDI 48) to C6 (MIDI 84): 22 white and 15 black keys
    keyboard.setAvailableRange (48, 84);
    keyboard.setLowestVisibleKey (48);
    keyboard.setBlackNoteLengthProportion (34.0f / 56.0f);
    keyboard.setScrollButtonsVisible (false);
    keyboard.setKeyPressBaseOctave (4);
    keyboard.setColour (juce::MidiKeyboardComponent::shadowColourId, juce::Colours::transparentBlack);
    keyboard.setTitle ("Keyboard, C3 to C6");

    orbit.picked = [this] (int i) { selectCandidate (i); };
    orbit.jumped = [this] (int k) { edit ([this, k] { jumpToPreset (k); }); };
    orbit.moved = [this] (LabOrbit::Position p)
    {
        edit (
            [this, p]
            {
                userActed = true;
                leaveBank();
                // The handle is clamped to the volume: an axis it did not move keeps its exact position
                for (size_t i = 0; i < 3; ++i)
                    if (! juce::exactlyEqual (p[i], juce::jlimit (0.0f, 1.0f, pos[i])))
                        pos[i] = p[i];
                applyPositions();
                refresh();
                decode();
            });
    };
    pad.moved = [this] (juce::Point<float> p)
    {
        edit (
            [this, p]
            {
                userActed = true;
                leaveBank();
                pos[0] = p.x;
                pos[1] = p.y;
                applyPositions (3);
                refresh();
                decode();
            });
    };
    list.assign = [this] (int slot, const juce::String& id) { assign (slot, id); };
    list.slid = [this] (const LatentAxis& slid, float p)
    {
        edit (
            [this, id = slid.id, p]
            {
                if (knobMode)
                {
                    // A knob group's slider moves the pad slot it is on
                    for (int i = 0; i < 3; ++i)
                        if (axes[(size_t) i] == id && groupOn (i) >= 0)
                        {
                            userActed = true;
                            pos[(size_t) i] = p;
                            applyPositions (1 << i);
                            refresh();
                            decode();
                        }
                    return;
                }
                auto* found = xy ? xy->find (id) : nullptr; // by id: a bundle may load while asking
                if (found == nullptr)
                    return;
                const auto& axis = *found;
                userActed = true;
                leaveBank();
                // Move along this axis and keep the pad's axes where they are (page latItem input)
                std::vector<const LatentAxis*> held { &axis };
                std::vector<float> targets { axisTarget (axis, p) };
                for (int i = 0; i < 3; ++i)
                    if (auto* a = axisOn (i); a && a != &axis)
                    {
                        held.push_back (a);
                        targets.push_back (axisTarget (*a, pos[(size_t) i]));
                    }
                z = constrain (z, held, targets);
                syncPositions();
                refresh (&axis);
                decode();
            });
    };
    const auto start = [this] (bool draw)
    {
        edit (
            [this, id = base.getSelectedId(), draw]
            {
                userActed = true;
                base.setSelectedId (id, juce::dontSendNotification);
                loadBase (id, draw);
            });
    };
    base.onChange = [start] { start (true); };
    resetButton.onClick = [start] { start (false); }; // back to the start point, the same random preset too
    category.onChange = [this]
    {
        edit (
            [this, id = category.getSelectedId()]
            {
                userActed = true;
                leaveBank();
                stopOrbit (false);
                category.setSelectedId (id, juce::dontSendNotification);
                updateDots();
                decode();
            });
    };
    lockButton.onClick = [this]
    {
        edit (
            [this]
            {
                userActed = true;
                if (lockedTail)
                    lockedTail.reset();
                else if (outputValid && current.layout.numNumeric > 0
                         && (size_t) current.layout.numNumeric <= current.codes.size())
                    lockedTail.emplace (current.codes.begin() + current.layout.numNumeric, current.codes.end());
                lockButton.setToggleState (lockedTail.has_value(), juce::dontSendNotification);
                decode();
            });
    };
    sendButton.onClick = [this]
    {
        if (! orbitSettings.on)
            return sendToSurge();
        // The next orbit step would switch back to Lite: stop, and send the centre once it is decoded
        sendAfterCentre = true;
        stopOrbit (true);
    };
    for (auto* c : std::initializer_list<juce::Component*> { &base, &category, &lockButton, &resetButton, &sendButton,
                                                             &orbit, &pad, &promptBox, &promptGo, &orbitButton })
        c->setEnabled (false);

    // Installed by scripts/install-models.sh, like the Prompt tab's models
    if (const auto installed = ml::appDataDir().getChildFile ("models/joint-vae/bundle.json"); installed.existsAsFile())
        loadBundle (installed);
    else
        setStatus ("No VAE bundle installed: run scripts/install-models.sh");
}

JointVaePanel::~JointVaePanel()
{
    base.setLookAndFeel (nullptr);
    category.setLookAndFeel (nullptr);
    setLookAndFeel (nullptr);
}

JointVaePanel::ComboLook::ComboLook (juce::Font f)
    : font (std::move (f))
{
    setColour (juce::ComboBox::textColourId, lab::ink);
    setColour (juce::PopupMenu::backgroundColourId, lab::dialog);
    setColour (juce::PopupMenu::textColourId, lab::ink);
    setColour (juce::PopupMenu::highlightedBackgroundColourId, lab::green);
    setColour (juce::PopupMenu::highlightedTextColourId, lab::onGreen);
}

void JointVaePanel::ComboLook::drawComboBox (juce::Graphics& g, int w, int h, bool, int, int, int, int,
                                             juce::ComboBox& box)
{
    g.setColour (box.hasKeyboardFocus (true) || box.isMouseOver (true) ? lab::green : lab::line);
    g.drawRect (0, 0, w, h, 1);
    const float cx = (float) w - 12.0f, cy = (float) h * 0.5f;
    juce::Path chevron;
    chevron.startNewSubPath (cx - 3.5f, cy - 1.5f);
    chevron.lineTo (cx, cy + 2.0f);
    chevron.lineTo (cx + 3.5f, cy - 1.5f);
    g.setColour (box.isEnabled() ? lab::muted : lab::line);
    g.strokePath (chevron, juce::PathStrokeType (1.2f));
}

void JointVaePanel::ComboLook::positionComboBoxText (juce::ComboBox& box, juce::Label& label)
{
    label.setBounds (8, 0, box.getWidth() - 28, box.getHeight());
    label.setBorderSize ({});
    label.setFont (font);
}

void JointVaePanel::sendToSurge()
{
    if (! outputValid || current.patch.isEmpty())
        return;
    const auto r = onPatch ? onPatch (current.patch) : juce::Result::fail ("Surge playback not connected");
    setStatus (r.wasOk() ? (bankMode ? "Playing the bank preset on Surge; select it again to return to Lite."
                                     : "Playing the decoded point on Surge; move the pad to return to Lite.")
                         : r.getErrorMessage(),
               r.failed());
}

void JointVaePanel::markEdited()
{
    stopOrbit (false); // the edit stays: the orbit never overwrites it
    if (! std::exchange (edited, true))
        setStatus ("Edited in Synth view");
}

void JointVaePanel::edit (std::function<void()> change)
{
    if (pendingEdit)
        return;    // asking: further Lab edits are ignored
    ++promptToken; // a Lab edit makes any prompt result still on its way obsolete
    if (! edited)
    {
        change();
        return;
    }
    pendingEdit = std::move (change);
    heldStatus.reset();
    showAsk (true);
}

void JointVaePanel::showAsk (bool show)
{
    for (auto* c : std::initializer_list<juce::Component*> { &askNote, &askReplace, &askKeep })
        c->setVisible (show);
    if (show)
        setStatus (kAsking);
    resized();
    repaint();
}

void JointVaePanel::answerReplace (bool replace) { resolveReplace (replace); }

void JointVaePanel::resolveReplace (bool replace)
{
    if (! pendingEdit)
        return;
    const auto change = std::exchange (pendingEdit, nullptr);
    showAsk (false);
    // A status written while asking shows now; without one, the answer's own
    const auto answered = [this, held = std::exchange (heldStatus, std::nullopt)] (const juce::String& text)
    {
        if (held && held->text.isNotEmpty())
            setStatus (held->text, held->error, held->map);
        else
            setStatus (text);
    };
    if (replace)
    {
        edited = false;
        answered ({});
        if (onReplace)
            onReplace();
        change();
        return;
    }
    // Keep editing: z, positions and codes were never touched; put the controls back on them
    answered ("Edited in Synth view");
    base.setSelectedId (shownBase, juce::dontSendNotification);
    category.setSelectedId (shownCategory, juce::dontSendNotification);
    clearPromptProgress(); // when the question was a Find's, its result is dropped
    refresh();
}

void JointVaePanel::setStatus (const juce::String& text, bool error, bool map)
{
    if (pendingEdit && text != kAsking)
    {
        heldStatus = StatusText { text, error, map }; // the question stays readable; this shows after the answer
        return;
    }
    errorShown = error;
    mapError = map;
    status.setText (text, juce::dontSendNotification);
}

const LatentAxis* JointVaePanel::axisOn (int slot) const
{
    return xy && ! knobMode && axes[(size_t) slot].isNotEmpty() ? xy->find (axes[(size_t) slot]) : nullptr;
}

int JointVaePanel::groupOn (int slot) const
{
    if (kg)
        for (size_t g = 0; g < kg->groups.size(); ++g)
            if (kg->groups[g].name == axes[(size_t) slot])
                return (int) g;
    return -1;
}

void JointVaePanel::setKnobMode (bool knob)
{
    if (! xy || ! kg || knob == knobMode)
        return;
    if (knob && (! outputValid || current.knobs.size() != (size_t) kg->names.size()))
        return;
    stopOrbit (true); // the knob pad has no orbit; it enters from the centre's output, which plays again
    if (knob)
    {
        latAxes = axes;
        knobVals = current.knobs;
        knobCodes = current.codes;
        axes = knobAxes;
    }
    else
    {
        knobAxes = axes;
        axes = latAxes;
    }
    knobMode = knob;
    latentButton.setToggleState (! knob, juce::dontSendNotification);
    knobButton.setToggleState (knob, juce::dontSendNotification);
    for (auto* c :
         std::initializer_list<juce::Component*> { &base, &category, &lockButton, &resetButton, &orbitButton })
        c->setEnabled (! knob);
    orbit.setVisible (! knob);
    pad.setVisible (knob);
    if (! knob)
        pad.setMap ({}); // the next knob pad computes its own map
    promptBox.setEnabled (! knob);
    promptGo.setEnabled (! knob);
    promptBox.setTooltip (knob ? "The prompt works on the latent pad" : "");
    promptGo.setTooltip (knob ? "The prompt works on the latent pad" : "Apply the prompt");
    if (knob)
        list.setGroups (*kg);
    else
        list.setAxes (xy->axes);
    syncPositions();
    refresh();
    updateDots();
    scheduleMap();
    updatePromptUi();
    if (! knob)
        decode(); // back to the sound of the kept z (or the bank preset's codes)
}

void JointVaePanel::loadBundle (const juce::File& file)
{
    stopOrbit (false); // a step the load supersedes never calls back
    setStatus ("Loading and checking joint VAE...");
    ++promptToken;
    worker.load (file,
                 [safe = juce::Component::SafePointer (this)] (juce::Result r, JointVaeWorker::Output out)
                 {
                     if (! safe)
                         return;
                     if (r.failed() || ! out.xypad || out.xypad->axes.size() < 2)
                     {
                         safe->setStatus (r.failed() ? r.getErrorMessage() : "Bundle has no pad axes", true);
                         return;
                     }
                     auto& self = *safe;
                     self.xy = out.xypad;
                     self.kg = out.knobGroups;
                     if (self.knobMode) // a new bundle starts on the latent pad
                     {
                         self.knobMode = false;
                         self.latentButton.setToggleState (true, juce::dontSendNotification);
                         self.knobButton.setToggleState (false, juce::dontSendNotification);
                         self.orbit.setVisible (true);
                         self.pad.setVisible (false);
                     }
                     for (size_t i = 0; i < 3; ++i)
                         self.knobAxes[i] = i < self.kg->groups.size() ? self.kg->groups[i].name : juce::String();
                     self.current = std::move (out);
                     self.outputValid = false;
                     // The prompt bundle is checked against this Lab bundle again
                     self.promptReady = false;
                     self.candidates.clear();
                     self.bankMode = false;
                     self.selected = -1;
                     self.randomPreset = -1; // another bundle's presets
                     const auto& all = self.xy->axes;
                     const auto firstDim =
                         std::find_if (all.begin(), all.end(), [] (const LatentAxis& a) { return a.kind == "dim"; });
                     self.axes = { self.xy->find ("s:brightness") ? "s:brightness" : all[0].id,
                                   self.xy->find ("s:length") ? "s:length" : all[1].id,
                                   firstDim != all.end() ? firstDim->id : juce::String() };
                     // Orbit needs a Z: the first latent dimension, else the first axis not on X or Y
                     if (self.axes[2].isEmpty() || self.axes[2] == self.axes[0] || self.axes[2] == self.axes[1])
                     {
                         self.axes[2] = {};
                         for (const auto& a : all)
                             if (a.id != self.axes[0] && a.id != self.axes[1])
                             {
                                 self.axes[2] = a.id;
                                 break;
                             }
                     }
                     self.list.setAxes (all);
                     self.base.clear (juce::dontSendNotification);
                     for (int i = 0; i < 8; ++i)
                         if (self.xy->typical.count (jointLabels[i]) != 0)
                             self.base.addItem (juce::String ("Typical ") + jointLabels[i], i + 1);
                     if (! self.xy->presetZ.empty())
                         self.base.addItem ("Random real preset", kRandomBase);
                     for (auto* c : std::initializer_list<juce::Component*> {
                              &self.base, &self.category, &self.lockButton, &self.resetButton, &self.orbit, &self.pad,
                              &self.promptBox, &self.promptGo, &self.knobButton, &self.orbitButton })
                         c->setEnabled (true);
                     self.base.setSelectedItemIndex (0, juce::dontSendNotification);
                     self.loadBase (self.base.getSelectedId());
                     self.loadPromptBundle (self.promptFile);
                 });
}

void JointVaePanel::loadBase (int id, bool draw)
{
    stopOrbit (false); // a new start (or Reset) ends the orbit
    if (! xy || id == 0)
        return;
    juce::String message;
    int label = 8;
    if (id == kRandomBase && ! xy->presetZ.empty())
    {
        if (draw || randomPreset < 0 || (size_t) randomPreset >= xy->presetZ.size())
            randomPreset = juce::Random::getSystemRandom().nextInt ((int) xy->presetZ.size());
        const auto k = (size_t) randomPreset;
        z = xy->presetZ[k];
        label = juce::jlimit (0, 8, xy->presetLabel[k]);
        message = "Started from " + quoted (presetName (*xy, k)) + " (" + jointLabels[label] + ").";
    }
    else if (auto it = xy->typical.find (jointLabels[juce::jlimit (1, 9, id) - 1]); it != xy->typical.end())
    {
        z = it->second;
        label = id - 1;
    }
    else
        return;
    category.setSelectedId (label + 1, juce::dontSendNotification);
    lockedTail.reset();
    lockButton.setToggleState (false, juce::dontSendNotification);
    bankMode = false; // a new start leaves the bank preset and drops Find's matches
    candidates.clear();
    selected = -1;
    updatePromptUi();
    setStatus (message);
    syncPositions();
    refresh();
    updateDots();
    decode();
}

void JointVaePanel::loadPromptBundle (const juce::File& file)
{
    promptFile = file;
    ++promptToken;
    ++promptEpoch;
    promptLoaded = promptLoading = false; // the model is checked against this Lab bundle when Find first needs it
    // Find is offered when a bundle is installed; a failed load on the first Find switches it off
    promptReady = xy != nullptr && file.existsAsFile();
    if (promptReady)
        findButton.setToggleState (true, juce::dontSendNotification);
    candidates.clear();
    selected = -1;
    updatePromptUi();
    updateDots();
}

void JointVaePanel::updatePromptUi()
{
    findButton.setVisible (promptReady);
    experimentalTag.setVisible (promptReady);
    if (! promptReady)
        nudgeButton.setToggleState (true, juce::dontSendNotification);
    const bool find = promptReady && findButton.getToggleState();
    promptBox.setTextToShowWhenEmpty (find ? "Describe a sound to find, like deep sub bass"
                                           : "Describe a sound: dark bass, then brighter",
                                      lab::muted);
    if (! knobMode)
        promptGo.setTooltip (find ? "Find presets for this description" : "Apply the word list");
    for (size_t i = 0; i < chips.size(); ++i)
    {
        chips[i].setVisible (i < candidates.size());
        if (i < candidates.size())
            chips[i].setButtonText (candidates[i].name);
        chips[i].setToggleState ((int) i == selected, juce::dontSendNotification); // as the pad's filled dot
    }
    resized();
}

void JointVaePanel::submitPrompt (const juce::String& text)
{
    if (promptReady && findButton.getToggleState())
        findSounds (text);
    else
        nudge (text);
}

void JointVaePanel::findSounds (const juce::String& text)
{
    if (! xy || ! promptReady || text.trim().isEmpty())
        return;
    const auto token = lastFind = ++promptToken;
    finding = true;
    if (! promptLoaded && ! promptLoading)
    {
        promptLoading = true;
        setStatus ("Loading prompt model...");
        // The worker thread runs this load before the find queued below
        prompter.load (promptFile, current.layout,
                       [safe = juce::Component::SafePointer (this), epoch = promptEpoch] (juce::Result r)
                       {
                           if (! safe || epoch != safe->promptEpoch)
                               return;
                           safe->promptLoading = false;
                           safe->promptLoaded = r.wasOk();
                           if (r.wasOk())
                               return;
                           safe->promptReady = false;
                           ++safe->promptToken; // the find queued behind it has nothing to apply
                           safe->setStatus ("Prompt search is off, using the word list: " + r.getErrorMessage(), true);
                           safe->updatePromptUi();
                       });
    }
    else if (! promptLoading)
        setStatus ("Finding sounds...");
    // The whole text goes to the learned model; no word-list nudges follow
    prompter.find (
        text,
        [safe = juce::Component::SafePointer (this), token] (juce::Result r, int, std::vector<PromptCandidate> found)
        {
            if (! safe)
                return;
            auto& self = *safe;
            if (token == self.lastFind)
                self.finding = false;
            // A newer prompt, edit, selection, reset or bundle came first, or "Replace your Synth edits?" is open
            if (token != self.promptToken || self.pendingEdit)
            {
                self.clearPromptProgress();
                return;
            }
            if (r.failed() || found.empty())
            {
                // The playing sound is kept; no automatic word-list fallback
                self.setStatus ((r.failed() ? r.getErrorMessage() : juce::String ("No presets found")) + dotSep
                                    + "the Word list can still pick a starting point",
                                true);
                return;
            }
            self.edit (
                [&self, rows = std::move (found)]
                {
                    self.candidates = rows;
                    self.applyCandidate (0);
                });
        });
}

void JointVaePanel::selectCandidate (int i)
{
    if (i >= 0 && (size_t) i < candidates.size())
        edit ([this, i] { applyCandidate (i); });
}

void JointVaePanel::applyCandidate (int i)
{
    // One state change: z, category and the row's exact codes (spec "User flow" 4)
    const auto& c = candidates[(size_t) i];
    userActed = true;
    selected = i;
    bankMode = true;
    bankCodes = c.codes;
    z = c.z;
    category.setSelectedId (juce::jlimit (0, 8, c.category) + 1, juce::dontSendNotification);
    lockedTail.reset();
    lockButton.setToggleState (false, juce::dontSendNotification);
    setStatus ("Selected a bank preset" + dotSep + "Lite preview");
    if (knobMode)
        setKnobMode (false); // decodes: the bank codes
    else
    {
        updatePromptUi();
        syncPositions();
        refresh();
        updateDots();
        decode();
    }
}

void JointVaePanel::leaveBank()
{
    if (knobMode || ! std::exchange (bankMode, false)) // the knob pad edits a copy of the bank codes instead
        return;
    if (selected >= 0 && (size_t) selected < candidates.size())
        setStatus ("Exploring from " + candidates[(size_t) selected].name);
}

void JointVaePanel::clearPromptProgress()
{
    const auto s = status.getText();
    if (! finding && (s.startsWith ("Loading prompt model") || s.startsWith ("Finding sounds")))
        setStatus (pendingEdit ? kAsking : juce::String ("Find cancelled"));
}

void JointVaePanel::nudge (const juce::String& text)
{
    if (! xy)
        return;
    const auto reply = interpretPrompt (
        text, userActed,
        *xy); // the auto-loaded base is not a chosen start (R18); the one call the text -> z search replaces
    if (! reply.start && reply.nudges.empty())
    {
        setStatus ("Word list" + dotSep + reply.message);
        return;
    }
    edit (
        [this, reply]
        {
            leaveBank();
            if (reply.start)
            {
                // Loads like a base; Lock is untouched
                z = reply.start->first;
                category.setSelectedId (reply.start->second + 1, juce::dontSendNotification);
                syncPositions();
            }
            for (const auto& [id, delta] : reply.nudges)
                if (auto* axis = xy->find (id))
                {
                    // As the list slider: move this axis, hold the pad's axes
                    std::vector<const LatentAxis*> held { axis };
                    std::vector<float> targets { axisTarget (
                        *axis, juce::jlimit (0.0f, 1.0f, axisPosition (*axis, z) + delta)) };
                    for (int i = 0; i < 3; ++i)
                        if (auto* a = axisOn (i); a && a != axis)
                        {
                            held.push_back (a);
                            targets.push_back (axisTarget (*a, pos[(size_t) i]));
                        }
                    z = constrain (z, held, targets);
                    syncPositions();
                }
            setStatus ("Word list" + dotSep + reply.message);
            userActed = true; // R14: applying a prompt is a user action
            refresh();
            updateDots();
            decode();
        });
}

void JointVaePanel::assign (int slot, const juce::String& id)
{
    // Page assign(): swap when the axis is on another slot. On the knob pad Z again clears Z; X, Y and
    // Orbit's Z never go empty.
    auto next = axes;
    if (next[(size_t) slot] == id)
    {
        if (slot != 2 || ! knobMode)
            return;
        next[2] = {};
    }
    else
    {
        for (size_t o = 0; o < 3; ++o)
            if ((int) o != slot && next[o] == id)
                next[o] = next[(size_t) slot];
        next[(size_t) slot] = id;
    }
    if (next[0].isEmpty() || next[1].isEmpty())
        return;
    axes = next;
    if (orbitSettings.on)
        z = centreZ; // the orbit goes on around the same sound, in the new axes
    syncPositions();
    refresh();
    updateDots();
    scheduleMap();
    resized();
    if (orbitSettings.on)
        decode(); // re-centres
}

void JointVaePanel::syncPositions()
{
    if (knobMode)
    {
        for (int i = 0; i < 3; ++i)
            if (const auto g = groupOn (i); g >= 0)
                if (const auto p = kg->position (g, knobVals); ! std::isnan (p))
                    pos[(size_t) i] = p;
                else
                    pos[(size_t) i] = 0.5f;
        return;
    }
    for (int i = 0; i < 3; ++i)
        if (auto* a = axisOn (i))
            pos[(size_t) i] = axisPosition (*a, z);
}

void JointVaePanel::applyPositions (int slots)
{
    if (knobMode)
    {
        // Page applyPos, then the group's knobs into the numeric codes by feature name
        for (int i = 0; i < 3; ++i)
            if (const auto g = groupOn (i); g >= 0 && (slots & (1 << i)))
            {
                kg->set (g, pos[(size_t) i], knobVals);
                for (const auto k : kg->groups[(size_t) g].knob)
                    if (const auto c = current.layout.names.indexOf (kg->names[k]);
                        c >= 0 && (size_t) c < knobCodes.size())
                        knobCodes[(size_t) c] = knobVals[(size_t) k];
            }
        return;
    }
    std::vector<const LatentAxis*> held;
    std::vector<float> targets;
    for (int i = 0; i < 3; ++i)
        if (auto* a = axisOn (i))
        {
            held.push_back (a);
            targets.push_back (axisTarget (*a, pos[(size_t) i]));
        }
    z = constrain (z, held, targets);
}

void JointVaePanel::refresh (const LatentAxis* dragged)
{
    // The handle sits at the assigned axes' positions of the current z (syncPositions)
    if (knobMode)
        pad.setDot ({ pos[0], pos[1] });
    else
        orbit.setHandle (pos);
    list.setAssigned (axes);
    if (knobMode)
        list.setSlotPositions (pos);
    else
        list.setPositions (z, dragged);
    repaint();
}

void JointVaePanel::updateDots()
{
    if (knobMode) // shortcut: no preset dots on the knob pad (the bundle holds no knob values per preset)
        return;
    const LatentAxis* on[3] = { axisOn (0), axisOn (1), axisOn (2) };
    const auto at = [&] (const JointLatent& p)
    {
        LabOrbit::Position r;
        for (size_t i = 0; i < 3; ++i)
            r[i] = on[i] ? axisPosition (*on[i], p) : pos[i];
        return r;
    };
    std::vector<LabOrbit::Position> dots;
    if (xy && on[0] && on[1])
        for (const auto& p : xy->presetZ)
            dots.push_back (at (p));
    orbit.setPresets (std::move (dots), xy ? xy->presetLabel : std::vector<int>(), category.getSelectedId() - 1);
    // Find's matches at their bank z
    std::vector<LabOrbit::Position> marks;
    if (on[0] && on[1])
        for (const auto& c : candidates)
            marks.push_back (at (c.z));
    orbit.setHighlights (std::move (marks), selected);
}

void JointVaePanel::jumpToPreset (int k)
{
    if (! xy || k < 0 || (size_t) k >= xy->presetZ.size())
        return;
    userActed = true; // a dot click is a user action (ruling R14)
    leaveBank();
    z = xy->presetZ[(size_t) k];
    const int l = juce::jlimit (0, 8, xy->presetLabel[(size_t) k]);
    category.setSelectedId (l + 1, juce::dontSendNotification);
    setStatus ("Jumped to the real preset " + quoted (presetName (*xy, (size_t) k)) + " (" + jointLabels[l] + ").");
    syncPositions();
    refresh();
    updateDots();
    decode();
}

void JointVaePanel::requestMap()
{
    if (! xy || ! knobMode) // the knob pad's category map; the worker's latent maps are unused (Orbit only)
        return;
    MapRequest request;
    request.z = z;
    request.category = category.getSelectedId() - 1;
    request.zPos = pos[2];
    request.kind = "cat";
    request.knobs = knobVals;
    request.groupX = groupOn (0);
    request.groupY = groupOn (1);
    request.lockedTail = lockedTail;
    worker.map (std::move (request),
                [safe = juce::Component::SafePointer (this)] (juce::Result r, MapResult m)
                {
                    if (! safe)
                        return;
                    if (r.failed())
                    {
                        safe->setStatus (r.getErrorMessage(), true, true);
                        return;
                    }
                    if (std::exchange (safe->mapError, false) && safe->errorShown)
                        safe->setStatus ({});
                    // Page drawPad: one pixel per cell, scaled up smoothly by the pad
                    const int n = m.n;
                    juce::Image image;
                    std::vector<std::pair<juce::String, juce::Point<float>>> labels;
                    if (m.cells.size() == (size_t) (n * n))
                    {
                        image = juce::Image (juce::Image::ARGB, n, n, true);
                        std::array<int, 9> count {};
                        std::array<float, 9> sx {}, sy {};
                        for (int k = 0; k < n * n; ++k)
                        {
                            const int c = juce::jlimit (0, 8, m.cells[(size_t) k]);
                            image.setPixelAt (k % n, k / n,
                                              lab::categories[c].withAlpha (juce::jmin (
                                                  1.0f, lab::mapAlpha * (0.45f + 0.9f * m.conf[(size_t) k]))));
                            ++count[(size_t) c];
                            sx[(size_t) c] += (float) (k % n);
                            sy[(size_t) c] += (float) (k / n);
                        }
                        for (size_t c = 0; c < 9; ++c)
                            if ((float) count[c] > (float) (n * n) * 0.04f)
                                labels.push_back ({ jointLabels[c],
                                                    { (sx[c] / (float) count[c] + 0.5f) / (float) n,
                                                      1.0f - (sy[c] / (float) count[c] + 0.5f) / (float) n } });
                    }
                    safe->pad.setMap (image, std::move (labels));
                    if (safe->onMapShown)
                        safe->onMapShown();
                });
}

void JointVaePanel::decode (bool orbitStep)
{
    if (! xy)
        return;
    if (orbitSettings.on && ! orbitStep)
        recentre();
    orbitStepPending = orbitStep;
    shownBase = base.getSelectedId();
    shownCategory = category.getSelectedId();
    scheduleMap(); // 150 ms after the last change
    LabRequest request;
    request.z = z;
    request.category = category.getSelectedId() - 1;
    request.lockedTail = lockedTail;
    if (knobMode)
        request.codes = knobCodes;
    else if (bankMode)
        request.codes = bankCodes; // no decoder: the stored preset plays exactly
    worker.lab (std::move (request),
                [safe = juce::Component::SafePointer (this)] (juce::Result r, JointVaeWorker::Output out)
                {
                    if (! safe)
                        return;
                    safe->orbitStepPending = false; // this, or a newer request that superseded the step
                    if (r.failed())
                    {
                        safe->sendAfterCentre = false;
                        safe->setStatus (r.getErrorMessage(), true);
                        if (safe->orbitSettings.on && ! safe->centreOut)
                            safe->stopOrbit (false); // no centre to hold
                        return;
                    }
                    auto& self = *safe;
                    self.current = std::move (out);
                    self.outputValid = true;
                    if (self.orbitSettings.on && self.current.patchResult.wasOk())
                    {
                        if (! self.centreOut)
                        {
                            // The centre's decode: Lock holds its structure, its switches hold every step's
                            if (! self.lockedTail && self.current.layout.numNumeric > 0
                                && (size_t) self.current.layout.numNumeric <= self.current.codes.size())
                            {
                                self.lockedTail.emplace (self.current.codes.begin() + self.current.layout.numNumeric,
                                                         self.current.codes.end());
                                self.lockedByOrbit = true;
                                self.lockButton.setToggleState (true, juce::dontSendNotification);
                            }
                            self.centreOut = self.current;
                        }
                        else // Lite's LFO destination follows route depths: flips would duck (Task 18)
                            holdDiscrete (self.current.lite, self.centreOut->lite);
                    }
                    self.showOutput();
                    const bool playable = safe->current.patchResult.wasOk();
                    safe->sendButton.setEnabled (playable);
                    if (! playable)
                        safe->setStatus (safe->current.patchResult.getErrorMessage(), true);
                    else
                    {
                        if (safe->errorShown)
                            safe->setStatus ({});
                        // Bundle load and showing the tab decode for display only (ruling R14)
                        if (safe->userActed && safe->onLite)
                            safe->onLite (safe->current.lite);
                    }
                    if (std::exchange (safe->sendAfterCentre, false) && playable)
                        safe->sendToSurge();
                });
}

void JointVaePanel::startOrbit()
{
    if (! xy || knobMode || orbitSettings.on)
        return;
    userActed = true; // R14: turning the orbit on is a user action
    orbitSettings.on = true;
    orbitStart = juce::Time::getMillisecondCounterHiRes();
    lockedByOrbit = false;
    orbitButton.setToggleState (true, juce::dontSendNotification);
    lockButton.setEnabled (false); // forced on while the orbit runs
    decode();                      // re-centres here: the centre plays, and its result sets Lock and the switches
    orbitTimer.startTimerHz (30);
}

void JointVaePanel::stopOrbit (bool play)
{
    if (! orbitSettings.on)
        return;
    orbitSettings.on = false;
    orbitTimer.stopTimer();
    orbitStepPending = false;
    if (centreOut) // the knob pad and Send read the output: the centre's, not the last step's
    {
        current = std::move (*centreOut);
        showOutput();
    }
    centreOut.reset();
    orbitButton.setToggleState (false, juce::dontSendNotification);
    lockButton.setEnabled (! knobMode);
    if (std::exchange (lockedByOrbit, false))
    {
        lockedTail.reset(); // Lock as the user left it
        lockButton.setToggleState (false, juce::dontSendNotification);
    }
    showOrbitPath();
    // Every stop returns to the centre, so a category change conditions on the preset, not on a step
    z = centreZ;
    pos = centre;
    refresh();
    if (play)
        decode();
}

void JointVaePanel::recentre()
{
    centre = pos;
    centreZ = z;
    centreOut.reset(); // ticks wait for the new centre's decode
    centreTime = juce::Time::getMillisecondCounterHiRes();
    showOrbitPath();
}

void JointVaePanel::orbitTick()
{
    if (! orbitSettings.on || ! centreOut || orbitStepPending || pendingEdit)
        return;
    leaveBank(); // the steps decode from z
    const auto now = juce::Time::getMillisecondCounterHiRes();
    auto ramped = orbitSettings;
    ramped.depth *= (float) juce::jmin (1.0, (now - centreTime) / 1000.0 / kOrbitRampSeconds);
    pos = orbitPosition (centre, ramped, (now - orbitStart) / 1000.0, axisOn (2) != nullptr);
    z = centreZ; // each step constrains from the centre, so the other directions stay at the anchor
    applyPositions();
    refresh();
    decode (true);
}

void JointVaePanel::showOrbitPath()
{
    std::vector<LabOrbit::Position> path;
    if (orbitSettings.on)
        // The X/Y circle at the centre's Z, projected; the handle's Z wobble (half the depth) rides off it
        for (int k = 0; k < 64; ++k)
            path.push_back (orbitPosition (centre, orbitSettings, k / (64.0 * orbitSettings.rateHz), false));
    orbit.setOrbit (std::move (path), centre);
}

void JointVaePanel::showOutput()
{
    // Page specCanvas: ((mel + 70) / 70)^0.8, green over the inset floor, low bands at the bottom
    const auto bands = (int) xy->bandHz.size();
    const auto frames = bands > 0 ? (int) current.mel.size() / bands : 0;
    juce::Image image = frames > 0 ? juce::Image (juce::Image::RGB, frames, bands, false) : juce::Image();
    for (int t = 0; t < frames; ++t)
        for (int b = 0; b < bands; ++b)
            image.setPixelAt (
                t, bands - 1 - b,
                lab::inset.interpolatedWith (
                    lab::green,
                    std::pow (juce::jlimit (0.0f, 1.0f, (current.mel[(size_t) (t * bands + b)] + 70) / 70), 0.8f)));
    spectrum.image = image;
    // The details popover: the same snapshot as the image (handoff, left sidebar)
    const auto sound = measures (*xy, current.mel);
    juce::String tip = "EXPECTED SOUND\n";
    tip << detailRow ("Brightness", juce::String (sound.brightness, 1) + " oct above the note")
        << detailRow ("Attack", sound.attack < 0.05f ? juce::String ("< 50 ms")
                                                     : juce::String (juce::roundToInt (sound.attack * 1000)) + " ms")
        << detailRow ("Length", juce::String (sound.length, 1) + " s")
        << detailRow ("Noise", juce::String (juce::roundToInt (sound.noise * 100)) + "%") << "\nDECODED PRESET"
        << (lockedTail ? dotSep + "structure locked" : juce::String()) << "\n";
    for (const auto& [k, v] : current.preset)
        tip << detailRow (k, v);
    tip << "\nA model prediction, not live audio.";
    spectrum.setTooltip (tip);
    spectrum.repaint();
    // CATEGORY MATCH: the knob forest's guess at this sound
    const auto top = (int) (std::max_element (current.guess.begin(), current.guess.end()) - current.guess.begin());
    for (auto& row : categoryRows)
    {
        row.match = current.guess[(size_t) row.index];
        row.closest = row.index == top;
        row.repaint();
    }
    repaint (categoryArea);
}

void JointVaePanel::CategoryRow::paintButton (juce::Graphics& g, bool over, bool)
{
    auto r = getLocalBounds().toFloat();
    if (getToggleState())
    {
        g.setColour (lab::green.withAlpha (0.07f));
        g.fillRect (r);
    }
    if (getToggleState() || hasKeyboardFocus (false))
    {
        g.setColour (lab::green);
        g.drawRect (r, 1.0f);
    }
    else if (over)
    {
        g.setColour (lab::line);
        g.drawRect (r, 1.0f);
    }
    r.reduce (6, 5);
    const auto bar = r.removeFromBottom (3);
    g.setFont (bodyFont (*this, 11.0f, closest));
    g.setColour (closest ? lab::green : lab::ink);
    g.drawText (getButtonText(), r, juce::Justification::centredLeft);
    g.drawText (juce::String (juce::roundToInt (match * 100)) + "%", r, juce::Justification::centredRight);
    g.setColour (lab::line);
    g.fillRect (bar);
    g.setColour (lab::categories[index]);
    g.fillRect (bar.withWidth (bar.getWidth() * juce::jlimit (0.0f, 1.0f, match)));
}

void JointVaePanel::paint (juce::Graphics& g)
{
    g.fillAll (lab::workspace);
    // Separators, not panels: one continuous workspace (handoff, layout contract)
    g.setColour (lab::line);
    g.drawVerticalLine (layout.left.getRight(), (float) layout.left.getY(), (float) layout.left.getBottom());
    g.drawVerticalLine (layout.right.getX(), (float) layout.right.getY(), (float) layout.right.getBottom());

    // Identity: the green swish mark and the lowercase wordmark
    const auto x = (float) layout.left.getX(), y = (float) layout.left.getY();
    juce::Path mark;
    mark.addArc (x + 2, y + 2, 29, 29, 0.45f, 5.8f, true);
    mark.startNewSubPath (x + 3, y + 23);
    mark.cubicTo (x + 10, y + 3, x + 24, y + 1, x + 33, y + 7);
    mark.startNewSubPath (x + 5, y + 29);
    mark.cubicTo (x + 16, y + 10, x + 27, y + 9, x + 34, y + 13);
    g.setColour (lab::green);
    g.strokePath (mark, juce::PathStrokeType (2.0f));
    g.setColour (lab::ink);
    g.setFont (bodyFont (*this, 25.0f, true));
    g.drawText ("pitcher", (int) x + 43, (int) y - 2, 180, 38, juce::Justification::centredLeft);

    auto heading = categoryArea;
    g.setFont (look.monoFont (15.0f).boldened());
    g.drawText ("CATEGORY MATCH", heading.removeFromTop (24), juce::Justification::centredLeft);
    if (outputValid)
    {
        const auto top = (int) (std::max_element (current.guess.begin(), current.guess.end()) - current.guess.begin());
        g.setFont (look.monoFont (10.0f));
        g.setColour (lab::green);
        g.drawText ("Closest: " + juce::String (jointLabels[top]) + dotSep
                        + juce::String (juce::roundToInt (current.guess[(size_t) top] * 100)) + "%",
                    heading, juce::Justification::centredLeft);
    }

    // The right sidebar's one tab: no Presets tab (Task 17 decision 3)
    g.setFont (look.monoFont (11.0f));
    g.setColour (lab::green);
    g.drawText (knobMode ? "KNOB GROUPS" : "AXES", axesTab.reduced (4, 0), juce::Justification::centredLeft);
    g.fillRect (axesTab.removeFromBottom (2));

    if (! askArea.isEmpty())
    {
        g.setColour (lab::green.withAlpha (0.07f));
        g.fillRect (askArea);
        g.setColour (lab::green);
        g.drawRect (askArea, 1);
    }
}

void JointVaePanel::resized()
{
    layout = PitcherLayout::calculate (getLocalBounds());

    // Left: identity, category match, the controls (Task 17 decision 2), the spectrogram at the bottom
    auto left = layout.left.withTrimmedRight (20);
    left.removeFromTop (52);
    categoryArea = left.removeFromTop (44);
    spectrum.setBounds (left.removeFromBottom (kSpectrumH));
    left.removeFromBottom (12);
    constexpr int controlsH = 8 * kRowH + 7 * kRowGap;
    const int rowH = juce::jlimit (22, 32, (left.getHeight() - controlsH - 12) / 8);
    for (auto& row : categoryRows)
        row.setBounds (left.removeFromTop (rowH));
    left.removeFromTop (12);
    const auto controlRow = [&left]
    {
        auto r = left.removeFromTop (kRowH);
        left.removeFromTop (kRowGap);
        return r;
    };
    auto modes = controlRow();
    latentButton.setBounds (modes.removeFromLeft (modes.getWidth() / 2).withTrimmedRight (3));
    knobButton.setBounds (modes.withTrimmedLeft (3));
    base.setBounds (controlRow());
    category.setBounds (controlRow());
    auto lockRow = controlRow();
    lockButton.setBounds (lockRow.removeFromLeft (lockRow.getWidth() * 3 / 5).withTrimmedRight (3));
    resetButton.setBounds (lockRow.withTrimmedLeft (3));
    sendButton.setBounds (controlRow());
    orbitButton.setBounds (controlRow());
    for (auto [slider, label] :
         { std::pair { &orbitRate, &orbitRateLabel }, std::pair { &orbitDepth, &orbitDepthLabel } })
    {
        auto r = controlRow();
        label->setBounds (r.removeFromLeft (r.getWidth() * 2 / 5));
        slider->setBounds (r);
    }

    // Right: the axis list, scrolling within its height
    auto right = layout.right.withTrimmedLeft (20);
    axesTab = right.removeFromTop (kTabH);
    right.removeFromTop (10);
    listView.setBounds (right);
    list.layOut (listView.getWidth() - listView.getScrollBarThickness()); // rows never sit under the scroll bar

    // Centre: the view, the Replace question and Find's chips, the prompt and its status, the keyboard
    keyboard.setBounds (layout.keyboard);
    keyboard.setKeyWidth ((float) layout.keyboard.getWidth() / 22.0f);
    promptBox.setBounds (layout.prompt);
    promptGo.setBounds (layout.submit);
    auto line = layout.status;
    if (findButton.isVisible())
    {
        findButton.setBounds (line.removeFromLeft (92));
        line.removeFromLeft (4);
    }
    nudgeButton.setBounds (line.removeFromLeft (80));
    line.removeFromLeft (8);
    if (experimentalTag.isVisible())
        experimentalTag.setBounds (line.removeFromLeft (92));
    status.setBounds (line);

    auto view = layout.orbit;
    if (! candidates.empty())
    {
        auto row = view.removeFromBottom (kChipRowH);
        view.removeFromBottom (kGap);
        const auto w = juce::jmin (96, row.getWidth() / (int) candidates.size());
        for (size_t i = 0; i < candidates.size() && i < chips.size(); ++i)
            chips[i].setBounds (row.removeFromLeft (w).withTrimmedRight (4));
    }
    askArea = {};
    if (pendingEdit)
    {
        askArea = view.removeFromBottom (kAskH);
        view.removeFromBottom (kGap);
        auto row = askArea.reduced (10, 5);
        askKeep.setBounds (row.removeFromRight (104).withSizeKeepingCentre (104, kRowH));
        row.removeFromRight (6);
        askReplace.setBounds (row.removeFromRight (80).withSizeKeepingCentre (80, kRowH));
        row.removeFromRight (10);
        askNote.setBounds (row);
    }
    orbit.setBounds (view);
    pad.setBounds (view);
}
#endif
} // namespace vae::experimental
