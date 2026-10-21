#include "JointVaePanel.h"
#include "plugin/ui/VaeLookAndFeel.h"

namespace vae::experimental
{
using LF = VaeLookAndFeel;

#if ! VAE_WITH_SURGE
JointVaePanel::JointVaePanel()
{
    note.setText ("VAE Lab needs a Surge build", juce::dontSendNotification);
    note.setJustificationType (juce::Justification::centred);
    addAndMakeVisible (note);
}
void JointVaePanel::loadBundle (const juce::File&) {}
void JointVaePanel::markEdited() {}
void JointVaePanel::resized() { note.setBounds (getLocalBounds()); }
void JointVaePanel::paint (juce::Graphics&) {}
#else
namespace
{
constexpr int kRandomBase = 100, kToolbarH = 28, kGap = 8, kReadoutH = 120, kPromptH = 76, kPromptRowH = 26,
              kChipRowH = 22, kXLabelH = 18, kZRowH = 24, kYLabelW = 18, kEyebrowH = 13;
// Background menu (page bgSel, latent mode): ids 1.. in this order
const char* const mapKinds[][2] = { { "cat", "Category guess" }, { "brightness", "Brightness" }, { "length", "Length" },
                                    { "attack", "Attack" },      { "noise", "Noise" },           { "none", "None" } };
const juce::String dotSep (juce::CharPointer_UTF8 (" \xc2\xb7 "));
const juce::String exploreFirst ("Explore generated variations to edit this preset with the pad or the word list");
juce::String quoted (const juce::String& name)
{
    return juce::String (juce::CharPointer_UTF8 ("\xe2\x80\x9c")) + name
           + juce::String (juce::CharPointer_UTF8 ("\xe2\x80\x9d"));
}
juce::String presetName (const XyPadData& xy, size_t k)
{
    return (int) k < xy.presetName.size() && xy.presetName[(int) k].isNotEmpty() ? xy.presetName[(int) k]
                                                                                 : "preset " + juce::String (k);
}
} // namespace

JointVaePanel::JointVaePanel()
{
    for (auto* c : std::initializer_list<juce::Component*> { &latentButton, &knobButton, &base, &category, &lockButton,
                                                             &resetButton, &dotsButton, &sendButton, &background, &pad,
                                                             &listView, &status, &promptBox, &promptGo, &nudgeButton })
        addAndMakeVisible (c);
    addChildComponent (zSlider);
    addChildComponent (findButton);
    addChildComponent (experimentalTag);
    experimentalTag.setFont (juce::FontOptions (11.0f));
    experimentalTag.setColour (juce::Label::textColourId, LF::quiet());
    experimentalTag.setTooltip ("Learned prompt search is unproven: see docs/P0-PROMPT-EVAL.md");
    addChildComponent (exploreButton);
    for (size_t i = 0; i < chips.size(); ++i)
    {
        addChildComponent (chips[i]);
        chips[i].onClick = [this, i] { selectCandidate ((int) i); };
        chips[i].setTooltip ("A preset from the prompt bundle's bank; selecting it plays its exact settings");
    }
    latentButton.setToggleState (true, juce::dontSendNotification);
    knobButton.setEnabled (false);
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
    lockButton.setTooltip ("Keep oscillator types, effects and other switches fixed while dragging");
    dotsButton.setTooltip ("Real presets as dots (colour = category)");
    dotsButton.setClickingTogglesState (true);
    dotsButton.setToggleState (true, juce::dontSendNotification); // page default: checked
    for (int i = 0; i < 6; ++i)
        background.addItem (mapKinds[i][1], i + 1);
    background.setSelectedId (2, juce::dontSendNotification); // page default in latent mode: brightness
    background.setTooltip ("Background: what the map shows at each point");
    background.onChange = [this] { scheduleMap(); };
    listView.setViewedComponent (&list, false);
    listView.setScrollBarsShown (true, false);
    zSlider.setRange (0.0, 1.0, 0.001);
    zSlider.setSliderStyle (juce::Slider::LinearHorizontal);
    zSlider.setTextBoxStyle (juce::Slider::NoTextBox, false, 0, 0);
    status.setFont (LF::labelFont (10.0f));
    status.setColour (juce::Label::textColourId, LF::quiet());
    status.setJustificationType (juce::Justification::topLeft);

    promptBox.setTextToShowWhenEmpty ("Describe a sound: dark bass, then brighter", LF::quiet());
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
    exploreButton.setTooltip ("Switch to the model's approximation of this preset, so the pad and the word list can "
                              "edit it. The sound may change; select the preset again to get it back.");
    exploreButton.onClick = [this] { explore(); };
    // Installed beside the Lab bundle by scripts/install-models.sh; loads after each Lab bundle
    promptFile = ml::appDataDir().getChildFile ("models/prompt-p0/prompt.json");
    updatePromptUi();

    pad.moved = [this] (juce::Point<float> p)
    {
        if (bankMode && ! knobMode)
        {
            setStatus (exploreFirst);
            refresh(); // the dot stays on the bank preset
            return;
        }
        edit (
            [this, p]
            {
                userActed = true;
                pos[0] = p.x;
                pos[1] = p.y;
                applyPositions (3);
                refresh();
                decode();
            });
    };
    zSlider.onValueChange = [this]
    {
        if (bankMode && ! knobMode)
        {
            refresh();
            return;
        }
        edit (
            [this, v = (float) zSlider.getValue()]
            {
                userActed = true;
                pos[2] = v;
                applyPositions (4);
                refresh();
                decode();
            });
    };
    pad.tapped = [this] (juce::Point<float> p)
    {
        // A tap's mouse-down already moved (and may be asking): the dot jump follows that move
        if (bankMode && ! knobMode)
            return;
        if (pendingEdit)
            pendingEdit = [this, p, move = std::move (pendingEdit)]
            {
                move();
                jumpToDot (p);
            };
        else
            edit ([this, p] { jumpToDot (p); });
    };
    list.assign = [this] (int slot, const juce::String& id) { assign (slot, id); };
    list.slid = [this] (const LatentAxis& slid, float p)
    {
        if (bankMode)
        {
            setStatus (exploreFirst);
            refresh();
            return;
        }
        edit (
            [this, id = slid.id, p]
            {
                auto* found = xy ? xy->find (id) : nullptr; // by id: a bundle may load while asking
                if (found == nullptr)
                    return;
                const auto& axis = *found;
                userActed = true;
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
    base.onChange = resetButton.onClick = [this]
    {
        edit (
            [this, id = base.getSelectedId()]
            {
                userActed = true;
                base.setSelectedId (id, juce::dontSendNotification);
                loadBase (id);
            });
    };
    category.onChange = [this]
    {
        edit (
            [this, id = category.getSelectedId()]
            {
                userActed = true;
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
    dotsButton.onClick = [this] { updateDots(); };
    sendButton.onClick = [this]
    {
        if (! outputValid || current.patch.isEmpty())
            return;
        const auto r = onPatch ? onPatch (current.patch) : juce::Result::fail ("Surge playback not connected");
        setStatus (r.wasOk() ? (bankMode ? "Playing the bank preset on Surge; select it again to return to Lite."
                                         : "Playing the decoded point on Surge; move the pad to return to Lite.")
                             : r.getErrorMessage(),
                   r.failed());
    };
    for (auto* c : std::initializer_list<juce::Component*> { &base, &category, &lockButton, &resetButton, &dotsButton,
                                                             &sendButton, &background, &pad, &zSlider })
        c->setEnabled (false);
    promptBox.setEnabled (false);
    promptGo.setEnabled (false);

    // Installed by scripts/install-models.sh, like the Prompt tab's models
    if (const auto installed = ml::appDataDir().getChildFile ("models/joint-vae/bundle.json"); installed.existsAsFile())
        loadBundle (installed);
    else
        setStatus ("No VAE bundle installed: run scripts/install-models.sh");
}

void JointVaePanel::markEdited()
{
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
    // Async, so the editor never blocks the host
    replaceBox = juce::AlertWindow::showScopedAsync (
        juce::MessageBoxOptions()
            .withIconType (juce::MessageBoxIconType::QuestionIcon)
            .withTitle ("Replace your Synth edits?")
            .withMessage ("The Lab's sound replaces the changes made in the Synth view.")
            .withButton ("Replace")
            .withButton ("Keep editing")
            .withAssociatedComponent (this),
        [safe = juce::Component::SafePointer (this)] (int result)
        {
            if (safe)
                safe->resolveReplace (result == 1);
        });
}

void JointVaePanel::answerReplace (bool replace)
{
    replaceBox.close();
    resolveReplace (replace);
}

void JointVaePanel::resolveReplace (bool replace)
{
    if (! pendingEdit)
        return;
    const auto change = std::exchange (pendingEdit, nullptr);
    if (replace)
    {
        edited = false;
        setStatus ({});
        change();
        return;
    }
    // Keep editing: z, positions and codes were never touched; put the controls back on them
    base.setSelectedId (shownBase, juce::dontSendNotification);
    category.setSelectedId (shownCategory, juce::dontSendNotification);
    refresh();
}

void JointVaePanel::setStatus (const juce::String& text, bool error)
{
    errorShown = error;
    mapError = false;
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

void JointVaePanel::fillBackground()
{
    background.clear (juce::dontSendNotification);
    if (knobMode)
    {
        background.addItem ("Category guess", 1);
        background.addItem ("None", 2);
        background.setSelectedId (1, juce::dontSendNotification);
        return;
    }
    for (int i = 0; i < 6; ++i)
        background.addItem (mapKinds[i][1], i + 1);
    background.setSelectedId (latBackground, juce::dontSendNotification);
}

void JointVaePanel::setKnobMode (bool knob)
{
    if (! xy || ! kg || knob == knobMode)
        return;
    if (knob && (! outputValid || current.knobs.size() != (size_t) kg->names.size()))
        return;
    if (knob)
    {
        latBackground = background.getSelectedId();
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
    for (auto* c : std::initializer_list<juce::Component*> { &base, &category, &lockButton, &resetButton, &dotsButton })
        c->setEnabled (! knob);
    fillBackground();
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
                         self.fillBackground();
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
                     const auto& all = self.xy->axes;
                     const auto firstDim =
                         std::find_if (all.begin(), all.end(), [] (const LatentAxis& a) { return a.kind == "dim"; });
                     self.axes = { self.xy->find ("s:brightness") ? "s:brightness" : all[0].id,
                                   self.xy->find ("s:length") ? "s:length" : all[1].id,
                                   firstDim != all.end() ? firstDim->id : juce::String() };
                     if (self.axes[2] == self.axes[0] || self.axes[2] == self.axes[1])
                         self.axes[2] = {};
                     self.list.setAxes (all);
                     self.base.clear (juce::dontSendNotification);
                     for (int i = 0; i < 8; ++i)
                         if (self.xy->typical.count (jointLabels[i]) != 0)
                             self.base.addItem (juce::String ("Typical ") + jointLabels[i], i + 1);
                     if (! self.xy->presetZ.empty())
                         self.base.addItem ("Random real preset", kRandomBase);
                     for (auto* c : std::initializer_list<juce::Component*> {
                              &self.base, &self.category, &self.lockButton, &self.resetButton, &self.dotsButton,
                              &self.background, &self.pad, &self.zSlider, &self.promptBox, &self.promptGo,
                              &self.knobButton })
                         c->setEnabled (true);
                     self.base.setSelectedItemIndex (0, juce::dontSendNotification);
                     self.loadBase (self.base.getSelectedId());
                     self.loadPromptBundle (self.promptFile);
                 });
}

void JointVaePanel::loadBase (int id)
{
    if (! xy || id == 0)
        return;
    juce::String message;
    int label = 8;
    if (id == kRandomBase && ! xy->presetZ.empty())
    {
        const auto k = (size_t) juce::Random::getSystemRandom().nextInt ((int) xy->presetZ.size());
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
    bankMode = false; // a new start leaves the bank preset
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
    promptLoaded = promptLoading = false; // the model is checked against this Lab bundle when Find first needs it
    // Find is offered when a bundle is installed; a failed load on the first Find switches it off
    promptReady = xy != nullptr && file.existsAsFile();
    if (promptReady)
        findButton.setToggleState (true, juce::dontSendNotification);
    candidates.clear();
    selected = -1;
    updatePromptUi();
}

void JointVaePanel::updatePromptUi()
{
    findButton.setVisible (promptReady);
    experimentalTag.setVisible (promptReady);
    if (! promptReady)
        nudgeButton.setToggleState (true, juce::dontSendNotification);
    const bool find = promptReady && findButton.getToggleState();
    promptBox.setTextToShowWhenEmpty (
        find ? "Find sounds: describe one, like deep sub bass" : "Word list: dark bass, then brighter", LF::quiet());
    if (! knobMode)
        promptGo.setTooltip (find ? "Find presets for this description" : "Apply the word list");
    exploreButton.setVisible (bankMode);
    for (size_t i = 0; i < chips.size(); ++i)
    {
        chips[i].setVisible (i < candidates.size());
        if (i < candidates.size())
            chips[i].setButtonText (candidates[i].name);
        chips[i].setToggleState (bankMode && (int) i == selected, juce::dontSendNotification);
    }
    // Bank mode: latent sound edits wait for Explore; the knob pad edits a copy of the codes
    category.setEnabled (xy != nullptr && ! knobMode && ! bankMode);
    pad.setEnabled (xy != nullptr && (! bankMode || knobMode));
    zSlider.setEnabled (xy != nullptr && (! bankMode || knobMode));
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
    const auto token = ++promptToken;
    if (! promptLoaded && ! promptLoading)
    {
        promptLoading = true;
        setStatus ("Loading prompt model...");
        // The worker thread runs this load before the find queued below
        prompter.load (promptFile, current.layout,
                       [safe = juce::Component::SafePointer (this)] (juce::Result r)
                       {
                           if (! safe)
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
            if (! safe || token != safe->promptToken)
                return; // a newer prompt, edit, selection, reset or bundle came first
            auto& self = *safe;
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

void JointVaePanel::explore()
{
    if (! bankMode)
        return;
    edit (
        [this]
        {
            userActed = true;
            bankMode = false;
            setStatus ("Exploring generated variations: the model's approximation of the preset");
            if (knobMode)
                setKnobMode (false); // decodes the bank z
            else
            {
                updatePromptUi();
                decode();
            }
        });
}

void JointVaePanel::nudge (const juce::String& text)
{
    if (! xy)
        return;
    if (bankMode)
    {
        setStatus (exploreFirst);
        return;
    }
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
    // Page assign(): swap when the axis is on another slot; Z again clears Z. X and Y never go empty.
    auto next = axes;
    if (next[(size_t) slot] == id)
    {
        if (slot != 2)
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
    syncPositions();
    refresh();
    updateDots();
    scheduleMap();
    resized();
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
    pad.setDot ({ pos[0], pos[1] });
    zSlider.setValue (juce::jlimit (0.0f, 1.0f, pos[2]), juce::dontSendNotification);
    list.setAssigned (axes);
    if (! knobMode)
        list.setPositions (z, dragged);
    repaint();
}

void JointVaePanel::updateDots()
{
    std::vector<std::tuple<float, float, int>> dots;
    if (knobMode) // shortcut: no preset dots on the knob pad (the bundle holds no knob values per preset)
    {
        pad.setPresetDots ({}, -1);
        return;
    }
    auto *ax = axisOn (0), *ay = axisOn (1);
    if (dotsButton.getToggleState() && ax && ay)
        for (size_t i = 0; i < xy->presetZ.size(); ++i)
            dots.emplace_back (axisPosition (*ax, xy->presetZ[i]), axisPosition (*ay, xy->presetZ[i]),
                               xy->presetLabel[i]);
    pad.setPresetDots (std::move (dots), category.getSelectedId() - 1);
}

void JointVaePanel::jumpToDot (juce::Point<float> p)
{
    auto *ax = axisOn (0), *ay = axisOn (1);
    if (! dotsButton.getToggleState() || ! ax || ! ay)
        return;
    const int label = category.getSelectedId() - 1;
    const auto w = (float) pad.getWidth(), h = (float) pad.getHeight();
    float best = 7.0f * 7.0f;
    int hit = -1;
    for (size_t i = 0; i < xy->presetZ.size(); ++i)
    {
        if (xy->presetLabel[i] != label)
            continue;
        const auto dx = (axisPosition (*ax, xy->presetZ[i]) - p.x) * w,
                   dy = (axisPosition (*ay, xy->presetZ[i]) - p.y) * h;
        if (dx * dx + dy * dy < best)
        {
            best = dx * dx + dy * dy;
            hit = (int) i;
        }
    }
    if (hit < 0)
        return;
    userActed = true; // a dot click is a user action (ruling R14)
    z = xy->presetZ[(size_t) hit];
    const int l = juce::jlimit (0, 8, xy->presetLabel[(size_t) hit]);
    category.setSelectedId (l + 1, juce::dontSendNotification);
    setStatus ("Jumped to the real preset " + quoted (presetName (*xy, (size_t) hit)) + " (" + jointLabels[l] + ").");
    syncPositions();
    refresh();
    updateDots();
    decode();
}

void JointVaePanel::requestMap()
{
    if (! xy)
        return;
    MapRequest request;
    request.z = z;
    request.category = category.getSelectedId() - 1;
    for (int i = 0; i < 3; ++i)
        if (axisOn (i))
            request.axisIds.push_back (axes[(size_t) i]);
    request.zPos = pos[2];
    request.kind = mapKinds[juce::jlimit (0, 5, background.getSelectedId() - 1)][0];
    if (knobMode)
    {
        request.kind = background.getSelectedId() == 2 ? "none" : "cat";
        request.knobs = knobVals;
        request.groupX = groupOn (0);
        request.groupY = groupOn (1);
    }
    request.lockedTail = lockedTail;
    worker.map (std::move (request),
                [safe = juce::Component::SafePointer (this)] (juce::Result r, MapResult m)
                {
                    if (! safe)
                        return;
                    if (r.failed())
                    {
                        safe->setStatus (r.getErrorMessage(), true);
                        safe->mapError = true;
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
                                              labCategoryColours[c].withAlpha (juce::jmin (
                                                  1.0f, labMapAlpha * (0.45f + 0.9f * m.conf[(size_t) k]))));
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
                    else if (m.val.size() == (size_t) (n * n))
                    {
                        image = juce::Image (juce::Image::ARGB, n, n, true);
                        const auto span = m.hi > m.lo ? m.hi - m.lo : 1.0f;
                        for (int k = 0; k < n * n; ++k)
                            image.setPixelAt (
                                k % n, k / n,
                                labHeat.withAlpha (
                                    0.03f + 0.72f * juce::jlimit (0.0f, 1.0f, (m.val[(size_t) k] - m.lo) / span)));
                    }
                    safe->pad.setMap (image, std::move (labels));
                    if (safe->onMapShown)
                        safe->onMapShown();
                });
}

void JointVaePanel::decode()
{
    if (! xy)
        return;
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
    worker.lab (
        std::move (request),
        [safe = juce::Component::SafePointer (this)] (juce::Result r, JointVaeWorker::Output out)
        {
            if (! safe)
                return;
            if (r.failed())
            {
                safe->setStatus (r.getErrorMessage(), true);
                return;
            }
            auto& self = *safe;
            self.current = std::move (out);
            self.outputValid = true;
            // Page specCanvas: ((mel + 70) / 70)^0.8, accent over the well, low bands at the bottom
            const auto bands = (int) self.xy->bandHz.size();
            const auto frames = bands > 0 ? (int) self.current.mel.size() / bands : 0;
            self.spec = frames > 0 ? juce::Image (juce::Image::RGB, frames, bands, false) : juce::Image();
            for (int t = 0; t < frames; ++t)
                for (int b = 0; b < bands; ++b)
                    self.spec.setPixelAt (
                        t, bands - 1 - b,
                        labWell.interpolatedWith (
                            labHeat,
                            std::pow (juce::jlimit (0.0f, 1.0f, (self.current.mel[(size_t) (t * bands + b)] + 70) / 70),
                                      0.8f)));
            self.sound = measures (*self.xy, self.current.mel);
            self.repaint();
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
        });
}

void JointVaePanel::paint (juce::Graphics& g)
{
    struct Label
    {
        juce::String name, low, high;
    };
    const auto label = [this] (int slot) -> Label
    {
        if (knobMode)
            return groupOn (slot) >= 0 ? Label { kg->groups[(size_t) groupOn (slot)].name, "low", "high" } : Label();
        auto* a = axisOn (slot);
        return a ? Label { a->name, a->low, a->high } : Label();
    };
    g.setFont (LF::labelFont (10.0f));
    if (const auto ax = label (0); ax.name.isNotEmpty())
    {
        g.setColour (LF::quiet());
        g.drawText (ax.low, xLabelArea, juce::Justification::centredLeft);
        g.drawText (ax.high, xLabelArea, juce::Justification::centredRight);
        g.setColour (LF::ink);
        g.drawText ("X " + juce::String (juce::CharPointer_UTF8 ("\xc2\xb7 ")) + ax.name, xLabelArea,
                    juce::Justification::centred);
    }
    if (const auto ay = label (1); ay.name.isNotEmpty() && ! yLabelArea.isEmpty())
    {
        // Rotated a quarter turn: reads bottom to top, low end at the bottom
        juce::Graphics::ScopedSaveState save (g);
        const auto c = yLabelArea.getCentre().toFloat();
        g.addTransform (juce::AffineTransform::rotation (-juce::MathConstants<float>::halfPi, c.x, c.y));
        const auto r = juce::Rectangle<int> (yLabelArea.getHeight(), yLabelArea.getWidth()).withCentre (c.toInt());
        g.setColour (LF::quiet());
        g.drawText (ay.low, r, juce::Justification::centredLeft);
        g.drawText (ay.high, r, juce::Justification::centredRight);
        g.setColour (LF::ink);
        g.drawText ("Y " + juce::String (juce::CharPointer_UTF8 ("\xc2\xb7 ")) + ay.name, r,
                    juce::Justification::centred);
    }
    if (zSlider.isVisible())
    {
        g.setColour (LF::ink);
        g.drawFittedText ("Z " + juce::String (juce::CharPointer_UTF8 ("\xc2\xb7 ")) + label (2).name, zLabelArea,
                          juce::Justification::centredLeft, 1, 0.7f);
    }
    paintReadouts (g);
}

void JointVaePanel::paintReadouts (juce::Graphics& g)
{
    // Page drawReadouts (latent mode), laid out as one strip
    g.setColour (LF::faint);
    g.drawRoundedRectangle (readoutArea.toFloat().reduced (0.5f), 5, 1);
    const auto eyebrow = [&g] (juce::Rectangle<int>& area, const juce::String& text)
    {
        g.setColour (LF::quiet());
        g.setFont (LF::labelFont (9.5f, true));
        g.drawText (text.toUpperCase(), area.removeFromTop (kEyebrowH), juce::Justification::topLeft);
    };
    const auto rows =
        [&g] (juce::Rectangle<int> area, const std::vector<std::pair<juce::String, juce::String>>& items, int keyW)
    {
        const int h = items.empty() ? 0 : juce::jmin (14, area.getHeight() / (int) items.size());
        g.setFont (LF::labelFont (10.0f));
        for (const auto& [k, v] : items)
        {
            auto row = area.removeFromTop (h);
            g.setColour (LF::quiet());
            g.drawText (k, row.removeFromLeft (keyW), juce::Justification::centredLeft);
            g.setColour (LF::ink);
            g.drawFittedText (v, row, juce::Justification::centredRight, 1, 0.75f);
        }
    };
    auto area = specArea;
    eyebrow (area, "Expected sound at the dot");
    auto caption = area.removeFromBottom (12);
    g.setColour (labWell);
    g.fillRoundedRectangle (area.toFloat(), 4);
    if (outputValid && spec.isValid())
        g.drawImage (spec, area.toFloat().reduced (1), juce::RectanglePlacement::stretchToFit);
    g.setColour (LF::faint);
    g.drawRoundedRectangle (area.toFloat().reduced (0.5f), 4, 1);
    g.setColour (LF::quiet());
    g.setFont (LF::labelFont (9.0f));
    g.drawText ("0 s", caption, juce::Justification::bottomLeft);
    g.drawText ("note off 2 s", caption, juce::Justification::centredBottom);
    g.drawText ("6 s", caption, juce::Justification::bottomRight);
    if (! outputValid)
        return;

    area = measureArea.withTrimmedTop (kEyebrowH);
    rows (area,
          { { "Brightness", juce::String (sound.brightness, 1) + " oct above the note" },
            { "Attack", sound.attack < 0.05f ? juce::String ("< 50 ms")
                                             : juce::String (juce::roundToInt (sound.attack * 1000)) + " ms" },
            { "Length", juce::String (sound.length, 1) + " s" },
            { "Noise", juce::String (juce::roundToInt (sound.noise * 100)) + "%" } },
          64);

    area = presetArea;
    eyebrow (area, juce::String ("Decoded preset") + (lockedTail ? dotSep + "structure locked" : juce::String()));
    rows (area, current.preset, 96);

    area = guessArea;
    eyebrow (area, "Category guess at the dot");
    const auto top = (int) (std::max_element (current.guess.begin(), current.guess.end()) - current.guess.begin());
    const int h = area.getHeight() / 8;
    for (int c = 0; c < 8; ++c)
    {
        auto row = area.removeFromTop (h);
        const auto p = current.guess[(size_t) c];
        g.setColour (c == top ? LF::ink : LF::quiet());
        g.setFont (LF::labelFont (9.5f, c == top));
        g.drawText (jointLabels[c], row.removeFromLeft (66), juce::Justification::centredLeft);
        g.setColour (LF::quiet());
        g.drawText (juce::String (juce::roundToInt (p * 100)) + "%", row.removeFromRight (32),
                    juce::Justification::centredRight);
        const auto bar = row.reduced (4, 0).withSizeKeepingCentre (row.getWidth() - 8, 6).toFloat();
        g.setColour (LF::faint);
        g.fillRoundedRectangle (bar, 3);
        g.setColour (labCategoryColours[c]);
        g.fillRoundedRectangle (bar.withWidth (bar.getWidth() * juce::jlimit (0.0f, 1.0f, p)), 3);
    }
}

void JointVaePanel::resized()
{
    auto r = getLocalBounds();
    auto bar = r.removeFromTop (kToolbarH);
    r.removeFromTop (kGap);
    const auto put = [&bar] (juce::Component& c, int w)
    {
        c.setBounds (bar.removeFromLeft (w));
        bar.removeFromLeft (6);
    };
    put (latentButton, 92);
    put (knobButton, 84);
    bar.removeFromLeft (10);
    put (base, 180);
    put (category, 120);
    put (lockButton, 124);
    put (resetButton, 64);
    put (dotsButton, 104);
    put (sendButton, 120);
    put (background, 130);

    auto left = r.removeFromLeft (r.getWidth() * 2 / 3);
    left.removeFromRight (12);
    listView.setBounds (r);
    list.layOut (listView.getWidth() - listView.getScrollBarThickness()); // rows never sit under the scroll bar

    readoutArea = left.removeFromBottom (kReadoutH);
    {
        auto strip = readoutArea.reduced (8);
        const auto w = strip.getWidth();
        specArea = strip.removeFromLeft (w * 22 / 100);
        strip.removeFromLeft (12);
        measureArea = strip.removeFromLeft (w * 25 / 100);
        strip.removeFromLeft (12);
        guessArea = strip.removeFromRight (w * 22 / 100);
        strip.removeFromRight (12);
        presetArea = strip;
    }
    left.removeFromBottom (kGap);
    // One prompt line, the candidates when there are any, then the status line
    const bool chipRow = ! candidates.empty() || bankMode;
    promptArea = left.removeFromBottom (kPromptH + (chipRow ? kChipRowH + 4 : 0));
    auto promptRow = promptArea.removeFromTop (kPromptRowH);
    if (findButton.isVisible())
    {
        findButton.setBounds (promptRow.removeFromLeft (88));
        experimentalTag.setBounds (promptRow.removeFromLeft (70));
    }
    nudgeButton.setBounds (promptRow.removeFromLeft (76));
    promptRow.removeFromLeft (6);
    promptGo.setBounds (promptRow.removeFromRight (kPromptRowH + 8));
    promptRow.removeFromRight (4);
    promptBox.setBounds (promptRow);
    promptArea.removeFromTop (4);
    if (chipRow)
    {
        auto row = promptArea.removeFromTop (kChipRowH);
        if (bankMode)
        {
            exploreButton.setBounds (row.removeFromRight (190));
            row.removeFromRight (6);
        }
        const auto w = juce::jmin (88, candidates.empty() ? 0 : row.getWidth() / (int) candidates.size());
        for (size_t i = 0; i < candidates.size() && i < chips.size(); ++i)
            chips[i].setBounds (row.removeFromLeft (w).withTrimmedRight (4));
        promptArea.removeFromTop (4);
    }
    status.setBounds (promptArea);
    left.removeFromBottom (kGap);
    const bool hasZ = hasSlot (2);
    zSlider.setVisible (hasZ);
    if (hasZ)
    {
        auto row = left.removeFromBottom (kZRowH).withTrimmedLeft (kYLabelW);
        zLabelArea = row.removeFromLeft (row.getWidth() / 3);
        zSlider.setBounds (row);
    }
    xLabelArea = left.removeFromBottom (kXLabelH).withTrimmedLeft (kYLabelW);
    yLabelArea = left.removeFromLeft (kYLabelW);
    pad.setBounds (left);
}
#endif
} // namespace vae::experimental
