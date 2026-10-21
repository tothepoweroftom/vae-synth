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
void JointVaePanel::resized() { note.setBounds (getLocalBounds()); }
void JointVaePanel::paint (juce::Graphics&) {}
#else
namespace
{
constexpr int kRandomBase = 100, kToolbarH = 28, kGap = 8, kReadoutH = 120, kPromptH = 76, kXLabelH = 18, kZRowH = 24,
              kYLabelW = 18;
}

JointVaePanel::JointVaePanel()
{
    for (auto* c :
         std::initializer_list<juce::Component*> { &latentButton, &knobButton, &base, &category, &lockButton,
                                                   &resetButton, &dotsButton, &sendButton, &pad, &listView, &status })
        addAndMakeVisible (c);
    addChildComponent (zSlider);
    latentButton.setToggleState (true, juce::dontSendNotification);
    knobButton.setEnabled (false); // shortcut: knob pad arrives with Task 10
    for (int i = 0; i < 9; ++i)
        category.addItem (jointLabels[i], i + 1);
    category.setTooltip ("Category the decoder is conditioned on");
    base.setTooltip ("Start from");
    lockButton.setTooltip ("Keep oscillator types, effects and other switches fixed while dragging");
    dotsButton.setTooltip ("Real presets as dots (colour = category)");
    dotsButton.setClickingTogglesState (true);
    listView.setViewedComponent (&list, false);
    listView.setScrollBarsShown (true, false);
    zSlider.setRange (0.0, 1.0, 0.001);
    zSlider.setSliderStyle (juce::Slider::LinearHorizontal);
    zSlider.setTextBoxStyle (juce::Slider::NoTextBox, false, 0, 0);
    status.setFont (LF::labelFont (10.0f));
    status.setColour (juce::Label::textColourId, LF::quiet());
    status.setJustificationType (juce::Justification::topLeft);

    pad.moved = [this] (juce::Point<float> p)
    {
        userActed = true;
        pos[0] = p.x;
        pos[1] = p.y;
        applyPositions();
        refresh();
        decode();
    };
    zSlider.onValueChange = [this]
    {
        userActed = true;
        pos[2] = (float) zSlider.getValue();
        applyPositions();
        refresh();
        decode();
    };
    list.assign = [this] (int slot, const juce::String& id) { assign (slot, id); };
    list.slid = [this] (const LatentAxis& axis, float p)
    {
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
    };
    base.onChange = resetButton.onClick = [this]
    {
        userActed = true;
        loadBase (base.getSelectedId());
    };
    category.onChange = [this]
    {
        userActed = true;
        updateDots();
        decode();
    };
    lockButton.onClick = [this]
    {
        userActed = true;
        if (lockedTail)
            lockedTail.reset();
        else if (outputValid && current.layout.numNumeric > 0
                 && (size_t) current.layout.numNumeric <= current.codes.size())
            lockedTail.emplace (current.codes.begin() + current.layout.numNumeric, current.codes.end());
        lockButton.setToggleState (lockedTail.has_value(), juce::dontSendNotification);
        decode();
    };
    dotsButton.onClick = [this] { updateDots(); };
    sendButton.onClick = [this]
    {
        if (! outputValid || current.patch.isEmpty())
            return;
        const auto r = onPatch ? onPatch (current.patch) : juce::Result::fail ("Surge playback not connected");
        setStatus (r.wasOk() ? "Playing the decoded point on Surge; move the pad to return to Lite."
                             : r.getErrorMessage(),
                   r.failed());
    };
    for (auto* c : std::initializer_list<juce::Component*> { &base, &category, &lockButton, &resetButton, &dotsButton,
                                                             &sendButton, &pad, &zSlider })
        c->setEnabled (false);

    // Installed by scripts/install-models.sh, like the Prompt tab's models
    if (const auto installed = ml::appDataDir().getChildFile ("models/joint-vae/bundle.json"); installed.existsAsFile())
        loadBundle (installed);
    else
        setStatus ("No VAE bundle installed: run scripts/install-models.sh");
}

void JointVaePanel::setStatus (const juce::String& text, bool error)
{
    errorShown = error;
    status.setText (text, juce::dontSendNotification);
}

const LatentAxis* JointVaePanel::axisOn (int slot) const
{
    return xy && axes[(size_t) slot].isNotEmpty() ? xy->find (axes[(size_t) slot]) : nullptr;
}

void JointVaePanel::loadBundle (const juce::File& file)
{
    setStatus ("Loading and checking joint VAE...");
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
                     self.current = std::move (out);
                     self.outputValid = false;
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
                              &self.pad, &self.zSlider })
                         c->setEnabled (true);
                     self.base.setSelectedItemIndex (0, juce::dontSendNotification);
                     self.loadBase (self.base.getSelectedId());
                     self.resized();
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
        const auto name = (int) k < xy->presetName.size() && xy->presetName[(int) k].isNotEmpty()
                              ? xy->presetName[(int) k]
                              : "preset " + juce::String (k);
        message = juce::String (juce::CharPointer_UTF8 ("Started from \xe2\x80\x9c")) + name
                  + juce::String (juce::CharPointer_UTF8 ("\xe2\x80\x9d (")) + jointLabels[label] + ").";
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
    setStatus (message);
    syncPositions();
    refresh();
    updateDots();
    decode();
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
    resized();
}

void JointVaePanel::syncPositions()
{
    for (int i = 0; i < 3; ++i)
        if (auto* a = axisOn (i))
            pos[(size_t) i] = axisPosition (*a, z);
}

void JointVaePanel::applyPositions()
{
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
    list.setPositions (z, dragged);
    repaint();
}

void JointVaePanel::updateDots()
{
    std::vector<std::tuple<float, float, int>> dots;
    auto *ax = axisOn (0), *ay = axisOn (1);
    if (dotsButton.getToggleState() && ax && ay)
        for (size_t i = 0; i < xy->presetZ.size(); ++i)
            dots.emplace_back (axisPosition (*ax, xy->presetZ[i]), axisPosition (*ay, xy->presetZ[i]),
                               xy->presetLabel[i]);
    pad.setPresetDots (std::move (dots), category.getSelectedId() - 1);
}

void JointVaePanel::decode()
{
    if (! xy)
        return;
    LabRequest request;
    request.z = z;
    request.category = category.getSelectedId() - 1;
    request.lockedTail = lockedTail;
    worker.lab (std::move (request),
                [safe = juce::Component::SafePointer (this)] (juce::Result r, JointVaeWorker::Output out)
                {
                    if (! safe)
                        return;
                    if (r.failed())
                    {
                        safe->setStatus (r.getErrorMessage(), true);
                        return;
                    }
                    safe->current = std::move (out);
                    safe->outputValid = true;
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
    const auto name = [this] (int slot)
    {
        auto* a = axisOn (slot);
        return a ? a->name : juce::String();
    };
    g.setFont (LF::labelFont (10.0f));
    if (auto* ax = axisOn (0))
    {
        g.setColour (LF::quiet());
        g.drawText (ax->low, xLabelArea, juce::Justification::centredLeft);
        g.drawText (ax->high, xLabelArea, juce::Justification::centredRight);
        g.setColour (LF::ink);
        g.drawText ("X " + juce::String (juce::CharPointer_UTF8 ("\xc2\xb7 ")) + name (0), xLabelArea,
                    juce::Justification::centred);
    }
    if (auto* ay = axisOn (1); ay && ! yLabelArea.isEmpty())
    {
        // Rotated a quarter turn: reads bottom to top, low end at the bottom
        juce::Graphics::ScopedSaveState save (g);
        const auto c = yLabelArea.getCentre().toFloat();
        g.addTransform (juce::AffineTransform::rotation (-juce::MathConstants<float>::halfPi, c.x, c.y));
        const auto r = juce::Rectangle<int> (yLabelArea.getHeight(), yLabelArea.getWidth()).withCentre (c.toInt());
        g.setColour (LF::quiet());
        g.drawText (ay->low, r, juce::Justification::centredLeft);
        g.drawText (ay->high, r, juce::Justification::centredRight);
        g.setColour (LF::ink);
        g.drawText ("Y " + juce::String (juce::CharPointer_UTF8 ("\xc2\xb7 ")) + name (1), r,
                    juce::Justification::centred);
    }
    if (zSlider.isVisible())
    {
        g.setColour (LF::ink);
        g.drawFittedText ("Z " + juce::String (juce::CharPointer_UTF8 ("\xc2\xb7 ")) + name (2), zLabelArea,
                          juce::Justification::centredLeft, 1, 0.7f);
    }
    // Reserved for the readout strip (Task 8)
    g.setColour (LF::faint);
    g.drawRoundedRectangle (readoutArea.toFloat().reduced (0.5f), 5, 1);
    g.setColour (LF::quiet());
    g.setFont (LF::labelFont (9.5f, true));
    g.drawText ("EXPECTED SOUND", readoutArea.reduced (8), juce::Justification::topLeft);
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
    put (base, 200);
    put (category, 120);
    put (lockButton, 124);
    put (resetButton, 64);
    put (dotsButton, 104);
    put (sendButton, 120);

    auto left = r.removeFromLeft (r.getWidth() * 2 / 3);
    left.removeFromRight (12);
    listView.setBounds (r);
    list.layOut (listView.getWidth() - listView.getScrollBarThickness()); // rows never sit under the scroll bar

    readoutArea = left.removeFromBottom (kReadoutH);
    left.removeFromBottom (kGap);
    promptArea = left.removeFromBottom (kPromptH); // Task 9's log and prompt; the status line until then
    status.setBounds (promptArea);
    left.removeFromBottom (kGap);
    const bool hasZ = axisOn (2) != nullptr;
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
