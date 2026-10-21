#pragma once
#include "JointVae.h"
#include "LabAxisList.h"
#include "LabPad.h"
#include "PromptModel.h"
#include "PromptWords.h"
#include <juce_gui_basics/juce_gui_basics.h>

namespace vae::experimental
{
// VAE Lab v2: the student's XY-pad page (vae/pitcher/xypad/page_v2.template.html) as a native screen.
// Every move decodes on the worker; results play on Lite through onLite, Send to Surge plays onPatch.
class JointVaePanel final : public juce::Component
{
public:
    JointVaePanel();
    void resized() override;
    void paint (juce::Graphics&) override;
    /** Plays a decoded point on the Lite engine (set by the editor; message thread). */
    std::function<void (const schema::Snapshot&)> onLite;
    /** Plays a decoded full Surge patch (set by the editor; message thread). */
    std::function<juce::Result (const juce::MemoryBlock&)> onPatch;
    /** The installed bundle loads on construction; tests load another one. */
    void loadBundle (const juce::File&);
    /** A change to the controls that did not come from onLite (Synth view, host): the sound is
        "edited" and the next Lab sound edit asks before replacing it (Task 16; message thread). */
    void markEdited();
#if VAE_WITH_SURGE
    bool isEdited() const { return edited; }
    bool isAskingToReplace() const { return pendingEdit != nullptr; }
    /** Answers "Replace your Synth edits?" (the dialog's buttons; tests call it directly). */
    void answerReplace (bool replace);
    /** Test hooks: called after a map result reaches the pad; whether a map is computing. */
    std::function<void()> onMapShown;
    bool isMapRunning() const { return worker.mapRunning(); }
    /** The prompt box's submit handler: Find sounds with a prompt bundle loaded and Find chosen, else Nudge
        (also driven by tests, which read the state back below). */
    void submitPrompt (const juce::String&);
    void findSounds (const juce::String&); // the learned prompt search (p0); results come back async
    void nudge (const juce::String&);      // the word list (interpretPrompt)
    /** The p0 prompt bundle loads after each Lab bundle; the installed one by default. */
    void loadPromptBundle (const juce::File&);
    bool isPromptReady() const { return promptReady; }
    const std::vector<PromptCandidate>& promptCandidates() const { return candidates; }
    int selectedCandidate() const { return selected; }
    void selectCandidate (int);
    bool inBankMode() const { return bankMode; }
    /** Leaves bank mode: decodes the bank z once, then the pad decodes as usual. */
    void explore();
    juce::String statusText() const { return status.getText(); }
    const XyPadData* padData() const { return xy.get(); }
    const JointLatent& latent() const { return z; }
    const JointVaeWorker::Output& output() const { return current; }
    const KnobGroups* knobGroups() const { return kg.get(); }
    void assignAxis (int slot, const juce::String& id) { assign (slot, id); }
#endif

private:
#if VAE_WITH_SURGE
    void loadBase (int index);
    void assign (int slot, const juce::String& id);
    void syncPositions();                // page latSyncPos
    void applyPositions (int slots = 7); // page latApplyPos; knob pad: only the slots (bit mask) the gesture moved
    void decode();
    void refresh (const LatentAxis* dragged = nullptr);
    void updateDots();
    void scheduleMap() { mapTimer.startTimer (150); }
    void requestMap();
    void jumpToDot (juce::Point<float>); // page pointerup: a tap within 7 px of a same-label dot
    void paintReadouts (juce::Graphics&);
    void setStatus (const juce::String&, bool error = false);
    const LatentAxis* axisOn (int slot) const;
    int groupOn (int slot) const; // knob pad: group index on the slot, -1 when none
    bool hasSlot (int slot) const { return knobMode ? groupOn (slot) >= 0 : axisOn (slot) != nullptr; }
    void setKnobMode (bool); // page mode switch: the decoded codes stay, the other pad's edits are dropped
    void fillBackground();
    void edit (std::function<void()> change); // a Lab sound edit: asks first while the sound is edited
    void resolveReplace (bool replace);
    void applyCandidate (int);
    void showCandidates();
    void updatePromptUi();

    JointVaeWorker worker;
    PromptWorker prompter;
    juce::File promptFile;
    bool promptReady = false;
    uint64_t promptToken = 0; // a prompt result applies only while this is unchanged
    std::vector<PromptCandidate> candidates;
    int selected = -1;
    bool bankMode = false;        // a bank preset plays: latent sound edits wait for Explore
    std::vector<float> bankCodes; // the selected row's codes, exactly as stored (NaN = inactive)
    std::shared_ptr<const XyPadData> xy;
    std::shared_ptr<const KnobGroups> kg;
    JointVaeWorker::Output current;
    bool outputValid = false;
    bool userActed = false;            // results play only after the user acts in the Lab (ruling R14)
    bool edited = false;               // the controls changed outside the Lab since its last write
    std::function<void()> pendingEdit; // the Lab edit waiting on "Replace your Synth edits?"
    juce::ScopedMessageBox replaceBox;
    int shownBase = 0, shownCategory = 0; // the combos at the last decode, put back by Keep editing
    bool errorShown = false;
    bool mapError = false; // the status line shows a map failure
    JointLatent z {};
    std::array<juce::String, 3> axes;
    std::array<float, 3> pos { 0.5f, 0.5f, 0.5f };
    std::optional<std::vector<float>> lockedTail;
    // Knob pad: flat 2D pad over the 61 knob groups (ruling R16). `axes` then holds group names.
    bool knobMode = false;
    std::array<juce::String, 3> latAxes, knobAxes;
    std::vector<float> knobVals, knobCodes; // the 123 knob values and the codes they are written into
    int latBackground = 2;

    juce::TextButton latentButton { "Latent pad" }, knobButton { "Knob pad" }, lockButton { "Lock structure" },
        resetButton { "Reset" }, dotsButton { "Preset dots" }, sendButton { "Send to Surge" };
    juce::ComboBox base, category, background;
    LabPad pad;
    juce::Slider zSlider;
    juce::Viewport listView;
    LabAxisList list;
    juce::Label status;
    juce::TextEditor promptBox;
    juce::TextButton promptGo { juce::String (juce::CharPointer_UTF8 ("\xe2\x86\x92")) };
    juce::TextButton findButton { "Find sounds" }, nudgeButton { "Word list" },
        exploreButton { "Explore generated variations" };
    std::array<juce::TextButton, 8> chips;
    juce::Rectangle<int> yLabelArea, xLabelArea, zLabelArea, promptArea, readoutArea, specArea, measureArea, presetArea,
        guessArea;
    juce::Image spec; // frames x 64 expected spectrogram of `current`
    SoundMeasures sound {};
    juce::TimedCallback mapTimer { [this]
                                   {
                                       mapTimer.stopTimer();
                                       requestMap();
                                   } };
#else
    juce::Label note;
#endif
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (JointVaePanel)
};
} // namespace vae::experimental
