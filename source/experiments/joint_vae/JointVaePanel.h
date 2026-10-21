#pragma once
#include "JointVae.h"
#include "LabAxisList.h"
#include "LabOrbit.h"
#include "LabPad.h"
#include "LabTheme.h"
#include "PitcherControls.h"
#include "PitcherLayout.h"
#include "PromptModel.h"
#include "PromptWords.h"
#include <juce_gui_basics/juce_gui_basics.h>

namespace vae::experimental
{
// VAE Lab v2: the student's XY-pad page (vae/pitcher/xypad/page_v2.template.html) as a native screen, in the
// brutalist layout of docs/design/PITCHER-BRUTALIST-JUCE.md: category match and the controls on the left, the
// orbit view (or the knob pad), the prompt and a 3-octave keyboard in the centre, the axis list on the right.
// Every move decodes on the worker; results play on Lite through onLite, Send to Surge plays onPatch.
class JointVaePanel final : public juce::Component
{
public:
    /** `keys`: the processor's keyboard state, which the Lab's keyboard plays. The typefaces are the
        design's Inter, Inter Bold and Space Mono (JUCE fallbacks without them). */
    explicit JointVaePanel (juce::MidiKeyboardState& keys, juce::Typeface::Ptr body = {}, juce::Typeface::Ptr bold = {},
                            juce::Typeface::Ptr mono = {});
    ~JointVaePanel() override;
    void resized() override;
    void paint (juce::Graphics&) override;
    /** Plays a decoded point on the Lite engine (set by the editor; message thread). */
    std::function<void (const schema::Snapshot&)> onLite;
    /** Plays a decoded full Surge patch (set by the editor; message thread). */
    std::function<juce::Result (const juce::MemoryBlock&)> onPatch;
    /** The user chose Replace: the edited sound is now the Lab's to replace (set by the editor). */
    std::function<void()> onReplace;
    /** The installed bundle loads on construction; tests load another one. */
    void loadBundle (const juce::File&);
    /** A change to the controls that did not come from onLite (Synth view, host): the sound is
        "edited" and the next Lab sound edit asks before replacing it (Task 16; message thread). */
    void markEdited();
#if VAE_WITH_SURGE
    bool isEdited() const { return edited; }
    bool isAskingToReplace() const { return pendingEdit != nullptr; }
    /** Answers "Replace your Synth edits?" (the banner's buttons; tests call it directly). */
    void answerReplace (bool replace);
    /** Test hooks: called after a map result reaches the knob pad; whether a map is computing. */
    std::function<void()> onMapShown;
    bool isMapRunning() const { return worker.mapRunning(); }
    bool hasOutput() const { return outputValid; } // a decode result has arrived
    const std::array<juce::String, 3>& assignedAxes() const { return axes; }
    const PitcherLayout& getLayout() const { return layout; }
    /** The prompt box's submit handler: Find sounds with a prompt bundle loaded and Find chosen, else Nudge
        (also driven by tests, which read the state back below). */
    void submitPrompt (const juce::String&);
    void findSounds (const juce::String&); // the learned prompt search (p0/p1); results come back async
    void nudge (const juce::String&);      // the word list (interpretPrompt)
    /** The prompt bundle (the installed one by default) is offered after each Lab bundle but only loads on the
        first Find (R20: it is ~625 MB, so an editor that never searches never pays for it). */
    void loadPromptBundle (const juce::File&);
    bool isPromptReady() const { return promptReady; }   // Find is offered
    bool isPromptLoaded() const { return promptLoaded; } // the model is in memory
    const std::vector<PromptCandidate>& promptCandidates() const { return candidates; }
    int selectedCandidate() const { return selected; }
    void selectCandidate (int);
    bool inBankMode() const { return bankMode; } // the selected candidate's exact codes play
    juce::String statusText() const { return status.getText(); }
    const XyPadData* padData() const { return xy.get(); }
    const JointLatent& latent() const { return z; }
    const JointVaeWorker::Output& output() const { return current; }
    const KnobGroups* knobGroups() const { return kg.get(); }
    void assignAxis (int slot, const juce::String& id) { assign (slot, id); }
#endif

private:
#if VAE_WITH_SURGE
    /** A CATEGORY MATCH row: name, percentage and a 3 px bar; a click toggles the cloud's highlight. */
    struct CategoryRow final : juce::Button
    {
        CategoryRow()
            : Button ({})
        {
            setClickingTogglesState (false);
        }
        void paintButton (juce::Graphics&, bool over, bool down) override;
        int index = 0;
        float match = 0;
        bool closest = false;
    };

    /** The start point and category boxes: square and monospaced, as the study's buttons (the study has none). */
    struct ComboLook final : juce::LookAndFeel_V4
    {
        explicit ComboLook (juce::Font);
        void drawComboBox (juce::Graphics&, int w, int h, bool down, int, int, int, int, juce::ComboBox&) override;
        juce::Font getComboBoxFont (juce::ComboBox&) override { return font; }
        void positionComboBoxText (juce::ComboBox&, juce::Label&) override;
        juce::Font getPopupMenuFont() override { return font; }
        juce::Font font;
    };

    void loadBase (int index, bool draw = true); // draw: Random real preset picks a new one (Reset keeps it)
    void assign (int slot, const juce::String& id);
    void syncPositions();                // page latSyncPos
    void applyPositions (int slots = 7); // page latApplyPos; knob pad: only the slots (bit mask) the gesture moved
    void decode();
    void refresh (const LatentAxis* dragged = nullptr);
    void updateDots();
    void scheduleMap()
    {
        if (knobMode) // Orbit is the latent pad's only view: only the knob pad has a map
            mapTimer.startTimer (150);
    }
    void requestMap();
    void jumpToPreset (int k); // a click on a preset dot of the current category
    void showOutput();         // the spectrogram, its details and the category match, from `current`
    void setStatus (const juce::String&, bool error = false);
    const LatentAxis* axisOn (int slot) const;
    int groupOn (int slot) const; // knob pad: group index on the slot, -1 when none
    bool hasSlot (int slot) const { return knobMode ? groupOn (slot) >= 0 : axisOn (slot) != nullptr; }
    void setKnobMode (bool); // page mode switch: the decoded codes stay, the other pad's edits are dropped
    void edit (std::function<void()> change); // a Lab sound edit: asks first while the sound is edited
    void resolveReplace (bool replace);
    void showAsk (bool); // the inline Replace question
    void applyCandidate (int);
    void leaveBank();           // the first sound edit after selecting a candidate decodes from its z
    void clearPromptProgress(); // a dropped Find must not leave "Finding sounds..." behind
    void updatePromptUi();

    PitcherLookAndFeel look; // before every child; detached in the destructor
    ComboLook comboLook { look.monoFont (11.0f) };
    JointVaeWorker worker;
    PromptWorker prompter;
    juce::File promptFile;
    bool promptReady = false, promptLoaded = false, promptLoading = false;
    uint64_t promptEpoch = 0; // a prompt load from before the last loadPromptBundle completes into nothing
    uint64_t promptToken = 0; // a prompt result applies only while this is unchanged
    uint64_t lastFind = 0;    // the token of the newest Find
    bool finding = false;     // the newest Find has not called back yet
    std::vector<PromptCandidate> candidates;
    int selected = -1;
    bool bankMode = false;        // the selected bank preset plays exactly, until the next sound edit
    std::vector<float> bankCodes; // the selected row's codes, exactly as stored (NaN = inactive)
    std::shared_ptr<const XyPadData> xy;
    std::shared_ptr<const KnobGroups> kg;
    JointVaeWorker::Output current;
    bool outputValid = false;
    bool userActed = false;               // results play only after the user acts in the Lab (ruling R14)
    bool edited = false;                  // the controls changed outside the Lab since its last write
    std::function<void()> pendingEdit;    // the Lab edit waiting on "Replace your Synth edits?"
    int randomPreset = -1;                // the Random real preset start, for Reset
    int shownBase = 0, shownCategory = 0; // the combos at the last decode, put back by Keep editing
    int highlighted = -1;                 // the category the cloud highlights (visual only)
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

    juce::TextButton latentButton { "Latent pad" }, knobButton { "Knob pad" }, lockButton { "Lock structure" },
        resetButton { "Reset" }, sendButton { "Send to Surge" };
    juce::ComboBox base, category;
    std::array<CategoryRow, 8> categoryRows;
    Spectrum spectrum;
    LabOrbit orbit;
    LabPad pad;
    juce::Viewport listView;
    LabAxisList list;
    juce::Label status;
    juce::TextEditor promptBox;
    juce::TextButton promptGo { ">" };
    juce::Label experimentalTag { {}, "experimental" };
    // R21: the Replace question sits in the Lab, not in a top-level window a host can open behind the plug-in
    juce::Label askNote { {},
                          "Replace your Synth edits? The Lab's sound replaces the changes made in the Synth view." };
    juce::TextButton askReplace { "Replace" }, askKeep { "Keep editing" };
    juce::TextButton findButton { "Find sounds" }, nudgeButton { "Word list" };
    std::array<juce::TextButton, 8> chips;
    PitcherKeyboard keyboard;
    PitcherLayout layout;
    juce::Rectangle<int> categoryArea, askArea, axesTab;
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
