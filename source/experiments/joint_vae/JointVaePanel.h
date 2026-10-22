#pragma once
#include "JointVae.h"
#include "LabAxisList.h"
#include "LabOrbit.h"
#include "LabPad.h"
#include "LabTheme.h"
#include "LatentOrbit.h"
#include "PitcherControls.h"
#include "PitcherLayout.h"
#include "PromptModel.h"
#include "PromptWords.h"
#include <juce_gui_basics/juce_gui_basics.h>

namespace vae::experimental
{
/** Task 22: hides the Lab's control group (Latent pad / Knob pad, Start from, Category, Lock structure, Reset,
    Send to Surge) and the prompt's Find sounds / Word list switch. They still work (tests drive them); true shows
    them again. */
inline constexpr bool kShowLabControls = false;

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
    /** The top of the left column that the Lab leaves to the logo (36 px) and the editor's view tabs, which sit
        over the panel (Task 22; Task 23 removed the Init and engine row under them). */
    static constexpr int kLeftTop = 90;
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
    /** Task 23: the saved presets as the processor stores them (the editor sets them on open), and their changes
        (message thread). */
    void setPresets (const juce::XmlElement&);
    std::function<void (const juce::XmlElement&)> onPresetsChanged;
#if VAE_WITH_SURGE
    static constexpr int kMaxPresets = 64;
    /** Save preset: opens the inline name field, prefilled with the last prompt or the category guess. */
    void startSaving();
    juce::String nameFieldText() const { return nameBox.getText(); }
    bool isNaming() const { return naming != -2; }
    /** The name field's Save: a new preset (or the rename it was opened for); an empty name is refused. */
    bool commitName (const juce::String&);
    int numPresets() const { return (int) presets.size(); }
    juce::String presetNameAt (int i) const { return presets[(size_t) i].name; }
    const std::vector<float>& presetCodes (int i) const { return presets[(size_t) i].codes; }
    bool presetPlayable (int) const; // saved with the loaded bundle
    void loadPreset (int);           // through the edit gate; plays the stored codes exactly
    void renamePreset (int);         // opens the name field on it
    void deletePreset (int);         // asks first
    void showPresetsTab (bool);
    /** A Delete question (the inline banner, as Replace); `answer` is its buttons. */
    bool isAsking() const { return pendingAnswer != nullptr; }
    void answer (bool yes);
    /** Tune space to my presets: a dummy that only shows progress for 10 s (trains nothing, writes nothing). */
    void startTuneDemo();
    void closeTuneDemo();
    bool isTuning() const { return tune.isVisible(); }
    double tuneProgress() const { return tune.progress; }
    juce::String tuneText() const { return tune.text(); }
    void fastForwardTune (double seconds); // test hook
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
    bool isOrbiting() const { return orbitSettings.on; }
    const std::array<float, 3>& orbitCentre() const { return centre; } // the orbit LFO's centre, pad units
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
    void decode (bool orbitStep = false);
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
    void setStatus (const juce::String&, bool error = false, bool map = false); // map: a knob-map failure
    const LatentAxis* axisOn (int slot) const;
    int groupOn (int slot) const; // knob pad: group index on the slot, -1 when none
    bool hasSlot (int slot) const { return knobMode ? groupOn (slot) >= 0 : axisOn (slot) != nullptr; }
    void setKnobMode (bool); // page mode switch: the decoded codes stay, the other pad's edits are dropped
    void edit (std::function<void()> change); // a Lab sound edit: asks first while the sound is edited
    void resolveReplace (bool replace);
    void showAsk (bool); // the inline Replace (or Delete) question
    void ask (const juce::String& note, const juce::String& yes, std::function<void()> onYes);
    void applyPreset (int);
    void presetsChanged (bool notify = true); // rebuilds the rows and (notify) tells the editor
    void layOutPresets();
    void endNaming();
    void tuneTick();
    void applyCandidate (int);
    void leaveBank();           // the first sound edit after selecting a candidate decodes from its z
    void clearPromptProgress(); // a dropped Find must not leave "Finding sounds..." behind
    void updatePromptUi();
    // The orbit LFO (Task 19): a 30 Hz tick moves the handle around `centre` and decodes like a drag
    void startOrbit();
    void stopOrbit (bool play); // back to the centre; play: decode it (false when the caller decodes next)
    void sendToSurge();
    void recentre(); // a drag, nudge or Find pick moves the centre to the new point
    void orbitTick();
    void showOrbitPath();

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
    std::function<void()> pendingAnswer;  // the Yes of a Delete preset question
    int randomPreset = -1;                // the Random real preset start, for Reset
    int shownBase = 0, shownCategory = 0; // the combos at the last decode, put back by Keep editing
    int highlighted = -1;                 // the category the cloud highlights (visual only)
    bool errorShown = false;
    bool mapError = false; // the status line shows a map failure
    struct StatusText
    {
        juce::String text;
        bool error = false, map = false;
    };
    std::optional<StatusText> heldStatus; // the latest status written while the Replace question is open
    JointLatent z {};
    std::array<juce::String, 3> axes;
    std::array<float, 3> pos { 0.5f, 0.5f, 0.5f };
    std::optional<std::vector<float>> lockedTail;
    // Knob pad: flat 2D pad over the 61 knob groups (ruling R16). `axes` then holds group names.
    bool knobMode = false;
    std::array<juce::String, 3> latAxes, knobAxes;
    std::vector<float> knobVals, knobCodes; // the 123 knob values and the codes they are written into
    // Orbit LFO: the centre's z and pad positions; its decoded Lite values hold every discrete control
    OrbitSettings orbitSettings;
    JointLatent centreZ {};
    std::array<float, 3> centre {};
    std::optional<JointVaeWorker::Output> centreOut; // empty until the centre's decode returns: ticks wait
    bool sendAfterCentre = false;                    // Send to Surge during an orbit: send the centre once decoded
    bool lockedByOrbit = false;                      // Lock was off before the orbit forced it on
    bool orbitStepPending = false;                   // one orbit decode at a time, so none is superseded unplayed
    double orbitStart = 0;                           // ms, the phase origin
    double centreTime = 0;                           // ms, the last start or re-centre: the radius ramps in from it

    juce::TextButton latentButton { "Latent pad" }, knobButton { "Knob pad" }, lockButton { "Lock structure" },
        resetButton { "Reset" }, sendButton { "Send to Surge" }, orbitButton { "Orbit LFO" };
    juce::Slider orbitRate, orbitDepth;
    juce::Label orbitRateLabel, orbitDepthLabel;
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
    // R21: the Replace question sits in the Lab, not in a top-level window a host can open behind the plug-in
    juce::Label askNote { {},
                          "Replace your Synth edits? The Lab's sound replaces the changes made in the Synth view." };
    juce::TextButton askReplace { "Replace" }, askKeep { "Keep editing" };
    juce::TextButton findButton { "Find sounds" }, nudgeButton { "Word list" };
    std::array<juce::TextButton, 8> chips;
    PitcherKeyboard keyboard;
    PitcherLayout layout;
    juce::Rectangle<int> categoryArea, askArea;

    // Presets (Task 23): the right column's Save preset, the name field and the Axes | Presets tabs
    struct Preset
    {
        juce::String name, bundle;
        int category = 8;
        JointLatent z {};
        std::array<juce::String, 3> axes;
        std::array<float, 3> pos {};
        std::vector<float> codes; // bundle feature order, NaN = inactive
    };
    struct PresetRow final : juce::Component
    {
        PresetRow()
        {
            for (auto* b : { &load, &rename, &remove })
                addAndMakeVisible (b);
        }
        juce::TextButton load, rename { "Rename" }, remove { "Delete" };
        void resized() override;
    };
    /** The demo's dialog: a title, the demo line, a progress bar and Close. */
    struct TuneOverlay final : juce::Component
    {
        TuneOverlay();
        void paint (juce::Graphics&) override;
        void resized() override;
        juce::String text() const;
        juce::TextButton close { "Close" };
        double progress = 0;
    };
    std::vector<Preset> presets;
    int naming = -2; // the name field: -2 closed, -1 a new preset, else the preset it renames
    juce::TextButton saveButton { "Save preset" }, nameSave { "Save" }, nameCancel { "Cancel" },
        axesTabButton { "AXES" }, presetsTabButton { "PRESETS" }, tuneButton { "Tune space to my presets" };
    juce::TextEditor nameBox;
    juce::Viewport presetView;
    juce::Component presetList;
    juce::OwnedArray<PresetRow> presetRows;
    juce::Label presetHint { {}, "No presets yet: Save preset keeps the current sound here." };
    TuneOverlay tune;
    double tuneStart = 0, tuneSkip = 0; // ms
    static constexpr double kTuneMs = 10000;
    juce::TimedCallback mapTimer { [this]
                                   {
                                       mapTimer.stopTimer();
                                       requestMap();
                                   } };
    juce::TimedCallback orbitTimer { [this] { orbitTick(); } };
    juce::TimedCallback tuneTimer { [this] { tuneTick(); } };
#else
    juce::Label note;
#endif
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (JointVaePanel)
};
} // namespace vae::experimental
