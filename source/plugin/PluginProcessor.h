/*
  The plug-in: host parameters are exactly the schema controls (stable,
  model-facing ids), plus one hidden "engine" choice. Parameter changes reach
  the active engine on the audio thread through ParamStore::dispatch().
  Prompt -> preset generation runs off the audio thread (ml::PresetGenerator)
  and lands as ordinary host-notified parameter changes, so automation, undo
  and the UI all see it.
*/

#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_audio_utils/juce_audio_utils.h>

#include "engine/SynthEngine.h"
#include "ml/PresetGenerator.h"
#include "ml/SameSearch.h"
#include "params/ParamStore.h"

#include <array>
#include <atomic>
#include <optional>
#include <vector>

namespace vae
{

class PluginProcessor final : public juce::AudioProcessor
{
public:
    static constexpr const char* kEngineParamId = "engine";

    PluginProcessor();
    ~PluginProcessor() override;

    //==========================================================================
    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    bool isBusesLayoutSupported (const BusesLayout&) const override;
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;
    using AudioProcessor::processBlock;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return JucePlugin_Name; }
    bool acceptsMidi() const override { return true; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int) override {}
    const juce::String getProgramName (int) override { return {}; }
    void changeProgramName (int, const juce::String&) override {}

    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

    //==========================================================================
    params::ParamStore& getParams() noexcept { return params; }
    juce::MidiKeyboardState& getKeyboardState() noexcept { return keyboardState; }
    ml::SameSearchWorker& getReferenceSearch() noexcept { return referenceSearch; }
    ml::PresetGenerator& getGenerator() noexcept { return generator; }

    /** Engines compiled into this build, in "engine" choice order. */
    const std::vector<EngineKind>& getEngineKinds() const noexcept { return engineKinds; }

    /** Current schema vector as the host sees it (message thread). */
    schema::Snapshot getControls() const;

    /** Play a full Surge patch (joint VAE experiment) on the Surge engine and select it
        (message thread). The next applyControls returns that engine to the schema vector. */
    juce::Result loadSurgePatch (const juce::MemoryBlock& patch);

    /** Lab playback (message thread): selects the Lite engine, leaves Surge patch mode and sets the
        controls that changed without host gestures. The host is still told about each change, so a host
        in write or latch mode can record pad moves as automation. */
    void playLiteControls (const schema::Snapshot&);

    /** True while the Surge engine plays a full patch from loadSurgePatch. */
    bool isPlayingSurgePatch() const noexcept;

    /** The Lab/Synth gate (Task 16), kept here so it survives closing the editor (message thread,
        session only). markLabControls records the current controls as the Lab's: after its write, or when
        the user chose Replace. editedSinceLab: the controls differ from that record (from the defaults
        before the first one), so something other than the Lab changed the sound. */
    void markLabControls() { labControls = getControls(); }
    bool editedSinceLab() const;

    /** The engine of that kind, or nullptr when it is not in this build. */
    SynthEngine* findEngine (EngineKind) const noexcept;

    /** Apply a whole vector as one host gesture per control (message thread). */
    void applyControls (const schema::Snapshot&);

    /** Load a versioned retrieval preset on the message thread; reject before changing parameters. */
    juce::Result loadPresetFile (const juce::File&);

    /** Prompt -> preset. `done` (message thread) gets the result note. */
    void generateFromPrompt (const juce::String& prompt, std::function<void (const juce::String& note)> done);
    void generateRandom (std::function<void (const juce::String& note)> done);

    /** Load a decoder model (default-constructed File = embedded stub). The model path is
        recorded (and saved with the state) only when the load succeeds; a
        rejected file leaves the previous decoder in use. */
    void loadDecoderModel (const juce::File& file, std::function<void (juce::Result)> done);
    juce::File getDecoderModelFile() const;
    juce::String getLastPrompt() const;
    void setLastPrompt (const juce::String& prompt);
    /** The editor's scale (Task 22): saved with the state, not a host parameter. 0 until an editor sets it. */
    float getEditorScale() const;
    void setEditorScale (float);

    /** The Lab's saved presets (Task 23): an opaque element the Lab writes and reads, saved with the state as a
        child element. Message thread. */
    juce::XmlElement getLabPresets() const;
    void setLabPresets (const juce::XmlElement&);

    /** Generator status, plus a note when the saved model file is missing. */
    juce::String describeModel() const;

private:
    void selectEngine (int choiceIndex); // audio thread, from dispatch()
    void requestModel (const juce::File& toLoad, const juce::File& toRecord, bool missing,
                       std::function<void (juce::Result)> done);

    std::vector<EngineKind> engineKinds;
    std::vector<std::unique_ptr<SynthEngine>> engines;
    std::atomic<int> requestedEngine { 0 };
    SynthEngine* activeEngine = nullptr; // audio thread only

    // Last values delivered on the audio thread, so a newly selected engine
    // can be brought up to date.
    std::array<float, schema::kControlCount> audioControls {};
    std::array<params::ParamIndex, schema::kControlCount> controlIndex {};

    params::ParamStore params;
    juce::MidiKeyboardState keyboardState;

    // Read by the host's state save on any thread; written on the message thread.
    mutable juce::CriticalSection stateLock;
    juce::File modelPath;      // saved with the state; may point at a missing file
    bool modelMissing = false; // modelPath did not exist when the state was loaded
    juce::String lastPrompt;
    float editorScale = 0.0f;
    juce::XmlElement labPresets { "LabPresets" };
    std::atomic<uint64_t> modelRequest { 0 }; // latest loadDecoderModel / state load wins

    std::optional<schema::Snapshot> labControls; // message thread

    ml::PresetGenerator generator;
    ml::SameSearchWorker referenceSearch;
    juce::Random random;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (PluginProcessor)
};

} // namespace vae
