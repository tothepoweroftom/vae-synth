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
#include "params/ParamStore.h"

#include <array>
#include <atomic>
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
    ml::PresetGenerator& getGenerator() noexcept { return generator; }

    /** Engines compiled into this build, in "engine" choice order. */
    const std::vector<EngineKind>& getEngineKinds() const noexcept { return engineKinds; }

    /** Current schema vector as the host sees it (message thread). */
    schema::Snapshot getControls() const;

    /** Apply a whole vector as one host gesture per control (message thread). */
    void applyControls (const schema::Snapshot&);

    /** Prompt -> preset. `done` (message thread) gets the result note. */
    void generateFromPrompt (const juce::String& prompt, std::function<void (const juce::String& note)> done);
    void generateRandom (std::function<void (const juce::String& note)> done);

    /** Load a decoder model (empty file = embedded stub). The model path is
        recorded (and saved with the state) only when the load succeeds; a
        rejected file leaves the previous decoder in use. */
    void loadDecoderModel (const juce::File& file, std::function<void (juce::Result)> done);
    juce::File getDecoderModelFile() const;
    juce::String getLastPrompt() const;

    /** Generator status, plus a note when the saved model file is missing. */
    juce::String describeModel() const;

private:
    void selectEngine (int choiceIndex);   // audio thread, from dispatch()
    void requestModel (const juce::File& toLoad, const juce::File& toRecord, bool missing,
                       std::function<void (juce::Result)> done);

    std::vector<EngineKind> engineKinds;
    std::vector<std::unique_ptr<SynthEngine>> engines;
    std::atomic<int> requestedEngine { 0 };
    SynthEngine* activeEngine = nullptr;   // audio thread only

    // Last values delivered on the audio thread, so a newly selected engine
    // can be brought up to date.
    std::array<float, schema::kControlCount> audioControls {};
    std::array<params::ParamIndex, schema::kControlCount> controlIndex {};

    params::ParamStore params;
    juce::MidiKeyboardState keyboardState;

    // Read by the host's state save on any thread; written on the message thread.
    mutable juce::CriticalSection stateLock;
    juce::File modelPath;          // saved with the state; may point at a missing file
    bool modelMissing = false;     // modelPath did not exist when the state was loaded
    juce::String lastPrompt;
    std::atomic<uint64_t> modelRequest { 0 };   // latest loadDecoderModel / state load wins

    ml::PresetGenerator generator;
    juce::Random random;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (PluginProcessor)
};

} // namespace vae
