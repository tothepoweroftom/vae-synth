#pragma once
#include "OnnxModel.h"
#include "StableAudioGenerator.h"
#include "schema/ControlSchema.h"
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_events/juce_events.h>
#include <array>
#include <atomic>

namespace vae::ml
{
// Where scripts/install-models.sh puts the assets; loaded on first use unless a picker chose others.
inline juce::File installedModels (const char* relative) { return installedModelFile (relative); }

// All methods allocate and must run off the audio thread.
class SameSearch
{
public:
    static constexpr int samples = 176400, dimensions = 512;
    using Embedding = std::array<float, dimensions>;
    struct Match
    {
        juce::String name;
        schema::Snapshot controls;
        float distance;
        int note;
    };
    juce::Result load (const juce::File& bank);
    bool isLoaded() const noexcept { return model != nullptr; }
    juce::Result search (const juce::File& audio, std::vector<Match>& matches);
    static juce::Result preprocess (const juce::File&, std::vector<float>& stereo);
    static juce::Result pool (std::span<const float> latents, Embedding&);
    juce::Result encode (std::span<const float> stereo, Embedding&);

private:
    struct Entry
    {
        Match match;
        Embedding embedding;
    };
    std::unique_ptr<OnnxModel> model;
    std::vector<Entry> entries;
};

class SameSearchWorker
{
public:
    using Callback = std::function<void (juce::Result, std::vector<SameSearch::Match>)>;
    SameSearchWorker() = default;
    ~SameSearchWorker();
    void load (const juce::File&, Callback);
    void search (const juce::File&, Callback);
    void loadGenerator (const juce::File&, Callback);
    void generate (const juce::String&, uint32_t seed, const juce::File& output, Callback,
                   std::function<void (juce::String)> progress = {});
    /** Load the installed bank and generator (no-op for what is already loaded). */
    void preload (Callback, std::function<void (juce::String)> progress);
    void cancel() { ++state->request; }
    bool isBusy() const noexcept { return state->busy.load() > 0; }

private:
    struct State
    {
        SameSearch search;
        StableAudioGenerator generator;
        std::atomic<uint64_t> request { 0 };
        std::atomic<int> busy { 0 };
    };
    using Notify = std::function<void (juce::String)>;
    static Notify notifier (std::shared_ptr<State>, uint64_t id, Notify progress);
    static juce::Result loadInstalled (State&, bool withGenerator, const Notify&);
    void submit (juce::File, bool loadBank, Callback);
    using Work = std::function<juce::Result (State&, uint64_t, std::vector<SameSearch::Match>&)>;
    void enqueue (Work, Callback);
    std::shared_ptr<State> state = std::make_shared<State>();
    juce::ThreadPool worker { juce::ThreadPoolOptions {}.withThreadName ("vae-same").withNumberOfThreads (1) };
};
} // namespace vae::ml
