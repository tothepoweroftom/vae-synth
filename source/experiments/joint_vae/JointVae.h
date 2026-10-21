#pragma once

#include "StudentCodec.h"
#include "SurgePatchBuilder.h"
#include "ml/OnnxModel.h"
#include <array>
#include <atomic>
#include <juce_events/juce_events.h>

namespace vae::experimental
{
// Independent of ControlSchema: these are the student's full Surge features.
using JointLatent = std::array<float, 32>;
using SoundCode = std::array<float, 48>;
inline constexpr const char* jointLabels[] = { "Bass",  "Lead",       "Pad",        "Keys",   "Pluck",
                                               "Synth", "Atmosphere", "Orchestral", "Unknown" };

struct JointLayout
{
    juce::String id;
    juce::StringArray names;
    static juce::Result parse (const juce::var&, JointLayout&);
};

// Synchronous, worker-thread only. Failed loads preserve the previous bundle.
class JointVae
{
public:
    juce::Result load (const juce::File& manifest);
    juce::Result decode (const JointLatent&, int category, std::vector<float>& codes);
    juce::Result encode (const SoundCode&, int category, JointLatent&);
    /** Decoded codes -> full Surge patch data (student codec + template). Needs a VAE_WITH_SURGE build. */
    juce::Result patch (const std::vector<float>& codes, juce::MemoryBlock&);
    const JointLayout& layout() const { return metadata; }
    static juce::Result slice (const JointLatent& anchor, int xDim, int yDim, float x, float y, float radius,
                               JointLatent& result);

    /** Label demo: the category whose noun from the student's prompt vocabulary
        (vae/pitcher/text2preset/vocabulary_v2.yaml) appears in the text, longest match first.
        Unknown (8) with an empty noun when none does. */
    struct LabelMatch
    {
        int category;
        juce::String noun;
    };
    static LabelMatch labelFor (const juce::String& prompt);

private:
    JointLayout metadata;
    std::unique_ptr<ml::OnnxModel> decoder, encoder;
    StudentCodec codec;
#if VAE_WITH_SURGE
    std::unique_ptr<SurgePatchBuilder> builder; // created on first use: loads Surge's data folder
#endif
};

// Owns model lifetime on a serial worker. UI callbacks run on the message thread.
// No engine or parameter-store dependency, and no mock/fallback output.
class JointVaeWorker
{
public:
    struct Output
    {
        JointLayout layout;
        JointLatent latent {};
        std::vector<float> codes;
        juce::MemoryBlock patch; // empty when patchResult failed
        juce::Result patchResult = juce::Result::ok();
    };
    using Callback = std::function<void (juce::Result, Output)>;
    ~JointVaeWorker();
    void load (juce::File, Callback);
    void cancel() { ++state->request; }
    void decode (JointLatent, int category, Callback);
    void encode (SoundCode, int category, Callback);

private:
    struct State
    {
        JointVae model;
        std::atomic<uint64_t> request { 0 };
    };
    using Work = std::function<juce::Result (JointVae&, Output&)>;
    void submit (Work, Callback);
    std::shared_ptr<State> state = std::make_shared<State>();
    juce::ThreadPool pool { juce::ThreadPoolOptions {}.withThreadName ("vae-joint").withNumberOfThreads (1) };
};
} // namespace vae::experimental
