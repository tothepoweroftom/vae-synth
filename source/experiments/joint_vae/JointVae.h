#pragma once

#include "StudentCodec.h"
#include "SurgePatchBuilder.h"
#include "XyPad.h"
#include "ml/OnnxModel.h"
#include "schema/ControlSchema.h"
#include <array>
#include <atomic>
#include <juce_events/juce_events.h>
#include <optional>

namespace vae::experimental
{
struct JointLayout
{
    juce::String id;
    juce::StringArray names;
    int numNumeric = 0; // index of the first "bool" feature: bools, categoricals and stepped values follow
    static juce::Result parse (const juce::var&, JointLayout&);
};

// Synchronous, worker-thread only. Failed loads preserve the previous bundle.
class JointVae
{
public:
    juce::Result load (const juce::File& manifest);
    juce::Result decode (const JointLatent&, int category, std::vector<float>& codes);
    /** Latent + category -> 48-number sound code (sound_decoder.onnx). */
    juce::Result decodeSound (const JointLatent&, int category, SoundCode&);
    juce::Result encode (const SoundCode&, int category, JointLatent&);
    /** Decoded codes -> full Surge patch data (student codec + template). Needs a VAE_WITH_SURGE build.
        `plan`, when given, receives the codec plan even when the build fails. */
    juce::Result patch (const std::vector<float>& codes, juce::MemoryBlock&, StudentPlan* plan = nullptr);
    const JointLayout& layout() const { return metadata; }
    const XyPadData& xypad() const { return xy; }
    const CategoryForest& forest() const { return categoryForest; }
    const KnobGroups& knobGroups() const { return groups; }
    const SoundPca& soundPca() const { return pca; }
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
    std::unique_ptr<ml::OnnxModel> decoder, encoder, soundDecoder;
    StudentCodec codec;
    XyPadData xy;
    CategoryForest categoryForest;
    KnobGroups groups;
    SoundPca pca;
#if VAE_WITH_SURGE
    std::unique_ptr<SurgePatchBuilder> builder; // created on first use: loads Surge's data folder
#endif
};

struct LabRequest
{
    JointLatent z {};
    int category = 8;
    std::optional<std::vector<float>> lockedTail; // codes[layout().numNumeric..]
    std::optional<std::vector<float>> codes;      /* knob pad: skip the decoder */
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
        schema::Snapshot lite = schema::Snapshot::defaults(); // from `patch`; defaults when patchResult failed
        SoundCode sound {};
        std::vector<float> mel;   // melFromSound (frames x 64 dB)
        std::vector<float> knobs; // KnobGroups::names order; NaN when the plan writes no Value for it
    };
    using Callback = std::function<void (juce::Result, Output)>;
    ~JointVaeWorker();
    void load (juce::File, Callback);
    void cancel() { ++state->request; }
    void decode (JointLatent, int category, Callback);
    void encode (SoundCode, int category, Callback);
    /** decode (unless codes) -> apply lockedTail -> plan -> patch -> lite + sound. */
    void lab (LabRequest, Callback);

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
