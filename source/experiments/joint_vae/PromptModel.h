#pragma once

#include "JointVae.h"
#include <span>

namespace vae::experimental
{
struct PromptCandidate
{
    juce::String bundleId;
    int row;
    int category;
    JointLatent z;
    std::vector<float> codes; // Lab feature order; NaN = inactive
    juce::String name;
};

// p0/p1 prompt bundle (prompt.json): text -> CLAP text encoder -> prior -> nearest bank presets.
// Synchronous, worker-thread only. Failed loads keep the previous model.
class PromptModel
{
public:
    PromptModel();
    ~PromptModel();
    juce::Result load (const juce::File& promptJson, const JointLayout& lab); // validates against lab.id and names
    juce::Result load (const juce::File& promptJson, const JointVae& lab) { return load (promptJson, lab.layout()); }
    juce::Result find (const juce::String& text, std::vector<PromptCandidate>& out, int& category);

    /** Means with weight >= minWeight, highest first (ties: component index); each takes the nearest unused row of
        `category` by squared distance on z (ties: row index). At most maxResults rows. */
    static std::vector<int> select (std::span<const float> weights, std::span<const float> mu,
                                    std::span<const float> bankZ, std::span<const int> bankCategory, int category,
                                    float minWeight, int maxResults);

private:
    friend class PromptModelTests;
    /** Intermediate values, for parity tests. */
    struct Trace
    {
        std::vector<int64_t> ids, mask;
        std::vector<float> embedding, logits, weights, mu; // mu: k x 32
    };
    /** Test-only seam: forceCategory >= 0 replaces the argmax, so tests reach categories a prompt rarely predicts. */
    juce::Result find (const juce::String& text, std::vector<PromptCandidate>& out, int& category, int forceCategory,
                       Trace* trace);
    struct Model;
    std::unique_ptr<Model> model;
};

// The Lab's prompt search on its own thread (ruling R17), so a slow prompt never delays pad decoding.
// Latest request wins; callbacks run on the message thread.
class PromptWorker
{
public:
    using Found = std::function<void (juce::Result, int category, std::vector<PromptCandidate>)>;
    ~PromptWorker();
    /** A newer load replaces a queued one; a failed load keeps the previous model. */
    void load (juce::File promptJson, JointLayout lab, std::function<void (juce::Result)>);
    void find (juce::String text, Found);

private:
    struct State
    {
        PromptModel model;
        std::atomic<uint64_t> request { 0 }, loads { 0 };
    };
    std::shared_ptr<State> state = std::make_shared<State>();
    juce::ThreadPool pool { juce::ThreadPoolOptions {}.withThreadName ("vae-prompt").withNumberOfThreads (1) };
};
} // namespace vae::experimental
