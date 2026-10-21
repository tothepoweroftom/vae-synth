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

// p0 prompt bundle (prompt.json): text -> CLAP text encoder -> prior -> nearest bank presets.
// Synchronous, worker-thread only. Failed loads keep the previous model.
class PromptModel
{
public:
    PromptModel();
    ~PromptModel();
    juce::Result load (const juce::File& promptJson, const JointVae& lab); // validates against lab.layout().id
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
    /** Test-only seam: forceCategory >= 0 replaces the argmax (p0 never predicts 3..7). */
    juce::Result find (const juce::String& text, std::vector<PromptCandidate>& out, int& category, int forceCategory,
                       Trace* trace);
    struct Model;
    std::unique_ptr<Model> model;
};
} // namespace vae::experimental
