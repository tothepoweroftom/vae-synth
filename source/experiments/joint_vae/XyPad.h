#pragma once

#include "JointVae.h"
#include <map>
#include <span>
#include <utility>
#include <vector>

namespace vae::experimental
{
// Port of the student's XY-pad page maths (vae/pitcher/xypad/page_v2.template.html). Engine-free.
struct LatentAxis
{
    juce::String id, kind /* "sound"|"dim" */, name, low, high, desc, key;
    JointLatent u {};
    float mean = 0, sd = 1, impact = 0;
    bool minor = false;
    std::vector<std::pair<juce::String, float>> sections;
};

struct XyPadData
{
    std::vector<LatentAxis> axes;                // sound axes first, then dims by impact (page order)
    std::map<juce::String, JointLatent> typical; // label name -> mean_z
    std::vector<JointLatent> presetZ;
    std::vector<int> presetLabel;
    juce::StringArray presetName;
    std::vector<float> bandHz;
    float dt = 0, f0 = 0;
    int holdFrames = 0;

    /** Fails without touching `out`. */
    static juce::Result load (const juce::File& xypadJson, XyPadData& out);
    const LatentAxis* find (const juce::String& id) const;
};

float axisTarget (const LatentAxis&, float pos);            // page tOf
float axisPosition (const LatentAxis&, const JointLatent&); // page posOf
/** Smallest change of z0 so that each axis reads its target. Dependent axes are dropped, not divided by. */
JointLatent constrain (const JointLatent& z0, std::span<const LatentAxis* const> axes, std::span<const float> targets);
} // namespace vae::experimental
