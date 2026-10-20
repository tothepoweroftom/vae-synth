// Renders an unmodified Surge XT preset (.fxp) through headless Surge.
#pragma once

#include <juce_core/juce_core.h>

#include <optional>
#include <vector>

namespace vae::compare
{
/** A note event of the probe: velocity 0 = note off. */
struct ProbeEvent
{
    int sample;
    int note;
    int velocity;
};

/** Mono render of the preset playing `events` (sorted by sample), or nullopt
    when Surge cannot load the file. */
std::optional<std::vector<float>> renderOriginal (const juce::File& fxp, double sampleRate,
                                                  const std::vector<ProbeEvent>& events, int numSamples);
} // namespace vae::compare
