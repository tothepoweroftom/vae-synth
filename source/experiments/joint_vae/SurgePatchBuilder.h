#pragma once

#include "StudentCodec.h"
#include <memory>

namespace vae::experimental
{
// Executes a StudentPlan the way the student's SurgeIO.write_plan does (vae/pitcher/core/surge_io.py),
// on a private headless Surge loaded with the template, and returns the patch as Surge patch data
// (wavetables embedded) for SurgeEngine::loadPatchData. Needs Surge's factory data folder for the
// template and wavetables. Not thread-safe: use from one worker thread.
class SurgePatchBuilder
{
public:
    explicit SurgePatchBuilder (const juce::File& surgeDataDir);
    ~SurgePatchBuilder();

    juce::Result build (const StudentPlan&, juce::MemoryBlock& patch);

    /** Value of a patch path in natural units after the last build (NaN if unknown), for tests. */
    float value (const juce::String& path) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
} // namespace vae::experimental
