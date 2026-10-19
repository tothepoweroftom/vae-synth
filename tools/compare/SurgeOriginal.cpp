#include "SurgeOriginal.h"

#include <memory>

// Surge's headers are not written for JUCE's recommended warning set.
JUCE_BEGIN_IGNORE_WARNINGS_GCC_LIKE ("-Wsign-conversion", "-Wfloat-equal", "-Wpedantic", "-Wshadow", "-Wconversion",
                                     "-Wfloat-conversion", "-Wold-style-cast", "-Wzero-as-null-pointer-constant",
                                     "-Wunused-parameter", "-Wswitch-enum", "-Wimplicit-int-float-conversion",
                                     "-Wshorten-64-to-32", "-Wextra-semi", "-Wcast-align", "-Wunused-value",
                                     "-Wunused-variable", "-Wunused-local-typedefs", "-Wstrict-aliasing",
                                     "-Wsign-compare")
#include "SurgeSynthesizer.h"
JUCE_END_IGNORE_WARNINGS_GCC_LIKE

namespace vae::compare
{
namespace
{
struct SilentPluginLayer final : SurgeSynthesizer::PluginLayer
{
    void surgeParameterUpdated (const SurgeSynthesizer::ID&, float) override {}
    void surgeMacroUpdated (long, float) override {}
};
} // namespace

std::optional<std::vector<float>> renderOriginal (const juce::File& fxp, double sampleRate,
                                                  const std::vector<ProbeEvent>& events, int numSamples)
{
    SilentPluginLayer layer;
    // Heap: SurgeSynthesizer is far too large for a worker thread's stack.
    auto owned = std::make_unique<SurgeSynthesizer> (&layer, SurgeStorage::skipPatchLoadDataPathSentinel);
    auto& synth = *owned;
    synth.setSamplerate (float (sampleRate));
    synth.time_data.tempo = 120.0;
    synth.process();

    if (! synth.loadPatchByPath (fxp.getFullPathName().toRawUTF8(), -1, fxp.getFileNameWithoutExtension().toRawUTF8()))
        return std::nullopt;
    for (int i = 0; i < 8; ++i)
        synth.process();
    synth.allNotesOff();
    synth.stopSound();
    for (int i = 0; i < 8; ++i)
        synth.process();

    // MIDI at 32-sample quantum boundaries, like the Surge engine adapter.
    std::vector<float> out ((size_t) numSamples);
    size_t next = 0;
    for (int pos = 0; pos < numSamples; pos += BLOCK_SIZE)
    {
        for (; next < events.size() && events[next].sample < pos + BLOCK_SIZE; ++next)
        {
            const auto& e = events[next];
            if (e.velocity > 0)
                synth.playNote (0, char (e.note), char (e.velocity), 0);
            else
                synth.releaseNote (0, char (e.note), 0);
        }
        synth.process();
        for (int i = 0; i < BLOCK_SIZE && pos + i < numSamples; ++i)
            out[(size_t) (pos + i)] = 0.5f * (synth.output[0][i] + synth.output[1][i]);
    }
    return out;
}

} // namespace vae::compare
