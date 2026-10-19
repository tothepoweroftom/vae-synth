#include "SurgeEngine.h"

// Surge's headers are not written for JUCE's recommended warning set.
JUCE_BEGIN_IGNORE_WARNINGS_GCC_LIKE ("-Wsign-conversion", "-Wfloat-equal", "-Wpedantic", "-Wshadow",
                                     "-Wconversion", "-Wfloat-conversion", "-Wold-style-cast",
                                     "-Wzero-as-null-pointer-constant", "-Wunused-parameter",
                                     "-Wswitch-enum", "-Wimplicit-int-float-conversion",
                                     "-Wshorten-64-to-32", "-Wextra-semi", "-Wcast-align", "-Wunused-value",
                                     "-Wunused-variable", "-Wunused-local-typedefs", "-Wstrict-aliasing",
                                     "-Wsign-compare")
JUCE_BEGIN_IGNORE_WARNINGS_MSVC (4244 4267 4100 4456 4458)
#include "SurgeSynthesizer.h"
JUCE_END_IGNORE_WARNINGS_MSVC
JUCE_END_IGNORE_WARNINGS_GCC_LIKE

#include <array>
#include <cstring>
#include <optional>

namespace vae
{
namespace
{
struct SilentPluginLayer final : SurgeSynthesizer::PluginLayer
{
    void surgeParameterUpdated (const SurgeSynthesizer::ID&, float) override {}
    void surgeMacroUpdated (long, float) override {}
};

constexpr int kQuantum = BLOCK_SIZE;
constexpr int kMaxSynthSideIds = 2048;
constexpr char kChannel = 0;
} // namespace

struct SurgeEngine::Impl
{
    SilentPluginLayer layer;
    std::unique_ptr<SurgeSynthesizer> synth;
    std::array<std::optional<SurgeSynthesizer::ID>, schema::kControlCount> ids {};
    schema::Snapshot controls = schema::Snapshot::defaults();
    schema::Snapshot written = schema::Snapshot::defaults();

    std::array<float, kQuantum> quantumL {}, quantumR {};
    int quantumPos = kQuantum;

    void resolveControls()
    {
        // Schema v2 placeholder: continuous controls that map 1:1 onto a Surge
        // parameter, by storage name. The full mapping replaces this.
        static const std::pair<int, const char*> kDirect[] = {
            { schema::kGlideTime, "a_portamento" },
            { schema::kOsc1Level, "a_level_o1" }, { schema::kOsc1Pitch, "a_osc1_pitch" },
            { schema::kOsc2Level, "a_level_o2" }, { schema::kOsc2Pitch, "a_osc2_pitch" },
            { schema::kOsc3Level, "a_level_o3" }, { schema::kOsc3Pitch, "a_osc3_pitch" },
            { schema::kFilter1Cutoff, "a_filter1_cutoff" }, { schema::kFilter1Resonance, "a_filter1_resonance" },
            { schema::kFilter1Env, "a_filter1_envmod" }, { schema::kFilter1Keytrack, "a_filter1_keytrack" },
            { schema::kFilter2Cutoff, "a_filter2_cutoff" }, { schema::kFilter2Resonance, "a_filter2_resonance" },
            { schema::kFilter2Env, "a_filter2_envmod" }, { schema::kFilter2Keytrack, "a_filter2_keytrack" },
            { schema::kAmpAttack, "a_env1_attack" }, { schema::kAmpDecay, "a_env1_decay" },
            { schema::kAmpSustain, "a_env1_sustain" }, { schema::kAmpRelease, "a_env1_release" },
            { schema::kFegAttack, "a_env2_attack" }, { schema::kFegDecay, "a_env2_decay" },
            { schema::kFegSustain, "a_env2_sustain" }, { schema::kFegRelease, "a_env2_release" },
            { schema::kMasterVolume, "volume" },
        };
        ids.fill (std::nullopt);
        for (auto* p : synth->storage.getPatch().param_ptr)
            for (const auto& [index, storageName] : kDirect)
                if (std::strcmp (p->get_storage_name(), storageName) == 0)
                    ids[(size_t) index] = synth->idForParameter (p);
    }

    void write (int index, float value)
    {
        if (const auto& id = ids[(size_t) index])
            synth->setParameter01 (*id, value, false, false);
        written[index] = value;
    }

    void applyMidi (const juce::MidiMessage& msg)
    {
        if (msg.isNoteOn())
            synth->playNote (kChannel, (char) msg.getNoteNumber(), (char) msg.getVelocity(), 0);
        else if (msg.isNoteOff())
            synth->releaseNote (kChannel, (char) msg.getNoteNumber(), (char) msg.getVelocity());
        else if (msg.isPitchWheel())
            synth->pitchBend (kChannel, msg.getPitchWheelValue() - 8192);
        else if (msg.isAllNotesOff() || msg.isAllSoundOff())
            synth->allNotesOff();
        else if (msg.isController())
            synth->channelController (kChannel, msg.getControllerNumber(), msg.getControllerValue());
    }

    void fillQuantum()
    {
        synth->process();
        std::copy (synth->output[0], synth->output[0] + kQuantum, quantumL.begin());
        std::copy (synth->output[1], synth->output[1] + kQuantum, quantumR.begin());
        quantumPos = 0;
    }
};

SurgeEngine::SurgeEngine() : impl (std::make_unique<Impl>()) {}
SurgeEngine::~SurgeEngine() = default;

void SurgeEngine::prepare (double sampleRate, int)
{
    // Build a fresh synth per prepare: setSamplerate on a running instance is
    // supported, but a clean instance also resets voices and FX tails.
    impl->synth = std::make_unique<SurgeSynthesizer> (&impl->layer, SurgeStorage::skipPatchLoadDataPathSentinel);
    impl->synth->setSamplerate (static_cast<float> (sampleRate));
    impl->synth->time_data.tempo = 120.0;
    impl->synth->process();   // let the default patch settle

    impl->resolveControls();
    for (int i = 0; i < schema::kControlCount; ++i)
        impl->write (i, impl->controls[i]);
    impl->quantumPos = kQuantum;
}

void SurgeEngine::reset()
{
    if (impl->synth != nullptr)
    {
        impl->synth->allNotesOff();
        impl->quantumPos = kQuantum;
    }
}

void SurgeEngine::setControl (int index, float normalised)
{
    if (index < 0 || index >= schema::kControlCount || ! std::isfinite (normalised))
        return;

    normalised = schema::sanitiseValue (index, normalised);
    impl->controls[index] = normalised;
    if (impl->synth != nullptr && ! juce::exactlyEqual (impl->written[index], normalised))
        impl->write (index, normalised);
}

int SurgeEngine::numResolvedControls() const noexcept
{
    int n = 0;
    for (const auto& id : impl->ids)
        n += id.has_value() ? 1 : 0;
    return n;
}

void SurgeEngine::process (juce::AudioBuffer<float>& buffer, const juce::MidiBuffer& midi)
{
    const int numSamples = buffer.getNumSamples();
    if (impl->synth == nullptr || buffer.getNumChannels() < 2)
    {
        buffer.clear();
        return;
    }


    auto* left = buffer.getWritePointer (0);
    auto* right = buffer.getWritePointer (1);
    auto nextEvent = midi.cbegin();

    int done = 0;
    while (done < numSamples)
    {
        if (impl->quantumPos >= kQuantum)
        {
            // Events that fall inside the quantum about to be rendered.
            const int quantumEnd = done + kQuantum;
            for (; nextEvent != midi.cend() && (*nextEvent).samplePosition < quantumEnd; ++nextEvent)
                impl->applyMidi ((*nextEvent).getMessage());
            impl->fillQuantum();
        }

        const int take = juce::jmin (kQuantum - impl->quantumPos, numSamples - done);
        std::copy_n (impl->quantumL.begin() + impl->quantumPos, take, left + done);
        std::copy_n (impl->quantumR.begin() + impl->quantumPos, take, right + done);
        impl->quantumPos += take;
        done += take;
    }

    // Anything left (offsets past the block end) applies now.
    for (; nextEvent != midi.cend(); ++nextEvent)
        impl->applyMidi ((*nextEvent).getMessage());
}

} // namespace vae
