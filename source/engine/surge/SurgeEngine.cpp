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
    bool oscTypeChanged = false;

    static bool isOscType (int index)
    {
        return index == schema::kOsc1Type || index == schema::kOsc2Type || index == schema::kOsc3Type;
    }

    /** Surge applies an osc type change lazily (loadOscalgos, inside process())
        and then resets that osc's parameters to the new type's defaults. Apply
        it now, before any note can spawn a voice with the new type but the old
        type's parameter layout, and re-write the schema's osc parameters so
        the vector still describes the patch. */
    void settleOscTypes()
    {
        if (! oscTypeChanged)
            return;
        oscTypeChanged = false;
        synth->loadOscalgos();
        for (int index : { int (schema::kOsc1Shape), int (schema::kOsc1Voices), int (schema::kOsc1Detune) })
            write (index, controls[index]);
    }

    void resolveControls()
    {
        ids.fill (std::nullopt);
        char name[TXT_SIZE] {};
        for (int i = 0; i < kMaxSynthSideIds; ++i)
        {
            SurgeSynthesizer::ID id;
            if (! synth->fromSynthSideId (i, id))
                continue;
            synth->getParameterName (id, name);
            for (int c = 0; c < schema::kControlCount; ++c)
                if (! ids[(size_t) c] && juce::String (name) == schema::controls()[(size_t) c].surgeParam)
                    ids[(size_t) c] = id;
        }
    }

    void write (int index, float value)
    {
        if (const auto& id = ids[(size_t) index])
            synth->setParameter01 (*id, value, false, false);
        written[index] = value;
        oscTypeChanged = oscTypeChanged || isOscType (index);
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
    impl->settleOscTypes();
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

    normalised = juce::jlimit (0.0f, 1.0f, normalised);
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

    impl->settleOscTypes();

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
