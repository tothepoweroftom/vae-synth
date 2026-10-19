#include "LiteEngine.h"

#include "ablaze/SynthEngine.h"
#include "schema/SurgeCurves.h"

namespace vae
{
namespace
{
using namespace schema;
namespace sc = schema::surge;

constexpr int kMaxVoices = 16;

float envMs (int index, float v) { return 1000.0f * sc::seconds (index, v); }
} // namespace

LiteEngine::LiteEngine() = default;
LiteEngine::~LiteEngine() = default;

void LiteEngine::prepare (double sampleRate, int)
{
    if (bank == nullptr)
        bank = std::make_shared<const ablaze::WavetableBank>();

    engine = std::make_unique<ablaze::SynthEngine> (static_cast<float> (sampleRate), kMaxVoices, bank);

    // Fixed parts of the Lite "patch" that the schema does not reach.
    setNative (ablaze::kParamOscAOn, 0.0f);
    setNative (ablaze::kParamOscBOn, 1.0f);
    setNative (ablaze::kParamSubOn, 0.0f);
    setNative (ablaze::kParamFilterVel, 0.0f);
    setNative (ablaze::kParamFilterDrive, 0.0f);

    for (int i = 0; i < kControlCount; ++i)
        apply (i);
    engine->snapParams();
}

void LiteEngine::reset()
{
    if (engine != nullptr)
        engine->reset();
}

void LiteEngine::setControl (int index, float normalised)
{
    if (index < 0 || index >= kControlCount || ! std::isfinite (normalised))
        return;

    normalised = schema::sanitiseValue (index, normalised);
    if (juce::exactlyEqual (controls[index], normalised))
        return;

    controls[index] = normalised;
    if (engine != nullptr)
        apply (index);
}

void LiteEngine::setNative (int nativeIndex, float value)
{
    engine->setParam (nativeIndex, value);
}

void LiteEngine::apply (int index)
{
    // Schema v2 placeholder mapping (osc 1, filter 1, envelopes, volume) until
    // the full Lite v2 voice lands.
    const auto& c = controls;
    switch (index)
    {
        case kOsc1On:
        case kOsc1Level:
            setNative (ablaze::kParamOscBLevel, c[kOsc1On] >= 0.5f ? sc::amplitudeGain (c[kOsc1Level]) : 0.0f);
            break;
        case kOsc1Wave:
        {
            static constexpr int kWave[] = { 0, 1, 2, 3, 3, 2, 0 };   // sine tri saw square noise wt fm
            setNative (ablaze::kParamOscBWave, float (kWave[asIndex (index, c[index])]));
            break;
        }
        case kOsc1Octave: setNative (ablaze::kParamOscBOctave, float (asIndex (index, c[index]) - 3)); break;
        case kFilter1Type:
        {
            static constexpr int kType[] = { 0, 0, 1, 3, 3, 2, 4 };   // off lp12 lp24 hp12 hp24 bp notch
            const int t = asIndex (index, c[index]);
            setNative (ablaze::kParamFilterOn, t == 0 ? 0.0f : 1.0f);
            setNative (ablaze::kParamFilterType, float (kType[t]));
            break;
        }
        case kFilter1Cutoff:    setNative (ablaze::kParamFilterCutoff, sc::cutoffHz (index, c[index])); break;
        case kFilter1Resonance: setNative (ablaze::kParamFilterReso, c[index]); break;
        case kFilter1Env:       setNative (ablaze::kParamFilterEnvAmt, sc::toUnits (index, c[index]) / 84.0f); break;
        case kFilter1Keytrack:  setNative (ablaze::kParamFilterKeytrack, juce::jlimit (0.0f, 1.0f, sc::bipolar (c[index]))); break;
        case kAmpAttack:  setNative (ablaze::kParamAmpAttack, envMs (index, c[index])); break;
        case kAmpDecay:   setNative (ablaze::kParamAmpDecay, envMs (index, c[index])); break;
        case kAmpSustain: setNative (ablaze::kParamAmpSustain, c[index]); break;
        case kAmpRelease: setNative (ablaze::kParamAmpRelease, envMs (index, c[index])); break;
        case kFegAttack:  setNative (ablaze::kParamFenvAttack, envMs (index, c[index])); break;
        case kFegDecay:   setNative (ablaze::kParamFenvDecay, envMs (index, c[index])); break;
        case kFegSustain: setNative (ablaze::kParamFenvSustain, c[index]); break;
        case kFegRelease: setNative (ablaze::kParamFenvRelease, envMs (index, c[index])); break;
        case kMasterVolume:
            setNative (ablaze::kParamGain, juce::Decibels::decibelsToGain (sc::toUnits (index, c[index]), -48.0f));
            break;
        default:
            break;
    }
}

void LiteEngine::process (juce::AudioBuffer<float>& buffer, const juce::MidiBuffer& midi)
{
    const int numSamples = buffer.getNumSamples();
    if (engine == nullptr || buffer.getNumChannels() < 2)
    {
        buffer.clear();
        return;
    }

    for (const auto meta : midi)
    {
        const auto msg = meta.getMessage();
        const int offset = juce::jlimit (0, juce::jmax (0, numSamples - 1), meta.samplePosition);

        if (msg.isNoteOn())
            engine->noteOn (msg.getNoteNumber(), float (juce::MidiMessage::getMidiNoteInHertz (msg.getNoteNumber())),
                            msg.getFloatVelocity(), offset);
        else if (msg.isNoteOff())
            engine->noteOff (msg.getNoteNumber(), offset);
        else if (msg.isAllNotesOff() || msg.isAllSoundOff())
            engine->allNotesOff (offset);
    }

    engine->render (buffer.getWritePointer (0), buffer.getWritePointer (1), numSamples);
}

} // namespace vae
