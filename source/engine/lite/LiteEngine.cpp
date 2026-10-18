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

// Surge's default patch mutes Osc 3 and Noise, and schema v1 has no control
// to unmute them, so on the reference engine those controls are inaudible.
// Lite mirrors that to keep a control vector sounding alike on both backends.
constexpr bool kOsc3Muted = true;
constexpr bool kNoiseMuted = true;

struct OscMapping
{
    int table;          // ablaze wavetable index for Osc 1
    bool shapeIsClassic;
    int analogWave;     // ablaze::AnalogWave for Osc 2
    int subWave;        // ablaze::SubWave for Osc 3
    bool silent;        // Surge "Audio In": no input on an instrument
};

// Indexed by sc::OscType. Tables: 0 Basic, 1 PWM, 2 Harmonic, 3 Formant, 4 Sync, 5 Fold.
// Analog: 0 Sine, 1 Triangle, 2 Saw, 3 Square, 4 Pulse. Sub: 0 Sine, 1 Triangle, 2 Square, 3 Pulse.
constexpr OscMapping kOscMap[] = {
    { 0, true,  2, 2, false },   // Classic
    { 0, false, 0, 0, false },   // Sine (Basic table at position 0 is a sine)
    { 2, false, 1, 1, false },   // Wavetable
    { 5, false, 3, 2, false },   // S&H Noise
    { 0, false, 0, 0, true  },   // Audio Input
    { 3, false, 0, 0, false },   // FM3
    { 4, false, 0, 0, false },   // FM2
    { 3, false, 1, 1, false },   // Window
    { 1, false, 2, 3, false },   // Modern
    { 2, false, 4, 3, false },   // String
    { 5, false, 3, 2, false },   // Twist
    { 4, false, 4, 3, false },   // Alias
};
static_assert (std::size (kOscMap) == 12);

int subOctaveChoice (int octave)
{
    // ablaze sub_octave choices: -1, -2, -3, 0, +1, +2, +3
    static constexpr int kChoiceForOctave[] = { 2, 1, 0, 3, 4, 5, 6 };
    return kChoiceForOctave[juce::jlimit (-3, 3, octave) + 3];
}

float envMs (float normalised) { return 1000.0f * sc::envSeconds (normalised); }
} // namespace

LiteEngine::LiteEngine() = default;
LiteEngine::~LiteEngine() = default;

void LiteEngine::prepare (double sampleRate, int)
{
    if (bank == nullptr)
        bank = std::make_shared<const ablaze::WavetableBank>();

    engine = std::make_unique<ablaze::SynthEngine> (static_cast<float> (sampleRate), kMaxVoices, bank);

    // Fixed parts of the Lite "patch" that the schema does not reach.
    setNative (ablaze::kParamOscAOn, 1.0f);
    setNative (ablaze::kParamOscBOn, 1.0f);
    setNative (ablaze::kParamSubOn, 1.0f);
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

    normalised = juce::jlimit (0.0f, 1.0f, normalised);
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
    const auto& c = controls;

    switch (index)
    {
        case kOsc1Type:
        case kOsc1Shape:
        {
            const auto& m = kOscMap[int (sc::oscType (c[kOsc1Type]))];
            // Classic: bipolar shape -1..1 morphs triangle -> saw -> square across
            // the Basic table (sine, triangle, saw, square). Others: shape is
            // Surge's first osc parameter, used as the table position.
            const float pos = m.shapeIsClassic ? (2.0f + sc::bipolar (c[kOsc1Shape])) / 3.0f
                                               : c[kOsc1Shape];
            setNative (ablaze::kParamOscATable, float (m.table));
            setNative (ablaze::kParamOscAPos, juce::jlimit (0.0f, 1.0f, pos));
            apply (kOsc1Volume);
            break;
        }
        case kOsc1Octave:   setNative (ablaze::kParamOscAOctave, float (sc::octave (c[index]))); break;
        case kOsc1Volume:
        {
            const bool silent = kOscMap[int (sc::oscType (c[kOsc1Type]))].silent;
            setNative (ablaze::kParamOscALevel, silent ? 0.0f : sc::amplitude (c[index]));
            break;
        }
        case kOsc1Voices:   setNative (ablaze::kParamOscAUnison, float (juce::jmin (7, sc::unisonVoices (c[index])))); break;
        case kOsc1Detune:   setNative (ablaze::kParamOscADetune, sc::unisonDetuneCents (c[index])); break;

        case kOsc2Type:
            setNative (ablaze::kParamOscBWave, float (kOscMap[int (sc::oscType (c[index]))].analogWave));
            apply (kOsc2Volume);
            break;
        case kOsc2Octave:   setNative (ablaze::kParamOscBOctave, float (sc::octave (c[index]))); break;
        case kOsc2Volume:
        case kOsc2Mute:
        {
            const bool silent = sc::toggle (c[kOsc2Mute]) || kOscMap[int (sc::oscType (c[kOsc2Type]))].silent;
            setNative (ablaze::kParamOscBLevel, silent ? 0.0f : sc::amplitude (c[kOsc2Volume]));
            break;
        }

        case kOsc3Type:
            setNative (ablaze::kParamSubWave, float (kOscMap[int (sc::oscType (c[index]))].subWave));
            apply (kOsc3Volume);
            break;
        case kOsc3Octave:   setNative (ablaze::kParamSubOctave, float (subOctaveChoice (sc::octave (c[index])))); break;
        case kOsc3Volume:
        {
            const bool silent = kOsc3Muted || kOscMap[int (sc::oscType (c[kOsc3Type]))].silent;
            setNative (ablaze::kParamSubLevel, silent ? 0.0f : sc::amplitude (c[index]));
            break;
        }
        case kNoiseVolume:  setNative (ablaze::kParamNoiseLevel, kNoiseMuted ? 0.0f : sc::amplitude (c[index])); break;

        case kFilterType:
        {
            using F = sc::FilterFamily;
            const auto family = sc::filterFamily (sc::filterType (c[index]));
            setNative (ablaze::kParamFilterOn, family == F::Off ? 0.0f : 1.0f);
            int type = 0;   // LP12 for LP12 and the effect/multi types Lite cannot model
            switch (family)
            {
                case F::LowPass24: type = 1; break;
                case F::BandPass:  type = 2; break;
                case F::HighPass:  type = 3; break;
                case F::Notch:     type = 4; break;
                case F::Off:
                case F::LowPass12:
                case F::Other:     break;
            }
            setNative (ablaze::kParamFilterType, float (type));
            break;
        }
        case kCutoff:       setNative (ablaze::kParamFilterCutoff, sc::cutoffHz (c[index])); break;
        case kResonance:    setNative (ablaze::kParamFilterReso, c[index]); break;
        case kFilterFeg:    setNative (ablaze::kParamFilterEnvAmt, sc::fegSemitones (c[index]) / (12.0f * 7.0f)); break;
        case kFilterKeytrack: setNative (ablaze::kParamFilterKeytrack, juce::jlimit (0.0f, 1.0f, sc::bipolar (c[index]))); break;

        case kAmpAttack:    setNative (ablaze::kParamAmpAttack, envMs (c[index])); break;
        case kAmpDecay:     setNative (ablaze::kParamAmpDecay, envMs (c[index])); break;
        case kAmpSustain:   setNative (ablaze::kParamAmpSustain, c[index]); break;
        case kAmpRelease:   setNative (ablaze::kParamAmpRelease, envMs (c[index])); break;
        case kFegAttack:    setNative (ablaze::kParamFenvAttack, envMs (c[index])); break;
        case kFegDecay:     setNative (ablaze::kParamFenvDecay, envMs (c[index])); break;
        case kFegSustain:   setNative (ablaze::kParamFenvSustain, c[index]); break;
        case kFegRelease:   setNative (ablaze::kParamFenvRelease, envMs (c[index])); break;

        case kGlobalVolume: setNative (ablaze::kParamGain, juce::Decibels::decibelsToGain (sc::globalVolumeDb (c[index]), -48.0f)); break;

        case kFilterSubtype:
        case kFilterBalance:
        case kLfo1Rate:
        case kLfo1Amplitude:
        default:
            break;   // not modelled (see header)
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
