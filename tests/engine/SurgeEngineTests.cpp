// Surge engine specific tests.
#include "EngineTestUtils.h"

#include "engine/surge/SurgeEngine.h"

#include <juce_dsp/juce_dsp.h>

JUCE_BEGIN_IGNORE_WARNINGS_GCC_LIKE ("-Wsign-conversion", "-Wfloat-equal", "-Wpedantic", "-Wshadow", "-Wconversion",
                                     "-Wfloat-conversion", "-Wold-style-cast", "-Wzero-as-null-pointer-constant",
                                     "-Wunused-parameter", "-Wswitch-enum", "-Wimplicit-int-float-conversion",
                                     "-Wshorten-64-to-32", "-Wextra-semi", "-Wcast-align", "-Wunused-value",
                                     "-Wunused-variable", "-Wunused-local-typedefs", "-Wstrict-aliasing",
                                     "-Wsign-compare")
#include "SurgeSynthesizer.h"
JUCE_END_IGNORE_WARNINGS_GCC_LIKE

#include <atomic>
#include <cstdlib>
#include <new>

// Surge enum values at the pinned revision (src/common/SurgeStorage.h), so the
// expectations below read as names. Kept independent of SurgeEngine.cpp's
// tables on purpose: the tests check those tables.
namespace surgeid
{
constexpr int oscClassic = 0, oscSine = 1, oscWavetable = 2, oscShNoise = 3, oscFm2 = 6, oscModern = 8;
constexpr int fxDelay = 1, fxPhaser = 3, fxDistortion = 5, fxEq = 6, fxChorus = 9, fxReverb2 = 11, fxFlanger = 12;
} // namespace surgeid

// Counting operator new for the audio-thread allocation test below. Counting
// is only on while that test processes blocks.
namespace
{
std::atomic<bool> gCountAllocations { false };
std::atomic<int> gAllocations { 0 };
} // namespace

// GCC cannot see that this operator new is malloc-backed and warns on free().
JUCE_BEGIN_IGNORE_WARNINGS_GCC_LIKE ("-Wmismatched-new-delete")
void* operator new (std::size_t size)
{
    if (gCountAllocations.load (std::memory_order_relaxed))
        gAllocations.fetch_add (1, std::memory_order_relaxed);
    if (void* p = std::malloc (size != 0 ? size : 1))
        return p;
    throw std::bad_alloc();
}
void operator delete (void* p) noexcept { std::free (p); }
void operator delete (void* p, std::size_t) noexcept { std::free (p); }
JUCE_END_IGNORE_WARNINGS_GCC_LIKE

namespace vae
{
namespace
{
using namespace schema;

int opt (int control, const char* name) { return *optionIndex (control, name); }

/** Apply a vector and run one block so lazy osc/FX type loads are live. */
void applyAndSettle (SurgeEngine& e, const Snapshot& s)
{
    e.setControls (s);
    juce::AudioBuffer<float> b (2, 32);
    juce::MidiBuffer m;
    e.process (b, m);
}

/** Fraction of spectral energy above `hz`, plus a coarse band profile. */
struct Spectrum
{
    float highRatio = 0.0f;
    std::array<float, 16> bands {};
};

Spectrum spectrumOf (const std::vector<float>& x, int start)
{
    constexpr int order = 13, size = 1 << order;
    juce::dsp::FFT fft (order);
    std::vector<float> data (size * 2, 0.0f);
    for (int i = 0; i < size; ++i)
        data[(size_t) i] = x[(size_t) (start + i)]
                           * (0.5f - 0.5f * std::cos (juce::MathConstants<float>::twoPi * float (i) / float (size)));
    fft.performFrequencyOnlyForwardTransform (data.data());

    Spectrum s;
    double total = 0.0, high = 0.0;
    for (int k = 1; k < size / 2; ++k)
    {
        const double e = double (data[(size_t) k]) * double (data[(size_t) k]);
        const double hz = double (k) * kRate / size;
        total += e;
        if (hz > 2000.0)
            high += e;
        // log-spaced bands from 100 Hz
        const int band = juce::jlimit (0, 15, int (std::log2 (juce::jmax (1.0, hz / 100.0)) * 2.0));
        s.bands[(size_t) band] += float (e);
    }
    s.highRatio = total > 0.0 ? float (high / total) : 0.0f;
    for (auto& b : s.bands)
        b = float (b / juce::jmax (1.0e-12, total));
    return s;
}

using testutil::rmsOf;
using testutil::zeroCrossingHz;

std::vector<float> capture (SurgeEngine& e, const Snapshot& s, std::vector<NoteEvent> events, int samples)
{
    e.reset();
    applyAndSettle (e, s);
    std::vector<float> left;
    renderEvents (e, { 64 }, samples, std::move (events), 0, &left);
    return left;
}
} // namespace

class SurgeEngineTests final : public juce::UnitTest
{
public:
    SurgeEngineTests()
        : juce::UnitTest ("Surge engine", "vae_engine")
    {
    }

    void runTest() override
    {
        SurgeEngine e;
        e.prepare (kRate, 512);

        beginTest ("every schema control resolves");
        expectEquals (e.numResolvedControls(), kControlCount);

        beginTest ("defaults land on Surge");
        applyAndSettle (e, Snapshot::defaults());
        expectEquals (e.surgeValue ("a_polymode"), 0.0f);
        expectEquals (e.surgeValue ("a_fb_config"), 3.0f); // parallel = Dual 1
        expectEquals (e.surgeValue ("a_mute_o1"), 0.0f);
        expectEquals (e.surgeValue ("a_mute_o2"), 1.0f);
        expectEquals (e.surgeValue ("a_route_o1"), 1.0f);
        expectEquals (e.surgeValue ("a_filter1_type"), 0.0f);
        expectEquals (e.surgeValue ("a_env1_release_shape"), 2.0f);
        expectEquals (e.surgeValue ("a_lfo0_trigmode"), 1.0f);
        expectWithinAbsoluteError (e.surgeValue ("a_filter1_cutoff"), 3.0f, 1.0e-3f);
        expectWithinAbsoluteError (e.surgeValue ("volume"), -2.0257f, 1.0e-3f);
        expectEquals (e.oscType (0), surgeid::oscClassic); // saw
        expectEquals (e.oscParam (0, 0), 0.0f);
        expectEquals (e.numModRoutes(), 0);
        // The fixed chain is always loaded; *_on = 0 disables the slot.
        const std::pair<int, int> chain[] = {
            { SurgeEngine::kDistSlot, surgeid::fxDistortion }, { SurgeEngine::kEqSlot, surgeid::fxEq },
            { SurgeEngine::kChorusSlot, surgeid::fxChorus },   { SurgeEngine::kModFxSlot, surgeid::fxPhaser },
            { SurgeEngine::kDelaySlot, surgeid::fxDelay },     { SurgeEngine::kReverbSlot, surgeid::fxReverb2 }
        };
        for (const auto& [slot, type] : chain)
        {
            expectEquals (e.fxType (slot), type);
            expect (! e.fxEnabled (slot));
        }

        beginTest ("categoricals write the expected Surge integers");
        struct Case
        {
            int control;
            const char* storage;
            std::vector<int> expected;
        };
        const std::vector<Case> cases = {
            { kPlayMode, "a_polymode", { 0, 1, 2 } },
            { kFilterRouting, "a_fb_config", { 0, 3 } },
            { kOsc2Octave, "a_osc2_octave", { -3, -2, -1, 0, 1, 2, 3 } },
            { kOsc3Filter, "a_route_o3", { 0, 1, 2 } },
            { kFilter1Type, "a_filter1_type", { 0, 1, 2, 4, 5, 6, 7 } },
            { kFilter2Type, "a_filter2_type", { 0, 1, 2, 4, 5, 6, 7 } },
            { kAmpAttackCurve, "a_env1_attack_shape", { 0, 1, 2 } },
            { kFegDecayCurve, "a_env2_decay_shape", { 0, 1, 2 } },
            { kFegReleaseCurve, "a_env2_release_shape", { 0, 1, 2 } },
            { kLfo1Shape, "a_lfo0_shape", { 0, 1, 2, 3, 5, 4 } },
            { kLfo2Shape, "a_lfo1_shape", { 0, 1, 2, 3, 5, 4 } },
            { kLfo1Retrigger, "a_lfo0_trigmode", { 0, 1 } },
            { kLfo2Unipolar, "a_lfo1_unipolar", { 0, 1 } },
            { kOsc2On, "a_mute_o2", { 1, 0 } },
        };
        for (const auto& c : cases)
            for (int k = 0; k < int (c.expected.size()); ++k)
            {
                auto s = Snapshot::defaults();
                s[c.control] = float (k);
                applyAndSettle (e, s);
                expectEquals (e.surgeValue (c.storage), float (c.expected[(size_t) k]),
                              juce::String (control (c.control).id) + " option " + juce::String (k));
            }
        auto withFilter = Snapshot::defaults();
        withFilter[kFilter1Type] = float (opt (kFilter1Type, "hp24"));
        applyAndSettle (e, withFilter);
        expectEquals (e.surgeValue ("a_filter1_subtype"), 0.0f);

        beginTest ("continuous controls are Surge-normalised");
        juce::Random rng (99);
        const std::pair<int, const char*> continuous[] = {
            { kGlideTime, "a_portamento" },
            { kOsc1Level, "a_level_o1" },
            { kOsc2Pitch, "a_osc2_pitch" },
            { kFilter1Cutoff, "a_filter1_cutoff" },
            { kFilter2Resonance, "a_filter2_resonance" },
            { kFilter1Env, "a_filter1_envmod" },
            { kFilter2Keytrack, "a_filter2_keytrack" },
            { kAmpAttack, "a_env1_attack" },
            { kFegSustain, "a_env2_sustain" },
            { kFegRelease, "a_env2_release" },
            { kLfo1Rate, "a_lfo0_rate" },
            { kLfo2Amount, "a_lfo1_magnitude" },
            { kLfo1Phase, "a_lfo0_phase" },
            { kLfo2FadeIn, "a_lfo1_attack" },
            { kMasterVolume, "volume" },
        };
        for (int trial = 0; trial < 5; ++trial)
        {
            auto s = Snapshot::defaults();
            for (const auto& [index, storage] : continuous)
                s[index] = rng.nextFloat();
            applyAndSettle (e, s);
            for (const auto& [index, storage] : continuous)
                expectWithinAbsoluteError (e.surgeValue (storage), surge::toUnits (index, s[index]), 1.0e-3f, storage);
        }

        beginTest ("osc waves select the documented Surge oscillators");
        const int expectedType[] = { surgeid::oscSine,    surgeid::oscModern,  surgeid::oscClassic,
                                     surgeid::oscClassic, surgeid::oscShNoise, surgeid::oscWavetable,
                                     surgeid::oscFm2 };
        for (int w = 0; w < control (kOsc1Wave).numOptions; ++w)
        {
            auto s = Snapshot::defaults();
            s[kOsc1Wave] = float (w);
            applyAndSettle (e, s);
            expectEquals (e.oscType (0), expectedType[w], control (kOsc1Wave).options[w]);
        }
        auto sq = Snapshot::defaults();
        sq[kOsc1Wave] = float (opt (kOsc1Wave, "square"));
        applyAndSettle (e, sq);
        expectEquals (e.oscParam (0, 0), -1.0f);
        sq[kOsc1Wave] = float (opt (kOsc1Wave, "triangle"));
        applyAndSettle (e, sq);
        expectEquals (e.oscParam (0, 2), 1.0f);
        expectEquals (e.oscParam (0, 0), 0.0f);

        beginTest ("osc waves sound distinct");
        std::vector<Spectrum> spectra;
        for (int w = 0; w < control (kOsc1Wave).numOptions; ++w)
        {
            auto s = Snapshot::defaults();
            s[kOsc1Wave] = float (w);
            const auto x = capture (e, s, { { 0, 57, 110 } }, 16384);
            spectra.push_back (spectrumOf (x, 4096));
            logMessage (juce::String ("  ") + control (kOsc1Wave).options[w] + " HF ratio "
                        + juce::String (spectra.back().highRatio, 4));
        }
        const int sine = opt (kOsc1Wave, "sine"), tri = opt (kOsc1Wave, "triangle"), saw = opt (kOsc1Wave, "saw"),
                  noise = opt (kOsc1Wave, "noise");
        expect (spectra[(size_t) sine].highRatio < 0.01f);
        expect (spectra[(size_t) tri].highRatio < spectra[(size_t) saw].highRatio);
        expect (spectra[(size_t) noise].highRatio > spectra[(size_t) saw].highRatio);
        for (size_t a = 0; a < spectra.size(); ++a)
            for (size_t b = a + 1; b < spectra.size(); ++b)
            {
                float d = 0.0f;
                for (size_t k = 0; k < 16; ++k)
                    d += std::abs (spectra[a].bands[k] - spectra[b].bands[k]);
                expect (d > 0.02f, juce::String (control (kOsc1Wave).options[a]) + " vs "
                                       + control (kOsc1Wave).options[b] + ": " + juce::String (d));
            }

        beginTest ("poly vs mono voice count");
        for (int mode = 0; mode < 3; ++mode)
        {
            auto s = Snapshot::defaults();
            s[kPlayMode] = float (mode);
            capture (e, s, { { 0, 60, 100 }, { 100, 64, 100 }, { 200, 67, 100 } }, 4800);
            expectEquals (e.numHeldVoices(), mode == 0 ? 3 : 1, control (kPlayMode).options[mode]);
        }

        beginTest ("legato does not retrigger the amp envelope, mono does");
        {
            auto s = Snapshot::defaults();
            s[kAmpAttack] = surge::fromUnits (kAmpAttack, 0.0f); // 1 s attack
            s[kPlayMode] = float (opt (kPlayMode, "mono"));
            const auto mono = capture (e, s, { { 0, 60, 100 }, { 24000, 64, 100 } }, 30000);
            s[kPlayMode] = float (opt (kPlayMode, "legato"));
            const auto legato = capture (e, s, { { 0, 60, 100 }, { 24000, 64, 100 } }, 30000);
            const float m = rmsOf (mono, 24100, 26000), l = rmsOf (legato, 24100, 26000);
            expect (l > 2.0f * m, "legato " + juce::String (l) + " mono " + juce::String (m));
        }

        beginTest ("glide");
        {
            auto s = Snapshot::defaults();
            s[kOsc1Wave] = float (opt (kOsc1Wave, "sine"));
            s[kPlayMode] = float (opt (kPlayMode, "legato"));
            const std::vector<NoteEvent> notes { { 0, 48, 100 }, { 9600, 72, 100 } };
            const auto dry = capture (e, s, notes, 19200);
            s[kGlideTime] = surge::fromUnits (kGlideTime, -1.0f); // 0.5 s
            const auto glide = capture (e, s, notes, 19200);
            const float f0 = zeroCrossingHz (dry, 10600, 12000), f1 = zeroCrossingHz (glide, 10600, 12000);
            logMessage ("  70 ms after the jump: " + juce::String (f0) + " Hz without glide, " + juce::String (f1)
                        + " Hz with");
            expectWithinAbsoluteError (f0, 523.25f, 20.0f);
            expect (f1 < 0.75f * f0);
        }

        beginTest ("FX units load, take their parameters and change the sound");
        struct FxCase
        {
            int on;
            int slot;
            int type;
            std::vector<std::pair<int, float>> set;
            int checkK;
            int checkControl;
        };
        const std::vector<FxCase> fxCases = {
            { kDistOn, SurgeEngine::kDistSlot, surgeid::fxDistortion, { { kDistDrive, 1.0f } }, 4, kDistDrive },
            { kEqOn,
              SurgeEngine::kEqSlot,
              surgeid::fxEq,
              { { kEqLowGain, 1.0f }, { kEqHighGain, 0.0f } },
              0,
              kEqLowGain },
            { kChorusOn, SurgeEngine::kChorusSlot, surgeid::fxChorus, { { kChorusDepth, 1.0f } }, 2, kChorusDepth },
            { kModfxOn,
              SurgeEngine::kModFxSlot,
              surgeid::fxPhaser,
              { { kModfxDepth, 1.0f }, { kModfxFeedback, 0.8f } },
              1,
              kModfxFeedback },
            { kDelayOn,
              SurgeEngine::kDelaySlot,
              surgeid::fxDelay,
              { { kDelayMix, 1.0f }, { kDelayTime, 0.3f } },
              1,
              kDelayTime },
            { kReverbOn, SurgeEngine::kReverbSlot, surgeid::fxReverb2, { { kReverbMix, 1.0f } }, 9, kReverbMix },
        };
        auto base = Snapshot::defaults();
        const std::vector<NoteEvent> pluck { { 0, 60, 100 }, { 4800, 60, 0 } };
        const auto dry = capture (e, base, pluck, 24000);
        for (const auto& c : fxCases)
        {
            auto s = base;
            s[c.on] = 1.0f;
            for (const auto& [index, value] : c.set)
                s[index] = value;
            const auto wet = capture (e, s, pluck, 24000);
            expectEquals (e.fxType (c.slot), c.type, control (c.on).id);
            expect (e.fxEnabled (c.slot), control (c.on).id);
            expectWithinAbsoluteError (e.fxParam (c.slot, c.checkK), surge::toUnits (c.checkControl, s[c.checkControl]),
                                       1.0e-3f, control (c.checkControl).id);
            double diff = 0.0, ref = 0.0;
            for (size_t i = 0; i < dry.size(); ++i)
            {
                diff += double (wet[i] - dry[i]) * double (wet[i] - dry[i]);
                ref += double (dry[i]) * double (dry[i]);
            }
            expect (diff > 0.01 * ref, juce::String (control (c.on).id) + " changes output");
        }

        beginTest ("mod FX type switch maps parameters per type");
        {
            auto s = base;
            s[kModfxOn] = 1.0f;
            s[kModfxType] = float (opt (kModfxType, "flanger"));
            s[kModfxMix] = 0.7f;
            s[kModfxRate] = 0.6f;
            applyAndSettle (e, s);
            expectEquals (e.fxType (SurgeEngine::kModFxSlot), surgeid::fxFlanger);
            expectWithinAbsoluteError (e.fxParam (SurgeEngine::kModFxSlot, 10), 0.7f,
                                       1.0e-3f); // mix, positive half of -1..1
            expectWithinAbsoluteError (e.fxParam (SurgeEngine::kModFxSlot, 2), surge::toUnits (kModfxRate, 0.6f),
                                       1.0e-3f);
            s[kModfxOn] = 0.0f;
            applyAndSettle (e, s);
            expect (! e.fxEnabled (SurgeEngine::kModFxSlot));
            expectEquals (e.fxType (SurgeEngine::kModFxSlot), surgeid::fxFlanger); // stays loaded
        }

        beginTest ("modulation routes follow the shared depth convention");
        {
            auto s = base;
            s[kLfo1Destination] = float (opt (kLfo1Destination, "filter1_cutoff"));
            s[kLfo1Depth] = 0.75f; // d = 0.5
            applyAndSettle (e, s);
            expectEquals (e.numModRoutes(), 1);
            expectWithinAbsoluteError (e.modRouteDepth (1, "a_filter1_cutoff"),
                                       0.5f * schema::surge::curveSpan ("cutoff"), 1.0e-3f);

            s[kLfo1Destination] = float (opt (kLfo1Destination, "pitch"));
            applyAndSettle (e, s);
            expect (std::isnan (e.modRouteDepth (1, "a_filter1_cutoff")), "old route cleared");
            expectWithinAbsoluteError (e.modRouteDepth (1, "a_pitch"), 0.5f * schema::surge::curveSpan ("pitch7"),
                                       1.0e-3f);
            expectEquals (e.numModRoutes(), 1);

            s[kFegDestination] = float (opt (kFegDestination, "pan"));
            s[kFegDepth] = 0.0f; // d = -1
            s[kLfo2Destination] = float (opt (kLfo2Destination, "volume"));
            s[kLfo2Depth] = 1.0f;
            applyAndSettle (e, s);
            expectEquals (e.numModRoutes(), 3);
            expectWithinAbsoluteError (e.modRouteDepth (0, "a_pan"), -schema::surge::curveSpan ("bipolar"), 1.0e-3f);
            expectWithinAbsoluteError (e.modRouteDepth (2, "a_volume"), schema::surge::curveSpan ("amplitude"),
                                       1.0e-3f);

            s[kLfo1Destination] = s[kLfo2Destination] = s[kFegDestination] = 0.0f;
            applyAndSettle (e, s);
            expectEquals (e.numModRoutes(), 0);
        }

        beginTest ("an LFO route is audible");
        {
            auto s = base;
            s[kLfo1Rate] = surge::fromUnits (kLfo1Rate, 3.0f); // 8 Hz
            const auto still = capture (e, s, { { 0, 60, 100 } }, 24000);
            s[kLfo1Destination] = float (opt (kLfo1Destination, "volume"));
            s[kLfo1Depth] = 0.0f;
            const auto wobble = capture (e, s, { { 0, 60, 100 } }, 24000);
            // Envelope of 10 ms windows: the routed LFO makes it fluctuate.
            auto spread = [] (const std::vector<float>& x)
            {
                float lo = 1.0e9f, hi = 0.0f;
                for (size_t w = 4800; w + 480 <= 24000; w += 480)
                {
                    const float r = rmsOf (x, w, w + 480);
                    lo = juce::jmin (lo, r);
                    hi = juce::jmax (hi, r);
                }
                return hi - lo;
            };
            expect (spread (wobble) > 4.0f * spread (still) + 1.0e-3f,
                    "still " + juce::String (spread (still)) + " wobble " + juce::String (spread (wobble)));
        }

        beginTest ("type switching while notes play stays finite");
        {
            juce::Random r (4321);
            e.reset();
            applyAndSettle (e, base);
            bool finite = true;
            float peak = 0.0f;
            for (int block = 0; block < 400; ++block)
            {
                auto s = randomSnapshot (r);
                s[kOsc1On] = 1.0f;
                s[kMasterVolume] = 0.8f;
                e.setControls (s);
                juce::AudioBuffer<float> b (2, 64);
                juce::MidiBuffer m;
                if (block % 50 == 0)
                    m.addEvent (juce::MidiMessage::noteOn (1, 48 + r.nextInt (24), (juce::uint8) 100), r.nextInt (64));
                if (block % 50 == 40)
                    m.addEvent (juce::MidiMessage::allNotesOff (1), 0);
                e.process (b, m);
                for (int ch = 0; ch < 2; ++ch)
                    for (int i = 0; i < 64; ++i)
                    {
                        finite = finite && std::isfinite (b.getSample (ch, i));
                        peak = juce::jmax (peak, std::abs (b.getSample (ch, i)));
                    }
            }
            expect (finite);
            expect (peak < 16.0f, "peak " + juce::String (peak));
        }

        beginTest ("FX on/off, filter types and continuous changes do not allocate on the audio thread");
        {
            SurgeEngine a;
            a.prepare (kRate, 256);
            applyAndSettle (a, Snapshot::defaults());
            juce::AudioBuffer<float> b (2, 256);
            juce::MidiBuffer noMidi, noteOn;
            noteOn.addEvent (juce::MidiMessage::noteOn (1, 60, (juce::uint8) 100), 0);
            a.process (b, noteOn);
            for (int i = 0; i < 8; ++i)
                a.process (b, noMidi); // warm up

            // The counter itself works.
            gAllocations = 0;
            gCountAllocations = true;
            auto probe = std::make_unique<int> (1);
            gCountAllocations = false;
            expectEquals (gAllocations.load(), 1);

            // Each step is one host block: a control change, then process().
            const std::pair<int, float> steps[] = {
                { kDistOn, 1.0f },      { kEqOn, 1.0f },          { kChorusOn, 1.0f },    { kModfxOn, 1.0f },
                { kDelayOn, 1.0f },     { kReverbOn, 1.0f },      { kReverbMix, 0.9f },   { kDelayTime, 0.2f },
                { kDistDrive, 0.9f },   { kModfxRate, 0.7f },     { kFilter1Type, 2.0f }, { kFilter2Type, 4.0f },
                { kFilter1Type, 6.0f }, { kFilter1Cutoff, 0.3f }, { kAmpRelease, 0.6f },  { kLfo1Shape, 3.0f },
                { kPlayMode, 1.0f },    { kDistOn, 0.0f },        { kEqOn, 0.0f },        { kChorusOn, 0.0f },
                { kModfxOn, 0.0f },     { kDelayOn, 0.0f },       { kReverbOn, 0.0f },    { kReverbOn, 1.0f },
                { kDelayOn, 1.0f },
            };
            for (const auto& [index, value] : steps)
            {
                gAllocations = 0;
                gCountAllocations = true;
                a.setControl (index, value);
                a.process (b, noMidi);
                gCountAllocations = false;
                expectEquals (gAllocations.load(), 0, juce::String (control (index).id) + " = " + juce::String (value));
            }
            expect (a.fxEnabled (SurgeEngine::kReverbSlot));
            expect (! a.fxEnabled (SurgeEngine::kDistSlot));
        }

        beginTest ("full patch data plays, ignores schema writes, then the schema vector returns");
        {
            struct Layer final : SurgeSynthesizer::PluginLayer
            {
                void surgeParameterUpdated (const SurgeSynthesizer::ID&, float) override {}
                void surgeMacroUpdated (long, float) override {}
            } layer;
            juce::MemoryBlock sine;
            {
                auto s = std::make_unique<SurgeSynthesizer> (&layer, VAE_SURGE_DATA_DIR); // too big for the stack
                expect (s->loadPatchByPath (VAE_SURGE_DATA_DIR "/patches_factory/Templates/Init Sine.fxp", -1,
                                            "Init Sine"));
                void* data = nullptr;
                const auto size = s->saveRaw (&data);
                sine.replaceAll (data, size);
            }
            SurgeEngine a;
            a.prepare (kRate, 256);
            applyAndSettle (a, Snapshot::defaults());
            juce::AudioBuffer<float> b (2, 256);
            juce::MidiBuffer noMidi;
            a.loadPatchData (sine);
            a.process (b, noMidi);
            expect (a.hasPatchData());
            expectEquals (a.oscType (0), surgeid::oscSine);
            a.setControl (kOsc1Wave, float (opt (kOsc1Wave, "square")));
            a.process (b, noMidi);
            expectEquals (a.oscType (0), surgeid::oscSine);

            a.prepare (kRate, 256); // survives a re-prepare
            a.process (b, noMidi);
            expectEquals (a.oscType (0), surgeid::oscSine);

            a.useSchemaControls();
            a.process (b, noMidi);
            expect (! a.hasPatchData());
            expectEquals (a.oscType (0), surgeid::oscClassic); // the square written while the patch played
            expectEquals (a.oscParam (0, 0), -1.0f);
            expectEquals (a.fxType (SurgeEngine::kReverbSlot), surgeid::fxReverb2);
        }
    }
};

static SurgeEngineTests surgeEngineTests;

} // namespace vae
