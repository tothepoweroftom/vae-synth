// Surge engine specific tests (owned by the Surge backend work).
#include "EngineTestUtils.h"

#include "engine/surge/SurgeEngine.h"

#include <juce_dsp/juce_dsp.h>

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
        data[(size_t) i] = x[(size_t) (start + i)] * (0.5f - 0.5f * std::cos (juce::MathConstants<float>::twoPi * float (i) / float (size)));
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

/** Zero-crossing frequency of x[from, to). */
float zeroCrossingHz (const std::vector<float>& x, int from, int to)
{
    int crossings = 0;
    for (int i = from + 1; i < to; ++i)
        if ((x[(size_t) i - 1] < 0.0f) != (x[(size_t) i] < 0.0f))
            ++crossings;
    return float (crossings) * 0.5f * float (kRate) / float (to - from);
}

float rmsOf (const std::vector<float>& x, int from, int to)
{
    double s = 0.0;
    for (int i = from; i < to; ++i)
        s += double (x[(size_t) i]) * double (x[(size_t) i]);
    return float (std::sqrt (s / double (to - from)));
}

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
    SurgeEngineTests() : juce::UnitTest ("Surge engine", "vae_engine") {}

    void runTest() override
    {
        SurgeEngine e;
        e.prepare (kRate, 512);

        beginTest ("every schema control resolves");
        expectEquals (e.numResolvedControls(), kControlCount);

        beginTest ("defaults land on Surge");
        applyAndSettle (e, Snapshot::defaults());
        expectEquals (e.surgeValue ("a_polymode"), 0.0f);
        expectEquals (e.surgeValue ("a_fb_config"), 3.0f);          // parallel = Dual 1
        expectEquals (e.surgeValue ("a_mute_o1"), 0.0f);
        expectEquals (e.surgeValue ("a_mute_o2"), 1.0f);
        expectEquals (e.surgeValue ("a_route_o1"), 1.0f);
        expectEquals (e.surgeValue ("a_filter1_type"), 0.0f);
        expectEquals (e.surgeValue ("a_env1_release_shape"), 2.0f);
        expectEquals (e.surgeValue ("a_lfo0_trigmode"), 1.0f);
        expectWithinAbsoluteError (e.surgeValue ("a_filter1_cutoff"), 3.0f, 1.0e-3f);
        expectWithinAbsoluteError (e.surgeValue ("volume"), -2.0257f, 1.0e-3f);
        expectEquals (e.oscType (0), 0);                               // Classic (saw)
        expectEquals (e.oscParam (0, 0), 0.0f);
        expectEquals (e.numModRoutes(), 0);
        for (int slot : { SurgeEngine::kDistSlot, SurgeEngine::kEqSlot, SurgeEngine::kChorusSlot,
                          SurgeEngine::kModFxSlot, SurgeEngine::kDelaySlot, SurgeEngine::kReverbSlot })
            expectEquals (e.fxType (slot), 0);

        beginTest ("categoricals write the expected Surge integers");
        struct Case { int control; const char* storage; std::vector<int> expected; };
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
            { kGlideTime, "a_portamento" }, { kOsc1Level, "a_level_o1" }, { kOsc2Pitch, "a_osc2_pitch" },
            { kFilter1Cutoff, "a_filter1_cutoff" }, { kFilter2Resonance, "a_filter2_resonance" },
            { kFilter1Env, "a_filter1_envmod" }, { kFilter2Keytrack, "a_filter2_keytrack" },
            { kAmpAttack, "a_env1_attack" }, { kFegSustain, "a_env2_sustain" }, { kFegRelease, "a_env2_release" },
            { kLfo1Rate, "a_lfo0_rate" }, { kLfo2Amount, "a_lfo1_magnitude" }, { kLfo1Phase, "a_lfo0_phase" },
            { kLfo2FadeIn, "a_lfo1_attack" }, { kMasterVolume, "volume" },
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
        const int expectedType[] = { 1, 8, 0, 0, 3, 2, 6 };   // Sine, Modern, Classic, Classic, S&H Noise, Wavetable, FM2
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
            logMessage (juce::String ("  ") + control (kOsc1Wave).options[w] + " HF ratio " + juce::String (spectra.back().highRatio, 4));
        }
        const int sine = opt (kOsc1Wave, "sine"), tri = opt (kOsc1Wave, "triangle"),
                  saw = opt (kOsc1Wave, "saw"), noise = opt (kOsc1Wave, "noise");
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
            s[kAmpAttack] = surge::fromUnits (kAmpAttack, 0.0f);   // 1 s attack
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
            s[kGlideTime] = surge::fromUnits (kGlideTime, -1.0f);   // 0.5 s
            const auto glide = capture (e, s, notes, 19200);
            const float f0 = zeroCrossingHz (dry, 10600, 12000), f1 = zeroCrossingHz (glide, 10600, 12000);
            logMessage ("  70 ms after the jump: " + juce::String (f0) + " Hz without glide, " + juce::String (f1) + " Hz with");
            expectWithinAbsoluteError (f0, 523.25f, 20.0f);
            expect (f1 < 0.75f * f0);
        }

        beginTest ("FX units load, take their parameters and change the sound");
        // Surge fx types: delay 1, phaser 3, distortion 5, EQ 6, chorus 9, reverb 2 11, flanger 12.
        struct FxCase { int on; int slot; int type; std::vector<std::pair<int, float>> set; int checkK; int checkControl; };
        const std::vector<FxCase> fxCases = {
            { kDistOn, SurgeEngine::kDistSlot, 5, { { kDistDrive, 1.0f } }, 4, kDistDrive },
            { kEqOn, SurgeEngine::kEqSlot, 6, { { kEqLowGain, 1.0f }, { kEqHighGain, 0.0f } }, 0, kEqLowGain },
            { kChorusOn, SurgeEngine::kChorusSlot, 9, { { kChorusDepth, 1.0f } }, 2, kChorusDepth },
            { kModfxOn, SurgeEngine::kModFxSlot, 3, { { kModfxDepth, 1.0f }, { kModfxFeedback, 0.8f } }, 1, kModfxFeedback },
            { kDelayOn, SurgeEngine::kDelaySlot, 1, { { kDelayMix, 1.0f }, { kDelayTime, 0.3f } }, 1, kDelayTime },
            { kReverbOn, SurgeEngine::kReverbSlot, 11, { { kReverbMix, 1.0f } }, 9, kReverbMix },
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
            expectWithinAbsoluteError (e.fxParam (c.slot, c.checkK), surge::toUnits (c.checkControl, s[c.checkControl]), 1.0e-3f,
                                       control (c.checkControl).id);
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
            expectEquals (e.fxType (SurgeEngine::kModFxSlot), 12);
            expectWithinAbsoluteError (e.fxParam (SurgeEngine::kModFxSlot, 10), 0.7f, 1.0e-3f);   // mix, positive half of -1..1
            expectWithinAbsoluteError (e.fxParam (SurgeEngine::kModFxSlot, 2), surge::toUnits (kModfxRate, 0.6f), 1.0e-3f);
            s[kModfxOn] = 0.0f;
            applyAndSettle (e, s);
            expectEquals (e.fxType (SurgeEngine::kModFxSlot), 0);
        }

        beginTest ("modulation routes follow the shared depth convention");
        {
            auto s = base;
            s[kLfo1Destination] = float (opt (kLfo1Destination, "filter1_cutoff"));
            s[kLfo1Depth] = 0.75f;   // d = 0.5
            applyAndSettle (e, s);
            expectEquals (e.numModRoutes(), 1);
            expectWithinAbsoluteError (e.modRouteDepth (1, "a_filter1_cutoff"), 0.5f * 130.0f, 1.0e-3f);

            s[kLfo1Destination] = float (opt (kLfo1Destination, "pitch"));
            applyAndSettle (e, s);
            expect (std::isnan (e.modRouteDepth (1, "a_filter1_cutoff")), "old route cleared");
            expectWithinAbsoluteError (e.modRouteDepth (1, "a_pitch"), 0.5f * 14.0f, 1.0e-3f);
            expectEquals (e.numModRoutes(), 1);

            s[kFegDestination] = float (opt (kFegDestination, "pan"));
            s[kFegDepth] = 0.0f;   // d = -1
            s[kLfo2Destination] = float (opt (kLfo2Destination, "volume"));
            s[kLfo2Depth] = 1.0f;
            applyAndSettle (e, s);
            expectEquals (e.numModRoutes(), 3);
            expectWithinAbsoluteError (e.modRouteDepth (0, "a_pan"), -2.0f, 1.0e-3f);
            expectWithinAbsoluteError (e.modRouteDepth (2, "a_volume"), 1.0f, 1.0e-3f);

            s[kLfo1Destination] = s[kLfo2Destination] = s[kFegDestination] = 0.0f;
            applyAndSettle (e, s);
            expectEquals (e.numModRoutes(), 0);
        }

        beginTest ("an LFO route is audible");
        {
            auto s = base;
            s[kLfo1Rate] = surge::fromUnits (kLfo1Rate, 3.0f);   // 8 Hz
            const auto still = capture (e, s, { { 0, 60, 100 } }, 24000);
            s[kLfo1Destination] = float (opt (kLfo1Destination, "volume"));
            s[kLfo1Depth] = 0.0f;
            const auto wobble = capture (e, s, { { 0, 60, 100 } }, 24000);
            // Envelope of 10 ms windows: the routed LFO makes it fluctuate.
            auto spread = [] (const std::vector<float>& x)
            {
                float lo = 1.0e9f, hi = 0.0f;
                for (int w = 4800; w + 480 <= 24000; w += 480)
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
    }
};

static SurgeEngineTests surgeEngineTests;

} // namespace vae
