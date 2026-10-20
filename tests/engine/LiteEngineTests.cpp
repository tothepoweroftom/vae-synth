// Lite engine specific tests.
#include "EngineTestUtils.h"

#include "engine/lite/LiteEngine.h"

namespace vae
{
namespace
{
using namespace schema;

float option (int index, const char* name) { return float (*optionIndex (index, name)); }

/** A clean single-osc patch: osc 1 sine into no filter. */
Snapshot basePatch (const char* wave = "sine")
{
    auto s = Snapshot::defaults();
    s[kOsc1Wave] = option (kOsc1Wave, wave);
    return s;
}

using testutil::rmsOf;
using testutil::zeroCrossingHz;

float correlation (const std::vector<float>& a, const std::vector<float>& b)
{
    double ab = 0.0, aa = 0.0, bb = 0.0;
    for (size_t i = 0; i < std::min (a.size(), b.size()); ++i)
    {
        ab += double (a[i]) * double (b[i]);
        aa += double (a[i]) * double (a[i]);
        bb += double (b[i]) * double (b[i]);
    }
    return aa > 0.0 && bb > 0.0 ? float (ab / std::sqrt (aa * bb)) : 1.0f;
}

std::vector<float> capture (LiteEngine& e, const Snapshot& s, int samples, std::vector<NoteEvent> events)
{
    e.reset();
    e.setControls (s);
    std::vector<float> out;
    renderEvents (e, { 64 }, samples, std::move (events), 0, &out);
    return out;
}
} // namespace

class LiteEngineTests final : public juce::UnitTest
{
public:
    LiteEngineTests()
        : juce::UnitTest ("Lite engine", "vae_engine")
    {
    }

    void runTest() override
    {
        LiteEngine e;
        e.prepare (kRate, 512);
        const int second = int (kRate);

        beginTest ("poly plays a chord, mono and legato play one voice");
        for (const char* mode : { "poly", "mono", "legato" })
        {
            auto s = basePatch();
            s[kPlayMode] = option (kPlayMode, mode);
            e.reset();
            e.setControls (s);
            renderEvents (e, { 64 }, 4800, { { 0, 60, 100 }, { 10, 64, 100 } });
            expectEquals (e.numActiveVoices(), juce::String (mode) == "poly" ? 2 : 1, mode);
        }

        beginTest ("mono returns to the held note; legato does not retrigger");
        {
            // Envelope decays to silence (sustain 0); a second key while the first
            // is held retriggers in mono, but stays silent in legato.
            auto s = basePatch();
            s[kAmpDecay] = fromUnitsFor (kAmpDecay, -4.0f); // 62.5 ms
            s[kAmpSustain] = 0.0f;
            for (const char* mode : { "mono", "legato" })
            {
                s[kPlayMode] = option (kPlayMode, mode);
                const auto out = capture (e, s, second, { { 0, 60, 100 }, { second / 2, 67, 100 } });
                const float after = rmsOf (out, size_t (second / 2 + 100), size_t (second / 2 + 2400));
                if (juce::String (mode) == "mono")
                    expect (after > 0.01f, "mono retrigger rms " + juce::String (after));
                else
                    expect (after < 1.0e-3f, "legato rms " + juce::String (after));
            }

            s[kPlayMode] = option (kPlayMode, "mono");
            s[kAmpSustain] = 1.0f;
            const auto out = capture (e, s, second, { { 0, 60, 100 }, { 4800, 72, 100 }, { second / 2, 72, 0 } });
            const float hz = zeroCrossingHz (out, size_t (second / 2 + 2400), size_t (second - 100));
            expectWithinAbsoluteError (hz, 261.6f, 15.0f); // back on note 60
        }

        beginTest ("glide moves the pitch over time");
        {
            auto s = basePatch();
            s[kPlayMode] = option (kPlayMode, "legato");
            s[kGlideTime] = 0.75f; // 2^-0.5 s ~ 0.7 s
            const auto out = capture (e, s, 2 * second, { { 0, 48, 100 }, { 4800, 72, 100 } });
            const float early = zeroCrossingHz (out, 4800, 4800 + 4800);
            const float late = zeroCrossingHz (out, size_t (second + 4800), size_t (second + 14400));
            expect (early < 0.7f * late, "early " + juce::String (early) + " late " + juce::String (late));
            expectWithinAbsoluteError (late, 523.25f, 20.0f);

            s[kGlideTime] = 0.0f;
            const auto jump = capture (e, s, second, { { 0, 48, 100 }, { 4800, 72, 100 } });
            expectWithinAbsoluteError (zeroCrossingHz (jump, 4900, 14400), 523.25f, 20.0f);
        }

        beginTest ("every osc wave is audible and distinct");
        {
            std::vector<std::vector<float>> waves;
            for (int w = 0; w < control (kOsc1Wave).numOptions; ++w)
            {
                auto s = basePatch();
                s[kOsc1Wave] = float (w);
                waves.push_back (capture (e, s, 8192, { { 0, 57, 100 } }));
                expect (rmsOf (waves.back(), 0, 8192) > 0.01f, control (kOsc1Wave).options[w]);
            }
            for (size_t a = 0; a < waves.size(); ++a)
                for (size_t b = a + 1; b < waves.size(); ++b)
                    expect (std::abs (correlation (waves[a], waves[b])) < 0.98f,
                            juce::String (control (kOsc1Wave).options[a]) + " vs " + control (kOsc1Wave).options[b]);
        }

        beginTest ("osc octave, pitch, level and routing");
        {
            auto s = basePatch();
            s[kOsc1Octave] = option (kOsc1Octave, "+1");
            expectWithinAbsoluteError (zeroCrossingHz (capture (e, s, 9600, { { 0, 57, 100 } }), 100, 9600), 440.0f,
                                       15.0f);
            s = basePatch();
            s[kOsc1Pitch] = fromUnitsFor (kOsc1Pitch, 7.0f);
            expectWithinAbsoluteError (zeroCrossingHz (capture (e, s, 9600, { { 0, 57, 100 } }), 100, 9600), 329.6f,
                                       15.0f);
            s = basePatch();
            const float full = rmsOf (capture (e, s, 4800, { { 0, 57, 100 } }), 0, 4800);
            s[kOsc1Level] = 0.5f; // x^3 = -18 dB
            const float low = rmsOf (capture (e, s, 4800, { { 0, 57, 100 } }), 0, 4800);
            expectWithinAbsoluteError (low / full, 0.125f, 0.01f);
        }

        beginTest ("serial and parallel filter routing differ");
        {
            auto s = basePatch ("saw");
            s[kFilter1Type] = option (kFilter1Type, "lp24");
            s[kFilter1Cutoff] = 0.35f;
            s[kFilter2Type] = option (kFilter2Type, "hp24");
            s[kFilter2Cutoff] = 0.85f;
            s[kFilterRouting] = option (kFilterRouting, "serial");
            const float serial = rmsOf (capture (e, s, 9600, { { 0, 48, 100 } }), 0, 9600);
            s[kFilterRouting] = option (kFilterRouting, "parallel");
            const float parallel = rmsOf (capture (e, s, 9600, { { 0, 48, 100 } }), 0, 9600);
            expect (serial < 0.3f * parallel,
                    "serial " + juce::String (serial) + " parallel " + juce::String (parallel));

            // An osc routed to filter 2 skips filter 1 in serial mode.
            s[kFilterRouting] = option (kFilterRouting, "serial");
            s[kFilter2Type] = option (kFilter2Type, "off");
            s[kOsc1Filter] = option (kOsc1Filter, "filter2");
            const float skipped = rmsOf (capture (e, s, 9600, { { 0, 48, 100 } }), 0, 9600);
            s[kOsc1Filter] = option (kOsc1Filter, "filter1");
            const float filtered = rmsOf (capture (e, s, 9600, { { 0, 48, 100 } }), 0, 9600);
            expect (filtered < 0.5f * skipped);
        }

        beginTest ("filter EG amount and keytrack move the cutoff");
        {
            auto s = basePatch ("saw");
            s[kFilter1Type] = option (kFilter1Type, "lp24");
            s[kFilter1Cutoff] = 0.3f;
            s[kOsc1Filter] = option (kOsc1Filter, "filter1");
            const float base = rmsOf (capture (e, s, 9600, { { 0, 48, 100 } }), 0, 9600);
            s[kFilter1Env] = 1.0f; // +96 st at full filter EG (sustain 1)
            const float opened = rmsOf (capture (e, s, 9600, { { 0, 48, 100 } }), 0, 9600);
            expect (opened > 2.0f * base);
            s[kFilter1Env] = 0.5f;
            s[kFilter1Keytrack] = 1.0f;
            const float tracked = rmsOf (capture (e, s, 9600, { { 0, 84, 100 } }), 0, 9600);
            s[kFilter1Keytrack] = 0.5f;
            const float untracked = rmsOf (capture (e, s, 9600, { { 0, 84, 100 } }), 0, 9600);
            expect (tracked > 2.0f * untracked);
        }

        beginTest ("high resonance self-oscillates from a near-silent input and stays finite");
        {
            auto s = basePatch ("noise");
            s[kOsc1Level] = 0.02f; // ~-100 dB
            s[kFilter1Type] = option (kFilter1Type, "hp24");
            s[kFilter1Cutoff] = 0.6f;
            const auto quiet = [&] (float res)
            {
                s[kFilter1Resonance] = res;
                e.reset();
                e.setControls (s);
                return render (e, { 512 }, second, -1, second / 2);
            };
            const auto ringing = quiet (0.92f);
            expect (ringing.finite);
            expect (ringing.rms() > 1.0e-3f, "rings: " + juce::String (ringing.rms()));
            expect (ringing.peak < 4.0f, "bounded: " + juce::String (ringing.peak));
            expect (quiet (0.3f).rms() < 1.0e-4f, "low resonance does not ring");
        }

        beginTest ("envelope curve options differ");
        {
            auto s = basePatch();
            s[kAmpAttack] = fromUnitsFor (kAmpAttack, 0.0f); // 1 s
            s[kAmpAttackCurve] = option (kAmpAttackCurve, "convex");
            const float convex = rmsOf (capture (e, s, second / 2, { { 0, 57, 100 } }), 9600, 14400);
            s[kAmpAttackCurve] = option (kAmpAttackCurve, "concave");
            const float concave = rmsOf (capture (e, s, second / 2, { { 0, 57, 100 } }), 9600, 14400);
            expect (convex > 2.0f * concave, "convex " + juce::String (convex) + " concave " + juce::String (concave));

            s = basePatch();
            s[kAmpDecay] = fromUnitsFor (kAmpDecay, 0.0f);
            s[kAmpSustain] = 0.0f;
            s[kAmpDecayCurve] = option (kAmpDecayCurve, "linear");
            const float linear = rmsOf (capture (e, s, second / 2, { { 0, 57, 100 } }), 9600, 14400);
            s[kAmpDecayCurve] = option (kAmpDecayCurve, "cubic");
            const float cubic = rmsOf (capture (e, s, second / 2, { { 0, 57, 100 } }), 9600, 14400);
            expect (linear > 1.5f * cubic, "linear " + juce::String (linear) + " cubic " + juce::String (cubic));
        }

        beginTest ("LFO -> volume creates amplitude variation; amount 0 does not");
        {
            for (int l = 0; l < 2; ++l)
            {
                const int base = l == 0 ? kLfo1Shape : kLfo2Shape;
                auto s = basePatch();
                s[base + 1] = fromUnitsFor (base + 1, 2.0f); // 4 Hz
                s[base + 7] = option (base + 7, "volume");
                s[base + 8] = 0.0f; // depth -1
                const auto out = capture (e, s, second, { { 0, 57, 100 } });
                float lo = 1.0e9f, hi = 0.0f;
                for (size_t w = 0; w + 480 <= out.size(); w += 480)
                {
                    lo = std::min (lo, rmsOf (out, w, w + 480));
                    hi = std::max (hi, rmsOf (out, w, w + 480));
                }
                expect (hi > 4.0f * lo + 1.0e-4f,
                        "lfo " + juce::String (l + 1) + " hi " + juce::String (hi) + " lo " + juce::String (lo));

                s[base + 2] = 0.0f; // amount 0
                const auto flat = capture (e, s, second, { { 0, 57, 100 } });
                expectWithinAbsoluteError (rmsOf (flat, 9600, 14400), rmsOf (flat, 24000, 28800), 0.002f);
            }
        }

        beginTest ("every provisional destination does something");
        for (int d = 1; d < control (kLfo1Destination).numOptions; ++d)
        {
            auto s = basePatch ("saw");
            s[kFilter1Type] = option (kFilter1Type, "lp12");
            s[kFilter2Type] = option (kFilter2Type, "lp12");
            s[kFilter1Cutoff] = 0.5f;
            s[kFilter2Cutoff] = 0.5f;
            const auto plain = capture (e, s, 9600, { { 0, 57, 100 } });
            s[kFegDestination] = float (d);
            s[kFegDepth] = 0.8f;
            const auto routed = capture (e, s, 9600, { { 0, 57, 100 } });
            expect (correlation (plain, routed) < 0.999f
                        || std::abs (rmsOf (plain, 0, 9600) - rmsOf (routed, 0, 9600)) > 1.0e-3f,
                    control (kFegDestination).options[d]);
        }

        beginTest ("each FX changes the output");
        {
            struct Case
            {
                int on;
                int tweak;
                float value;
            };
            const Case cases[] = {
                { kDistOn, kDistDrive, 0.9f },   { kEqOn, kEqLowGain, 0.8f },    { kChorusOn, kChorusDepth, 0.8f },
                { kModfxOn, kModfxDepth, 0.9f }, { kDelayOn, kDelayTime, 0.4f }, { kReverbOn, kReverbMix, 0.6f },
            };
            auto s = basePatch ("saw");
            const auto dry = capture (e, s, second / 2, { { 0, 57, 100 }, { 9600, 57, 0 } });
            for (const auto& c : cases)
                for (int type = 0; type < (c.on == kModfxOn ? 2 : 1); ++type)
                {
                    auto w = s;
                    w[c.on] = 1.0f;
                    w[c.tweak] = c.value;
                    w[kModfxType] = float (type);
                    const auto wet = capture (e, w, second / 2, { { 0, 57, 100 }, { 9600, 57, 0 } });
                    float diff = 0.0f;
                    for (size_t i = 0; i < dry.size(); ++i)
                        diff = std::max (diff, std::abs (dry[i] - wet[i]));
                    expect (diff > 1.0e-3f, juce::String (control (c.on).id) + " type " + juce::String (type));
                }
        }

        beginTest ("random patches and chords in every play mode stay finite");
        {
            juce::Random rng (99);
            for (int trial = 0; trial < 60; ++trial)
            {
                e.reset();
                e.setControls (randomSnapshot (rng));
                std::vector<NoteEvent> ev;
                for (int k = 0; k < 6; ++k)
                    ev.push_back ({ rng.nextInt (8000), 36 + rng.nextInt (60), rng.nextBool() ? 100 : 0 });
                const auto r = renderEvents (e, { 33, 128, 7 }, 12000, ev);
                expect (r.finite);
                expect (r.peak <= 4.0f);
            }
        }

        beginTest ("voice stealing past 16 voices");
        {
            e.reset();
            e.setControls (Snapshot::defaults());
            testutil::Player p (e);
            for (int n = 0; n < 20; ++n)
                p.block (64, { juce::MidiMessage::noteOn (1, 40 + n, (juce::uint8) 100) });
            expectEquals (e.numActiveVoices(), LiteEngine::kMaxVoices);
            for (int n = 0; n < 20; ++n)
                p.block (64, { juce::MidiMessage::noteOff (1, 40 + n) });
            p.run (2 * second);
            expectEquals (e.numActiveVoices(), 0);
            expect (p.finite);
        }

        beginTest ("delay switched off and on again starts empty (no stale echo)");
        {
            e.reset();
            auto s = Snapshot::defaults();
            s[kDelayOn] = 1.0f;
            s[kDelayMix] = 1.0f;
            s[kDelayFeedback] = 0.9f;
            s[kDelayTime] = fromUnitsFor (kDelayTime, -1.0f); // 0.5 s
            e.setControls (s);
            testutil::Player p (e);
            p.block (256, { juce::MidiMessage::noteOn (1, 60, (juce::uint8) 100) });
            p.run (4800);
            p.block (256, { juce::MidiMessage::noteOff (1, 60) });
            p.run (second);
            expect (p.tailPeak (4800) > 1.0e-3f, "echoes ring while on");
            e.setControl (kDelayOn, 0.0f);
            p.run (256);
            e.setControl (kDelayOn, 1.0f);
            p.run (second);
            expect (p.tailPeak (size_t (second)) < 1.0e-6f,
                    "re-enabled delay is silent: " + juce::String (p.tailPeak (size_t (second))));
        }
    }

private:
    static float fromUnitsFor (int index, float units) { return schema::surge::fromUnits (index, units); }
};

static LiteEngineTests liteEngineTests;

} // namespace vae
