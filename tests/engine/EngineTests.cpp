#include "EngineTestUtils.h"

namespace vae
{

//==============================================================================
class SchemaTests final : public juce::UnitTest
{
public:
    SchemaTests() : juce::UnitTest ("Schema", "vae_engine") {}

    void runTest() override
    {
        using namespace schema;

        beginTest ("generated table");
        expectEquals (kSchemaVersion, 2);
        expect (Snapshot::defaults().isValid());
        expect (indexOf ("filter1_cutoff").has_value());
        expect (! indexOf ("nope").has_value());
        expectEquals (*optionIndex (kPlayMode, "legato"), 2);
        int features = 0;
        for (const auto& c : controls())
        {
            expectEquals (c.firstFeature, features);
            features += featureWidth (c);
            if (c.kind == Kind::Continuous)
                expect (c.curve >= 0, c.id);
            if (c.kind == Kind::Categorical)
                expect (c.numOptions >= 2, c.id);
        }
        expectEquals (features, kFeatureCount);

        beginTest ("features round trip");
        juce::Random rng (7);
        for (int trial = 0; trial < 50; ++trial)
        {
            const auto s = randomSnapshot (rng);
            expect (s.isValid());
            expect (decodeFeatures (encodeFeatures (s)).values == s.values);
        }

        beginTest ("mock decoder keeps discrete defaults at the zero latent");
        const auto m = mockDecode ({});
        const auto d = Snapshot::defaults();
        for (int i = 0; i < kControlCount; ++i)
        {
            if (control (i).kind == Kind::Continuous)
                expectWithinAbsoluteError (m[i], 0.5f * d[i] + 0.25f, 1.0e-6f);
            else
                expectEquals (m[i], d[i]);
        }

        beginTest ("sanitise");
        auto bad = Snapshot::defaults();
        bad[kFilter1Cutoff] = std::numeric_limits<float>::quiet_NaN();
        bad[kOsc1Wave] = 99.0f;
        bad[kOsc2On] = 0.7f;
        expect (! bad.isValid());
        bad.sanitise();
        expect (bad.isValid());
        expectEquals (bad[kOsc1Wave], float (control (kOsc1Wave).numOptions - 1));
        expectEquals (bad[kOsc2On], 1.0f);

        beginTest ("curves");
        namespace sc = schema::surge;
        for (int steps : { 2, 7, 12, 16, 36 })
            for (int k = 0; k < steps; ++k)
                expectEquals (sc::stepFromNorm (sc::normFromStep (k, steps), steps), k);
        expectWithinAbsoluteError (sc::cutoffHz (kFilter1Cutoff, 0.0f), 13.75f, 0.01f);
        expectWithinAbsoluteError (sc::cutoffHz (kFilter1Cutoff, d[kFilter1Cutoff]), 523.25f, 0.1f);
        expectWithinAbsoluteError (sc::seconds (kAmpAttack, 8.0f / 13.0f), 1.0f, 1.0e-5f);
        expectEquals (sc::glideSeconds (0.0f), 0.0f);
        for (int i = 0; i < kControlCount; ++i)
            expect (! sc::describe (i, d[i]).empty());
    }
};

//==============================================================================
class EngineTests final : public juce::UnitTest
{
public:
    EngineTests() : juce::UnitTest ("Engines (generic)", "vae_engine") {}

    void runTest() override
    {
        using namespace schema;

        for (auto kind : availableEngines())
        {
            const juce::String label (engineLabel (kind));
            auto engine = createEngine (kind);
            expect (engine != nullptr);
            engine->prepare (kRate, 512);

            beginTest (label + ": default patch renders at awkward block sizes");
            const auto on = render (*engine, { 1, 31, 32, 33, 64, 512, 7 }, int (kRate));
            expect (on.finite);
            expect (on.peak > 1.0e-3f, "peak " + juce::String (on.peak));
            logMessage (label + " default patch: peak " + juce::String (on.peak, 4) + ", rms " + juce::String (on.rms(), 4));

            beginTest (label + ": release decays");
            engine->reset();
            const auto tail = render (*engine, { 256 }, int (kRate * 3.0), int (kRate * 0.5), int (kRate * 2.5));
            expect (tail.finite);
            expect (tail.peak < 1.0e-3f, "tail peak " + juce::String (tail.peak));

            beginTest (label + ": every control extreme / option stays finite");
            for (int i = 0; i < kControlCount; ++i)
            {
                const auto& c = control (i);
                const int n = c.kind == Kind::Categorical ? c.numOptions : 2;
                for (int k = 0; k < n; ++k)
                {
                    engine->reset();
                    auto s = Snapshot::defaults();
                    s[kOsc2On] = 1.0f;   // more of the patch in play
                    s[kFilter1Type] = 2.0f;
                    s[i] = float (k);
                    engine->setControls (s);
                    const auto r = render (*engine, { 128 }, 4096);
                    expect (r.finite, juce::String (c.id) + " = " + juce::String (k));
                    expect (r.peak < 16.0f, juce::String (c.id) + " peak " + juce::String (r.peak));
                }
            }

            beginTest (label + ": random vectors stay finite and bounded");
            juce::Random rng (1234);
            for (int trial = 0; trial < 40; ++trial)
            {
                engine->reset();
                engine->setControls (randomSnapshot (rng));
                const auto r = render (*engine, { 96 }, 12000);
                expect (r.finite);
                expect (r.peak < 16.0f, "peak " + juce::String (r.peak));
            }

            beginTest (label + ": filter 1 cutoff is audible");
            engine->reset();
            auto bright = Snapshot::defaults();
            bright[kFilter1Type] = float (*optionIndex (kFilter1Type, "lp24"));
            bright[kFilter1Cutoff] = 1.0f;
            bright[kOsc1Filter] = float (*optionIndex (kOsc1Filter, "filter1"));
            engine->setControls (bright);
            const auto open = render (*engine, { 256 }, 16384);
            engine->reset();
            auto dark = bright;
            dark[kFilter1Cutoff] = 0.3f;
            engine->setControls (dark);
            const auto closed = render (*engine, { 256 }, 16384);
            expect (closed.rms() < 0.5f * open.rms(),
                    "open " + juce::String (open.rms()) + " closed " + juce::String (closed.rms()));

            beginTest (label + ": osc on/off is audible");
            engine->reset();
            auto silent = Snapshot::defaults();
            silent[kOsc1On] = 0.0f;
            engine->setControls (silent);
            const auto off = render (*engine, { 256 }, 8192);
            expect (off.peak < 1.0e-4f, "peak with every osc off " + juce::String (off.peak));
        }
    }
};

//==============================================================================
class MidiEdgeCaseTests final : public juce::UnitTest
{
public:
    MidiEdgeCaseTests() : juce::UnitTest ("Engines (MIDI edge cases)", "vae_engine") {}

    void runTest() override
    {
        using namespace schema;
        using testutil::Player;
        using M = juce::MidiMessage;
        const int second = int (kRate);

        for (auto kind : availableEngines())
        {
            const juce::String label (engineLabel (kind));

            beginTest (label + ": switching out of poly with a chord held leaves no stuck notes");
            for (const char* mode : { "mono", "legato" })
            {
                auto engine = createEngine (kind);
                engine->prepare (kRate, 512);
                engine->setControls (Snapshot::defaults());
                Player p (*engine);
                p.block (256, { M::noteOn (1, 60, (juce::uint8) 100), M::noteOn (1, 64, (juce::uint8) 100),
                                M::noteOn (1, 67, (juce::uint8) 100) });
                p.run (4800);
                engine->setControl (kPlayMode, float (*optionIndex (kPlayMode, mode)));
                p.block (256, { M::noteOff (1, 60), M::noteOff (1, 64), M::noteOff (1, 67) });
                p.run (2 * second);
                expect (p.tailPeak (size_t (second / 2)) < 1.0e-3f,
                        juce::String (mode) + " tail peak " + juce::String (p.tailPeak (size_t (second / 2))));
            }

            beginTest (label + ": MIDI in a 0-sample block is not lost");
            {
                auto engine = createEngine (kind);
                engine->prepare (kRate, 512);
                engine->setControls (Snapshot::defaults());
                Player p (*engine);
                p.block (0, { M::noteOn (1, 60, (juce::uint8) 100) });
                p.run (9600);
                expect (p.peak (0, p.left.size()) > 1.0e-3f, "note-on in an empty block sounds");
                p.block (0, { M::noteOff (1, 60) });
                p.run (2 * second);
                expect (p.tailPeak (size_t (second / 2)) < 1.0e-3f,
                        "note-off in an empty block releases: tail " + juce::String (p.tailPeak (size_t (second / 2))));
                expect (p.finite);
            }

            beginTest (label + ": pitch bend moves pitch by up to 2 semitones");
            {
                auto engine = createEngine (kind);
                engine->prepare (kRate, 512);
                engine->setControls (testutil::sinePatch());
                Player p (*engine);
                p.block (256, { M::noteOn (1, 69, (juce::uint8) 100) });
                p.run (9600);
                const float f0 = p.zeroCrossingHz (p.left.size() - 4800, p.left.size(), kRate);
                p.block (256, { M::pitchWheel (1, 16383) });
                p.run (9600);
                const float f1 = p.zeroCrossingHz (p.left.size() - 4800, p.left.size(), kRate);
                logMessage ("  " + label + ": " + juce::String (f0) + " Hz -> " + juce::String (f1) + " Hz bent");
                expectWithinAbsoluteError (f0, 440.0f, 440.0f * 0.03f);
                expectWithinAbsoluteError (f1, 493.88f, 493.88f * 0.03f);
            }

            beginTest (label + ": sustain pedal holds released notes until pedal up");
            {
                auto engine = createEngine (kind);
                engine->prepare (kRate, 512);
                engine->setControls (Snapshot::defaults());
                Player p (*engine);
                p.block (256, { M::controllerEvent (1, 64, 127), M::noteOn (1, 60, (juce::uint8) 100) });
                p.run (4800);
                p.block (256, { M::noteOff (1, 60) });
                p.run (second);
                expect (p.tailPeak (4800) > 1.0e-3f, "still sounding with the pedal down: " + juce::String (p.tailPeak (4800)));
                p.block (256, { M::controllerEvent (1, 64, 0) });
                p.run (2 * second);
                expect (p.tailPeak (size_t (second / 2)) < 1.0e-3f, "released on pedal up: " + juce::String (p.tailPeak (size_t (second / 2))));
            }

            beginTest (label + ": finite and in tune at 22.05 / 44.1 / 96 / 192 kHz");
            for (double rate : { 22050.0, 44100.0, 96000.0, 192000.0 })
            {
                auto engine = createEngine (kind);
                engine->prepare (rate, 512);
                engine->setControls (testutil::sinePatch());
                Player p (*engine);
                p.block (256, { M::noteOn (1, 69, (juce::uint8) 100) });
                p.run (int (rate * 0.3));
                expect (p.finite, juce::String (rate));
                const float hz = p.zeroCrossingHz (p.left.size() - size_t (rate * 0.1), p.left.size(), rate);
                expectWithinAbsoluteError (hz, 440.0f, 440.0f * 0.03f, label + " at " + juce::String (rate) + " Hz");
            }
        }
    }
};

static SchemaTests schemaTests;
static EngineTests engineTests;
static MidiEdgeCaseTests midiEdgeCaseTests;

} // namespace vae
