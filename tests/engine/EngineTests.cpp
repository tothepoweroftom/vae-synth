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

static SchemaTests schemaTests;
static EngineTests engineTests;

} // namespace vae
