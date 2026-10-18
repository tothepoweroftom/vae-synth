#include <juce_core/juce_core.h>

#include "engine/SynthEngine.h"
#include "schema/SurgeCurves.h"

#if VAE_WITH_SURGE
 #include "engine/surge/SurgeEngine.h"
#endif

namespace vae
{
namespace
{
constexpr double kRate = 48000.0;

struct RenderStats
{
    float peak = 0.0f;
    double sumSquares = 0.0;
    int64_t samples = 0;
    bool finite = true;

    float rms() const { return samples > 0 ? float (std::sqrt (sumSquares / double (samples))) : 0.0f; }
};

/** Render `blocks` (sizes cycled) with a note on at sample 0 and off at noteOffAt. */
RenderStats render (SynthEngine& engine, std::initializer_list<int> blockSizes, int totalSamples,
                    int noteOffAt = -1, int startMeasuringAt = 0)
{
    RenderStats stats;
    int position = 0;
    auto size = blockSizes.begin();
    while (position < totalSamples)
    {
        const int n = juce::jmin (*size, totalSamples - position);
        if (++size == blockSizes.end())
            size = blockSizes.begin();

        juce::AudioBuffer<float> buffer (2, juce::jmax (1, n));
        buffer.setSize (2, n, false, false, true);
        juce::MidiBuffer midi;
        if (position == 0)
            midi.addEvent (juce::MidiMessage::noteOn (1, 60, (juce::uint8) 110), 0);
        if (noteOffAt >= position && noteOffAt < position + n)
            midi.addEvent (juce::MidiMessage::noteOff (1, 60), noteOffAt - position);

        engine.process (buffer, midi);

        for (int ch = 0; ch < 2; ++ch)
            for (int i = 0; i < n; ++i)
            {
                const float s = buffer.getSample (ch, i);
                stats.finite = stats.finite && std::isfinite (s);
                if (position + i >= startMeasuringAt)
                {
                    stats.peak = juce::jmax (stats.peak, std::abs (s));
                    stats.sumSquares += double (s) * double (s);
                    ++stats.samples;
                }
            }
        position += n;
    }
    return stats;
}

const char* engineLabel (EngineKind k) { return k == EngineKind::Surge ? "Surge" : "Lite"; }
} // namespace

//==============================================================================
class SchemaTests final : public juce::UnitTest
{
public:
    SchemaTests() : juce::UnitTest ("Schema", "vae_engine") {}

    void runTest() override
    {
        beginTest ("generated table");
        expectEquals (schema::kControlCount, 32);
        expectEquals (schema::kLatentDimensions, 8);
        expect (schema::Snapshot::defaults().isValid());
        expect (schema::indexOf ("cutoff").has_value());
        expect (! schema::indexOf ("nope").has_value());

        beginTest ("mock decoder matches VAE_SURGE snapshot_from_latent");
        schema::Latent zero {};
        const auto s = schema::mockDecode (zero);
        for (int i = 0; i < schema::kControlCount; ++i)
            expectWithinAbsoluteError (s[i], 0.5f * schema::controls()[(size_t) i].defaultValue + 0.25f, 1.0e-6f);

        beginTest ("sanitise");
        auto bad = schema::Snapshot::defaults();
        bad[0] = std::numeric_limits<float>::quiet_NaN();
        bad[1] = 3.0f;
        expect (! bad.isValid());
        bad.sanitise();
        expect (bad.isValid());
        expectEquals (bad[1], 1.0f);

        beginTest ("Surge curves");
        namespace sc = schema::surge;
        for (int steps : { 2, 7, 12, 16, 36 })
            for (int k = 0; k < steps; ++k)
                expectEquals (sc::stepFromNorm (sc::normFromStep (k, steps), steps), k);
        expectEquals (sc::octave (0.5f), 0);
        expectEquals (sc::filterType (0.0f), 0);   // Surge filter "Off"
        expectWithinAbsoluteError (sc::cutoffHz (0.0f), 13.75f, 0.01f);
        expectWithinAbsoluteError (sc::envSeconds (8.0f / 13.0f), 1.0f, 1.0e-5f);
        expectWithinAbsoluteError (sc::globalVolumeDb (1.0f), 0.0f, 1.0e-6f);
        for (int i = 0; i < schema::kControlCount; ++i)
            expect (! sc::describe (i, 0.37f).empty());
    }
};

//==============================================================================
class EngineTests final : public juce::UnitTest
{
public:
    EngineTests() : juce::UnitTest ("Engines", "vae_engine") {}

    void runTest() override
    {
        for (auto kind : availableEngines())
        {
            auto engine = createEngine (kind);
            expect (engine != nullptr);
            engine->prepare (kRate, 512);
            logMessage (juce::String (engineLabel (kind)) + " engine");

            beginTest (juce::String (engineLabel (kind)) + ": note renders at awkward block sizes");
            const auto on = render (*engine, { 1, 31, 32, 33, 64, 512, 7 }, int (kRate));
            expect (on.finite);
            expect (on.peak > 1.0e-3f, "peak " + juce::String (on.peak));
            logMessage ("  default patch: peak " + juce::String (on.peak, 4) + ", rms " + juce::String (on.rms(), 4));

            beginTest (juce::String (engineLabel (kind)) + ": release decays");
            engine->reset();
            const auto tail = render (*engine, { 256 }, int (kRate * 3.0), int (kRate * 0.5), int (kRate * 2.5));
            expect (tail.finite);
            expect (tail.peak < 1.0e-3f, "tail peak " + juce::String (tail.peak));

            beginTest (juce::String (engineLabel (kind)) + ": every control at 0 and 1 stays finite");
            for (int i = 0; i < schema::kControlCount; ++i)
                for (float v : { 0.0f, 1.0f })
                {
                    engine->reset();
                    auto s = schema::Snapshot::defaults();
                    s[i] = v;
                    engine->setControls (s);
                    const auto r = render (*engine, { 128 }, 4096);
                    expect (r.finite, juce::String (schema::controls()[(size_t) i].id) + " = " + juce::String (v));
                }

            beginTest (juce::String (engineLabel (kind)) + ": random vectors stay finite");
            juce::Random rng (1234);
            for (int trial = 0; trial < 24; ++trial)
            {
                engine->reset();
                schema::Snapshot s;
                for (auto& v : s.values)
                    v = rng.nextFloat();
                engine->setControls (s);
                const auto r = render (*engine, { 96 }, 8192);
                expect (r.finite);
                expect (r.peak < 8.0f, "peak " + juce::String (r.peak));
            }

            beginTest (juce::String (engineLabel (kind)) + ": cutoff is audible");
            engine->reset();
            auto bright = schema::Snapshot::defaults();
            bright[schema::kFilterType] = schema::surge::normFromStep (2, 36);   // LP 24
            bright[schema::kCutoff] = 1.0f;
            engine->setControls (bright);
            const auto open = render (*engine, { 256 }, 16384);
            engine->reset();
            auto dark = bright;
            dark[schema::kCutoff] = 0.15f;
            engine->setControls (dark);
            const auto closed = render (*engine, { 256 }, 16384);
            expect (closed.rms() < 0.5f * open.rms(),
                    "open " + juce::String (open.rms()) + " closed " + juce::String (closed.rms()));
        }

#if VAE_WITH_SURGE
        beginTest ("Surge: every schema control resolves to a Surge parameter");
        SurgeEngine surge;
        surge.prepare (kRate, 512);
        expectEquals (surge.numResolvedControls(), schema::kControlCount);
#endif
    }
};

static SchemaTests schemaTests;
static EngineTests engineTests;

} // namespace vae
