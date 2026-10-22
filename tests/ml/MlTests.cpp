#include <juce_events/juce_events.h>

#include <atomic>

#include "ml/OnnxModel.h"
#include "ml/PresetGenerator.h"

namespace vae::ml
{

class MlTests final : public juce::UnitTest
{
public:
    MlTests()
        : juce::UnitTest ("ONNX / prompt pipeline", "vae_ml")
    {
    }

    void runTest() override
    {
        const juce::File stub (VAE_STUB_DECODER_PATH);

        if (juce::SystemStats::getEnvironmentVariable ("VAE_DATA_DIR", {}).isNotEmpty())
        {
            beginTest ("explicit data directory isolates installed model lookup");
            expect (installedModels ("joint-vae/bundle.json")
                    == appDataDir().getChildFile ("models/joint-vae/bundle.json"));
            expect (installedModels ("surge-data") == appDataDir().getChildFile ("models/surge-data"));
        }

        beginTest ("runtime loads");
        OnnxModel model;
        logMessage (model.getRuntimeStatus());
        expect (model.isRuntimeAvailable(), model.getRuntimeStatus());

        beginTest ("stub decoder I/O");
        expect (model.load (stub).wasOk());
        expectEquals ((int) model.getInputs().size(), 1);
        expect (model.getInputs()[0].shape.back() == schema::kLatentDimensions);
        expect (model.getOutputs()[0].shape.back() == schema::kFeatureCount);

        beginTest ("tensor input validation rejects missing data and shapes");
        {
            std::vector<std::vector<float>> out;
            std::array<float, 8> values {};
            std::array<int64_t, 2> badShape { 1, 9 };
            expect (model.run (values, badShape, out).failed());
            expect (model.run (std::span<const FloatTensorView> {}, out).failed());
            const std::array<FloatTensorView, 2> extra { FloatTensorView { values, badShape }, { values, badShape } };
            expect (model.run (extra, out).failed());
        }

        beginTest ("missing model fails cleanly");
        OnnxModel missing;
        expect (missing.load (juce::File ("/definitely/not/here.onnx")).failed());
        expect (! missing.isLoaded());

        beginTest ("decoder: zero latent gives the schema defaults");
        LatentDecoder decoder;
        expect (decoder.loadModel (stub).wasOk());
        expect (decoder.hasModel());
        juce::String error;
        const auto s = decoder.decode ({}, &error);
        expect (error.isEmpty(), error);
        expect (s.isValid());
        for (int i = 0; i < schema::kControlCount; ++i)
            expectWithinAbsoluteError (s[i], schema::controls()[(size_t) i].defaultValue, 2.0e-3f);

        beginTest ("decoder: latents move the controls");
        schema::Latent l;
        l.fill (0.8f);
        const auto moved = decoder.decode (l);
        expect (moved.isValid());
        int changed = 0;
        for (int i = 0; i < schema::kControlCount; ++i)
            changed += std::abs (moved[i] - s[i]) > 0.05f ? 1 : 0;
        expect (changed >= 4, juce::String (changed) + " controls changed");

        beginTest ("decoder without a model falls back to the mock");
        LatentDecoder none;
        expect (! none.hasModel());
        const auto mock = none.decode (l);
        const auto expected = schema::mockDecode (l);
        expect (mock.values == expected.values);

        beginTest ("hash prompt encoder");
        HashPromptEncoder enc;
        const auto a = enc.encode ("Warm detuned pad");
        expect (a == enc.encode ("warm, detuned PAD!"));
        expect (a != enc.encode ("plucky bass"));
        for (auto v : a)
            expect (v >= -1.0f && v <= 1.0f);
        const auto empty = enc.encode ("   ");
        expect (std::all_of (empty.begin(), empty.end(), [] (float v) { return juce::exactlyEqual (v, 0.0f); }));

        beginTest ("generator end to end (worker thread -> message thread)");
        PresetGenerator generator;
        bool loaded = false, generated = false;
        PresetGenerator::Result result;
        generator.loadDecoder (stub,
                               [&] (juce::Result r)
                               {
                                   expect (r.wasOk());
                                   loaded = true;
                               });
        generator.generate ("glassy bell",
                            [&] (const PresetGenerator::Result& r)
                            {
                                result = r;
                                generated = true;
                            });
        for (int i = 0; i < 200 && ! (loaded && generated); ++i)
            juce::MessageManager::getInstance()->runDispatchLoopUntil (10);
        expect (loaded && generated);
        expect (result.controls.isValid());
        expectEquals (result.prompt, juce::String ("glassy bell"));
        logMessage ("note: " + result.note);
        expect (result.note.contains ("ONNX decoder"), result.note);

        beginTest ("rejected models keep the working decoder");
        const juce::File fixtures (VAE_ML_FIXTURES);
        LatentDecoder keep;
        expect (keep.loadModel (stub).wasOk());
        for (auto fixture : { "batch2.onnx", "rank1.onnx", "missing.onnx" })
        {
            const auto r = keep.loadModel (fixtures.getChildFile (fixture));
            expect (r.failed(), juce::String (fixture) + " should be rejected");
            expect (keep.hasModel(), juce::String (fixture) + " dropped the working model");
            expect (keep.describe().contains ("stub_decoder.onnx"), keep.describe());
        }

        beginTest ("non-finite decoder output falls back to the mock with an error");
        LatentDecoder nanDecoder;
        expect (nanDecoder.loadModel (fixtures.getChildFile ("nan.onnx")).wasOk());
        juce::String nanError;
        const auto fallback = nanDecoder.decode (l, &nanError);
        expect (nanError.contains ("non-finite"), nanError);
        expect (fallback.values == schema::mockDecode (l).values);

        beginTest ("describe() does not wait for a running job");
        struct BlockingEncoder final : PromptEncoder
        {
            std::atomic<bool>& entered;
            std::atomic<bool>& release;
            BlockingEncoder (std::atomic<bool>& e, std::atomic<bool>& r)
                : entered (e)
                , release (r)
            {
            }
            juce::String describe() const override { return "blocking"; }
            schema::Latent encode (const juce::String&) override
            {
                entered = true;
                while (! release.load())
                    juce::Thread::sleep (1);
                return {};
            }
        };
        std::atomic<bool> entered { false }, release { false };
        PresetGenerator busy;
        busy.setPromptEncoder (std::make_unique<BlockingEncoder> (entered, release));
        busy.generate ("x", [] (const PresetGenerator::Result&) {});
        for (int i = 0; i < 500 && ! entered.load(); ++i)
            juce::Thread::sleep (2);
        expect (entered.load(), "job started");
        const auto t0 = juce::Time::getMillisecondCounterHiRes();
        const auto status = busy.describe(); // the job holds the decoder lock right now
        expect (juce::Time::getMillisecondCounterHiRes() - t0 < 100.0, "describe() blocked");
        expect (status.startsWith ("blocking"), status);
        release = true;
    }
};

static MlTests mlTests;

} // namespace vae::ml
