#include <juce_events/juce_events.h>

#include "ml/OnnxModel.h"
#include "ml/PresetGenerator.h"

namespace vae::ml
{

class MlTests final : public juce::UnitTest
{
public:
    MlTests() : juce::UnitTest ("ONNX / prompt pipeline", "vae_ml") {}

    void runTest() override
    {
        const juce::File stub (VAE_STUB_DECODER_PATH);

        beginTest ("runtime loads");
        OnnxModel model;
        logMessage (model.getRuntimeStatus());
        expect (model.isRuntimeAvailable(), model.getRuntimeStatus());

        beginTest ("stub decoder I/O");
        expect (model.load (stub).wasOk());
        expectEquals ((int) model.getInputs().size(), 1);
        expect (model.getInputs()[0].shape.back() == schema::kLatentDimensions);
        expect (model.getOutputs()[0].shape.back() == schema::kFeatureCount);

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
        generator.loadDecoder (stub, [&] (juce::Result r) { expect (r.wasOk()); loaded = true; });
        generator.generate ("glassy bell", [&] (const PresetGenerator::Result& r) { result = r; generated = true; });
        for (int i = 0; i < 200 && ! (loaded && generated); ++i)
            juce::MessageManager::getInstance()->runDispatchLoopUntil (10);
        expect (loaded && generated);
        expect (result.controls.isValid());
        expectEquals (result.prompt, juce::String ("glassy bell"));
        logMessage ("note: " + result.note);
        expect (result.note.contains ("ONNX decoder"), result.note);
    }
};

static MlTests mlTests;

} // namespace vae::ml
