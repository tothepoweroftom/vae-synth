#include <juce_events/juce_events.h>

#include "PluginProcessor.h"
#include "../engine/EngineTestUtils.h"

namespace vae
{
namespace
{
/** Pump the message loop until `done` or the timeout. */
bool pumpUntil (const std::function<bool()>& done, int timeoutMs = 5000)
{
    const auto end = juce::Time::getMillisecondCounter() + (juce::uint32) timeoutMs;
    while (! done())
    {
        if (juce::Time::getMillisecondCounter() > end)
            return false;
        juce::MessageManager::getInstance()->runDispatchLoopUntil (5);
    }
    return true;
}

/** Wait until every decoder job queued so far has called back: the worker
    runs jobs in order and callbacks are posted in order, so a job queued now
    calls back after all of them. */
void settleModel (PluginProcessor& p)
{
    bool done = false;
    p.getGenerator().generateFromLatent ({}, [&done] (const ml::PresetGenerator::Result&) { done = true; });
    pumpUntil ([&] { return done; });
}

juce::MemoryBlock stateOf (PluginProcessor& p)
{
    juce::MemoryBlock mb;
    p.getStateInformation (mb);
    return mb;
}

struct GestureCounter final : juce::AudioProcessorListener
{
    void audioProcessorParameterChanged (juce::AudioProcessor*, int, float) override {}
    void audioProcessorChanged (juce::AudioProcessor*, const ChangeDetails&) override {}
    void audioProcessorParameterChangeGestureBegin (juce::AudioProcessor*, int) override { ++begins; }
    void audioProcessorParameterChangeGestureEnd (juce::AudioProcessor*, int) override { ++ends; }
    int begins = 0, ends = 0;
};
} // namespace

class PluginTests final : public juce::UnitTest
{
public:
    PluginTests()
        : juce::UnitTest ("Plug-in processor", "vae_plugin")
    {
    }

    void runTest() override
    {
        const auto dir = juce::File::createTempFile ("vae_plugin_tests");
        dir.createDirectory();
        const auto model = dir.getChildFile ("roundtrip_decoder.onnx");
        juce::File (VAE_STUB_DECODER_PATH).copyFileTo (model);
        juce::Random rng (99);

        beginTest ("state round trip: controls, engine, prompt, model path");
        {
            PluginProcessor a;
            const auto controls = randomSnapshot (rng);
            a.applyControls (controls);
            a.getParams().set (PluginProcessor::kEngineParamId, float (a.getEngineKinds().size() - 1));

            bool loaded = false, generated = false;
            a.loadDecoderModel (model,
                                [&] (juce::Result r)
                                {
                                    expect (r.wasOk(), r.getErrorMessage());
                                    loaded = true;
                                });
            expect (pumpUntil ([&] { return loaded; }));
            a.generateFromPrompt ("dusty tape pad", [&] (const juce::String&) { generated = true; });
            expect (pumpUntil ([&] { return generated; }));
            a.applyControls (controls); // the generation replaced them

            const auto state = stateOf (a);
            PluginProcessor b;
            b.setStateInformation (state.getData(), (int) state.getSize());
            expect (pumpUntil ([&] { return b.describeModel().contains ("roundtrip_decoder.onnx"); }),
                    b.describeModel());

            expect (b.getControls().values == a.getControls().values, "controls differ after round trip");
            expectEquals (b.getParams().get (PluginProcessor::kEngineParamId),
                          a.getParams().get (PluginProcessor::kEngineParamId));
            expectEquals (b.getLastPrompt(), juce::String ("dusty tape pad"));
            expect (b.getDecoderModelFile() == model);
        }

        beginTest ("a missing model file keeps its saved path and says so");
        {
            PluginProcessor a;
            bool loaded = false;
            a.loadDecoderModel (model, [&] (juce::Result) { loaded = true; });
            expect (pumpUntil ([&] { return loaded; }));
            const auto state = stateOf (a);

            const auto moved = dir.getChildFile ("moved.onnx");
            expect (model.moveFileTo (moved));

            PluginProcessor b;
            b.setStateInformation (state.getData(), (int) state.getSize());
            settleModel (b); // every pending load (constructor stub, state load) has called back
            expect (b.getDecoderModelFile() == model, b.getDecoderModelFile().getFullPathName());
            expect (b.describeModel().contains ("not found"), b.describeModel());
            expect (b.describeModel().contains ("stub_decoder.onnx (embedded)"), b.describeModel());

            const auto saved = stateOf (b);
            const auto again = juce::AudioProcessor::getXmlFromBinary (saved.getData(), (int) saved.getSize());
            expect (again != nullptr && again->getStringAttribute ("decoderModel") == model.getFullPathName());
            expect (moved.moveFileTo (model));
        }

        beginTest ("a different schema version keeps the default controls");
        {
            PluginProcessor a;
            a.applyControls (randomSnapshot (rng));
            const auto block = stateOf (a);
            auto xml = juce::AudioProcessor::getXmlFromBinary (block.getData(), (int) block.getSize());
            expect (xml != nullptr);
            xml->setAttribute ("schemaVersion", schema::kSchemaVersion - 1);
            juce::MemoryBlock old;
            juce::AudioProcessor::copyXmlToBinary (*xml, old);

            PluginProcessor b;
            b.setStateInformation (old.getData(), (int) old.getSize());
            expect (b.getControls().values == schema::Snapshot::defaults().values);
            settleModel (b);
        }

        beginTest ("applyControls sets every control with balanced gestures");
        {
            PluginProcessor p;
            GestureCounter gestures;
            p.addListener (&gestures);
            const auto controls = randomSnapshot (rng);
            p.applyControls (controls);
            p.removeListener (&gestures);
            expectEquals (gestures.begins, schema::kControlCount);
            expectEquals (gestures.ends, schema::kControlCount);
            auto expected = controls;
            expected.sanitise();
            for (int i = 0; i < schema::kControlCount; ++i)
                expectWithinAbsoluteError (p.getControls()[i], expected[i], 1.0e-5f);
            settleModel (p);
        }

        beginTest ("a rejected decoder keeps the previous model and its saved path");
        {
            PluginProcessor p;
            bool loaded = false;
            p.loadDecoderModel (model,
                                [&] (juce::Result r)
                                {
                                    expect (r.wasOk());
                                    loaded = true;
                                });
            expect (pumpUntil ([&] { return loaded; }));

            juce::Result rejected = juce::Result::ok();
            bool done = false;
            p.loadDecoderModel (juce::File (VAE_ML_FIXTURES).getChildFile ("rank1.onnx"),
                                [&] (juce::Result r)
                                {
                                    rejected = r;
                                    done = true;
                                });
            expect (pumpUntil ([&] { return done; }));
            expect (rejected.failed());
            expect (p.getDecoderModelFile() == model, p.getDecoderModelFile().getFullPathName());
            expect (p.describeModel().contains ("roundtrip_decoder.onnx"), p.describeModel());
        }

        dir.deleteRecursively();
    }
};

static PluginTests pluginTests;

} // namespace vae
