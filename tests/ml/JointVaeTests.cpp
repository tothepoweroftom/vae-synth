#include "experiments/joint_vae/JointVae.h"
#include <cmath>
#include <limits>

namespace vae::experimental
{
class JointVaeTests final : public juce::UnitTest
{
public:
    JointVaeTests()
        : juce::UnitTest ("Joint VAE experiment", "vae_ml")
    {
    }
    void runTest() override
    {
        beginTest ("local slice preserves anchor and untouched dimensions");
        JointLatent anchor, point;
        for (size_t i = 0; i < anchor.size(); ++i)
            anchor[i] = (float) i / 10;
        expect (JointVae::slice (anchor, 2, 27, 0, 0, 2, point).wasOk());
        expect (point == anchor);
        expect (JointVae::slice (anchor, 2, 27, 0.5f, -1, 2, point).wasOk());
        for (size_t i = 0; i < anchor.size(); ++i)
            expectWithinAbsoluteError (point[i], anchor[i] + (i == 2 ? 1.0f : i == 27 ? -2.0f : 0.0f), 1e-6f);
        beginTest ("invalid slices reject without overwriting output");
        const auto previous = point;
        expect (JointVae::slice (anchor, 2, 2, 0, 0, 1, point).failed());
        expect (JointVae::slice (anchor, 32, 1, 0, 0, 1, point).failed());
        expect (JointVae::slice (anchor, 0, 1, 2, 0, 1, point).failed());
        expect (JointVae::slice (anchor, 0, 1, 0, 0, -1, point).failed());
        anchor[3] = std::numeric_limits<float>::quiet_NaN();
        expect (JointVae::slice (anchor, 0, 1, 0, 0, 1, point).failed());
        expect (point == previous);
        beginTest ("no model never returns a mock preset");
        JointVae model;
        std::vector<float> codes { 42 };
        expect (model.decode ({}, 0, codes).failed());
        expectEquals (codes[0], 42.0f);
        expect (model.load (juce::File ("/nonexistent/bundle.json")).failed());

        beginTest ("cancelled requests cannot publish stale UI results");
        {
            JointVaeWorker worker;
            bool stale = false, done = false;
            worker.decode ({}, 0, [&] (auto, auto) { stale = true; });
            worker.cancel();
            worker.decode ({}, 0,
                           [&] (juce::Result r, auto)
                           {
                               expect (juce::MessageManager::getInstance()->isThisTheMessageThread());
                               expect (r.failed());
                               done = true;
                           });
            const auto deadline = juce::Time::getMillisecondCounter() + 5000;
            while (! done && juce::Time::getMillisecondCounter() < deadline)
                juce::MessageManager::getInstance()->runDispatchLoopUntil (5);
            expect (done);
            expect (! stale);
        }

        // Optional private real-model integration: also run by the export workflow locally.
        const auto path = juce::SystemStats::getEnvironmentVariable ("VAE_JOINT_BUNDLE", "");
        if (path.isEmpty())
        {
            logMessage ("Set VAE_JOINT_BUNDLE to run native parity against a trained bundle.");
            return;
        }
        beginTest ("real bundle native inference matches Python reference vectors");
        juce::File manifest (path);
        auto r = model.load (manifest);
        expect (r.wasOk(), r.getErrorMessage());
        if (r.failed())
            return;
        const auto parity = juce::JSON::parse (manifest.getSiblingFile ("parity.json"));
        const auto* cases = parity["references"].getArray();
        expect (cases != nullptr && cases->size() == 36);
        if (cases == nullptr)
            return;
        for (const auto& row : *cases)
        {
            JointLatent z;
            SoundCode sound;
            for (int i = 0; i < 32; ++i)
                z[(size_t) i] = (float) row["latent"][i];
            for (int i = 0; i < 48; ++i)
                sound[(size_t) i] = (float) row["sound"][i];
            expect (model.decode (z, (int) row["category"], codes).wasOk());
            expectEquals ((int) codes.size(), model.layout().names.size());
            for (size_t i = 0; i < codes.size(); ++i)
                expectWithinAbsoluteError (codes[i], (float) row["codes"][(int) i], 2e-5f);
            expect (model.encode (sound, (int) row["category"], z).wasOk());
            for (int i = 0; i < 32; ++i)
                expectWithinAbsoluteError (z[(size_t) i], (float) row["encoded_latent"][i], 2e-5f);
        }
        beginTest ("failed replacement leaves working model loaded");
        const auto id = model.layout().id;
        expect (model.load (manifest.getSiblingFile ("missing.json")).failed());
        expectEquals (model.layout().id, id);
        expect (model.decode ({}, 0, codes).wasOk());
        expect (model.decode ({}, 9, codes).failed());
        beginTest ("manifest rejects wrong category order and duplicate features");
        auto data = juce::JSON::parse (manifest);
        JointLayout layout;
        data["labels"].getArray()->set (0, "Pad");
        expect (JointLayout::parse (data, layout).failed());
        data = juce::JSON::parse (manifest);
        data["features"].getArray()->set (1, data["features"][0]);
        expect (JointLayout::parse (data, layout).failed());
    }
};
static JointVaeTests jointVaeTests;
} // namespace vae::experimental
