#include <juce_events/juce_events.h>

#include "PluginProcessor.h"
#include "PluginEditor.h"
#include "experiments/joint_vae/LabPad.h"
#include "schema/PresetFile.h"
#include "../engine/EngineTestUtils.h"
#if VAE_WITH_SURGE
#include "engine/surge/SurgeEngine.h"
#endif

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

/** Depth-first search for the first descendant of type T. */
template <typename T>
T* findChild (juce::Component& parent)
{
    for (auto* child : parent.getChildren())
    {
        if (auto* hit = dynamic_cast<T*> (child))
            return hit;
        if (auto* hit = findChild<T> (*child))
            return hit;
    }
    return nullptr;
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

        beginTest ("VAE Lab is separate and tab switching does not change synth parameters");
        {
            PluginProcessor processor;
            settleModel (processor);
            const auto before = processor.getControls();
            std::unique_ptr<juce::AudioProcessorEditor> editor (processor.createEditor());
            experimental::JointVaePanel* lab = nullptr;
            juce::TextButton* labButton = nullptr;
            juce::TextButton* synthButton = nullptr;
            for (auto* child : editor->getChildren())
            {
                if (auto* panel = dynamic_cast<experimental::JointVaePanel*> (child))
                    lab = panel;
                if (auto* button = dynamic_cast<juce::TextButton*> (child))
                {
                    if (button->getButtonText() == "VAE Lab")
                        labButton = button;
                    if (button->getButtonText() == "Synth")
                        synthButton = button;
                }
            }
            expect (lab != nullptr && labButton != nullptr && synthButton != nullptr);
            if (lab && labButton && synthButton)
            {
                expect (! lab->isVisible());
                labButton->onClick();
                expect (lab->isVisible());
                expect (! lab->getBounds().isEmpty());
                synthButton->onClick();
                expect (! lab->isVisible());
            }
            const auto after = processor.getControls();
            for (int i = 0; i < schema::kControlCount; ++i)
                expectEquals (after[i], before[i]);
        }

        beginTest ("VAE Lab v2 layout");
        {
            PluginProcessor processor;
            settleModel (processor);
            std::unique_ptr<juce::AudioProcessorEditor> editor (processor.createEditor());
            auto* lab = findChild<experimental::JointVaePanel> (*editor);
            juce::TextButton* labButton = nullptr;
            for (auto* child : editor->getChildren())
                if (auto* button = dynamic_cast<juce::TextButton*> (child);
                    button && button->getButtonText() == "VAE Lab")
                    labButton = button;
            expect (lab != nullptr && labButton != nullptr);
            if (lab && labButton)
            {
                labButton->onClick();
                auto* pad = findChild<experimental::LabPad> (*lab);
                auto* list = findChild<juce::Viewport> (*lab);
#if VAE_WITH_SURGE
                expect (pad != nullptr && list != nullptr);
                if (pad && list)
                {
                    const auto p = lab->getLocalArea (pad, pad->getLocalBounds());
                    const auto l = lab->getLocalArea (list, list->getLocalBounds());
                    expect (p.getWidth() >= 0.6 * lab->getWidth(), "pad width " + juce::String (p.getWidth()));
                    expect (p.getHeight() >= 0.55 * lab->getHeight(), "pad height " + juce::String (p.getHeight()));
                    expect (l.getX() >= p.getRight() && ! l.intersects (p));
                    expect (l.getHeight() >= p.getHeight());
                    const auto bundle = juce::SystemStats::getEnvironmentVariable ("VAE_JOINT_BUNDLE", "");
                    if (bundle.isNotEmpty())
                    {
                        lab->loadBundle (juce::File (bundle));
                        expect (
                            pumpUntil ([&] { return list->getViewedComponent()->getNumChildComponents() > 30; }, 30000),
                            "axis rows");
                        pumpUntil ([] { return false; }, 300); // the first decode, for the screenshot
                        expect (list->getViewedComponent()->getWidth() <= list->getMaximumVisibleWidth(),
                                "rows clear the scroll bar: " + juce::String (list->getViewedComponent()->getWidth())
                                    + " > " + juce::String (list->getMaximumVisibleWidth()));
                    }
                }
#else
                expect (pad == nullptr && list == nullptr);
                auto* note = findChild<juce::Label> (*lab);
                expect (note != nullptr && note->getText() == "VAE Lab needs a Surge build");
#endif
                auto image = editor->createComponentSnapshot (editor->getLocalBounds());
                expect (image.isValid());
                const auto screenshot = juce::SystemStats::getEnvironmentVariable ("VAE_LAB_SCREENSHOT", "");
                if (screenshot.isNotEmpty())
                {
                    auto stream = juce::File (screenshot).createOutputStream();
                    if (stream)
                        juce::PNGImageFormat().writeImageToStream (image, *stream);
                }
            }
        }

#if VAE_WITH_SURGE
        if (const auto bundle = juce::SystemStats::getEnvironmentVariable ("VAE_JOINT_BUNDLE", ""); bundle.isNotEmpty())
        {
            beginTest ("VAE Lab plays nothing until the user acts in it");
            PluginProcessor processor;
            settleModel (processor);
            processor.getParams().set (PluginProcessor::kEngineParamId, float (processor.getEngineKinds().size() - 1));
            const auto engine = processor.getParams().get (PluginProcessor::kEngineParamId);
            const auto before = processor.getControls();
            std::unique_ptr<juce::AudioProcessorEditor> editor (processor.createEditor());
            auto* lab = findChild<experimental::JointVaePanel> (*editor);
            expect (lab != nullptr && ! lab->isVisible());
            if (lab)
            {
                int played = 0;
                auto play = lab->onLite;
                lab->onLite = [&] (const schema::Snapshot& s)
                {
                    ++played;
                    if (play)
                        play (s);
                };
                juce::TextButton* send = nullptr;
                std::function<void (juce::Component&)> find = [&] (juce::Component& c)
                {
                    for (auto* child : c.getChildren())
                    {
                        if (auto* b = dynamic_cast<juce::TextButton*> (child);
                            b && b->getButtonText() == "Send to Surge")
                            send = b;
                        find (*child);
                    }
                };
                find (*lab);
                expect (send != nullptr);
                lab->loadBundle (juce::File (bundle));
                // Send to Surge enables when the first decode result arrives
                expect (send && pumpUntil ([&] { return send->isEnabled(); }, 30000), "decode result for display");
                pumpUntil ([] { return false; }, 200);
                expectEquals (played, 0);
                expectEquals (processor.getParams().get (PluginProcessor::kEngineParamId), engine);
                const auto after = processor.getControls();
                for (int i = 0; i < schema::kControlCount; ++i)
                    expectEquals (after[i], before[i]);
                if (auto* pad = findChild<experimental::LabPad> (*lab))
                    pad->keyPressed (juce::KeyPress (juce::KeyPress::rightKey));
                expect (pumpUntil ([&] { return played > 0; }, 30000), "a pad move plays");
            }
        }
        else
            logMessage ("Set VAE_JOINT_BUNDLE to run the VAE Lab bundle tests.");

        if (const auto bundle = juce::SystemStats::getEnvironmentVariable ("VAE_JOINT_BUNDLE", ""); bundle.isNotEmpty())
        {
            beginTest ("knob pad moves only its group");
            PluginProcessor processor;
            settleModel (processor);
            std::unique_ptr<juce::AudioProcessorEditor> editor (processor.createEditor());
            auto* lab = findChild<experimental::JointVaePanel> (*editor);
            auto* pad = lab ? findChild<experimental::LabPad> (*lab) : nullptr;
            juce::TextButton *knobButton = nullptr, *latentButton = nullptr;
            std::function<void (juce::Component&)> find = [&] (juce::Component& c)
            {
                for (auto* child : c.getChildren())
                {
                    if (auto* b = dynamic_cast<juce::TextButton*> (child); b && b->getButtonText() == "Knob pad")
                        knobButton = b;
                    else if (b && b->getButtonText() == "Latent pad")
                        latentButton = b;
                    find (*child);
                }
            };
            if (lab)
                find (*lab);
            expect (pad != nullptr && knobButton != nullptr && latentButton != nullptr);
            if (pad && knobButton && latentButton)
            {
                bool ready = false;
                lab->onMapShown = [&ready] { ready = true; };
                lab->loadBundle (juce::File (bundle));
                expect (pumpUntil ([&] { return ready; }, 60000), "bundle loaded");
                const auto before = lab->output().codes;
                const auto z0 = lab->latent();
                expect (knobButton->isEnabled(), "knob pad is available");
                if (knobButton->isEnabled())
                {
                    knobButton->onClick();
                    ready = false;
                    expect (pumpUntil ([&] { return ready; }, 60000), "knob map shown");
                    expect (lab->output().codes == before, "switching keeps the decoded codes");
                    const auto* kg = lab->knobGroups();
                    // Z: a multi-knob group away from groups 0 and 1, preferring one with an inactive knob
                    const auto& beforeKnobs = lab->output().knobs;
                    int zGroup = -1;
                    if (kg)
                        for (int pass = 0; pass < 2 && zGroup < 0; ++pass)
                            for (int g = 2; g < (int) kg->groups.size() && zGroup < 0; ++g)
                            {
                                const auto& gr = kg->groups[(size_t) g];
                                bool overlap = false, inactive = false;
                                for (int k : gr.knob)
                                {
                                    inactive |= std::isnan (beforeKnobs[(size_t) k]);
                                    for (int o = 0; o < 2; ++o)
                                        overlap |= std::find (kg->groups[(size_t) o].knob.begin(),
                                                              kg->groups[(size_t) o].knob.end(), k)
                                                   != kg->groups[(size_t) o].knob.end();
                                }
                                if (gr.knob.size() >= 2 && ! overlap && (pass == 1 || inactive))
                                    zGroup = g;
                            }
                    expect (zGroup >= 0, "a multi-knob Z group");
                    if (kg && zGroup >= 0)
                        lab->assignAxis (2, kg->groups[(size_t) zGroup].name);
                    const auto knobsBefore = beforeKnobs;
                    pad->moved ({ 1.0f, 0.5f });
                    expect (kg != nullptr && kg->groups.size() == 61);
                    if (kg)
                    {
                        const auto& names = lab->output().layout.names;
                        std::vector<float> expected = before;
                        for (int g = 0; g < 2; ++g)
                        {
                            auto knobs = std::vector<float> (kg->names.size(), 0.0f);
                            kg->set (g, g == 0 ? 1.0f : 0.5f, knobs);
                            for (int k : kg->groups[(size_t) g].knob)
                                if (const auto i = names.indexOf (kg->names[k]); i >= 0)
                                    expected[(size_t) i] = knobs[(size_t) k];
                        }
                        const auto& g0 = kg->groups[0];
                        expect (pumpUntil ([&] { return lab->output().codes == expected; }, 30000),
                                "only groups X and Y change");
                        // Everything outside groups X and Y, inactive (NaN) knobs included, is untouched
                        for (size_t k = 0; k < knobsBefore.size(); ++k)
                        {
                            const auto in = [&] (int g)
                            {
                                const auto& v = kg->groups[(size_t) g].knob;
                                return std::find (v.begin(), v.end(), (int) k) != v.end();
                            };
                            if (! in (0) && ! in (1))
                                expect (std::memcmp (&knobsBefore[k], &lab->output().knobs[k], sizeof (float)) == 0,
                                        "knob " + kg->names[(int) k] + " unchanged");
                        }
                        for (size_t k = 0; k < g0.knob.size(); ++k)
                            expectWithinAbsoluteError (
                                lab->output().codes[(size_t) names.indexOf (kg->names[g0.knob[k]])],
                                g0.sign[k] > 0 ? kg->p95[(size_t) g0.knob[k]] : kg->p5[(size_t) g0.knob[k]], 1e-5f);
                    }
                    latentButton->onClick();
                    expect (lab->latent() == z0, "back on the latent pad, z is kept");
                }
            }
        }

        if (const auto bundle = juce::SystemStats::getEnvironmentVariable ("VAE_JOINT_BUNDLE", ""); bundle.isNotEmpty())
        {
            beginTest ("prompt starts and nudges");
            PluginProcessor processor;
            settleModel (processor);
            std::unique_ptr<juce::AudioProcessorEditor> editor (processor.createEditor());
            auto* lab = findChild<experimental::JointVaePanel> (*editor);
            juce::ComboBox* category = nullptr;
            std::function<void (juce::Component&)> find = [&] (juce::Component& c)
            {
                for (auto* child : c.getChildren())
                {
                    if (auto* box = dynamic_cast<juce::ComboBox*> (child); box && box->getItemText (0) == "Bass")
                        category = box;
                    find (*child);
                }
            };
            if (lab)
                find (*lab);
            expect (lab != nullptr && category != nullptr);
            if (lab && category)
            {
                int played = 0;
                lab->onLite = [&played] (const schema::Snapshot&) { ++played; };
                bool ready = false;
                lab->onMapShown = [&ready] { ready = true; };
                lab->loadBundle (juce::File (bundle));
                expect (pumpUntil ([&] { return ready; }, 60000), "bundle loaded");
                auto* xy = lab->padData();
                auto* brightness = xy ? xy->find ("s:brightness") : nullptr;
                expect (brightness != nullptr);
                if (brightness)
                {
                    const auto typical = axisPosition (*brightness, xy->typical.at ("Bass"));
                    // "zorp" has nothing usable: no change, no sound
                    const auto z0 = lab->latent();
                    lab->submitPrompt ("zorp");
                    pumpUntil ([] { return false; }, 300);
                    expect (lab->latent() == z0);
                    expect (lab->statusText().contains ("Nothing to change"), lab->statusText());
                    expectEquals (played, 0);
                    lab->submitPrompt ("dark bass");
                    expectEquals (category->getText(), juce::String ("Bass"));
                    expect (lab->statusText().contains ("Bass"), lab->statusText());
                    expectWithinAbsoluteError (axisPosition (*brightness, lab->latent()), typical - 0.1f, 1e-3f);
                    lab->submitPrompt ("brighter");
                    expectWithinAbsoluteError (axisPosition (*brightness, lab->latent()), typical, 1e-3f);
                    expect (pumpUntil ([&] { return played > 0; }, 30000), "a prompt plays");
                }
            }
        }

        if (const auto bundle = juce::SystemStats::getEnvironmentVariable ("VAE_JOINT_BUNDLE", ""); bundle.isNotEmpty())
        {
            beginTest ("prompt first message falls back to the Unknown start until the user acts");
            // A fresh panel with the bundle; returns brightness position after the given steps
            const auto run = [&] (const std::function<void (experimental::JointVaePanel&)>& steps)
            {
                PluginProcessor processor;
                settleModel (processor);
                std::unique_ptr<juce::AudioProcessorEditor> editor (processor.createEditor());
                auto* lab = findChild<experimental::JointVaePanel> (*editor);
                bool ready = false;
                if (lab)
                {
                    lab->onMapShown = [&ready] { ready = true; };
                    lab->loadBundle (juce::File (bundle));
                }
                expect (lab != nullptr && pumpUntil ([&] { return ready; }, 60000), "bundle loaded");
                if (lab && ready)
                    steps (*lab);
            };
            const auto bright = [] (experimental::JointVaePanel& lab)
            { return axisPosition (*lab.padData()->find ("s:brightness"), lab.latent()); };
            const auto unknown = [&] (experimental::JointVaePanel& lab)
            {
                auto* box = [&]
                {
                    juce::ComboBox* found = nullptr;
                    std::function<void (juce::Component&)> f = [&] (juce::Component& c)
                    {
                        for (auto* ch : c.getChildren())
                        {
                            if (auto* b = dynamic_cast<juce::ComboBox*> (ch); b && b->getItemText (0) == "Bass")
                                found = b;
                            f (*ch);
                        }
                    };
                    f (lab);
                    return found;
                }();
                return box != nullptr && box->getSelectedId() == 9;
            };
            float darkPos = 0, brightPos = 0, movedBefore = 0;
            run (
                [&] (experimental::JointVaePanel& lab)
                {
                    lab.submitPrompt ("something dark");
                    expect (unknown (lab), "category Unknown");
                    expect (lab.statusText().contains ("no sound type found"), lab.statusText());
                    darkPos = bright (lab);
                });
            run (
                [&] (experimental::JointVaePanel& lab)
                {
                    lab.submitPrompt ("brighter");
                    expect (unknown (lab), "category Unknown");
                    expect (lab.statusText().contains ("no sound type found"), lab.statusText());
                    brightPos = bright (lab);
                });
            expectWithinAbsoluteError (brightPos - darkPos, 0.2f, 1e-3f);
            run (
                [&] (experimental::JointVaePanel& lab)
                {
                    if (auto* pad = findChild<experimental::LabPad> (lab))
                        pad->keyPressed (juce::KeyPress (juce::KeyPress::upKey));
                    movedBefore = bright (lab);
                    lab.submitPrompt ("brighter");
                    expect (! lab.statusText().contains ("no sound type found"), lab.statusText());
                    expectWithinAbsoluteError (bright (lab), movedBefore + 0.1f, 1e-3f);
                });
        }

        if (const auto bundle = juce::SystemStats::getEnvironmentVariable ("VAE_JOINT_BUNDLE", ""); bundle.isNotEmpty())
        {
            beginTest ("editor closes mid-map");
            PluginProcessor processor;
            settleModel (processor);
            std::unique_ptr<juce::AudioProcessorEditor> editor (processor.createEditor());
            auto* lab = findChild<experimental::JointVaePanel> (*editor);
            auto* pad = lab ? findChild<experimental::LabPad> (*lab) : nullptr;
            expect (pad != nullptr);
            juce::ComboBox* background = nullptr;
            std::function<void (juce::Component&)> find = [&] (juce::Component& c)
            {
                for (auto* child : c.getChildren())
                {
                    if (auto* box = dynamic_cast<juce::ComboBox*> (child);
                        box && box->getItemText (0) == "Category guess")
                        background = box;
                    find (*child);
                }
            };
            if (lab)
                find (*lab);
            expect (background != nullptr);
            if (pad && background)
            {
                int shown = 0;
                lab->onMapShown = [&shown] { ++shown; };
                lab->loadBundle (juce::File (bundle));
                expect (pumpUntil ([&] { return shown > 0; }, 60000), "the first map arrives");
                // The category map is the slow one (~0.8 s): it is surely computing at the reset
                background->setSelectedId (1, juce::sendNotificationSync);
                expect (pumpUntil ([&] { return lab->isMapRunning(); }, 5000), "a category map is computing");
                const int before = shown;
                const auto t0 = juce::Time::getMillisecondCounter();
                editor.reset();
                expect (juce::Time::getMillisecondCounter() - t0 < 300, "closing does not wait for the whole map");
                pumpUntil ([] { return false; }, 500);
                expectEquals (shown, before); // no map result reached the deleted panel
            }
        }
        else
            logMessage ("Set VAE_JOINT_BUNDLE to run \"editor closes mid-map\".");
#else
        logMessage ("VAE Lab bundle tests (\"editor closes mid-map\") need a Surge build.");
#endif

        beginTest ("SAME invalid latents and cancelled worker callbacks");
        {
            ml::SameSearch::Embedding embedding;
            expect (ml::SameSearch::pool ({}, embedding).failed());
            std::vector<float> invalid (256 * 44, std::numeric_limits<float>::quiet_NaN());
            expect (ml::SameSearch::pool (invalid, embedding).failed());
            ml::SameSearchWorker worker;
            bool stale = false, complete = false;
            worker.load (dir.getChildFile ("missing-bank.json"), [&] (auto, auto) { stale = true; });
            worker.cancel();
            worker.search (dir.getChildFile ("missing.wav"),
                           [&] (juce::Result result, auto matches)
                           {
                               expect (juce::MessageManager::getInstance()->isThisTheMessageThread());
                               expect (result.failed());
                               expect (matches.empty());
                               complete = true;
                           });
            expect (pumpUntil ([&] { return complete; }));
            expect (! stale);
        }

        beginTest ("Native generation errors do not change controls and callbacks stay on message thread");
        {
            ml::StableAudioGenerator generator;
            expect (generator.load (dir.getChildFile ("missing-generator.json")).failed());
            const auto output = dir.getChildFile ("should-not-exist.wav");
            expect (generator.generate ("bell", 42, output).failed());
            expect (! output.exists());
            ml::SameSearchWorker worker;
            bool stale = false, complete = false;
            worker.loadGenerator (dir.getChildFile ("missing.json"), [&] (auto, auto) { stale = true; });
            worker.cancel();
            worker.generate ("bell", 42, output,
                             [&] (juce::Result result, auto matches)
                             {
                                 expect (juce::MessageManager::getInstance()->isThisTheMessageThread());
                                 expect (result.failed());
                                 expect (matches.empty());
                                 complete = true;
                             });
            expect (pumpUntil ([&] { return complete; }));
            expect (! stale);
            expect (! output.exists());
        }

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

        beginTest ("retrieved presets load without a decoder and invalid input leaves controls unchanged");
        {
            PluginProcessor p;
            settleModel (p);
            const auto controls = randomSnapshot (rng);
            const auto preset = dir.getChildFile ("retrieved.json");
            auto json = schema::presetJson (controls, "Retrieved", "lite");
            expect (preset.replaceWithText (juce::JSON::toString (json)));
            expect (p.loadPresetFile (preset).wasOk());
            for (int i = 0; i < schema::kControlCount; ++i)
                expectWithinAbsoluteError (p.getControls()[i], controls[i], 1.0e-5f);
            const auto before = p.getControls();
            for (int failure = 0; failure < 4; ++failure)
            {
                json = schema::presetJson (controls, "Broken", "lite");
                if (failure == 0)
                    json.getDynamicObject()->setProperty ("schema_version", -1);
                if (failure == 1)
                    json.getDynamicObject()->setProperty ("engine", "unknown");
                if (failure == 2)
                    json["values"].getArray()->set (0, "not a number");
                if (failure == 3)
                    json["values"].getArray()->set (0, 99999);
                expect (preset.replaceWithText (juce::JSON::toString (json)));
                expect (p.loadPresetFile (preset).failed());
                expect (p.getControls().values == before.values);
            }
            PluginProcessor restored;
            const auto state = stateOf (p);
            restored.setStateInformation (state.getData(), int (state.getSize()));
            expect (restored.getControls().values == before.values);
            settleModel (restored);
        }

#if VAE_WITH_SURGE
        beginTest ("Lite after Surge");
        {
            PluginProcessor p;
            settleModel (p);
            p.prepareToPlay (kRate, 512);
            juce::MemoryBlock fxp;
            expect (juce::File (VAE_SURGE_DATA_DIR "/patches_factory/Templates/Init Sine.fxp").loadFileAsData (fxp));
            expect (p.loadSurgePatch (fxp).wasOk());
            juce::AudioBuffer<float> buffer (2, 480); // 10 ms
            juce::MidiBuffer midi;
            p.processBlock (buffer, midi);
            auto* surge = dynamic_cast<SurgeEngine*> (p.findEngine (EngineKind::Surge));
            expect (surge != nullptr && surge->hasPatchData());

            GestureCounter gestures;
            p.addListener (&gestures);
            auto lite = schema::Snapshot::defaults();
            p.playLiteControls (lite);
            const auto liteIndex =
                std::distance (p.getEngineKinds().begin(),
                               std::find (p.getEngineKinds().begin(), p.getEngineKinds().end(), EngineKind::Lite));
            expectEquals (juce::roundToInt (p.getParams().get (PluginProcessor::kEngineParamId)), (int) liteIndex);
            expect (surge != nullptr && ! surge->hasPatchData());
            p.processBlock (buffer, midi); // the engine switch lands
            midi.addEvent (juce::MidiMessage::noteOn (1, 60, (juce::uint8) 100), 0);
            auto rmsOf50ms = [&]
            {
                double sum = 0;
                for (int b = 0; b < 5; ++b)
                {
                    p.processBlock (buffer, midi);
                    midi.clear();
                    for (int i = 0; i < buffer.getNumSamples(); ++i)
                        sum += buffer.getSample (0, i) * buffer.getSample (0, i);
                }
                return std::sqrt (sum / (5 * buffer.getNumSamples()));
            };
            expectGreaterThan (rmsOf50ms(), 0.0);
            lite[schema::kFilter1Cutoff] = 0.3f; // a pad move while the note is held
            p.playLiteControls (lite);
            expectGreaterThan (rmsOf50ms(), 0.0);
            p.removeListener (&gestures);
            expectEquals (gestures.begins, 0);
            expectEquals (gestures.ends, 0);
            expectWithinAbsoluteError (p.getControls()[schema::kFilter1Cutoff], 0.3f, 1e-6f);
            p.releaseResources();
        }
#endif

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
