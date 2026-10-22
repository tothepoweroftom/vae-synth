#include "experiments/joint_vae/JointVae.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#if VAE_WITH_SURGE
JUCE_BEGIN_IGNORE_WARNINGS_GCC_LIKE ("-Wsign-conversion", "-Wfloat-equal", "-Wpedantic", "-Wshadow", "-Wconversion",
                                     "-Wfloat-conversion", "-Wold-style-cast", "-Wzero-as-null-pointer-constant",
                                     "-Wunused-parameter", "-Wswitch-enum", "-Wimplicit-int-float-conversion",
                                     "-Wshorten-64-to-32", "-Wextra-semi", "-Wcast-align", "-Wunused-value",
                                     "-Wunused-variable", "-Wunused-local-typedefs", "-Wstrict-aliasing",
                                     "-Wsign-compare")
#include "SurgeSynthesizer.h"
JUCE_END_IGNORE_WARNINGS_GCC_LIKE
#endif

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
        beginTest ("label demo: prompt nouns pick the student's category");
        const std::pair<const char*, int> prompts[] = {
            { "dark reese bass", 0 },
            { "Synth Strings, slow", 7 },
            { "warm string pad", 2 },
            { "a Rhodes-style piano", 3 },
            { "glassy bell tone", 4 },
            { "subtle evolving", 8 },
            { "", 8 },
            { "808", 0 },
            { "hoover synth", 5 },
        };
        for (const auto& [text, expected] : prompts)
            expectEquals (JointVae::labelFor (text).category, expected, text);
        expectEquals (JointVae::labelFor ("dark reese bass").noun, juce::String ("reese bass"));

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
        beginTest ("native codec matches the student's Python codec plans");
        // VAE_CODEC_PARITY: another plan file, e.g. the p0 bank's (codec_parity.py --bank)
        const auto parityFile = juce::SystemStats::getEnvironmentVariable ("VAE_CODEC_PARITY", "");
        const auto codecParity = juce::JSON::parse (
            parityFile.isNotEmpty() ? juce::File (parityFile) : manifest.getSiblingFile ("codec-parity.json"));
        const auto* plans = codecParity["cases"].getArray();
        expect (plans != nullptr, "run tools/joint_vae/codec_parity.py --bundle <bundle dir>");
        StudentCodec codec;
        expect (StudentCodec::load (manifest.getParentDirectory(), codec).wasOk());
        std::vector<StudentPlan> nativePlans;
        for (int c = 0; plans != nullptr && c < plans->size(); ++c)
        {
            const auto& expected = (*plans)[c]["plan"];
            std::vector<float> rowCodes;
            for (const auto& v : *(*plans)[c]["codes"].getArray()) // null = NaN = inactive (bank rows)
                rowCodes.push_back (v.isVoid() ? std::numeric_limits<float>::quiet_NaN() : (float) v);
            StudentPlan plan;
            expect (codec.plan (model.layout().names, rowCodes, plan).wasOk());
            auto sameSize = [&] (const char* key, size_t n)
            {
                expectEquals ((int) n, expected[key].size(), juce::String (key) + " in case " + juce::String (c));
                return n == (size_t) expected[key].size();
            };
            if (sameSize ("types", plan.types.size()))
                for (size_t i = 0; i < plan.types.size(); ++i)
                    expect (plan.types[i].path == expected["types"][(int) i][0].toString()
                            && plan.types[i].value == (int) expected["types"][(int) i][1]);
            if (sameSize ("subtypes", plan.subtypes.size()))
                for (size_t i = 0; i < plan.subtypes.size(); ++i)
                    expect (plan.subtypes[i].path == expected["subtypes"][(int) i][0].toString()
                            && plan.subtypes[i].value == (int) expected["subtypes"][(int) i][1]);
            if (sameSize ("flags", plan.flags.size()))
                for (size_t i = 0; i < plan.flags.size(); ++i)
                {
                    const auto& e = expected["flags"][(int) i];
                    expect (plan.flags[i].path == e[0].toString() && plan.flags[i].method == e[1].toString()
                            && plan.flags[i].value == (int) e[2]);
                }
            if (sameSize ("values", plan.values.size()))
                for (size_t i = 0; i < plan.values.size(); ++i)
                {
                    const auto& e = expected["values"][(int) i];
                    expect (plan.values[i].path == e[0].toString() && plan.values[i].kind == e[1].toString()
                                && std::abs (plan.values[i].code - (float) e[2]) < 1e-6f
                                && (e[3].isVoid() || plan.values[i].natural == (int) e[3]),
                            plan.values[i].path);
                }
            if (sameSize ("wavetables", plan.wavetables.size()))
                for (size_t i = 0; i < plan.wavetables.size(); ++i)
                    expect (plan.wavetables[i].osc == (int) expected["wavetables"][(int) i][0]
                            && plan.wavetables[i].name == expected["wavetables"][(int) i][1].toString());
            if (sameSize ("routings", plan.routings.size()))
                for (size_t i = 0; i < plan.routings.size(); ++i)
                {
                    const auto& e = expected["routings"][(int) i];
                    expect (plan.routings[i].source == (int) e[0] && plan.routings[i].path == e[1].toString()
                            && std::abs (plan.routings[i].depth - (float) e[2]) < 1e-6f);
                }
            nativePlans.push_back (std::move (plan));
        }
#if VAE_WITH_SURGE
        beginTest ("plans become full Surge patches");
        {
            SurgePatchBuilder builder ((juce::File (VAE_SURGE_DATA_DIR)));
            for (size_t c = 0; c < juce::jmin<size_t> (nativePlans.size(), 6); ++c)
            {
                const auto& plan = nativePlans[c];
                juce::MemoryBlock patch;
                const auto r = builder.build (plan, patch);
                expect (r.wasOk(), r.getErrorMessage());
                expect (patch.getSize() > 1000);
                for (const auto& t : plan.types)
                    if (t.path != "fx_disable")
                        expectEquals (builder.value (t.path), (float) t.value, t.path);
                for (const auto& t : plan.subtypes)
                    expectEquals (builder.value (t.path), (float) t.value, t.path);
                for (const auto& v : plan.values)
                    if (v.kind == "cat")
                        expectEquals (builder.value (v.path), (float) v.natural, v.path);
                // whole-patch check: decoding the same plan twice gives identical patch data
                juce::MemoryBlock again;
                expect (builder.build (plan, again).wasOk());
                expect (again == patch);

                // Plays on a Surge without the factory data, like the plug-in's engine: the
                // wavetables travel inside the patch data. 1 s of C3, RMS logged.
                struct Layer final : SurgeSynthesizer::PluginLayer
                {
                    void surgeParameterUpdated (const SurgeSynthesizer::ID&, float) override {}
                    void surgeMacroUpdated (long, float) override {}
                } layer;
                auto player = std::make_unique<SurgeSynthesizer> (
                    &layer, SurgeStorage::skipPatchLoadDataPathSentinel); // too big for the stack
                player->setSamplerate (48000.0f);
                player->loadRaw (patch.getData(), (int) patch.getSize());
                player->playNote (0, 48, 100, 0);
                double sum = 0;
                bool finite = true;
                const int blocks = 48000 / BLOCK_SIZE;
                for (int i = 0; i < blocks; ++i)
                {
                    player->process();
                    for (int k = 0; k < BLOCK_SIZE; ++k)
                    {
                        finite = finite && std::isfinite (player->output[0][k]);
                        sum += player->output[0][k] * player->output[0][k];
                    }
                }
                const auto rms = std::sqrt (sum / (blocks * BLOCK_SIZE));
                logMessage ("case " + juce::String ((int) c) + ": patch " + juce::String ((int) patch.getSize())
                            + " bytes, C3 RMS " + juce::String (20 * std::log10 (rms + 1e-12), 1) + " dBFS");
                expect (finite);
            }
            juce::MemoryBlock viaModel; // the worker's path: last decoded codes -> patch
            const auto r = model.patch (codes, viaModel);
            expect (r.wasOk(), r.getErrorMessage());
            expect (viaModel.getSize() > 1000);
        }
        beginTest ("builds do not depend on earlier builds");
        {
            const auto n = juce::jmin<size_t> (nativePlans.size(), 6);
            SurgePatchBuilder used ((juce::File (VAE_SURGE_DATA_DIR)));
            const auto t0 = juce::Time::getMillisecondCounterHiRes();
            for (size_t c = 0; c < n; ++c)
            {
                juce::MemoryBlock fresh, afterOthers;
                SurgePatchBuilder one ((juce::File (VAE_SURGE_DATA_DIR)));
                expect (one.build (nativePlans[c], fresh).wasOk());
                for (size_t o = 0; o < n; ++o)
                    if (o != c)
                        expect (used.build (nativePlans[o], afterOthers).wasOk());
                expect (used.build (nativePlans[c], afterOthers).wasOk());
                expect (fresh == afterOthers, "case " + juce::String ((int) c));
            }
            logMessage ("mean build incl. fresh builders: "
                        + juce::String ((juce::Time::getMillisecondCounterHiRes() - t0) / double (n * n + n), 1)
                        + " ms");
        }
#endif
        {
            JointVaeWorker worker;
            auto wait = [&] (auto submit)
            {
                std::optional<std::pair<juce::Result, JointVaeWorker::Output>> got;
                submit ([&] (juce::Result res, JointVaeWorker::Output o) { got.emplace (res, std::move (o)); });
                const auto deadline = juce::Time::getMillisecondCounter() + 20000;
                while (! got && juce::Time::getMillisecondCounter() < deadline)
                    juce::MessageManager::getInstance()->runDispatchLoopUntil (1);
                expect (got.has_value(), "worker timed out");
                return got ? std::move (*got) : std::pair { juce::Result::fail ("timeout"), JointVaeWorker::Output {} };
            };
            auto lab = [&] (LabRequest req)
            { return wait ([&] (auto cb) { worker.lab (std::move (req), std::move (cb)); }); };
            expect (wait ([&] (auto cb) { worker.load (manifest, std::move (cb)); }).first.wasOk());
            LabRequest a;
            const auto& case0 = (*cases)[0];
            for (int i = 0; i < 32; ++i)
                a.z[(size_t) i] = (float) case0["latent"][i];
            a.category = (int) case0["category"];
            const auto [ra, outA] = lab (a);
            expect (ra.wasOk(), ra.getErrorMessage());
            const int numNumeric = model.layout().numNumeric;
            expect (numNumeric > 0 && numNumeric < model.layout().names.size());
            expectEquals ((int) outA.knobs.size(), 123);
            expect (std::any_of (outA.knobs.begin(), outA.knobs.end(), [] (float k) { return std::isfinite (k); }));
            expectEquals ((int) outA.mel.size(), model.soundPca().frames * 64);
            expect (! outA.mel.empty());
#if VAE_WITH_SURGE
            beginTest ("decoded point plays on Lite");
            expect (outA.patchResult.wasOk(), outA.patchResult.getErrorMessage());
            expect (outA.patch.getSize() > 0);
            for (int i = 0; i < schema::kControlCount; ++i)
                expectEquals (schema::sanitiseValue (i, outA.lite[i]), outA.lite[i], schema::control (i).id);
            LabRequest knobPad = a; // given codes skip the decoder: same codes, same patch
            knobPad.codes = outA.codes;
            knobPad.z = {};
            const auto [rk, outK] = lab (knobPad);
            expect (rk.wasOk() && outK.codes == outA.codes && outK.patch == outA.patch);
#endif
            beginTest ("lock keeps structure");
            LabRequest b;
            b.z = a.z;
            for (auto& v : b.z)
                v = -v;
            b.category = a.category;
            const auto [rFree, outFree] = lab (b);
            b.lockedTail = std::vector<float> (outA.codes.begin() + numNumeric, outA.codes.end());
            const auto [rb, outB] = lab (b);
            expect (rb.wasOk() && rFree.wasOk(), rb.getErrorMessage());
            const auto n = (size_t) model.layout().names.size();
            expect (outA.codes.size() == n && outB.codes.size() == n && outFree.codes.size() == n);
            if (outA.codes.size() == n && outB.codes.size() == n && outFree.codes.size() == n)
            {
                expect (
                    std::equal (outA.codes.begin() + numNumeric, outA.codes.end(), outB.codes.begin() + numNumeric));
                expect (std::equal (outFree.codes.begin(), outFree.codes.begin() + numNumeric, outB.codes.begin()));
                expect (outFree.codes != outB.codes, "point B's own structure differs from A's");
            }
            LabRequest wrong = b;
            wrong.lockedTail->push_back (0);
            expect (lab (wrong).first.failed());

            beginTest ("lab timing");
            const auto t0 = juce::Time::getMillisecondCounterHiRes();
            for (int i = 0; i < 50; ++i)
            {
                LabRequest moved = a;
                moved.z[0] += 0.01f * (float) i;
                lab (moved);
            }
            logMessage ("mean lab() time over 50 calls: "
                        + juce::String ((juce::Time::getMillisecondCounterHiRes() - t0) / 50.0, 1) + " ms");

            beginTest ("readouts");
            expectEquals ((int) outA.guess.size(), 8);
            expectWithinAbsoluteError (std::accumulate (outA.guess.begin(), outA.guess.end(), 0.0f), 1.0f, 1e-3f);
            expect (std::any_of (outA.preset.begin(), outA.preset.end(),
                                 [] (const auto& row) { return row.first == "Amp A / D / S / R"; }));
            expect (std::any_of (outA.preset.begin(), outA.preset.end(),
                                 [] (const auto& row) { return row.first == "Osc 1"; }));

            auto map = [&] (MapRequest req)
            {
                std::optional<std::pair<juce::Result, MapResult>> got;
                worker.map (std::move (req), [&] (juce::Result res, MapResult o) { got.emplace (res, std::move (o)); });
                const auto deadline = juce::Time::getMillisecondCounter() + 60000;
                while (! got && juce::Time::getMillisecondCounter() < deadline)
                    juce::MessageManager::getInstance()->runDispatchLoopUntil (1);
                expect (got.has_value(), "map timed out");
                return got ? std::move (*got) : std::pair { juce::Result::fail ("timeout"), MapResult {} };
            };
            MapRequest m;
            m.z = a.z;
            m.category = a.category;
            m.axisIds = { "s:brightness", "s:length" };
            m.zPos = 0.5f;
            beginTest ("map shape");
            for (const auto* kind : { "brightness", "cat" })
            {
                m.kind = kind;
                const auto t = juce::Time::getMillisecondCounterHiRes();
                const auto [rm, outM] = map (m);
                logMessage (juce::String ("24x24 ") + kind
                            + " map: " + juce::String (juce::Time::getMillisecondCounterHiRes() - t, 0) + " ms");
                expect (rm.wasOk(), rm.getErrorMessage());
                expectEquals (outM.n, 24);
                if (m.kind == "brightness")
                {
                    expectEquals ((int) outM.val.size(), 576);
                    expect (std::all_of (outM.val.begin(), outM.val.end(), [] (float v) { return std::isfinite (v); }));
                    expect (outM.lo < outM.hi);
                }
                else
                {
                    expectEquals ((int) outM.cells.size(), 576);
                    expectEquals ((int) outM.conf.size(), 576);
                    expect (std::all_of (outM.cells.begin(), outM.cells.end(), [] (int c) { return c >= 0 && c < 8; }));
                }
            }

            beginTest ("map cancels");
            {
                m.kind = "cat";
                bool mapFired = false, mapOk = false;
                std::optional<juce::Result> labDone;
                worker.map (m,
                            [&] (juce::Result res, MapResult)
                            {
                                mapFired = true;
                                mapOk = res.wasOk();
                            });
                const auto started = juce::Time::getMillisecondCounter() + 5000;
                while (! worker.mapRunning() && juce::Time::getMillisecondCounter() < started)
                    juce::Thread::sleep (1);
                expect (worker.mapRunning(), "the map job started before the lab request");
                const auto t0 = juce::Time::getMillisecondCounterHiRes();
                worker.lab (a, [&] (juce::Result res, JointVaeWorker::Output) { labDone = res; });
                const auto deadline = juce::Time::getMillisecondCounter() + 20000;
                while (! labDone && juce::Time::getMillisecondCounter() < deadline)
                    juce::MessageManager::getInstance()->runDispatchLoopUntil (1);
                const auto labMs = juce::Time::getMillisecondCounterHiRes() - t0;
                logMessage ("lab result " + juce::String (labMs, 1) + " ms after cancelling a running category map");
                expect (labMs < 300, "the lab result does not wait for the map: " + juce::String (labMs, 0) + " ms");
                juce::MessageManager::getInstance()->runDispatchLoopUntil (200);
                expect (labDone.has_value() && labDone->wasOk(), "lab result after a cancelled map");
                expect (! mapFired || ! mapOk, "map was cancelled");
            }
        }
        beginTest ("failed replacement leaves working model loaded");
        const auto id = model.layout().id;
        expect (model.load (manifest.getSiblingFile ("missing.json")).failed());
        expectEquals (model.layout().id, id);
        expect (model.decode ({}, 0, codes).wasOk());
        expect (model.decode ({}, 9, codes).failed());
        beginTest ("old bundle rejected");
        {
            const auto old = juce::File::createTempFile ("old-bundle");
            old.createDirectory();
            for (const auto& f : manifest.getParentDirectory().findChildFiles (juce::File::findFiles, false))
                if (f.getFileName() != "xypad.json")
                    f.copyFileTo (old.getChildFile (f.getFileName()));
            const auto rejected = model.load (old.getChildFile (manifest.getFileName()));
            expect (rejected.failed() && rejected.getErrorMessage().contains ("export_xypad.py"),
                    rejected.getErrorMessage());
            expectEquals (model.layout().id, id);
            expect (model.decode ({}, 0, codes).wasOk());
            old.deleteRecursively();
        }
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
