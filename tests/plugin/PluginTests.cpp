#include <juce_events/juce_events.h>

#include <thread>

#include "PluginProcessor.h"
#include "PluginEditor.h"
#include "experiments/joint_vae/LabOrbit.h"
#include "experiments/joint_vae/LabPad.h"
#include "PitcherControls.h"
#include "schema/PresetFile.h"
#include "../engine/EngineTestUtils.h"
#include "engine/lite/LiteEngine.h"
#if VAE_WITH_SURGE
#include "engine/surge/SurgeEngine.h"
#include "experiments/joint_vae/JointVae.h"
#include "experiments/joint_vae/LatentOrbit.h"
#include "experiments/joint_vae/LiteTranslation.h"
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

/** Every descendant of type T, depth first. */
template <typename T>
std::vector<T*> findAll (juce::Component& parent)
{
    std::vector<T*> all;
    for (auto* child : parent.getChildren())
    {
        if (auto* hit = dynamic_cast<T*> (child))
            all.push_back (hit);
        for (auto* hit : findAll<T> (*child))
            all.push_back (hit);
    }
    return all;
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
    void audioProcessorParameterChanged (juce::AudioProcessor*, int, float) override { ++changes; }
    void audioProcessorChanged (juce::AudioProcessor*, const ChangeDetails&) override {}
    void audioProcessorParameterChangeGestureBegin (juce::AudioProcessor*, int) override { ++begins; }
    void audioProcessorParameterChangeGestureEnd (juce::AudioProcessor*, int) override { ++ends; }
    int begins = 0, ends = 0, changes = 0;
};
/** Depth-first search for a TextButton with this text. */
juce::TextButton* findButton (juce::Component& parent, const juce::String& text)
{
    for (auto* child : parent.getChildren())
    {
        if (auto* b = dynamic_cast<juce::TextButton*> (child); b && b->getButtonText() == text)
            return b;
        if (auto* hit = findButton (*child, text))
            return hit;
    }
    return nullptr;
}

/** The Synth view's knob for a continuous control (its tooltip starts with the control id). */
juce::Slider* synthKnob (juce::Component& editor, int control)
{
    const juce::String id (schema::control (control).id);
    for (auto* child : editor.getChildren())
        if (auto* section = dynamic_cast<ControlSection*> (child))
            for (auto* cell : section->getChildren())
                for (auto* c : cell->getChildren())
                    if (auto* s = dynamic_cast<juce::Slider*> (c);
                        s && s->getTooltip().upToFirstOccurrenceOf (" ", false, false) == id)
                        return s;
    return nullptr;
}

/** First continuous control index. */
int firstContinuous()
{
    for (int i = 0; i < schema::kControlCount; ++i)
        if (schema::control (i).kind == schema::Kind::Continuous)
            return i;
    return -1;
}

/** The editor's header status label (a direct child; the Lab has its own). */
juce::Label* headerStatus (juce::Component& editor)
{
    for (auto* child : editor.getChildren())
        if (auto* l = dynamic_cast<juce::Label*> (child))
            return l;
    return nullptr;
}

#if VAE_WITH_SURGE
// Task 18: Lab results along a drag, 2 s at 60 per second, made as JointVaeWorker::lab makes them.
enum class Drag
{
    pad,
    padLocked,
    slider
};
std::vector<schema::Snapshot> labDrag (experimental::JointVae& model, Drag drag)
{
    using namespace experimental;
    const auto& xy = model.xypad();
    auto z = xy.typical.at ("Lead");
    const auto* ax = xy.find ("s:brightness");
    const auto* ay = xy.find ("s:length");
    const auto* dim = &*std::find_if (xy.axes.begin(), xy.axes.end(), [] (const auto& a) { return a.kind == "dim"; });
    std::optional<std::vector<float>> tail;
    std::vector<schema::Snapshot> out;
    for (int k = 0; k < 120; ++k)
    {
        const float t = float (k) / 119.0f;
        if (drag == Drag::slider)
        {
            // The list slider: move one latent dim, hold the pad's axes
            const LatentAxis* held[] { dim, ax, ay };
            const float targets[] { axisTarget (*dim, 0.2f + 0.6f * t), axisTarget (*ax, 0.5f),
                                    axisTarget (*ay, 0.5f) };
            z = constrain (z, held, targets);
        }
        else
        {
            const LatentAxis* held[] { ax, ay };
            const float targets[] { axisTarget (*ax, 0.2f + 0.6f * t),
                                    axisTarget (*ay, 0.5f + 0.3f * std::sin (juce::MathConstants<float>::twoPi * t)) };
            z = constrain (z, held, targets);
        }
        std::vector<float> codes;
        if (model.decode (z, 1, codes).failed())
            return {};
        const auto from = codes.begin() + model.layout().numNumeric;
        if (drag == Drag::padLocked && ! tail)
            tail.emplace (from, codes.end());
        if (tail)
            std::copy (tail->begin(), tail->end(), from);
        juce::MemoryBlock patch;
        auto lite = schema::Snapshot::defaults();
        if (model.patch (codes, patch).failed() || liteControls (patch, lite).failed())
            return {};
        out.push_back (lite);
    }
    return out;
}

struct DragStats
{
    double discretePerSecond = 0, edgeRatio = 0, meanMs = 0, maxMs = 0;
    int ducks = 0;
    int64_t duckedSamples = 0;
    float maxStep = 0;
    juce::String describe() const
    {
        return juce::String (discretePerSecond, 1) + " discrete changes/s, " + juce::String (ducks) + " ducks, "
               + juce::String (duckedSamples) + " samples ducked, max step " + juce::String (maxStep, 4)
               + ", block-edge/inside step " + juce::String (edgeRatio, 2) + ", process " + juce::String (meanMs, 3)
               + " ms mean / " + juce::String (maxMs, 3) + " ms max";
    }
};

/** Task 19: the orbit LFO's results for 2 s at the panel's 30 per second, made as the panel makes them: each
    step constrains the pad axes from the centre z, Lock holds the centre's structure codes, and every discrete
    Lite control holds the centre's value. */
std::vector<schema::Snapshot> labOrbit (experimental::JointVae& model, experimental::OrbitSettings settings)
{
    using namespace experimental;
    const auto& xy = model.xypad();
    const auto centreZ = xy.typical.at ("Lead");
    const auto* dim = &*std::find_if (xy.axes.begin(), xy.axes.end(), [] (const auto& a) { return a.kind == "dim"; });
    const LatentAxis* held[] { xy.find ("s:brightness"), xy.find ("s:length"), dim };
    std::array<float, 3> centre;
    for (size_t i = 0; i < 3; ++i)
        centre[i] = axisPosition (*held[i], centreZ);
    const auto lite = [&] (const JointLatent& z, std::optional<std::vector<float>>& tail, schema::Snapshot& out)
    {
        std::vector<float> codes;
        if (model.decode (z, 1, codes).failed())
            return false;
        const auto from = codes.begin() + model.layout().numNumeric;
        if (! tail)
            tail.emplace (from, codes.end());
        std::copy (tail->begin(), tail->end(), from);
        juce::MemoryBlock patch;
        out = schema::Snapshot::defaults();
        return model.patch (codes, patch).wasOk() && liteControls (patch, out).wasOk();
    };
    std::optional<std::vector<float>> tail;
    schema::Snapshot centreLite;
    if (! lite (centreZ, tail, centreLite))
        return {};
    std::vector<schema::Snapshot> out { centreLite };
    for (int k = 1; k < 60; ++k)
    {
        auto ramped = settings; // the radius ramps in, as the panel's
        ramped.depth *= (float) std::min (1.0, k / 30.0 / kOrbitRampSeconds);
        const auto p = orbitPosition (centre, ramped, k / 30.0, true);
        const float targets[] { axisTarget (*held[0], p[0]), axisTarget (*held[1], p[1]), axisTarget (*held[2], p[2]) };
        schema::Snapshot s;
        if (! lite (constrain (centreZ, held, targets), tail, s))
            return {};
        holdDiscrete (s, centreLite);
        out.push_back (s);
    }
    return out;
}

/** Holds a note on Lite and plays `results` at `perSecond` through playLiteControls (2 s, 512-sample blocks). */
DragStats playDrag (const std::vector<schema::Snapshot>& results, int perSecond = 60)
{
    DragStats stats;
    for (size_t k = 1; k < results.size(); ++k)
        for (int i = 0; i < schema::kControlCount; ++i)
            stats.discretePerSecond += schema::control (i).kind != schema::Kind::Continuous
                                       && ! juce::exactlyEqual (results[k][i], results[k - 1][i]);
    stats.discretePerSecond /= 2.0;

    PluginProcessor p;
    settleModel (p);
    p.prepareToPlay (kRate, 512);
    p.playLiteControls (results[0]);
    juce::AudioBuffer<float> buffer (2, 512);
    juce::MidiBuffer midi;
    p.processBlock (buffer, midi); // the engine switch lands
    auto& lite = dynamic_cast<LiteEngine&> (*p.findEngine (EngineKind::Lite));
    midi.addEvent (juce::MidiMessage::noteOn (1, 60, (juce::uint8) 100), 0);
    for (int b = 0; b < 24; ++b) // 0.25 s into the note
    {
        p.processBlock (buffer, midi);
        midi.clear();
    }
    lite.diagnostics = {};
    float last = buffer.getSample (0, 511);
    size_t next = 1;
    double sumMs = 0, edge = 0, inside = 0;
    const int blocks = 2 * int (kRate) / 512;
    for (int b = 0; b < blocks; ++b)
    {
        while (next < results.size() && b * 512 >= int (next) * int (kRate) / perSecond)
            p.playLiteControls (results[next++]);
        const auto t0 = juce::Time::getHighResolutionTicks();
        p.processBlock (buffer, midi);
        const auto ms = juce::Time::highResolutionTicksToSeconds (juce::Time::getHighResolutionTicks() - t0) * 1000.0;
        sumMs += ms;
        stats.maxMs = std::max (stats.maxMs, ms);
        for (int i = 0; i < 512; ++i)
        {
            const float x = buffer.getSample (0, i), step = std::abs (x - last);
            stats.maxStep = std::max (stats.maxStep, step);
            (i == 0 ? edge : inside) += step;
            last = x;
        }
    }
    stats.edgeRatio = (edge / blocks) / (inside / (blocks * 511));
    stats.meanMs = sumMs / blocks;
    stats.ducks = lite.diagnostics.ducksStarted;
    stats.duckedSamples = lite.diagnostics.duckedSamples;
    p.releaseResources();
    return stats;
}
#endif

#if VAE_WITH_SURGE
/** The Lab's orbit handle moved to x, y with Z kept, as a handle drag reports it. */
void moveHandle (experimental::LabOrbit& orbit, float x, float y) { orbit.moved ({ x, y, orbit.getHandle()[2] }); }

/** A mouse event on `c` at `at` (local); `dragged` marks a release after a drag. */
juce::MouseEvent mouseAt (juce::Component& c, juce::Point<float> at, juce::Point<float> down, bool dragged = false)
{
    const auto now = juce::Time::getCurrentTime();
    return { juce::Desktop::getInstance().getMainMouseSource(),
             at,
             juce::ModifierKeys::leftButtonModifier,
             0,
             0,
             0,
             0,
             0,
             &c,
             &c,
             now,
             down,
             now,
             1,
             dragged };
}
#endif
} // namespace

class PluginTests final : public juce::UnitTest
{
public:
    PluginTests()
        : juce::UnitTest ("Plug-in processor", "vae_plugin")
    {
    }

#if VAE_WITH_SURGE
    /** Task 14: p0 prompt search in the Lab. Needs VAE_JOINT_BUNDLE and VAE_PROMPT_BUNDLE. */
    void runPromptTests()
    {
        const auto bundle = juce::SystemStats::getEnvironmentVariable ("VAE_JOINT_BUNDLE", "");
        const auto promptJson = juce::File (juce::SystemStats::getEnvironmentVariable ("VAE_PROMPT_BUNDLE", ""));
        if (bundle.isEmpty() || ! promptJson.existsAsFile())
        {
            logMessage ("Set VAE_JOINT_BUNDLE and VAE_PROMPT_BUNDLE to run the prompt search tests.");
            return;
        }
        // The fixture's rows; z and codes from bank.bin, the stored source of truth
        const auto meta = juce::JSON::parse (promptJson);
        const int rows = meta["bank"]["rows"], numCodes = meta["bank"]["codes"];
        juce::MemoryBlock bank;
        promptJson.getSiblingFile ("bank.bin").loadFileAsData (bank);
        const auto* bankF = static_cast<const float*> (bank.getData());
        const auto bankZ = [&] (int row)
        {
            experimental::JointLatent z {};
            std::copy_n (bankF + row * 32, 32, z.begin());
            return z;
        };
        const auto bankCodes = [&] (int row)
        {
            const auto* c = bankF + rows * 32 + row * numCodes;
            return std::vector<float> (c, c + numCodes);
        };
        const auto same = [] (const std::vector<float>& a, const std::vector<float>& b)
        {
            if (a.size() != b.size())
                return false;
            for (size_t i = 0; i < a.size(); ++i)
                if (std::isnan (a[i]) ? ! std::isnan (b[i]) : ! juce::exactlyEqual (a[i], b[i]))
                    return false;
            return true;
        };
        std::vector<int> bassRows;
        const auto parity = juce::JSON::parse (promptJson.getSiblingFile ("prompt-parity.json"));
        for (const auto& c : *parity["cases"].getArray())
            if (c["name"].toString() == "bass")
                for (const auto& r : *c["rows"].getArray())
                    bassRows.push_back ((int) r);
        expect (bassRows.size() > 2, "fixture bass rows");
        if (bassRows.size() <= 2)
            return;

        struct Lab
        {
            std::unique_ptr<PluginProcessor> processor = std::make_unique<PluginProcessor>();
            std::unique_ptr<juce::AudioProcessorEditor> editor;
            experimental::JointVaePanel* lab = nullptr;
            int played = 0;
        };
        // A panel with both bundles loaded; `prompt` replaces the fixture prompt bundle
        const auto open = [&] (const juce::File& prompt)
        {
            auto t = std::make_unique<Lab>();
            settleModel (*t->processor);
            t->editor.reset (t->processor->createEditor());
            t->lab = findChild<experimental::JointVaePanel> (*t->editor);
            expect (t->lab != nullptr);
            if (t->lab == nullptr)
                return t;
            t->lab->onLite = [p = t.get()] (const schema::Snapshot&) { ++p->played; };
            t->lab->loadBundle (juce::File (bundle));
            expect (pumpUntil ([&] { return t->lab->hasOutput(); }, 60000), "Lab bundle loaded");
            t->lab->loadPromptBundle (prompt);
            expect (pumpUntil ([&] { return t->lab->isPromptReady(); }, 60000), "Find offered");
            return t;
        };
        const auto found = [] (experimental::JointVaePanel& lab)
        { return pumpUntil ([&] { return ! lab.promptCandidates().empty(); }, 30000); };
        // The lab() result for the current request has arrived when the output codes match
        const auto playing = [&] (experimental::JointVaePanel& lab, const std::vector<float>& codes)
        { return pumpUntil ([&] { return same (lab.output().codes, codes); }, 30000); };

        beginTest ("prompt model loads on the first Find, not on open (R20)");
        {
            auto t = open (promptJson);
            if (t->lab && t->lab->isPromptReady())
            {
                auto& lab = *t->lab;
                pumpUntil ([] { return false; }, 1500); // a load on open would be done by now
                expect (! lab.isPromptLoaded(), "opening the editor does not load p0");
                lab.findSounds ("deep sub bass");
                expect (lab.statusText().startsWith ("Loading prompt model"), lab.statusText());
                expect (found (lab), "candidates");
                expect (lab.isPromptLoaded());
            }
        }

        beginTest ("a prompt load that a Lab bundle change overtakes is ignored");
        {
            auto t = open (promptJson);
            if (t->lab && t->lab->isPromptReady())
            {
                auto& lab = *t->lab;
                lab.findSounds ("deep sub bass");  // starts the load
                lab.loadPromptBundle (promptJson); // what a Lab bundle change does, before the load completes
                pumpUntil ([] { return false; }, 2000);
                expect (! lab.isPromptLoaded(), "the stale load did not count");
                expect (! lab.statusText().startsWith ("Loading prompt model"), lab.statusText());
                expect (lab.isPromptReady());
                lab.findSounds ("deep sub bass");
                expect (found (lab), "a fresh load and Find succeed");
                expect (lab.isPromptLoaded());
            }
        }

        beginTest ("find selects fixture row");
        {
            auto t = open (promptJson);
            if (t->lab && t->lab->isPromptReady())
            {
                auto& lab = *t->lab;
                lab.findSounds ("deep sub bass");
                expect (found (lab), "candidates");
                const auto& c = lab.promptCandidates();
                expectEquals ((int) c.size(), (int) bassRows.size());
                for (size_t i = 0; i < c.size() && i < bassRows.size(); ++i)
                    expectEquals (c[i].row, bassRows[i]);
                expectEquals (lab.selectedCandidate(), 0);
                expect (lab.inBankMode());
                expect (playing (lab, bankCodes (bassRows[0])), "the bank row's codes reach lab()");
                expect (pumpUntil ([&] { return t->played > 0; }, 5000), "it plays on Lite");
                expect (lab.latent() == bankZ (bassRows[0]), "the dot sits at the bank z");
                if (auto* orbit = findChild<experimental::LabOrbit> (lab); orbit && lab.padData())
                {
                    const auto* x = lab.padData()->find ("s:brightness");
                    expect (
                        x != nullptr
                        && std::abs (orbit->getHandle()[0] - juce::jlimit (0.0f, 1.0f, axisPosition (*x, lab.latent())))
                               < 1e-4f);
                }
                expectEquals (lab.statusText(),
                              juce::String (juce::CharPointer_UTF8 ("Selected a bank preset \xc2\xb7 Lite preview")));
                expect (c[0].name.startsWith ("Bass"), c[0].name);
                if (const auto shot = juce::SystemStats::getEnvironmentVariable ("VAE_LAB_BANK_SCREENSHOT", "");
                    shot.isNotEmpty())
                {
                    for (auto* child : t->editor->getChildren())
                        if (auto* b = dynamic_cast<juce::TextButton*> (child); b && b->getButtonText() == "VAE Lab")
                            b->onClick();
                    pumpUntil ([] { return false; }, 500);
                    if (auto stream = juce::File (shot).createOutputStream())
                        juce::PNGImageFormat().writeImageToStream (
                            t->editor->createComponentSnapshot (t->editor->getLocalBounds()), *stream);
                }
            }
        }

        beginTest ("alternative restores exact codes");
        {
            auto t = open (promptJson);
            if (t->lab && t->lab->isPromptReady())
            {
                auto& lab = *t->lab;
                lab.findSounds ("deep sub bass");
                expect (found (lab), "candidates");
                lab.selectCandidate (2);
                expect (playing (lab, bankCodes (bassRows[2])), "row 2's codes");
                expect (lab.latent() == bankZ (bassRows[2]));
                // Send to Surge gets the codec's patch of those codes: no decoder
                juce::MemoryBlock sent;
                lab.onPatch = [&sent] (const juce::MemoryBlock& m)
                {
                    sent = m;
                    return juce::Result::ok();
                };
                for (auto* b : findAll<juce::TextButton> (lab))
                    if (b->getButtonText() == "Send to Surge")
                        b->onClick();
                experimental::JointVae model;
                juce::MemoryBlock expected;
                expect (model.load (juce::File (bundle)).wasOk());
                expect (model.patch (bankCodes (bassRows[2]), expected).wasOk());
                expect (! sent.isEmpty() && sent == expected, "Send to Surge plays the bank codes");
                // A knob-pad edit changes a copy; selecting the row again restores it exactly
                auto* knob = [&]
                {
                    for (auto* b : findAll<juce::TextButton> (lab))
                        if (b->getButtonText() == "Knob pad")
                            return b;
                    return (juce::TextButton*) nullptr;
                }();
                auto* pad = findChild<experimental::LabPad> (lab);
                expect (knob != nullptr && pad != nullptr);
                if (knob && pad)
                {
                    knob->onClick();
                    for (int i = 0; i < 10; ++i)
                        pad->keyPressed (
                            juce::KeyPress (juce::KeyPress::rightKey, juce::ModifierKeys::shiftModifier, 0));
                    expect (pumpUntil ([&] { return ! same (lab.output().codes, bankCodes (bassRows[2])); }, 30000),
                            "the knob pad edits the codes");
                    lab.selectCandidate (1);
                    lab.selectCandidate (2);
                    expect (playing (lab, bankCodes (bassRows[2])), "row 2 restored exactly");
                    pumpUntil ([] { return false; }, 300);
                    expect (same (lab.output().codes, bankCodes (bassRows[2])), "and stays");
                    expect (lab.inBankMode());
                }
            }
        }

        // A click on the orbit at a normalised XYZ point that does not drag
        const auto click = [] (experimental::LabOrbit& orbit, experimental::LabOrbit::Position p)
        {
            if (auto* lab = orbit.getParentComponent(); lab && orbit.getWidth() == 0)
                lab->setSize (1240, 700); // the Lab tab is hidden: lay it out
            const auto px = orbit.project (p);
            const auto e = mouseAt (orbit, px, px);
            orbit.mouseDown (e);
            orbit.mouseUp (e);
        };
        const auto near = [] (experimental::LabOrbit::Position a, experimental::LabOrbit::Position b)
        { return std::abs (a[0] - b[0]) < 1e-4f && std::abs (a[1] - b[1]) < 1e-4f && std::abs (a[2] - b[2]) < 1e-4f; };
        const auto progress = [] (const juce::String& s)
        { return s.startsWith ("Loading prompt model") || s.startsWith ("Finding sounds"); };

        beginTest ("find highlights candidates");
        {
            auto t = open (promptJson);
            auto* pad = t->lab ? findChild<experimental::LabOrbit> (*t->lab) : nullptr;
            if (pad && t->lab->isPromptReady())
            {
                auto& lab = *t->lab;
                lab.findSounds ("deep sub bass");
                expect (found (lab), "candidates");
                const auto check = [&] (const char* x, const char* y)
                {
                    const auto* ax = lab.padData()->find (x);
                    const auto* ay = lab.padData()->find (y);
                    const auto* az = lab.padData()->find (lab.assignedAxes()[2]);
                    const auto& c = lab.promptCandidates();
                    const auto& h = pad->getHighlights();
                    expectEquals ((int) h.size(), (int) c.size());
                    for (size_t i = 0; i < c.size() && i < h.size() && ax && ay && az; ++i)
                        expect (near (h[i], { axisPosition (*ax, c[i].z), axisPosition (*ay, c[i].z),
                                              axisPosition (*az, c[i].z) }),
                                "highlight at the bank z");
                };
                check ("s:brightness", "s:length");
                lab.assignAxis (0, "s:attack"); // reprojects
                check ("s:attack", "s:length");
                lab.findSounds ("lush ambient pad"); // the next Find replaces them
                expect (pumpUntil ([&] { return lab.promptCandidates()[0].category == 2; }, 30000));
                check ("s:attack", "s:length");
                for (auto* b : findAll<juce::TextButton> (lab))
                    if (b->getButtonText() == "Reset")
                        b->onClick(); // Reset clears them
                expect (pad->getHighlights().empty() && lab.promptCandidates().empty(), "Reset clears the highlights");
            }
        }

        beginTest ("dot click selects candidate");
        {
            auto t = open (promptJson);
            auto* pad = t->lab ? findChild<experimental::LabOrbit> (*t->lab) : nullptr;
            if (pad && t->lab->isPromptReady())
            {
                auto& lab = *t->lab;
                lab.findSounds ("deep sub bass");
                expect (found (lab), "candidates");
                expect (playing (lab, bankCodes (bassRows[0])));
                expect (pad->getHighlights().size() > 2);
                if (pad->getHighlights().size() > 2)
                {
                    const auto at = pad->getHighlights()[2];
                    click (*pad, at);
                    expectEquals (lab.selectedCandidate(), 2);
                    expect (lab.inBankMode());
                    expect (playing (lab, bankCodes (bassRows[2])), "row 2's exact codes");
                    pumpUntil ([] { return false; }, 300);
                    expect (same (lab.output().codes, bankCodes (bassRows[2])), "and stays");
                    const auto clamped = [] (experimental::LabOrbit::Position p)
                    {
                        for (auto& v : p)
                            v = juce::jlimit (0.0f, 1.0f, v);
                        return p;
                    };
                    expect (near (pad->getHandle(), clamped (at)), "the handle moves to the dot");
                }
            }
        }

        beginTest ("drag explores from candidate");
        {
            auto t = open (promptJson);
            auto* pad = t->lab ? findChild<experimental::LabOrbit> (*t->lab) : nullptr;
            if (pad && t->lab->isPromptReady())
            {
                auto& lab = *t->lab;
                lab.findSounds ("deep sub bass");
                expect (found (lab), "candidates");
                const auto codes = bankCodes (bassRows[0]);
                expect (playing (lab, codes));
                const auto z0 = lab.latent();
                const auto* ax = lab.padData()->find ("s:brightness");
                const auto* ay = lab.padData()->find ("s:length");
                const auto played = t->played;
                pad->keyPressed (juce::KeyPress (juce::KeyPress::upKey));
                expect (! lab.inBankMode(), "the first move leaves the exact preset");
                expect (lab.latent() != z0, "the pad is not locked");
                if (ax && ay)
                {
                    expect (std::abs (axisPosition (*ax, lab.latent()) - axisPosition (*ax, z0)) < 1e-3f,
                            "from the bank z");
                    expect (std::abs (axisPosition (*ay, lab.latent()) - (axisPosition (*ay, z0) + 0.02f)) < 1e-3f);
                }
                expect (pumpUntil ([&] { return ! same (lab.output().codes, codes); }, 30000), "decoded, no codes");
                expect (pumpUntil ([&] { return t->played > played; }, 5000), "it plays");
                expectEquals (lab.statusText(), "Exploring from " + lab.promptCandidates()[0].name);
                // Nothing else is blocked
                const auto z1 = lab.latent();
                lab.nudge ("brighter");
                expect (lab.latent() != z1, "the word list edits");
                const auto z2 = lab.latent();
                lab.assignAxis (2, "s:noise");
                // Page Up moves the handle along Z (Orbit has no separate Z slider)
                pad->keyPressed (juce::KeyPress (pad->getHandle()[2] > 0.5f ? juce::KeyPress::pageDownKey
                                                                            : juce::KeyPress::pageUpKey));
                expect (lab.latent() != z2, "Z edits");
                expectEquals (lab.selectedCandidate(), 0, "the dot stays marked");
            }
        }

        beginTest ("reselect restores exact codes");
        {
            auto t = open (promptJson);
            auto* pad = t->lab ? findChild<experimental::LabOrbit> (*t->lab) : nullptr;
            if (pad && t->lab->isPromptReady())
            {
                auto& lab = *t->lab;
                lab.findSounds ("deep sub bass");
                expect (found (lab), "candidates");
                const auto codes = bankCodes (bassRows[1]);
                lab.selectCandidate (1);
                expect (playing (lab, codes));
                pad->keyPressed (juce::KeyPress (juce::KeyPress::rightKey, juce::ModifierKeys::shiftModifier, 0));
                expect (pumpUntil ([&] { return ! same (lab.output().codes, codes); }, 30000), "exploring");
                if (pad->getHighlights().size() > 1)
                    click (*pad, pad->getHighlights()[1]); // the dot
                expect (lab.inBankMode());
                expect (playing (lab, codes), "the exact codes again");
                expect (lab.latent() == bankZ (bassRows[1]));
                pad->keyPressed (juce::KeyPress (juce::KeyPress::leftKey));
                expect (pumpUntil ([&] { return ! same (lab.output().codes, codes); }, 30000), "exploring again");
                lab.selectCandidate (1); // the chip
                expect (playing (lab, codes), "the chip restores them too");
            }
        }

        beginTest ("send to surge exact then decoded");
        {
            auto t = open (promptJson);
            auto* pad = t->lab ? findChild<experimental::LabOrbit> (*t->lab) : nullptr;
            if (pad && t->lab->isPromptReady())
            {
                auto& lab = *t->lab;
                lab.findSounds ("deep sub bass");
                expect (found (lab), "candidates");
                const auto codes = bankCodes (bassRows[0]);
                expect (playing (lab, codes));
                juce::MemoryBlock sent;
                lab.onPatch = [&sent] (const juce::MemoryBlock& m)
                {
                    sent = m;
                    return juce::Result::ok();
                };
                const auto send = [&]
                {
                    sent.reset();
                    for (auto* b : findAll<juce::TextButton> (lab))
                        if (b->getButtonText() == "Send to Surge")
                            b->onClick();
                };
                experimental::JointVae model;
                juce::MemoryBlock exact;
                expect (model.load (juce::File (bundle)).wasOk());
                expect (model.patch (codes, exact).wasOk());
                send();
                expect (! sent.isEmpty() && sent == exact, "the exact preset's codes");
                pad->keyPressed (juce::KeyPress (juce::KeyPress::upKey, juce::ModifierKeys::shiftModifier, 0));
                expect (pumpUntil ([&] { return ! same (lab.output().codes, codes); }, 30000));
                pumpUntil ([] { return false; }, 300);
                juce::MemoryBlock decoded;
                expect (model.patch (lab.output().codes, decoded).wasOk());
                send();
                expect (! sent.isEmpty() && sent == decoded && sent != exact, "the decoded patch after exploring");
            }
        }

        beginTest ("a dropped find never leaves a progress status");
        {
            auto t = open (promptJson);
            auto* pad = t->lab ? findChild<experimental::LabOrbit> (*t->lab) : nullptr;
            if (pad && t->lab->isPromptReady())
            {
                auto& lab = *t->lab;
                // Overtaken by a drag (cold: the load, then the find)
                lab.findSounds ("deep sub bass");
                pad->keyPressed (juce::KeyPress (juce::KeyPress::upKey));
                expect (pumpUntil ([&] { return lab.isPromptLoaded() && ! progress (lab.statusText()); }, 30000),
                        lab.statusText());
                pumpUntil ([] { return false; }, 300);
                expect (lab.promptCandidates().empty(), "dropped");
                // Arrives while "Replace your Synth edits?" is open
                lab.markEdited();
                pad->keyPressed (juce::KeyPress (juce::KeyPress::upKey));
                expect (lab.isAskingToReplace());
                lab.findSounds ("deep sub bass");
                pumpUntil ([] { return false; }, 1500);
                expect (lab.promptCandidates().empty() && ! progress (lab.statusText()), lab.statusText());
                lab.answerReplace (false);
                // Its own question answered Keep editing
                lab.findSounds ("deep sub bass");
                expect (pumpUntil ([&] { return lab.isAskingToReplace(); }, 30000));
                lab.answerReplace (false);
                expect (lab.promptCandidates().empty() && ! progress (lab.statusText()), lab.statusText());
            }
        }

        beginTest ("stale prompt result dropped");
        {
            auto t = open (promptJson);
            if (t->lab && t->lab->isPromptReady())
            {
                auto& lab = *t->lab;
                lab.findSounds ("deep sub bass");
                lab.findSounds ("lush ambient pad");
                expect (found (lab), "candidates");
                pumpUntil ([] { return false; }, 1500);
                expect (! lab.promptCandidates().empty() && lab.promptCandidates()[0].category == 2,
                        "the newer prompt's (Pad) results");
                // A manual edit after a prompt drops its result
                const auto pads = lab.promptCandidates();
                lab.findSounds ("deep sub bass");
                if (auto* pad = findChild<experimental::LabOrbit> (lab))
                    pad->keyPressed (juce::KeyPress (juce::KeyPress::upKey));
                const auto z1 = lab.latent();
                pumpUntil ([] { return false; }, 1500);
                expect (lab.latent() == z1 && ! lab.inBankMode(), "the drag wins");
                expect (! progress (lab.statusText()), lab.statusText());
                expect (lab.promptCandidates().size() == pads.size() && lab.promptCandidates()[0].row == pads[0].row);
            }
        }

        beginTest ("no-bank category keeps sound");
        {
            // The fixture bundle with every Bass row relabelled Keys: "deep sub bass" (Bass) has no bank rows
            const auto dir = juce::File::createTempFile ("vae_prompt_nobank");
            dir.createDirectory();
            for (auto name : { "text_encoder.onnx", "prior.onnx", "tokenizer.json", "bank.bin", "prompt.json" })
                expect (promptJson.getSiblingFile (name).createSymbolicLink (dir.getChildFile (name), true));
            auto bankJson = juce::JSON::parse (promptJson.getSiblingFile ("bank.json"));
            if (auto* cats = bankJson["category"].getArray())
                for (auto& c : *cats)
                    if ((int) c == 0)
                        c = 3;
            expect (dir.getChildFile ("bank.json").replaceWithText (juce::JSON::toString (bankJson)));
            auto t = open (dir.getChildFile ("prompt.json"));
            if (t->lab && t->lab->isPromptReady())
            {
                auto& lab = *t->lab;
                lab.nudge ("pad"); // a chosen start, so something plays
                expect (pumpUntil ([&] { return t->played > 0; }, 30000));
                pumpUntil ([] { return false; }, 300);
                const auto z0 = lab.latent();
                const auto codes = lab.output().codes;
                const auto played = t->played;
                lab.findSounds ("deep sub bass");
                expect (pumpUntil ([&] { return lab.statusText().contains ("no Bass presets"); }, 30000),
                        lab.statusText());
                pumpUntil ([] { return false; }, 300);
                expect (lab.latent() == z0 && same (lab.output().codes, codes), "the sound is kept");
                expectEquals (t->played, played);
                expect (lab.promptCandidates().empty() && ! lab.inBankMode());
                expect (lab.statusText().contains ("Word list"), lab.statusText());
            }
            t.reset();
            dir.deleteRecursively();
        }

        beginTest ("editor closes mid-prompt");
        {
            auto t = open (promptJson);
            if (t->lab && t->lab->isPromptReady())
            {
                const auto before = t->played;
                juce::String longText;
                for (int i = 0; i < 200; ++i)
                    longText << "warm evolving analog pad ";
                t->lab->findSounds (longText);
                pumpUntil ([] { return false; }, 50); // the job is running
                const auto t0 = juce::Time::getMillisecondCounter();
                t->editor.reset();
                logMessage ("closing mid-prompt took " + juce::String (juce::Time::getMillisecondCounter() - t0)
                            + " ms");
                pumpUntil ([] { return false; }, 1000);
                expectEquals (t->played, before); // nothing reached the deleted panel
            }
        }
    }
#endif

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

        beginTest ("VAE Lab layout: equal sidebars, a centre column, a centre-only keyboard");
        {
            PluginProcessor processor;
            settleModel (processor);
            std::unique_ptr<juce::AudioProcessorEditor> editor (processor.createEditor());
            auto* lab = findChild<experimental::JointVaePanel> (*editor);
            auto* labButton = findButton (*editor, "VAE Lab");
            auto* editorKeys = findChild<FlatKeyboard> (*editor);
            expect (lab != nullptr && labButton != nullptr && editorKeys != nullptr);
            if (lab && labButton && editorKeys)
            {
                labButton->onClick();
                const auto shoot = [&] (const char* variable)
                {
                    const auto file = juce::SystemStats::getEnvironmentVariable (variable, "");
                    if (file.isNotEmpty())
                        if (auto stream = juce::File (file).createOutputStream())
                            juce::PNGImageFormat().writeImageToStream (
                                editor->createComponentSnapshot (editor->getLocalBounds()), *stream);
                };
#if VAE_WITH_SURGE
                auto* orbit = findChild<experimental::LabOrbit> (*lab);
                auto* list = findChild<juce::Viewport> (*lab);
                auto* keys = findChild<PitcherKeyboard> (*lab);
                auto* prompt = findChild<juce::TextEditor> (*lab);
                expect (orbit != nullptr && list != nullptr && keys != nullptr && prompt != nullptr);
                if (orbit && list && keys && prompt)
                {
                    const auto& l = lab->getLayout();
                    const auto in = [&] (juce::Component* c) { return lab->getLocalArea (c, c->getLocalBounds()); };
                    expectEquals (l.left.getWidth(), l.right.getWidth(), "equal sidebar widths");
                    expect (l.left.getY() == l.right.getY() && l.left.getHeight() == l.right.getHeight(),
                            "equal sidebar heights");
                    expect (l.left.getRight() < l.centre.getX() && l.centre.getRight() < l.right.getX(), "no overlap");
                    for (auto* c : std::initializer_list<juce::Component*> { orbit, prompt, keys })
                        expect (l.centre.contains (in (c)), "in the centre column");
                    expect (in (orbit).getBottom() <= in (prompt).getY() && in (prompt).getBottom() <= in (keys).getY(),
                            "orbit, prompt, keyboard from top to bottom");
                    expect (in (keys).getX() >= l.centre.getX() && in (keys).getRight() <= l.centre.getRight()
                                && in (keys).getBottom() <= l.left.getBottom(),
                            "the keyboard stays in the centre column");
                    expect (l.right.contains (in (list)), "the axis list is in the right sidebar");
                    expect (in (orbit).getWidth() >= 0.45 * lab->getWidth(),
                            "orbit width " + juce::String (in (orbit).getWidth()));
                    expect (in (orbit).getHeight() >= 0.5 * lab->getHeight(),
                            "orbit height " + juce::String (in (orbit).getHeight()));
                    expect (keys->getRangeStart() == 48 && keys->getRangeEnd() == 84, "C3 to C6: 37 keys");
                    expect (keys->getKeyStartPosition (84) + keys->getKeyWidth() <= (float) keys->getWidth() + 1.0f,
                            "every key fits");
                    expect (keys->isVisible() && ! editorKeys->isVisible(), "the Lab keyboard replaces the editor's");
                    const auto bundle = juce::SystemStats::getEnvironmentVariable ("VAE_JOINT_BUNDLE", "");
                    if (bundle.isNotEmpty())
                    {
                        lab->loadBundle (juce::File (bundle));
                        expect (
                            pumpUntil ([&] { return list->getViewedComponent()->getNumChildComponents() > 30; }, 30000),
                            "axis rows");
                        expect (pumpUntil ([&] { return lab->hasOutput(); }, 30000), "the first decode");
                        expect (list->getViewedComponent()->getWidth() <= list->getMaximumVisibleWidth(),
                                "rows clear the scroll bar: " + juce::String (list->getViewedComponent()->getWidth())
                                    + " > " + juce::String (list->getMaximumVisibleWidth()));
                        expect (lab->assignedAxes()[2].isNotEmpty(), "Orbit has a Z axis");
                        shoot ("VAE_LAB_SCREENSHOT");

                        // The inline Replace question: its text fits, and other status writes leave it showing
                        lab->markEdited();
                        moveHandle (*orbit, 0.7f, 0.6f);
                        expect (lab->isAskingToReplace());
                        juce::Label* note = nullptr;
                        for (auto* label : findAll<juce::Label> (*lab))
                            if (label->getText().startsWith ("Replace your Synth edits?"))
                                note = label;
                        expect (note != nullptr && note->isVisible());
                        if (note)
                        {
                            juce::AttributedString text (note->getText());
                            text.setFont (note->getFont());
                            juce::TextLayout layout;
                            layout.createLayout (text, (float) note->getWidth());
                            expect (layout.getHeight() <= (float) note->getHeight()
                                        && layout.getWidth() <= (float) note->getWidth(),
                                    "the banner text is not clipped");
                            expect (l.centre.contains (in (note)), "the banner is in the centre column");
                        }
                        if (auto* send = findButton (*lab, "Send to Surge"))
                            send->onClick();
                        expect (lab->statusText().contains ("Replace your Synth edits?"), lab->statusText());
                        shoot ("VAE_LAB_ASK_SCREENSHOT");
                        lab->answerReplace (false);

                        // The knob pad keeps its flat pad and category map
                        bool shown = false;
                        lab->onMapShown = [&shown] { shown = true; };
                        if (auto* knob = findButton (*lab, "Knob pad"))
                            knob->onClick();
                        auto* pad = findChild<experimental::LabPad> (*lab);
                        expect (pad != nullptr && pad->isVisible() && ! orbit->isVisible(),
                                "the knob pad replaces Orbit");
                        expect (pumpUntil ([&] { return shown; }, 30000), "knob map");
                        lab->onMapShown = nullptr;
                        shoot ("VAE_LAB_KNOB_SCREENSHOT");
                    }
                }
#else
                expect (findChild<experimental::LabOrbit> (*lab) == nullptr
                        && findChild<juce::Viewport> (*lab) == nullptr);
                auto* note = findChild<juce::Label> (*lab);
                expect (note != nullptr && note->getText() == "VAE Lab needs a Surge build");
                shoot ("VAE_LAB_SCREENSHOT");
#endif
                expect (editor->createComponentSnapshot (editor->getLocalBounds()).isValid());
                findButton (*editor, "Synth")->onClick();
                expect (editorKeys->isVisible(), "the editor keyboard is back in the Synth view");
            }
        }

#if ! VAE_PROMPT_TAB // Task 16: the Prompt tab is compiled out by default
        beginTest ("two views only");
        {
            PluginProcessor processor;
            std::unique_ptr<juce::AudioProcessorEditor> editor (processor.createEditor());
            expect (findButton (*editor, "Synth") != nullptr);
            expect (findButton (*editor, "VAE Lab") != nullptr);
            expect (findButton (*editor, "Prompt") == nullptr);
        }

        beginTest ("prompt tab compiled out");
        {
            expectEquals (VAE_PROMPT_TAB, 0);
            PluginProcessor processor;
            std::unique_ptr<juce::AudioProcessorEditor> editor (processor.createEditor());
            expect (findChild<XYPad> (*editor) == nullptr, "no latent XY pad");
            expect (findChild<ParamList> (*editor) == nullptr, "no parameter sidebar");
            expect (findButton (*editor, "Generate") == nullptr && findButton (*editor, "SAME bank...") == nullptr);
            auto* status = headerStatus (*editor);
            expect (status != nullptr && ! status->getText().contains ("SAME")
                        && ! status->getText().contains ("model"),
                    status ? status->getText() : juce::String());
        }
#endif

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
                if (auto* orbit = findChild<experimental::LabOrbit> (*lab))
                    orbit->keyPressed (juce::KeyPress (juce::KeyPress::rightKey));
                expect (pumpUntil ([&] { return played > 0; }, 30000), "a handle move plays");
            }
        }
        else
            logMessage ("Set VAE_JOINT_BUNDLE to run the VAE Lab bundle tests.");

        if (const auto bundle = juce::SystemStats::getEnvironmentVariable ("VAE_JOINT_BUNDLE", ""); bundle.isNotEmpty())
        {
            // An editor on `processor` whose Lab has loaded the bundle; nullptr when it did not load
            const auto openLab = [&] (PluginProcessor& processor)
            {
                std::unique_ptr<juce::AudioProcessorEditor> editor (processor.createEditor());
                auto* lab = findChild<experimental::JointVaePanel> (*editor);
                auto* orbit = lab ? findChild<experimental::LabOrbit> (*lab) : nullptr;
                if (lab)
                    lab->loadBundle (juce::File (bundle));
                const bool ready = lab && pumpUntil ([&] { return lab->hasOutput(); }, 60000);
                expect (orbit != nullptr && ready, "bundle loaded");
                if (orbit == nullptr || ! ready)
                    editor.reset();
                return editor;
            };
            // A fresh editor whose Lab has loaded the bundle; `steps` get the editor and its Lab
            const auto withLab =
                [&] (const std::function<void (juce::AudioProcessorEditor&, PluginProcessor&,
                                               experimental::JointVaePanel&, experimental::LabOrbit&)>& steps)
            {
                PluginProcessor processor;
                settleModel (processor);
                processor.prepareToPlay (48000.0, 512); // Surge needs its synth to leave patch mode
                if (auto editor = openLab (processor))
                {
                    auto* lab = findChild<experimental::JointVaePanel> (*editor);
                    steps (*editor, processor, *lab, *findChild<experimental::LabOrbit> (*lab));
                }
                processor.releaseResources();
            };
            const auto liteOf = [] (const experimental::JointVaePanel& lab)
            {
                schema::Snapshot s;
                for (int i = 0; i < schema::kControlCount; ++i)
                    s[i] = schema::sanitiseValue (i, lab.output().lite[i]);
                return s;
            };
            const auto same = [] (const schema::Snapshot& a, const schema::Snapshot& b)
            {
                for (int i = 0; i < schema::kControlCount; ++i)
                    if (std::abs (a[i] - b[i]) > 1e-5f)
                        return false;
                return true;
            };

            beginTest ("lab preset shows in synth view");
            withLab (
                [&] (juce::AudioProcessorEditor& editor, PluginProcessor& processor, experimental::JointVaePanel& lab,
                     experimental::LabOrbit& pad)
                {
                    findButton (editor, "VAE Lab")->onClick();
                    const auto before = lab.output().codes;
                    moveHandle (pad, 0.85f, 0.2f);
                    expect (
                        pumpUntil (
                            [&]
                            { return lab.output().codes != before && same (processor.getControls(), liteOf (lab)); },
                            30000),
                        "the Lab's point is on the parameters");
                    findButton (editor, "Synth")->onClick();
                    int checked = 0;
                    for (int i = 0; i < schema::kControlCount; ++i)
                        if (auto* knob = synthKnob (editor, i))
                        {
                            ++checked;
                            expectWithinAbsoluteError ((float) knob->getValue(), liteOf (lab)[i], 1e-4f,
                                                       schema::control (i).id);
                        }
                    expect (checked > 40, "synth knobs found: " + juce::String (checked));
                    if (const auto shot = juce::SystemStats::getEnvironmentVariable ("VAE_SYNTH_SCREENSHOT", "");
                        shot.isNotEmpty())
                        if (auto stream = juce::File (shot).createOutputStream())
                            juce::PNGImageFormat().writeImageToStream (
                                editor.createComponentSnapshot (editor.getLocalBounds()), *stream);
                    expect (! lab.isEdited(), "showing the Synth view is not an edit");

                    // Send to Surge: the Synth view says so; the first knob move returns to Lite
                    auto* send = findButton (lab, "Send to Surge");
                    expect (send != nullptr && send->isEnabled());
                    if (send)
                        send->onClick();
                    auto* status = headerStatus (editor);
                    expect (status != nullptr && status->isVisible()
                                && status->getText()
                                       == "Playing the full Surge patch from the Lab. Knobs return to Lite",
                            status ? status->getText() : juce::String());
                    auto* surge = dynamic_cast<SurgeEngine*> (processor.findEngine (EngineKind::Surge));
                    expect (surge != nullptr && surge->hasPatchData());
                    if (auto* knob = synthKnob (editor, firstContinuous()))
                        knob->setValue (knob->getValue() > 0.5 ? 0.1 : 0.9, juce::sendNotificationSync);
                    pumpUntil ([] { return false; }, 100);
                    expectEquals (processor.getParams().get (PluginProcessor::kEngineParamId), 0.0f); // Lite is first
                    expect (surge != nullptr && ! surge->hasPatchData());
                    expect (status != nullptr && status->getText().isEmpty(),
                            status ? status->getText() : juce::String());
                });

            beginTest ("lab writes do not mark edited");
            withLab (
                [&] (juce::AudioProcessorEditor&, PluginProcessor& processor, experimental::JointVaePanel& lab,
                     experimental::LabOrbit& pad)
                {
                    for (auto p : { juce::Point<float> (0.2f, 0.8f), juce::Point<float> (0.7f, 0.4f) })
                    {
                        const auto before = lab.output().codes;
                        moveHandle (pad, p.x, p.y);
                        expect (pumpUntil (
                            [&]
                            { return lab.output().codes != before && same (processor.getControls(), liteOf (lab)); },
                            30000));
                    }
                    pumpUntil ([] { return false; }, 300); // every async parameter notification lands
                    expect (! lab.isEdited());
                    expect (! lab.statusText().contains ("Edited"), lab.statusText());
                });

            beginTest ("synth edit marks lab edited");
            withLab (
                [&] (juce::AudioProcessorEditor& editor, PluginProcessor& processor, experimental::JointVaePanel& lab,
                     experimental::LabOrbit& pad)
                {
                    auto before = lab.output().codes;
                    moveHandle (pad, 0.3f, 0.3f);
                    expect (pumpUntil (
                        [&] { return lab.output().codes != before && same (processor.getControls(), liteOf (lab)); },
                        30000));
                    // A Synth-view knob move, as the user makes it
                    auto* knob = synthKnob (editor, firstContinuous());
                    expect (knob != nullptr);
                    if (knob == nullptr)
                        return;
                    knob->setValue (knob->getValue() > 0.5 ? 0.1 : 0.9, juce::sendNotificationSync);
                    expect (pumpUntil ([&] { return lab.isEdited(); }, 1000), "knob move marks edited");
                    expectEquals (lab.statusText(), juce::String ("Edited in Synth view"));
                    const auto edited = processor.getControls();
                    const auto z0 = lab.latent();

                    // Keep editing: sound and Lab state unchanged
                    moveHandle (pad, 0.9f, 0.9f);
                    expect (lab.isAskingToReplace(), "a Lab move asks");
                    // R21: asked inline in the Lab, never in a top-level window a host can hide
                    auto* keep = findButton (lab, "Keep editing");
                    auto* replace = findButton (lab, "Replace");
                    expect (keep != nullptr && keep->isVisible() && replace != nullptr && replace->isVisible(),
                            "the question's buttons are in the Lab");
                    expect (lab.statusText().contains ("Replace your Synth edits?"), lab.statusText());
                    pumpUntil ([] { return false; }, 50);
                    for (int w = 0; w < juce::TopLevelWindow::getNumTopLevelWindows(); ++w)
                        expect (dynamic_cast<juce::AlertWindow*> (juce::TopLevelWindow::getTopLevelWindow (w))
                                    == nullptr,
                                "no top-level dialog");
                    moveHandle (pad, 0.1f, 0.1f); // ignored while asking
                    if (keep != nullptr)
                        keep->triggerClick();
                    pumpUntil ([] { return false; }, 50); // triggerClick is async
                    expect (! lab.isAskingToReplace(), "Keep editing answers");
                    expect (keep == nullptr || ! keep->isVisible(), "the question goes away");
                    expectEquals (lab.statusText(), juce::String ("Edited in Synth view"));
                    pumpUntil ([] { return false; }, 500);
                    expect (same (processor.getControls(), edited), "Keep editing leaves the parameters");
                    expect (lab.latent() == z0, "Keep editing leaves the Lab");
                    expect (lab.isEdited());

                    // Replace: the new point plays
                    before = lab.output().codes;
                    moveHandle (pad, 0.9f, 0.9f);
                    expect (lab.isAskingToReplace(), "asks again");
                    lab.answerReplace (true);
                    expect (! lab.isAskingToReplace() && ! lab.isEdited());
                    expect (
                        pumpUntil (
                            [&]
                            { return lab.output().codes != before && same (processor.getControls(), liteOf (lab)); },
                            30000),
                        "Replace applies the new point");
                    expect (! (lab.latent() == z0));
                    pumpUntil ([] { return false; }, 200);
                    expect (! lab.isEdited(), "the replacing write is the Lab's");

                    // Host automation from another thread also marks edited
                    const auto index = processor.getParams().indexOf (schema::control (firstContinuous()).id);
                    std::thread host (
                        [&]
                        { processor.getParams().set (index, processor.getParams().get (index) > 0.5f ? 0.2f : 0.8f); });
                    host.join();
                    expect (pumpUntil ([&] { return lab.isEdited(); }, 1000), "automation marks edited");
                });

            beginTest ("an in-flight lab decode never overwrites a non-Lab write");
            withLab (
                [&] (juce::AudioProcessorEditor&, PluginProcessor& processor, experimental::JointVaePanel& lab,
                     experimental::LabOrbit& pad)
                {
                    const auto index = processor.getParams().indexOf (schema::control (firstContinuous()).id);
                    const auto before = lab.output().codes;
                    moveHandle (pad, 0.8f, 0.7f);              // not edited: the decode is queued
                    processor.getParams().set (index, 0.123f); // a host write lands before its result
                    expect (pumpUntil ([&] { return lab.output().codes != before; }, 30000), "the decode arrives");
                    pumpUntil ([] { return false; }, 300);
                    expectWithinAbsoluteError (processor.getParams().get (index), 0.123f, 1e-6f);
                    expect (lab.isEdited(), "the host write marks edited");
                    moveHandle (pad, 0.2f, 0.3f);
                    expect (lab.isAskingToReplace(), "the next Lab move asks");
                    lab.answerReplace (true);
                    const auto replaced = lab.output().codes;
                    expect (
                        pumpUntil (
                            [&]
                            { return lab.output().codes != replaced && same (processor.getControls(), liteOf (lab)); },
                            30000),
                        "Replace still plays");
                });

            beginTest ("reset returns to the same random real preset");
            withLab (
                [&] (juce::AudioProcessorEditor&, PluginProcessor&, experimental::JointVaePanel& lab,
                     experimental::LabOrbit&)
                {
                    juce::ComboBox* base = nullptr;
                    for (auto* c : findAll<juce::ComboBox> (lab))
                        for (int i = 0; i < c->getNumItems(); ++i)
                            if (c->getItemText (i) == "Random real preset")
                                base = c;
                    auto* reset = findButton (lab, "Reset");
                    expect (base != nullptr && reset != nullptr);
                    if (base == nullptr || reset == nullptr)
                        return;
                    const auto choose = [&] (int id)
                    {
                        base->setSelectedId (base->getItemId (0), juce::sendNotificationSync);
                        base->setSelectedId (id, juce::sendNotificationSync);
                        return lab.latent();
                    };
                    const auto start = choose (100);
                    for (int k = 0; k < 4; ++k)
                    {
                        reset->onClick();
                        expect (lab.latent() == start, "Reset returns to the drawn preset");
                    }
                    bool redrawn = false;
                    for (int k = 0; k < 8 && ! redrawn; ++k)
                        redrawn = ! (choose (100) == start);
                    expect (redrawn, "choosing Random real preset again draws a new one");
                });

            beginTest ("camera gestures never decode or play");
            withLab (
                [&] (juce::AudioProcessorEditor& editor, PluginProcessor&, experimental::JointVaePanel& lab,
                     experimental::LabOrbit& orbit)
                {
                    findButton (editor, "VAE Lab")->onClick();
                    int played = 0;
                    auto play = lab.onLite;
                    lab.onLite = [&] (const schema::Snapshot& s)
                    {
                        ++played;
                        if (play)
                            play (s);
                    };
                    const auto before = lab.output().codes;
                    moveHandle (orbit, 0.8f, 0.3f); // the user acts, so a decode would play
                    expect (pumpUntil ([&] { return played > 0 && lab.output().codes != before; }, 30000));
                    pumpUntil ([] { return false; }, 300);
                    const auto codes = lab.output().codes;
                    const auto z = lab.latent();
                    const auto handle = orbit.getHandle();
                    const int count = played;
                    const auto seen = orbit.project (handle);
                    auto* view = findChild<OrbitView> (orbit);
                    expect (view != nullptr);
                    if (view == nullptr)
                        return;
                    // Orbit: a drag on empty space (a corner, away from the handle), then its release
                    const juce::Point<float> corner { 4.0f, (float) view->getHeight() - 4.0f };
                    const auto end = corner + juce::Point<float> (90.0f, -40.0f);
                    view->mouseDown (mouseAt (*view, corner, corner));
                    view->mouseDrag (mouseAt (*view, end, corner));
                    view->mouseUp (mouseAt (*view, end, corner, true));
                    orbit.mouseUp (mouseAt (orbit, end, corner, true));
                    expect (orbit.project (handle).getDistanceFrom (seen) > 1.0f, "the drag orbits the camera");
                    // Zoom: wheel and the - / + buttons; then Reset view and Home
                    juce::MouseWheelDetails wheel {};
                    wheel.deltaY = 0.5f;
                    view->mouseWheelMove (mouseAt (*view, corner, corner), wheel);
                    for (auto* text : { "+", "-", "Reset view" })
                        if (auto* b = findButton (orbit, text))
                            b->onClick();
                        else
                            expect (false, juce::String ("camera button ") + text);
                    expect (orbit.project (handle).getDistanceFrom (seen) < 1e-3f, "Reset view restores the camera");
                    view->keyPressed (juce::KeyPress (juce::KeyPress::homeKey));
                    pumpUntil ([] { return false; }, 500);
                    expectEquals (played, count, "camera gestures play nothing");
                    expect (lab.latent() == z && lab.output().codes == codes, "and decode nothing");
                    expect (orbit.getHandle() == handle, "the handle stays");
                });

            beginTest ("an orbit handle drag constrains the assigned axes, then decodes");
            withLab (
                [&] (juce::AudioProcessorEditor& editor, PluginProcessor& processor, experimental::JointVaePanel& lab,
                     experimental::LabOrbit& orbit)
                {
                    findButton (editor, "VAE Lab")->onClick();
                    auto* view = findChild<OrbitView> (orbit);
                    expect (view != nullptr);
                    if (view == nullptr)
                        return;
                    const auto before = lab.output().codes;
                    const auto start = orbit.getHandle();
                    const auto at = orbit.project (start);
                    const auto to = at + juce::Point<float> (30.0f, -20.0f);
                    view->mouseDown (mouseAt (*view, at, at));
                    view->mouseDrag (mouseAt (*view, to, at));
                    view->mouseUp (mouseAt (*view, to, at, true));
                    const auto handle = orbit.getHandle();
                    expect (handle != start, "the handle moved");
                    const auto& ids = lab.assignedAxes();
                    for (int i = 0; i < 3; ++i)
                        if (auto* axis = lab.padData()->find (ids[(size_t) i]))
                            expectWithinAbsoluteError (axisPosition (*axis, lab.latent()), handle[(size_t) i], 1e-3f,
                                                       "slot " + juce::String (i) + " reads the handle");
                        else
                            expect (false, "slot " + juce::String (i) + " is assigned");
                    expect (
                        pumpUntil (
                            [&]
                            { return lab.output().codes != before && same (processor.getControls(), liteOf (lab)); },
                            30000),
                        "the constrained z decodes and plays");
                });

            beginTest ("the orbit LFO moves around the point with Lock on and every switch held (Task 19)");
            withLab (
                [&] (juce::AudioProcessorEditor& editor, PluginProcessor& processor, experimental::JointVaePanel& lab,
                     experimental::LabOrbit& orbit)
                {
                    findButton (editor, "VAE Lab")->onClick();
                    auto* button = findButton (lab, "Orbit LFO");
                    auto* lock = findButton (lab, "Lock structure");
                    expect (button != nullptr && button->isEnabled() && lock != nullptr, "the Orbit LFO button");
                    if (button == nullptr || lock == nullptr)
                        return;
                    const auto z0 = lab.latent();
                    const auto centre = orbit.getHandle();
                    std::vector<schema::Snapshot> played;
                    std::vector<std::vector<float>> codes;
                    std::vector<experimental::LabOrbit::Position> handles;
                    auto play = lab.onLite;
                    lab.onLite = [&] (const schema::Snapshot& s)
                    {
                        played.push_back (s);
                        codes.push_back (lab.output().codes);
                        handles.push_back (orbit.getHandle());
                        if (play)
                            play (s);
                    };
                    button->onClick(); // R14: turning the orbit on is a user action, so it plays
                    expect (lab.isOrbiting() && button->getToggleState());
                    pumpUntil ([] { return false; }, 1000);
                    logMessage ("orbit results in 1 s: " + juce::String ((int) played.size()));
                    expect (played.size() >= 20, "results in 1 s: " + juce::String ((int) played.size()));
                    if (played.size() < 2)
                        return;
                    const auto first = std::hypot (handles[1][0] - centre[0], handles[1][1] - centre[1]);
                    expect (first < 0.3f * 0.08f, "the radius ramps in: no jump at the start, " + juce::String (first));
                    expect (lock->getToggleState() && ! lock->isEnabled(), "Lock is forced on");
                    const auto numeric = (size_t) lab.output().layout.numNumeric;
                    expect (numeric > 0);
                    for (size_t k = 1; k < codes.size(); ++k)
                        expect (std::equal (codes[k].begin() + (long) numeric, codes[k].end(),
                                            codes[0].begin() + (long) numeric),
                                "the structure codes stay (Lock)");
                    int continuousMoves = 0;
                    for (size_t k = 1; k < played.size(); ++k)
                        for (int i = 0; i < schema::kControlCount; ++i)
                            if (schema::control (i).kind != schema::Kind::Continuous)
                                expectEquals (played[k][i], played[0][i], schema::control (i).id);
                            else
                                continuousMoves += ! juce::exactlyEqual (played[k][i], played[0][i]);
                    expect (continuousMoves > 0, "continuous controls move");
                    expect (orbit.getHandle() != centre, "the dot moves");
                    const auto at = orbit.getHandle();
                    float d2 = 0;
                    for (size_t i = 0; i < 3; ++i)
                        d2 += (at[i] - centre[i]) * (at[i] - centre[i]);
                    expect (std::sqrt (d2) <= 0.08f * 1.12f + 1e-4f, "the dot stays within the depth");
                    const auto& ids = lab.assignedAxes();
                    for (size_t i = 0; i < 2; ++i)
                        if (auto* axis = lab.padData()->find (ids[i]))
                            expectWithinAbsoluteError (axisPosition (*axis, lab.latent()), at[i], 1e-3f,
                                                       "the axis sliders follow the handle");
                    if (const auto shot = juce::SystemStats::getEnvironmentVariable ("VAE_ORBIT_SCREENSHOT", "");
                        shot.isNotEmpty())
                    {
                        for (auto* s : findAll<juce::Slider> (lab)) // the widest path reads best in a still
                            if (s->getTitle() == "Orbit depth")
                                s->setValue (0.25, juce::sendNotificationSync);
                        pumpUntil ([] { return false; }, 1500);
                        juce::File (shot).deleteFile();
                        if (auto stream = juce::File (shot).createOutputStream())
                            juce::PNGImageFormat().writeImageToStream (
                                editor.createComponentSnapshot (editor.getLocalBounds()), *stream);
                    }

                    // Off: the sound returns to the centre point, and Lock to how the user left it
                    button->onClick();
                    expect (! lab.isOrbiting() && ! button->getToggleState());
                    expect (lab.latent() == z0, "the centre's z");
                    expect (orbit.getHandle() == centre, "the handle is back at the centre");
                    expect (pumpUntil (
                                [&]
                                {
                                    return same (processor.getControls(), liteOf (lab)) && lab.latent() == z0
                                           && lab.output().latent == z0;
                                },
                                30000),
                            "the centre plays");
                    pumpUntil ([] { return false; }, 300);
                    expect (! lock->getToggleState() && lock->isEnabled(), "Lock is back off");
                    expect (lab.output().codes == codes[0], "the centre's exact codes");
                });

            beginTest ("a drag re-centres the orbit; Reset, a category, the knob pad and a Synth edit stop it");
            withLab (
                [&] (juce::AudioProcessorEditor& editor, PluginProcessor&, experimental::JointVaePanel& lab,
                     experimental::LabOrbit& orbit)
                {
                    findButton (editor, "VAE Lab")->onClick();
                    auto* button = findButton (lab, "Orbit LFO");
                    expect (button != nullptr);
                    if (button == nullptr)
                        return;
                    const auto start = [&]
                    {
                        if (! lab.isOrbiting())
                            button->onClick();
                        pumpUntil ([] { return false; }, 200);
                        return lab.isOrbiting();
                    };
                    expect (start());
                    moveHandle (orbit, 0.8f, 0.3f);
                    expect (lab.isOrbiting(), "a drag keeps the orbit");
                    expectWithinAbsoluteError (lab.orbitCentre()[0], 0.8f, 1e-5f, "re-centred X");
                    expectWithinAbsoluteError (lab.orbitCentre()[1], 0.3f, 1e-5f, "re-centred Y");
                    pumpUntil ([] { return false; }, 600);
                    const auto at = orbit.getHandle();
                    expect (at != lab.orbitCentre() && std::hypot (at[0] - 0.8f, at[1] - 0.3f) < 0.1f,
                            "it circles the new centre");

                    findButton (lab, "Reset")->onClick();
                    expect (! lab.isOrbiting(), "Reset stops it");
                    expect (start());
                    juce::ComboBox* category = nullptr;
                    for (auto* c : findAll<juce::ComboBox> (lab))
                        if (c->getItemText (0) == juce::String (experimental::jointLabels[0]))
                            category = c;
                    expect (category != nullptr);
                    if (category)
                        category->setSelectedId (category->getSelectedId() % 9 + 1, juce::sendNotificationSync);
                    expect (! lab.isOrbiting(), "a category change stops it");
                    expect (start());
                    findButton (lab, "Knob pad")->onClick();
                    expect (! lab.isOrbiting(), "the knob pad stops it");
                    expect (! button->isEnabled(), "and has no orbit");
                    findButton (lab, "Latent pad")->onClick();
                    expect (start());
                    lab.markEdited(); // a Synth or host edit, as the editor reports it
                    expect (! lab.isOrbiting(), "a non-Lab edit stops it rather than overwriting");
                    button->onClick();
                    expect (lab.isAskingToReplace() && ! lab.isOrbiting(), "starting while edited asks first");
                    lab.answerReplace (true);
                    expect (lab.isOrbiting(), "Replace starts it");
                });

            beginTest (
                "every orbit stop returns to the centre: Category conditions on it, the knob pad enters from it");
            withLab (
                [&] (juce::AudioProcessorEditor& editor, PluginProcessor&, experimental::JointVaePanel& lab,
                     experimental::LabOrbit& orbit)
                {
                    findButton (editor, "VAE Lab")->onClick();
                    auto* button = findButton (lab, "Orbit LFO");
                    juce::ComboBox* category = nullptr;
                    for (auto* c : findAll<juce::ComboBox> (lab))
                        if (c->getItemText (0) == juce::String (experimental::jointLabels[0]))
                            category = c;
                    expect (button != nullptr && category != nullptr);
                    if (button == nullptr || category == nullptr)
                        return;
                    const auto z0 = lab.latent();
                    const auto centre = orbit.getHandle();
                    std::vector<float> centreCodes;
                    auto play = lab.onLite;
                    lab.onLite = [&] (const schema::Snapshot& s)
                    {
                        if (centreCodes.empty())
                            centreCodes = lab.output().codes;
                        if (play)
                            play (s);
                    };
                    button->onClick();
                    pumpUntil ([] { return false; }, 900);
                    expect (! (lab.latent() == z0), "orbiting away from the centre");
                    category->setSelectedId (category->getSelectedId() % 9 + 1, juce::sendNotificationSync);
                    expect (! lab.isOrbiting());
                    expect (lab.latent() == z0 && orbit.getHandle() == centre,
                            "a category change starts from the centre");
                    expect (pumpUntil ([&] { return lab.output().latent == z0 && lab.output().codes != centreCodes; },
                                       30000),
                            "the new category decodes the centre's z");

                    category->setSelectedId (category->getSelectedId() == 1 ? 9 : category->getSelectedId() - 1,
                                             juce::sendNotificationSync); // back to the start's category
                    centreCodes.clear();
                    button->onClick();
                    pumpUntil ([] { return false; }, 900);
                    expect (! centreCodes.empty() && ! (lab.latent() == z0));
                    findButton (lab, "Knob pad")->onClick();
                    expect (! lab.isOrbiting() && lab.knobGroups() != nullptr);
                    expect (lab.latent() == z0, "the knob pad leaves the latent at the centre");
                    expect (lab.output().codes == centreCodes, "the knob pad enters from the centre's output");
                });

            beginTest ("Send to Surge during an orbit stops it and sends the centre");
            withLab (
                [&] (juce::AudioProcessorEditor& editor, PluginProcessor& processor, experimental::JointVaePanel& lab,
                     experimental::LabOrbit&)
                {
                    findButton (editor, "VAE Lab")->onClick();
                    auto* button = findButton (lab, "Orbit LFO");
                    auto* send = findButton (lab, "Send to Surge");
                    expect (button != nullptr && send != nullptr);
                    if (button == nullptr || send == nullptr)
                        return;
                    int lites = 0;
                    juce::MemoryBlock centrePatch;
                    auto play = lab.onLite;
                    lab.onLite = [&] (const schema::Snapshot& s)
                    {
                        if (lites++ == 0)
                            centrePatch = lab.output().patch;
                        if (play)
                            play (s);
                    };
                    std::vector<juce::MemoryBlock> sent;
                    auto toSurge = lab.onPatch;
                    lab.onPatch = [&] (const juce::MemoryBlock& patch)
                    {
                        sent.push_back (patch);
                        return toSurge (patch);
                    };
                    button->onClick();
                    pumpUntil ([] { return false; }, 600);
                    expect (lab.isOrbiting() && lites > 3);
                    send->onClick();
                    expect (! lab.isOrbiting(), "Send stops the orbit");
                    expect (pumpUntil ([&] { return ! sent.empty(); }, 30000), "then sends");
                    const int after = lites;
                    pumpUntil ([] { return false; }, 300);
                    expectEquals (lites, after, "no Lite write follows");
                    auto* surge = dynamic_cast<SurgeEngine*> (processor.findEngine (EngineKind::Surge));
                    expect (surge != nullptr && surge->hasPatchData(), "Surge keeps playing");
                    expect (processor.getParams().get (PluginProcessor::kEngineParamId) != 0.0f, "not back on Lite");
                    expect (sent.size() == 1 && sent[0] == centrePatch, "the centre's patch");
                    expect (lab.statusText().contains ("on Surge"), lab.statusText());
                });

            beginTest ("a status written while the Replace question is open shows after the answer");
            withLab (
                [&] (juce::AudioProcessorEditor& editor, PluginProcessor&, experimental::JointVaePanel& lab,
                     experimental::LabOrbit& orbit)
                {
                    const auto before = lab.output().codes;
                    moveHandle (orbit, 0.3f, 0.3f); // the user acts, so the word list only nudges
                    expect (pumpUntil ([&] { return lab.output().codes != before; }, 30000));
                    for (const bool replace : { false, true })
                    {
                        if (auto* knob = synthKnob (editor, firstContinuous()))
                            knob->setValue (knob->getValue() > 0.5 ? 0.1 : 0.9, juce::sendNotificationSync);
                        expect (pumpUntil ([&] { return lab.isEdited(); }, 1000));
                        moveHandle (orbit, 0.6f, 0.4f);
                        expect (lab.isAskingToReplace());
                        lab.nudge ("qwertyuiop"); // no words it knows: a status line only, no edit
                        expect (lab.statusText().contains ("Replace your Synth edits?"), "the question stays");
                        lab.answerReplace (replace);
                        expect (lab.statusText().startsWith ("Word list"), lab.statusText());
                    }
                });

            beginTest ("a reopened editor keeps the lab gate and the Surge patch state");
            {
                PluginProcessor processor;
                settleModel (processor);
                processor.prepareToPlay (48000.0, 512);
                const auto index = processor.getParams().indexOf (schema::control (firstContinuous()).id);
                const auto labOf = [] (juce::AudioProcessorEditor& e)
                { return findChild<experimental::JointVaePanel> (e); };
                if (auto editor = openLab (processor))
                {
                    auto& lab = *labOf (*editor);
                    const auto before = lab.output().codes;
                    moveHandle (*findChild<experimental::LabOrbit> (lab), 0.85f, 0.2f);
                    expect (pumpUntil (
                        [&] { return lab.output().codes != before && same (processor.getControls(), liteOf (lab)); },
                        30000));
                }
                // Reopened with the sound as the Lab left it: not edited, a move plays
                if (auto editor = openLab (processor))
                    expect (! labOf (*editor)->isEdited(), "the Lab's own sound is not an edit");
                // A Synth or host edit while the editor is closed: the first pad move asks
                processor.getParams().set (index, processor.getParams().get (index) > 0.5f ? 0.1f : 0.9f);
                const auto edited = processor.getControls();
                if (auto editor = openLab (processor))
                {
                    auto& lab = *labOf (*editor);
                    expect (lab.isEdited(), "an edit made while closed");
                    moveHandle (*findChild<experimental::LabOrbit> (lab), 0.3f, 0.6f);
                    expect (lab.isAskingToReplace(), "the first pad move asks");
                    lab.answerReplace (false);
                    pumpUntil ([] { return false; }, 500);
                    expect (same (processor.getControls(), edited), "the edit is kept");
                    // Send to Surge, then close and reopen: the banner shows and a knob returns to Lite
                    const auto shown = lab.output().codes;
                    moveHandle (*findChild<experimental::LabOrbit> (lab), 0.7f, 0.6f);
                    lab.answerReplace (true);
                    expect (pumpUntil (
                                [&]
                                { return lab.output().codes != shown && same (processor.getControls(), liteOf (lab)); },
                                30000),
                            "Replace plays");
                    auto* send = findButton (lab, "Send to Surge");
                    expect (send != nullptr && send->isEnabled());
                    if (send)
                        send->onClick();
                }
                auto* surge = dynamic_cast<SurgeEngine*> (processor.findEngine (EngineKind::Surge));
                expect (surge != nullptr && surge->hasPatchData());
                if (auto editor = openLab (processor))
                {
                    expect (surge != nullptr && surge->hasPatchData(), "opening the editor keeps the patch (R14)");
                    findButton (*editor, "Synth")->onClick();
                    auto* status = headerStatus (*editor);
                    expect (status != nullptr
                                && status->getText()
                                       == "Playing the full Surge patch from the Lab. Knobs return to Lite",
                            status ? status->getText() : juce::String());
                    if (auto* knob = synthKnob (*editor, firstContinuous()))
                        knob->setValue (knob->getValue() > 0.5 ? 0.1 : 0.9, juce::sendNotificationSync);
                    pumpUntil ([] { return false; }, 100);
                    expectEquals (processor.getParams().get (PluginProcessor::kEngineParamId), 0.0f);
                    expect (surge != nullptr && ! surge->hasPatchData(), "the knob returns to Lite");
                    expect (status != nullptr && status->getText().isEmpty());
                }
                processor.releaseResources();

                // No Lab write yet and the controls are not the defaults: the first Lab edit asks
                PluginProcessor other;
                settleModel (other);
                other.getParams().set (index, other.getParams().get (index) > 0.5f ? 0.1f : 0.9f);
                if (auto editor = openLab (other))
                    expect (labOf (*editor)->isEdited(), "a non-default sound counts as edited");
                PluginProcessor fresh;
                settleModel (fresh);
                if (auto editor = openLab (fresh))
                    expect (! labOf (*editor)->isEdited(), "the default sound does not");
            }
        }

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
                expect (pumpUntil ([&] { return lab->hasOutput(); }, 60000), "bundle loaded");
                expect (! ready, "the latent view requests no map (Orbit only)");
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
                lab->loadBundle (juce::File (bundle));
                expect (pumpUntil ([&] { return lab->hasOutput(); }, 60000), "bundle loaded");
                auto* xy = lab->padData();
                auto* brightness = xy ? xy->find ("s:brightness") : nullptr;
                expect (brightness != nullptr);
                if (brightness)
                {
                    const auto typical = axisPosition (*brightness, xy->typical.at ("Bass"));
                    // "zorp" has nothing usable: no change, no sound
                    const auto z0 = lab->latent();
                    lab->nudge ("zorp");
                    pumpUntil ([] { return false; }, 300);
                    expect (lab->latent() == z0);
                    expect (lab->statusText().contains ("Nothing to change"), lab->statusText());
                    expectEquals (played, 0);
                    lab->nudge ("dark bass");
                    expectEquals (category->getText(), juce::String ("Bass"));
                    expect (lab->statusText().contains ("Bass"), lab->statusText());
                    expectWithinAbsoluteError (axisPosition (*brightness, lab->latent()), typical - 0.1f, 1e-3f);
                    lab->nudge ("brighter");
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
                if (lab)
                    lab->loadBundle (juce::File (bundle));
                const bool ready = lab != nullptr && pumpUntil ([&] { return lab->hasOutput(); }, 60000);
                expect (ready, "bundle loaded");
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
                    lab.nudge ("something dark");
                    expect (unknown (lab), "category Unknown");
                    expect (lab.statusText().contains ("no sound type found"), lab.statusText());
                    darkPos = bright (lab);
                });
            run (
                [&] (experimental::JointVaePanel& lab)
                {
                    lab.nudge ("brighter");
                    expect (unknown (lab), "category Unknown");
                    expect (lab.statusText().contains ("no sound type found"), lab.statusText());
                    brightPos = bright (lab);
                });
            expectWithinAbsoluteError (brightPos - darkPos, 0.2f, 1e-3f);
            run (
                [&] (experimental::JointVaePanel& lab)
                {
                    if (auto* orbit = findChild<experimental::LabOrbit> (lab))
                        orbit->keyPressed (juce::KeyPress (juce::KeyPress::upKey));
                    movedBefore = bright (lab);
                    lab.nudge ("brighter");
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
            auto* knob = lab ? findButton (*lab, "Knob pad") : nullptr;
            expect (knob != nullptr);
            if (knob)
            {
                int shown = 0;
                lab->onMapShown = [&shown] { ++shown; };
                lab->loadBundle (juce::File (bundle));
                expect (pumpUntil ([&] { return lab->hasOutput(); }, 60000), "the bundle loads");
                // Orbit requests no maps; the knob pad's category map is quick, so it may already be done here.
                // The worker's cancel-on-close is unchanged (shortcut: no slow map left to close in the middle of)
                knob->onClick();
                expect (pumpUntil ([&] { return lab->isMapRunning() || shown > 0; }, 5000), "a knob map started");
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
        runPromptTests();
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
            gestures.changes = 0;
            p.playLiteControls (lite); // the same point again: nothing to tell the host
            expectEquals (gestures.changes, 0);
            lite[schema::kFilter1Cutoff] = 0.3f; // a pad move while the note is held
            p.playLiteControls (lite);
            expectEquals (gestures.changes, 1);
            expectGreaterThan (rmsOf50ms(), 0.0);
            p.removeListener (&gestures);
            expectEquals (gestures.begins, 0);
            expectEquals (gestures.ends, 0);
            expectWithinAbsoluteError (p.getControls()[schema::kFilter1Cutoff], 0.3f, 1e-6f);
            p.releaseResources();
        }
#endif

#if VAE_WITH_SURGE
        if (const auto bundle = juce::SystemStats::getEnvironmentVariable ("VAE_JOINT_BUNDLE", ""); bundle.isNotEmpty())
        {
            beginTest ("Lab drags play without clicks (Task 18)");
            experimental::JointVae lab;
            expect (lab.load (juce::File (bundle)).wasOk());
            for (auto drag : { Drag::pad, Drag::padLocked, Drag::slider })
            {
                const auto results = labDrag (lab, drag);
                expectEquals ((int) results.size(), 120);
                if (results.size() != 120)
                    continue;
                const juce::String name = drag == Drag::pad         ? "pad"
                                          : drag == Drag::padLocked ? "pad, Lock on"
                                                                    : "slider";
                const auto moving = playDrag (results);
                const auto still = playDrag (std::vector<schema::Snapshot> (120, results[0]));
                logMessage (name + ": " + moving.describe());
                logMessage (name + " (still): " + still.describe());
                // The loudest of the path's start, middle and end held still: the steps the sound has anyway.
                // Before the ramps a drag reached 3.8x (pad), 3.5x (Lock on), 1.23x (slider); after, 1.02x at most
                float stillStep = still.maxStep;
                for (size_t k : { 60, 119 })
                    stillStep =
                        std::max (stillStep, playDrag (std::vector<schema::Snapshot> (120, results[k])).maxStep);
                logMessage (name + ": max step " + juce::String (moving.maxStep / stillStep, 2) + "x the still path's");
                expect (moving.maxStep < 1.15f * stillStep, name + ": " + moving.describe());
                // Before the fix a moving delay time stepped every block edge (pad 3.4x, slider 1.5x the other steps)
                expect (moving.edgeRatio < 1.3, name + ": " + moving.describe());
                expectEquals (still.ducks, 0); // 93 values written again: no duck for an unchanged one
                expect (moving.ducks <= 2 * moving.discretePerSecond, name + ": " + moving.describe());
                expect (moving.duckedSamples < kRate / 10, name + ": " + moving.describe()); // < 5% of 2 s
                expect (moving.maxMs < 512 * 1000 / kRate, name + ": " + moving.describe()); // the block budget
            }
        }
#endif

#if VAE_WITH_SURGE
        if (const auto bundle = juce::SystemStats::getEnvironmentVariable ("VAE_JOINT_BUNDLE", ""); bundle.isNotEmpty())
        {
            beginTest ("the orbit LFO plays without ducks or clicks (Task 19)");
            experimental::JointVae lab;
            expect (lab.load (juce::File (bundle)).wasOk());
            experimental::OrbitSettings slow, fast;
            fast.rateHz = 2.0f; // the fastest and widest the controls allow
            fast.depth = 0.25f;
            for (const auto& settings : { slow, fast })
            {
                const auto results = labOrbit (lab, settings);
                expectEquals ((int) results.size(), 60);
                if (results.size() != 60)
                    continue;
                const auto name = juce::String (settings.rateHz, 2) + " Hz, depth " + juce::String (settings.depth, 2);
                const auto moving = playDrag (results, 30);
                float stillStep = 0;
                for (size_t k : { 0, 15, 30, 45, 59 })
                    stillStep =
                        std::max (stillStep, playDrag (std::vector<schema::Snapshot> (60, results[k]), 30).maxStep);
                logMessage ("orbit " + name + ": " + moving.describe());
                logMessage ("orbit " + name + ": max step " + juce::String (moving.maxStep / stillStep, 2)
                            + "x the still points'");
                expectEquals (moving.discretePerSecond, 0.0, name + ": every switch held");
                expectEquals (moving.ducks, 0, name + ": " + moving.describe());
                expect (moving.maxStep < 1.15f * stillStep, name + ": " + moving.describe());
                expect (moving.maxMs < 512 * 1000 / kRate, name + ": " + moving.describe());
            }
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
