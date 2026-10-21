/*
  vae_p0_eval: two modes for the p0 prompt feature.

    vae_p0_eval eval <outDir> <bundle.json> <prompt.json>
      Renders, per fixed prompt, the p0 Find first candidate, the category-only typical start and the word-list start
      through Lite and Surge to WAV, and writes results.md (rows for docs/P0-PROMPT-EVAL.md).
    vae_p0_eval profile <bundle.json> <prompt.json>
      20 Finds through PromptWorker (odd ones while a Lab drag runs) with a held note rendering on Lite.
    vae_p0_eval mem <editors> <0|1 load p0> <bundle.json> <prompt.json>
      Resident memory with N Lab workers, optionally each with its p0 model loaded.

  Experimental tooling; it makes no judgement of sound quality.
*/

#include <juce_audio_formats/juce_audio_formats.h>

#include "../compare/Features.h"
#include "engine/SynthEngine.h"
#include "engine/surge/SurgeEngine.h"
#include "experiments/joint_vae/JointVae.h"
#include "experiments/joint_vae/LiteTranslation.h"
#include "experiments/joint_vae/PromptModel.h"
#include "experiments/joint_vae/PromptWords.h"

#include <mach/mach.h>
#include <sys/resource.h>

#include <atomic>
#include <iostream>
#include <mutex>

using namespace vae;
using namespace vae::experimental;

namespace
{
constexpr double kRate = 48000.0;
constexpr int kBlock = 256;

const std::vector<juce::String> kPrompts = {
    "dark bass",           "bright bass",      "short pad",        "long pad",    "soft lead",
    "sharp lead",          "deep sub bass",    "lush ambient pad", "plucky bell", "warm electric piano keys",
    "aggressive saw lead", "evolving texture",
};

double residentMb()
{
    mach_task_basic_info info {};
    mach_msg_type_number_t n = MACH_TASK_BASIC_INFO_COUNT;
    task_info (mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t) &info, &n);
    return (double) info.resident_size / 1048576.0;
}
double peakMb()
{
    rusage u {};
    getrusage (RUSAGE_SELF, &u);
    return (double) u.ru_maxrss / 1048576.0; // bytes on macOS
}
void pump (int ms) { juce::MessageManager::getInstance()->runDispatchLoopUntil (ms); }
bool pumpUntil (const std::function<bool()>& done, int timeoutMs)
{
    const auto end = juce::Time::getMillisecondCounter() + (juce::uint32) timeoutMs;
    while (! done() && juce::Time::getMillisecondCounter() < end)
        pump (5);
    return done();
}

std::vector<float> render (SynthEngine& engine)
{
    const int total = int (2.2 * kRate), off = int (1.5 * kRate);
    std::vector<float> out ((size_t) total);
    juce::AudioBuffer<float> buffer (2, kBlock);
    for (int pos = 0; pos < total; pos += kBlock)
    {
        const int n = std::min (kBlock, total - pos);
        buffer.setSize (2, n, false, false, true);
        juce::MidiBuffer midi;
        if (pos == 0)
            midi.addEvent (juce::MidiMessage::noteOn (1, 60, (juce::uint8) 100), 0);
        if (off >= pos && off < pos + n)
            midi.addEvent (juce::MidiMessage::noteOff (1, 60), off - pos);
        engine.process (buffer, midi);
        for (int i = 0; i < n; ++i)
            out[(size_t) (pos + i)] = 0.5f * (buffer.getSample (0, i) + buffer.getSample (1, i));
    }
    return out;
}

void writeWav (const juce::File& file, const std::vector<float>& mono)
{
    file.deleteFile();
    std::unique_ptr<juce::OutputStream> stream = std::make_unique<juce::FileOutputStream> (file);
    juce::WavAudioFormat wav;
    auto options = juce::AudioFormatWriterOptions().withSampleRate (kRate).withNumChannels (1).withBitsPerSample (16);
    if (auto writer = wav.createWriterFor (stream, options))
    {
        const float* channels[] = { mono.data() };
        writer->writeFromFloatArrays (channels, 1, (int) mono.size());
    }
}

juce::String slug (juce::String s) { return s.replaceCharacter (' ', '-'); }

struct Start
{
    juce::String name; // "p0", "typical", "words"
    bool ok = false;
    juce::String note;
    juce::MemoryBlock patch;
    schema::Snapshot lite = schema::Snapshot::defaults();
    int category = -1;
    SoundMeasures meas {};
};

void finish (JointVae& lab, Start& s, const std::vector<float>& codes, const JointLatent& z)
{
    SoundCode sound {};
    if (auto r = lab.decodeSound (z, s.category, sound); r.failed())
        s.note += " (sound decoder: " + r.getErrorMessage() + ")";
    else
        s.meas = measures (lab.xypad(), melFromSound (lab.soundPca(), sound));
    if (auto r = lab.patch (codes, s.patch); r.failed())
    {
        s.note += " patch failed: " + r.getErrorMessage();
        return;
    }
    if (auto r = liteControls (s.patch, s.lite); r.failed())
    {
        s.note += " lite failed: " + r.getErrorMessage();
        return;
    }
    s.ok = true;
}

int runEval (const juce::File& out, const juce::File& bundle, const juce::File& promptJson)
{
    JointVae lab;
    PromptModel p0;
    if (auto r = lab.load (bundle); r.failed() || p0.load (promptJson, lab).failed())
    {
        std::cerr << "load failed: " << r.getErrorMessage() << "\n";
        return 1;
    }
    out.createDirectory();
    const auto& xy = lab.xypad();
    juce::String table;
    juce::String coverage;
    for (const auto& prompt : kPrompts)
    {
        std::vector<Start> starts (3);
        starts[0].name = "p0 Find";
        starts[1].name = "typical";
        starts[2].name = "word list";
        std::vector<PromptCandidate> found;
        int category = -1;
        const auto fr = p0.find (prompt, found, category);
        juce::String rows;
        for (const auto& c : found)
            rows << (rows.isEmpty() ? "" : ", ") << c.row << " " << c.name;
        coverage << "| " << prompt << " | " << (category >= 0 ? juce::String (jointLabels[category]) : "-") << " | "
                 << (fr.wasOk() && ! found.empty() ? juce::String ((int) found.size()) + " rows: " + rows
                                                   : "FALLBACK: " + (fr.failed() ? fr.getErrorMessage() : "no rows"))
                 << " |\n";
        if (fr.wasOk() && ! found.empty())
        {
            starts[0].category = found[0].category;
            starts[0].note = "row " + juce::String (found[0].row) + " " + found[0].name;
            finish (lab, starts[0], found[0].codes, found[0].z);
        }
        else
            starts[0].note = "none: " + (fr.failed() ? fr.getErrorMessage() : juce::String ("no rows"));
        // Category-only typical start for p0's predicted category
        if (category >= 0 && xy.typical.count (jointLabels[category]) != 0)
        {
            const auto z = xy.typical.at (jointLabels[category]);
            starts[1].category = category;
            starts[1].note = juce::String ("typical ") + jointLabels[category];
            std::vector<float> codes;
            if (auto r = lab.decode (z, category, codes); r.wasOk())
                finish (lab, starts[1], codes, z);
        }
        else
            starts[1].note = "no typical start";
        // Word-list start (nudges applied one axis at a time; the panel also holds the pad's other axes)
        {
            const auto reply = interpretPrompt (prompt, false, xy);
            JointLatent z = reply.start ? reply.start->first : xy.typical.at ("Unknown");
            const int cat = reply.start ? reply.start->second : 8;
            for (const auto& [id, delta] : reply.nudges)
                if (auto* axis = xy.find (id))
                {
                    const LatentAxis* held[] = { axis };
                    const float target[] = { axisTarget (*axis,
                                                         juce::jlimit (0.0f, 1.0f, axisPosition (*axis, z) + delta)) };
                    z = constrain (z, held, target);
                }
            starts[2].category = cat;
            starts[2].note = juce::String (jointLabels[cat]) + (reply.start ? "" : " (no category word: Unknown base)")
                             + (reply.nudges.empty() ? "" : ", " + juce::String ((int) reply.nudges.size()) + " nudge");
            std::vector<float> codes;
            if (auto r = lab.decode (z, cat, codes); r.wasOk())
                finish (lab, starts[2], codes, z);
        }
        for (auto& s : starts)
            for (auto kind : { EngineKind::Lite, EngineKind::Surge })
            {
                const juce::String engineName = kind == EngineKind::Lite ? "Lite" : "Surge";
                juce::String path = "-", rms = "-", centroid = "-";
                if (s.ok)
                {
                    auto engine = createEngine (kind);
                    engine->prepare (kRate, kBlock);
                    if (kind == EngineKind::Lite)
                        engine->setControls (s.lite);
                    else
                    {
                        // The patch loads on the next blocks; a note sent before that is lost
                        static_cast<SurgeEngine&> (*engine).loadPatchData (s.patch);
                        juce::AudioBuffer<float> settle (2, kBlock);
                        for (int i = 0; i < 16; ++i)
                            engine->process (settle, {});
                    }
                    const auto audio = render (*engine);
                    const auto file =
                        out.getChildFile (slug (prompt) + "__" + slug (s.name) + "__" + engineName + ".wav");
                    writeWav (file, audio);
                    const auto f = compare::analyse (audio, kRate, {});
                    path = "`" + file.getRelativePathFrom (juce::File::getCurrentWorkingDirectory()) + "`";
                    rms = f.silent ? "silent" : juce::String (f.levelDb, 1) + " dB";
                    centroid = f.silent ? "-" : juce::String ((int) f.centroidHz) + " Hz";
                }
                table << "| " << prompt << " | " << s.name << " | " << engineName << " | " << path << " | " << s.note
                      << " | " << (s.ok ? juce::String (s.meas.brightness, 2) : "-") << " | "
                      << (s.ok ? juce::String (s.meas.length, 2) : "-") << " | " << rms << " | " << centroid
                      << " |  |\n";
            }
    }
    out.getChildFile ("results.md").replaceWithText ("## Renders\n\n" + table + "\n## Coverage\n\n" + coverage);
    std::cout << "wrote " << out.getFullPathName() << "\n";
    return 0;
}

// --- profiling ---------------------------------------------------------------------------------------------
struct Audio : juce::Thread
{
    Audio()
        : juce::Thread ("p0-eval-audio")
    {
    }
    void run() override
    {
        auto engine = createEngine (EngineKind::Lite);
        engine->prepare (kRate, kBlock);
        engine->setControls (schema::Snapshot::defaults());
        juce::AudioBuffer<float> buffer (2, kBlock);
        const double period = 1000.0 * kBlock / kRate;
        double next = juce::Time::getMillisecondCounterHiRes();
        bool first = true;
        while (! threadShouldExit())
        {
            {
                const std::unique_lock lock (mutex, std::try_to_lock);
                if (lock && pending)
                {
                    engine->setControls (*pending);
                    pending.reset();
                }
            }
            juce::MidiBuffer midi;
            if (first)
                midi.addEvent (juce::MidiMessage::noteOn (1, 48, (juce::uint8) 100), 0);
            first = false;
            const auto t0 = juce::Time::getMillisecondCounterHiRes();
            engine->process (buffer, midi);
            const auto ms = juce::Time::getMillisecondCounterHiRes() - t0;
            sum += ms;
            ++blocks;
            worst = std::max (worst.load(), ms);
            if (ms > period)
                ++over;
            next += period;
            while (juce::Time::getMillisecondCounterHiRes() < next && ! threadShouldExit())
                juce::Thread::sleep (1);
            // a render thread that fell behind is an underrun in a real host; resync instead of bursting
            if (juce::Time::getMillisecondCounterHiRes() > next + 4 * period)
                next = juce::Time::getMillisecondCounterHiRes();
        }
    }
    std::mutex mutex;
    std::optional<schema::Snapshot> pending;
    std::atomic<double> worst { 0 };
    double sum = 0;
    std::atomic<int> blocks { 0 }, over { 0 };
};

int runProfile (const juce::File& bundle, const juce::File& promptJson)
{
    JointVaeWorker jw;
    JointLayout layout;
    bool loaded = false;
    jw.load (bundle,
             [&] (juce::Result r, JointVaeWorker::Output o)
             {
                 if (r.wasOk())
                     layout = o.layout;
                 loaded = true;
             });
    pumpUntil ([&] { return loaded; }, 60000);
    std::cout << "RSS after Lab bundle: " << residentMb() << " MB\n";
    JointVae probe;
    probe.load (bundle);
    const auto typical = probe.xypad().typical.at ("Bass");

    Audio audio;
    audio.startThread();
    pump (500);

    PromptWorker pw;
    std::vector<double> cold, warmIdle, warmDrag;
    double loadMs = 0;
    int dragSent = 0, dragDone = 0;
    const auto drag = [&]
    {
        LabRequest req;
        req.z = typical;
        req.z[0] += 0.8f * std::sin (0.07f * (float) dragSent);
        req.z[1] += 0.8f * std::cos (0.05f * (float) dragSent);
        req.category = 0;
        ++dragSent;
        jw.lab (req,
                [&] (juce::Result r, JointVaeWorker::Output o)
                {
                    ++dragDone;
                    if (r.wasOk() && o.patchResult.wasOk())
                    {
                        const std::lock_guard lock (audio.mutex);
                        audio.pending = o.lite;
                    }
                });
    };
    for (int i = 0; i < 20; ++i)
    {
        const bool dragging = i % 2 == 1;
        const auto text = kPrompts[(size_t) i % kPrompts.size()];
        bool done = false;
        const auto t0 = juce::Time::getMillisecondCounterHiRes();
        if (i == 0)
        {
            bool ok = false, ready = false;
            pw.load (promptJson, layout,
                     [&] (juce::Result r)
                     {
                         ok = r.wasOk();
                         loadMs = juce::Time::getMillisecondCounterHiRes() - t0;
                         ready = true;
                     });
            pumpUntil ([&] { return ready; }, 120000);
            std::cout << "p0 load " << loadMs << " ms ok=" << ok << " RSS " << residentMb() << " MB\n";
            // the cold Find includes the load, as the first Find in the editor does
        }
        auto t1 = i == 0 ? t0 : juce::Time::getMillisecondCounterHiRes();
        pw.find (text, [&] (juce::Result, int, std::vector<PromptCandidate>) { done = true; });
        auto nextDrag = juce::Time::getMillisecondCounterHiRes();
        while (! done)
        {
            if (dragging && juce::Time::getMillisecondCounterHiRes() >= nextDrag)
            {
                drag();
                nextDrag += 16.0;
            }
            pump (2);
        }
        const auto ms = juce::Time::getMillisecondCounterHiRes() - t1;
        (i == 0 ? cold : dragging ? warmDrag : warmIdle).push_back (ms);
        // a drag goes on for a second around each odd Find
        for (auto end = juce::Time::getMillisecondCounterHiRes() + (dragging ? 800.0 : 200.0);
             juce::Time::getMillisecondCounterHiRes() < end;)
        {
            if (dragging)
                drag();
            pump (16);
        }
    }
    audio.stopThread (2000);
    const auto stats = [] (std::vector<double> v)
    {
        std::sort (v.begin(), v.end());
        double sum = 0;
        for (auto x : v)
            sum += x;
        return juce::String (v.empty() ? 0 : sum / (double) v.size(), 0) + " mean / "
               + juce::String (v.empty() ? 0 : v[v.size() / 2], 0) + " median / "
               + juce::String (v.empty() ? 0 : v.back(), 0) + " max ms over " + juce::String ((int) v.size());
    };
    std::cout << "cold Find (load + first find): " << stats (cold) << "\n";
    std::cout << "warm Find, idle: " << stats (warmIdle) << "\n";
    std::cout << "warm Find, during Lab drag: " << stats (warmDrag) << " (" << dragDone << "/" << dragSent
              << " drag decodes)\n";
    std::cout << "process(): " << audio.blocks << " blocks of " << kBlock << " @ " << kRate << ", mean "
              << audio.sum / std::max (1, audio.blocks.load()) << " ms, max " << audio.worst.load() << " ms, budget "
              << 1000.0 * kBlock / kRate << " ms, over budget " << audio.over << "\n";
    std::cout << "RSS now " << residentMb() << " MB, peak " << peakMb() << " MB\n";
    return 0;
}

int runMem (int editors, bool withP0, const juce::File& bundle, const juce::File& promptJson)
{
    std::vector<std::unique_ptr<JointVaeWorker>> labs;
    std::vector<std::unique_ptr<PromptWorker>> prompts;
    std::cout << "RSS start " << residentMb() << " MB\n";
    for (int i = 0; i < editors; ++i)
    {
        labs.push_back (std::make_unique<JointVaeWorker>());
        JointLayout layout;
        bool done = false;
        labs.back()->load (bundle,
                           [&] (juce::Result, JointVaeWorker::Output o)
                           {
                               layout = o.layout;
                               done = true;
                           });
        pumpUntil ([&] { return done; }, 60000);
        if (withP0)
        {
            prompts.push_back (std::make_unique<PromptWorker>());
            bool ready = false;
            prompts.back()->load (promptJson, layout, [&] (juce::Result) { ready = true; });
            pumpUntil ([&] { return ready; }, 120000);
        }
        std::cout << "after editor " << i + 1 << ": RSS " << residentMb() << " MB\n";
    }
    pump (300);
    std::cout << editors << " editor(s), p0 " << (withP0 ? "loaded" : "not loaded") << ": RSS " << residentMb()
              << " MB, peak " << peakMb() << " MB\n";
    return 0;
}
} // namespace

int main (int argc, char** argv)
{
    juce::ScopedJuceInitialiser_GUI gui;
    juce::StringArray a;
    for (int i = 1; i < argc; ++i)
        a.add (argv[i]);
    if (a.size() == 4 && a[0] == "eval")
        return runEval (juce::File::getCurrentWorkingDirectory().getChildFile (a[1]), juce::File (a[2]),
                        juce::File (a[3]));
    if (a.size() == 3 && a[0] == "profile")
        return runProfile (juce::File (a[1]), juce::File (a[2]));
    if (a.size() == 5 && a[0] == "mem")
        return runMem (a[1].getIntValue(), a[2] == "1", juce::File (a[3]), juce::File (a[4]));
    std::cerr << "usage: see the header of tools/p0_eval/P0EvalMain.cpp\n";
    return 2;
}
