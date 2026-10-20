/*
  vae_compare: how different do the Lite and Surge backends sound for the same
  schema vector, and how much does the schema itself lose against the original
  Surge preset?

  For every preset in the importer's JSONL (tools/surge_preset_to_vector.py)
  it renders the same MIDI probe three ways - the original .fxp in headless
  Surge, the schema vector on the Surge backend, the schema vector on the Lite
  backend - and writes one CSV row of audio features and pairwise distances.
  tools/compare_report.py turns the CSV into a report. See docs/COMPARE.md.
*/

#include <juce_audio_formats/juce_audio_formats.h>

#include "Features.h"
#include "SurgeOriginal.h"
#include "engine/SynthEngine.h"

#include <atomic>
#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <thread>

namespace vae::compare
{
namespace
{
constexpr double kRate = 48000.0;
constexpr int kBlock = 256;
constexpr int kVelocity = 100;
const ProbeTimes kProbe; // note 48 at 0..1.5 s, chord 60/64/67 at 2.5..3.5 s
constexpr double kChordOff = 3.5, kEnd = 4.5;

std::vector<ProbeEvent> probeEvents()
{
    auto at = [] (double s) { return int (s * kRate); };
    return { { at (kProbe.note1On), 48, kVelocity },
             { at (kProbe.note1Off), 48, 0 },
             { at (kProbe.chordOn), 60, kVelocity },
             { at (kProbe.chordOn), 64, kVelocity },
             { at (kProbe.chordOn), 67, kVelocity },
             { at (kChordOff), 60, 0 },
             { at (kChordOff), 64, 0 },
             { at (kChordOff), 67, 0 } };
}

const int kNumSamples = int (kEnd * kRate);

/** Mono render of a schema vector on one backend, fresh engine per call. */
std::vector<float> renderVector (EngineKind kind, const schema::Snapshot& controls,
                                 const std::vector<ProbeEvent>& events)
{
    auto engine = createEngine (kind);
    engine->setControls (controls);
    engine->prepare (kRate, kBlock);

    std::vector<float> out ((size_t) kNumSamples);
    juce::AudioBuffer<float> buffer (2, kBlock);
    size_t next = 0;
    for (int pos = 0; pos < kNumSamples; pos += kBlock)
    {
        const int n = std::min (kBlock, kNumSamples - pos);
        buffer.setSize (2, n, false, false, true);
        juce::MidiBuffer midi;
        for (; next < events.size() && events[next].sample < pos + n; ++next)
        {
            const auto& e = events[next];
            midi.addEvent (e.velocity > 0 ? juce::MidiMessage::noteOn (1, e.note, (juce::uint8) e.velocity)
                                          : juce::MidiMessage::noteOff (1, e.note),
                           e.sample - pos);
        }
        engine->process (buffer, midi);
        for (int i = 0; i < n; ++i)
            out[(size_t) (pos + i)] = 0.5f * (buffer.getSample (0, i) + buffer.getSample (1, i));
    }
    return out;
}

/** Preset category from its folder: patches_factory/<Category>/ or
    patches_3rdparty/<Author>/<Category>/, normalised; else "unknown". */
juce::String categoryOf (const juce::String& path)
{
    static const std::map<juce::String, juce::String> canonical = {
        { "basses", "bass" },       { "bass", "bass" },      { "brass", "brass" },
        { "chords", "chord" },      { "chord", "chord" },    { "fx", "fx" },
        { "keys", "keys" },         { "leads", "lead" },     { "lead", "lead" },
        { "pads", "pad" },          { "pad", "pad" },        { "percussion", "perc" },
        { "drums", "perc" },        { "plucks", "pluck" },   { "pluck", "pluck" },
        { "sequences", "seq" },     { "rhythms", "seq" },    { "arps", "seq" },
        { "strings", "strings" },   { "vox", "vox" },        { "voices", "vox" },
        { "vocals", "vox" },        { "winds", "winds" },    { "woodwinds", "winds" },
        { "polysynths", "poly" },   { "drones", "drone" },   { "organs", "organ" },
        { "mallets", "mallet" },    { "bells", "mallet" },   { "atmospheres", "atmos" },
        { "soundscapes", "atmos" }, { "ambiance", "atmos" }, { "ambiances", "atmos" },
        { "textures", "atmos" },
    };
    juce::StringArray parts;
    parts.addTokens (path.replaceCharacter ('\\', '/'), "/", {});
    for (int i = 0; i < parts.size(); ++i)
    {
        juce::String folder;
        if (parts[i] == "patches_factory" && i + 2 < parts.size())
            folder = parts[i + 1];
        else if (parts[i] == "patches_3rdparty" && i + 3 < parts.size())
            folder = parts[i + 2];
        else
            continue;
        const auto it = canonical.find (folder.toLowerCase());
        return it != canonical.end() ? it->second : juce::String ("unknown");
    }
    return "unknown";
}

struct Preset
{
    juce::String file;
    schema::Snapshot controls;
    int lossy = 0;
};

std::optional<Preset> parseLine (const juce::String& line)
{
    const auto json = juce::JSON::parse (line);
    const auto* values = json["values"].getArray();
    if (values == nullptr || values->size() != schema::kControlCount)
        return std::nullopt;
    Preset p;
    p.file = json["file"].toString();
    for (int i = 0; i < schema::kControlCount; ++i)
        p.controls[i] = float (double ((*values)[i]));
    p.controls.sanitise();
    if (const auto* lossy = json["lossy"].getArray())
        p.lossy = lossy->size();
    return p;
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

const char* kRenders[] = { "orig", "svec", "lite" };
const char* kPairs[] = { "os", "sl", "ol" }; // orig-svec (schema loss), svec-lite (engine gap), orig-lite

juce::String header()
{
    juce::StringArray cols { "file", "category", "n_lossy", "orig_ok" };
    for (auto* r : kRenders)
    {
        for (auto* f : { "silent", "level_db", "centroid_hz", "flatness", "attack_s", "release_s" })
            cols.add (juce::String (r) + "_" + f);
        for (int b = 0; b < kMelBands; ++b)
            cols.add (juce::String (r) + "_mel" + juce::String (b));
    }
    for (auto* p : kPairs)
        for (auto* d : { "mel_db", "frame_mel_db", "env_corr", "d_level_db", "d_centroid_oct", "d_flatness" })
            cols.add (juce::String (p) + "_" + d);
    return cols.joinIntoString (",");
}

juce::String featureCells (const std::optional<Features>& f)
{
    juce::StringArray cells;
    if (! f)
    {
        for (int i = 0; i < 6 + kMelBands; ++i)
            cells.add ("");
        return cells.joinIntoString (",");
    }
    cells.add (f->silent ? "1" : "0");
    for (float v : { f->levelDb, f->centroidHz, f->flatness, f->attackS, f->releaseS })
        cells.add (juce::String (v, 4));
    for (float v : f->meanMel)
        cells.add (juce::String (v, 3));
    return cells.joinIntoString (",");
}

juce::String distanceCells (const std::optional<Features>& a, const std::optional<Features>& b)
{
    if (! a || ! b)
        return ",,,,,";
    const auto d = compare (*a, *b);
    juce::StringArray cells;
    for (float v : { d.melDb, d.frameMelDb, d.envCorr, d.levelDb, d.centroidOct, d.flatness })
        cells.add (juce::String (v, 4));
    return cells.joinIntoString (",");
}

juce::String csvQuote (const juce::String& s) { return "\"" + s.replace ("\"", "\"\"") + "\""; }

struct Options
{
    juce::File input, output, wavDir, wavList;
    int jobs = int (std::thread::hardware_concurrency());
    int limit = -1;
};

std::optional<Options> parseArgs (const juce::StringArray& args)
{
    Options o;
    for (int i = 0; i < args.size(); ++i)
    {
        const auto& a = args[i];
        auto value = [&] { return i + 1 < args.size() ? args[++i] : juce::String(); };
        if (a == "-o" || a == "--output")
            o.output = juce::File::getCurrentWorkingDirectory().getChildFile (value());
        else if (a == "--jobs")
            o.jobs = std::max (1, value().getIntValue());
        else if (a == "--limit")
            o.limit = value().getIntValue();
        else if (a == "--wav-dir")
            o.wavDir = juce::File::getCurrentWorkingDirectory().getChildFile (value());
        else if (a == "--wav-list")
            o.wavList = juce::File::getCurrentWorkingDirectory().getChildFile (value());
        else if (! a.startsWith ("-"))
            o.input = juce::File::getCurrentWorkingDirectory().getChildFile (a);
        else
            return std::nullopt;
    }
    if (o.input == juce::File() || o.output == juce::File())
        return std::nullopt;
    return o;
}

int run (const Options& options)
{
    juce::StringArray lines;
    options.input.readLines (lines);
    lines.removeEmptyStrings();
    if (options.limit >= 0 && lines.size() > options.limit)
        lines.removeRange (options.limit, lines.size());

    std::set<juce::String> wavWanted;
    if (options.wavList.existsAsFile())
    {
        juce::StringArray wanted;
        options.wavList.readLines (wanted);
        for (const auto& w : wanted)
            if (w.trim().isNotEmpty())
                wavWanted.insert (w.trim());
        options.wavDir.createDirectory();
    }

    const auto events = probeEvents();
    std::vector<juce::String> rows ((size_t) lines.size());
    std::atomic<int> nextJob { 0 }, done { 0 }, failed { 0 };
    std::mutex logLock;

    auto worker = [&]
    {
        for (int i = nextJob++; i < lines.size(); i = nextJob++)
        {
            try
            {
                const auto preset = parseLine (lines[i]);
                if (! preset)
                    throw std::runtime_error ("unreadable JSONL line");

                std::optional<std::vector<float>> orig =
                    renderOriginal (juce::File (preset->file), kRate, events, kNumSamples);
                const auto svec = renderVector (EngineKind::Surge, preset->controls, events);
                const auto lite = renderVector (EngineKind::Lite, preset->controls, events);

                std::optional<Features> fo, fs = analyse (svec, kRate, kProbe), fl = analyse (lite, kRate, kProbe);
                if (orig)
                    fo = analyse (*orig, kRate, kProbe);

                juce::StringArray row { csvQuote (preset->file), categoryOf (preset->file),
                                        juce::String (preset->lossy), orig ? "1" : "0" };
                row.add (featureCells (fo));
                row.add (featureCells (fs));
                row.add (featureCells (fl));
                row.add (distanceCells (fo, fs));
                row.add (distanceCells (fs, fl));
                row.add (distanceCells (fo, fl));
                rows[(size_t) i] = row.joinIntoString (",");

                if (wavWanted.count (preset->file) > 0)
                {
                    const auto stem =
                        juce::String (i).paddedLeft ('0', 5) + "_"
                        + juce::File (preset->file)
                              .getFileNameWithoutExtension()
                              .retainCharacters ("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-");
                    if (orig)
                        writeWav (options.wavDir.getChildFile (stem + "_orig.wav"), *orig);
                    writeWav (options.wavDir.getChildFile (stem + "_svec.wav"), svec);
                    writeWav (options.wavDir.getChildFile (stem + "_lite.wav"), lite);
                }
            }
            catch (const std::exception& e)
            {
                ++failed;
                const std::scoped_lock sl (logLock);
                std::cerr << "skip line " << i + 1 << ": " << e.what() << std::endl;
            }

            if (const int n = ++done; n % 100 == 0)
            {
                const std::scoped_lock sl (logLock);
                std::cerr << n << " / " << lines.size() << std::endl;
            }
        }
    };

    std::vector<std::thread> pool;
    for (int t = 0; t < options.jobs; ++t)
        pool.emplace_back (worker);
    for (auto& t : pool)
        t.join();

    juce::FileOutputStream out (options.output);
    if (! out.openedOk())
    {
        std::cerr << "cannot write " << options.output.getFullPathName() << std::endl;
        return 1;
    }
    out.setPosition (0);
    out.truncate();
    out << header() << "\n";
    for (const auto& r : rows)
        if (r.isNotEmpty())
            out << r << "\n";

    std::cerr << "compared " << (lines.size() - failed) << " presets, skipped " << failed.load() << std::endl;
    return 0;
}
} // namespace
} // namespace vae::compare

int main (int argc, char** argv)
{
    juce::StringArray args;
    for (int i = 1; i < argc; ++i)
        args.add (argv[i]);

    const auto options = vae::compare::parseArgs (args);
    if (! options)
    {
        std::cerr << "usage: vae_compare vectors.jsonl -o compare.csv [--jobs N] [--limit N]"
                     " [--wav-dir DIR --wav-list FILE]\n";
        return 2;
    }
    return vae::compare::run (*options);
}
