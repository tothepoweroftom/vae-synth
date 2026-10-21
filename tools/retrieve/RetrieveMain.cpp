// Offline reference audio -> nearest playable controls. No model or training dependency.
#include <juce_audio_formats/juce_audio_formats.h>
#include "engine/SynthEngine.h"
#include "schema/PresetFile.h"
#include "../compare/Features.h"
#include <iostream>
#include <set>
#include <stdexcept>

namespace
{
constexpr double rate = 48000;
constexpr int samples = 4 * 48000, block = 256, descriptorSize = 40 + 24;
using Descriptor = std::array<double, descriptorSize>;
void require (bool condition, const juce::String& message)
{
    if (! condition)
        throw std::runtime_error (message.toStdString());
}
juce::File file (const juce::String& path) { return juce::File::getCurrentWorkingDirectory().getChildFile (path); }
void save (const juce::File& path, const juce::var& data)
{
    require (path.replaceWithText (juce::JSON::toString (data)), "Cannot write " + path.getFullPathName());
}
std::vector<float> render (const vae::schema::Snapshot& controls, int note)
{
    auto engine = vae::createEngine (vae::EngineKind::Lite);
    engine->setControls (controls);
    engine->prepare (rate, block);
    std::vector<float> out (samples);
    juce::AudioBuffer<float> buffer (2, block);
    juce::MidiBuffer midi;
    midi.ensureSize (128);
    for (int pos = 0; pos < samples; pos += block)
    {
        const int n = std::min (block, samples - pos);
        buffer.setSize (2, n, false, false, true);
        midi.clear();
        if (pos == 0)
            midi.addEvent (juce::MidiMessage::noteOn (1, note, (juce::uint8) 100), 0);
        constexpr int off = 72000;
        if (pos <= off && off < pos + n)
            midi.addEvent (juce::MidiMessage::noteOff (1, note), off - pos);
        engine->process (buffer, midi);
        for (int i = 0; i < n; ++i)
            out[(size_t) (pos + i)] = 0.5f * (buffer.getSample (0, i) + buffer.getSample (1, i));
    }
    return out;
}
std::vector<float> readAudio (const juce::File& path)
{
    juce::AudioFormatManager formats;
    formats.registerBasicFormats();
    std::unique_ptr<juce::AudioFormatReader> reader (formats.createReaderFor (path));
    require (reader != nullptr, "Cannot read audio");
    require (reader->sampleRate >= 8000 && reader->sampleRate <= 192000 && reader->lengthInSamples > 0
                 && reader->lengthInSamples <= reader->sampleRate * 30 && reader->numChannels <= 2,
             "Use mono/stereo audio, 8–192 kHz, at most 30 seconds");
    const int n = int (reader->lengthInSamples);
    juce::AudioBuffer<float> input (2, n);
    require (reader->read (&input, 0, n, 0, true, true), "Audio read failed");
    // Offline windowed-sinc sample-rate conversion; all feature extraction uses 48 kHz.
    std::vector<float> mono ((size_t) n + 256, 0.0f);
    for (int i = 0; i < n; ++i)
    {
        const auto v = reader->numChannels == 1 ? input.getSample (0, i)
                                                : 0.5f * (input.getSample (0, i) + input.getSample (1, i));
        require (std::isfinite (v), "Audio contains non-finite samples");
        mono[(size_t) i] = v;
    }
    std::vector<float> out ((size_t) std::ceil (n * rate / reader->sampleRate));
    juce::WindowedSincInterpolator resampler;
    resampler.process (reader->sampleRate / rate, mono.data(), out.data(), int (out.size()));
    return out;
}
Descriptor describe (std::vector<float> audio)
{
    float peak = 0;
    for (float v : audio)
    {
        require (std::isfinite (v), "Non-finite render");
        peak = std::max (peak, std::abs (v));
    }
    require (peak > 1.0e-5f, "Silent audio");
    // Align onset, cap to four seconds, and normalise gain for both bank and query.
    const auto first =
        std::find_if (audio.begin(), audio.end(), [peak] (float v) { return std::abs (v) > peak * 0.01f; });
    audio.erase (audio.begin(), first);
    audio.resize (samples, 0);
    for (auto& v : audio)
        v /= peak;
    const auto f = vae::compare::analyse (audio, rate, {});
    require (! f.silent, "Silent audio");
    Descriptor d {};
    for (size_t i = 0; i < 40; ++i)
        d[i] = f.meanMel[i] / 20.0;
    for (size_t i = 0; i < 24; ++i)
    {
        const auto begin = i * f.envelope.size() / 24, end = (i + 1) * f.envelope.size() / 24;
        for (auto j = begin; j < end; ++j)
            d[40 + i] += f.envelope[j] / double (end - begin);
    }
    return d;
}
juce::var descriptorJson (const Descriptor& d)
{
    juce::Array<juce::var> a;
    for (auto v : d)
        a.add (v);
    return a;
}
Descriptor readDescriptor (const juce::var& value)
{
    const auto* a = value.getArray();
    require (a != nullptr && a->size() == descriptorSize, "Invalid bank descriptor");
    Descriptor d {};
    for (int i = 0; i < descriptorSize; ++i)
    {
        require ((*a)[i].isDouble() || (*a)[i].isInt(), "Invalid descriptor value");
        d[(size_t) i] = double ((*a)[i]);
        require (std::isfinite (d[(size_t) i]), "Non-finite descriptor");
    }
    return d;
}
double distance (const Descriptor& a, const Descriptor& b)
{
    double spectrum = 0, envelope = 0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        const auto delta = a[i] - b[i];
        (i < 40 ? spectrum : envelope) += delta * delta;
    }
    return std::sqrt (spectrum / 40) + 0.5 * std::sqrt (envelope / 24);
}
void wav (const juce::File& path, const std::vector<float>& audio)
{
    require (! path.exists(), "Output already exists: " + path.getFullPathName());
    std::unique_ptr<juce::OutputStream> stream = std::make_unique<juce::FileOutputStream> (path);
    juce::WavAudioFormat format;
    auto writer = format.createWriterFor (
        stream, juce::AudioFormatWriterOptions().withSampleRate (rate).withNumChannels (1).withBitsPerSample (24));
    require (writer != nullptr, "Cannot create WAV");
    const float* data[] = { audio.data() };
    require (writer->writeFromFloatArrays (data, 1, int (audio.size())), "Cannot write WAV");
}
void build (const juce::File& source, const juce::File& output)
{
    require (source.existsAsFile() && ! output.exists(), "Need an existing JSONL input and a new bank path");
    juce::StringArray lines;
    lines.addLines (source.loadFileAsString());
    juce::Array<juce::var> entries;
    std::set<std::array<float, vae::schema::kControlCount>> seen;
    int invalid = 0, silent = 0, duplicates = 0;
    for (const auto& line : lines)
    {
        if (line.trim().isEmpty())
            continue;
        const auto json = juce::JSON::parse (line);
        vae::schema::Snapshot controls;
        if (vae::schema::readPresetValues (json, controls).failed())
        {
            ++invalid;
            continue;
        }
        if (! seen.insert (controls.values).second)
        {
            ++duplicates;
            continue;
        }
        for (int note : { 48, 60, 72 })
        {
            try
            {
                const auto d = describe (render (controls, note));
                auto entry = vae::schema::presetJson (
                    controls, file (json["file"].toString()).getFileNameWithoutExtension(), "lite");
                entry.getDynamicObject()->setProperty ("source", json["file"]);
                entry.getDynamicObject()->setProperty ("note", note);
                entry.getDynamicObject()->setProperty ("descriptor", descriptorJson (d));
                entries.add (entry);
            }
            catch (const std::exception&)
            {
                ++silent;
            }
        }
        if (seen.size() % 100 == 0)
            std::cerr << "Rendered " << seen.size() << " unique presets\n";
    }
    require (! entries.isEmpty(), "No usable presets");
    auto* bank = new juce::DynamicObject();
    bank->setProperty ("format", "vae-retrieval-1");
    bank->setProperty ("schema_version", vae::schema::kSchemaVersion);
    bank->setProperty ("engine", "lite");
    bank->setProperty ("entries", entries);
    save (output, juce::var (bank));
    std::cout << entries.size() << " searchable renders; " << seen.size() << " distinct control vectors; " << duplicates
              << " duplicate vectors; " << invalid << " invalid rows; " << silent << " silent/failed renders\n";
}
// Export the exact Lite probe audio for learned encoders, preserving entry order.
void exportAudio (const juce::File& bankFile, const juce::File& output)
{
    const auto bank = juce::JSON::parse (bankFile);
    require (bank["format"].toString() == "vae-retrieval-1"
                 && int (bank["schema_version"]) == vae::schema::kSchemaVersion && bank["engine"].toString() == "lite",
             "Incompatible bank");
    const auto* entries = bank["entries"].getArray();
    require (entries != nullptr && ! entries->isEmpty(), "Empty bank");
    require (! output.exists(), "Choose a new audio directory");
    require (output.createDirectory().wasOk(), "Cannot create audio directory");
    juce::Array<juce::var> manifest;
    for (const auto& entry : *entries)
    {
        vae::schema::Snapshot controls;
        require (vae::schema::readPresetValues (entry, controls).wasOk(), "Invalid bank controls");
        const int note = int (entry["note"]);
        require (note == 48 || note == 60 || note == 72, "Invalid bank note");
        const auto filename = juce::String (manifest.size()).paddedLeft ('0', 5) + ".wav";
        wav (output.getChildFile (filename), render (controls, note));
        auto preset = vae::schema::presetJson (controls, entry["name"].toString(), "lite");
        preset.getDynamicObject()->setProperty ("note", note);
        preset.getDynamicObject()->setProperty ("source", entry["source"]);
        preset.getDynamicObject()->setProperty ("audio", filename);
        manifest.add (preset);
        if (manifest.size() % 100 == 0)
            std::cerr << "Exported " << manifest.size() << " renders\n";
    }
    save (output.getChildFile ("manifest.json"), manifest);
}

void search (const juce::File& bankFile, const juce::File& reference, const juce::File& output)
{
    const auto bank = juce::JSON::parse (bankFile);
    require (bank["format"].toString() == "vae-retrieval-1"
                 && int (bank["schema_version"]) == vae::schema::kSchemaVersion && bank["engine"].toString() == "lite",
             "Incompatible bank");
    const auto* entries = bank["entries"].getArray();
    require (entries != nullptr && ! entries->isEmpty(), "Empty bank");
    const auto query = describe (readAudio (reference));
    struct Match
    {
        double score;
        juce::var entry;
        vae::schema::Snapshot controls;
    };
    std::vector<Match> matches;
    for (const auto& entry : *entries)
    {
        vae::schema::Snapshot controls;
        require (vae::schema::readPresetValues (entry, controls).wasOk(), "Invalid bank controls");
        const int note = int (entry["note"]);
        require (note == 48 || note == 60 || note == 72, "Invalid bank probe note");
        matches.push_back ({ distance (query, readDescriptor (entry["descriptor"])), entry, controls });
    }
    std::stable_sort (matches.begin(), matches.end(), [] (const auto& a, const auto& b) { return a.score < b.score; });
    require (! output.exists(), "Choose a new result directory");
    require (output.createDirectory().wasOk(), "Cannot create result directory");
    std::set<std::array<float, vae::schema::kControlCount>> seen;
    juce::Array<juce::var> results;
    for (auto& match : matches)
    {
        if (! seen.insert (match.controls.values).second)
            continue;
        const auto stem = "match-" + juce::String (results.size() + 1);
        auto preset = vae::schema::presetJson (match.controls, match.entry["name"].toString(), "lite");
        preset.getDynamicObject()->setProperty ("score", match.score);
        preset.getDynamicObject()->setProperty ("source", match.entry["source"]);
        preset.getDynamicObject()->setProperty ("note", match.entry["note"]);
        save (output.getChildFile (stem + ".json"), preset);
        wav (output.getChildFile (stem + ".wav"), render (match.controls, int (match.entry["note"])));
        results.add (preset);
        std::cout << stem << "  " << match.entry["name"].toString() << "  distance=" << match.score << '\n';
        if (results.size() == 5)
            break;
    }
    save (output.getChildFile ("results.json"), results);
}
} // namespace

int main (int argc, char** argv)
{
    try
    {
        if (argc == 4 && juce::String (argv[1]) == "build")
            build (file (argv[2]), file (argv[3]));
        else if (argc == 4 && juce::String (argv[1]) == "export")
            exportAudio (file (argv[2]), file (argv[3]));
        else if (argc == 5 && juce::String (argv[1]) == "search")
            search (file (argv[2]), file (argv[3]), file (argv[4]));
        else
            throw std::runtime_error ("Usage: vae_retrieve build vectors.jsonl bank.json\n       vae_retrieve search "
                                      "bank.json reference.wav new-result-directory\n       vae_retrieve export "
                                      "bank.json new-audio-directory");
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
