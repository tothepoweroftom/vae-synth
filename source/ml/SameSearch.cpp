#include "SameSearch.h"
#include "schema/PresetFile.h"
#include <juce_cryptography/juce_cryptography.h>
#include <numeric>
#include <set>

namespace vae::ml
{
juce::Result SameSearch::preprocess (const juce::File& file, std::vector<float>& out)
{
    juce::AudioFormatManager formats;
    formats.registerBasicFormats();
    std::unique_ptr<juce::AudioFormatReader> reader (formats.createReaderFor (file));
    if (! reader || reader->sampleRate < 8000 || reader->sampleRate > 192000
        || reader->sampleRate != std::floor (reader->sampleRate) || reader->numChannels < 1 || reader->numChannels > 2
        || reader->lengthInSamples <= 0 || reader->lengthInSamples > reader->sampleRate * 30)
        return juce::Result::fail ("Use mono/stereo audio, 8–192 kHz, up to 30 seconds");
    const int n = int (reader->lengthInSamples), channels = int (reader->numChannels), rate = int (reader->sampleRate);
    juce::AudioBuffer<float> audio (channels, n);
    if (! reader->read (&audio, 0, n, 0, true, true))
        return juce::Result::fail ("Could not read audio");
    float peak = 0;
    for (int c = 0; c < channels; ++c)
        for (int i = 0; i < n; ++i)
        {
            const float v = audio.getSample (c, i);
            if (! std::isfinite (v))
                return juce::Result::fail ("Non-finite audio");
            peak = std::max (peak, std::abs (v));
        }
    if (peak < 1e-5f)
        return juce::Result::fail ("Silent audio");
    int first = 0;
    for (; first < n; ++first)
    {
        float v = 0;
        for (int c = 0; c < channels; ++c)
            v = std::max (v, std::abs (audio.getSample (c, first)));
        if (v > peak * .01f)
            break;
    }
    const int length = n - first;
    out.assign (2 * samples, 0);
    const int count = std::min (samples, int (std::ceil (double (length) * 44100 / rate)));
    // Match torchaudio's default sinc_interp_hann: width 6, rolloff .99,
    // rational polyphase FIR. Only evaluate each phase's non-zero support.
    const int gcd = std::gcd (rate, 44100), orig = rate / gcd, dest = 44100 / gcd;
    const float base = float (std::min (orig, dest)) * .99f;
    const int width = int (std::ceil (6 * orig / base));
    struct Tap
    {
        int offset;
        float value;
    };
    std::vector<std::vector<Tap>> phases;
    if (rate != 44100)
    {
        phases.resize ((size_t) dest);
        for (int p = 0; p < dest; ++p)
        {
            const float centre = float (p) * float (orig) / float (dest);
            const int lo = std::max (-width, int (std::floor (centre - width)) - 1);
            const int hi = std::min (width + orig - 1, int (std::ceil (centre + width)) + 1);
            for (int k = lo; k <= hi; ++k)
            {
                const float t = (float (k) / float (orig) - float (p) / float (dest)) * base;
                if (std::abs (t) >= 6)
                    continue;
                const float angle = t * juce::MathConstants<float>::pi;
                const float window = std::cos (angle / 12);
                const float value =
                    (std::abs (angle) < 1e-12f ? 1 : std::sin (angle) / angle) * window * window * base / float (orig);
                phases[(size_t) p].push_back ({ k, value });
            }
        }
    }
    peak = 0;
    for (int c = 0; c < 2; ++c)
        for (int i = 0; i < count; ++i)
        {
            float v = 0;
            if (rate == 44100)
                v = audio.getSample (std::min (c, channels - 1), first + i);
            else
            {
                const int start = (i / dest) * orig;
                for (const auto& tap : phases[(size_t) (i % dest)])
                {
                    const int at = start + tap.offset;
                    if (at >= 0 && at < length)
                        v += tap.value * audio.getSample (std::min (c, channels - 1), first + at);
                }
            }
            out[(size_t) c * samples + (size_t) i] = v;
            peak = std::max (peak, std::abs (v));
        }
    if (peak < 1e-5f)
        return juce::Result::fail ("Silent retained audio");
    for (auto& v : out)
        v /= peak;
    return juce::Result::ok();
}

juce::Result SameSearch::pool (std::span<const float> z, Embedding& out)
{
    if (z.size() != 256 * 44)
        return juce::Result::fail ("Expected SAME latents [1,256,44]");
    double norm = 0;
    for (size_t c = 0; c < 256; ++c)
    {
        double mean = 0, variance = 0;
        for (size_t t = 0; t < 44; ++t)
        {
            const float v = z[c * 44 + t];
            if (! std::isfinite (v))
                return juce::Result::fail ("Non-finite SAME output");
            mean += v / 44.0;
        }
        for (size_t t = 0; t < 44; ++t)
        {
            const double d = z[c * 44 + t] - mean;
            variance += d * d / 44.0;
        }
        out[c] = float (mean);
        out[256 + c] = float (std::sqrt (variance));
        norm += mean * mean + variance;
    }
    if (norm < 1e-16)
        return juce::Result::fail ("Degenerate SAME embedding");
    for (auto& v : out)
        v /= float (std::sqrt (norm));
    return juce::Result::ok();
}

juce::Result SameSearch::load (const juce::File& file)
{
    if (! file.existsAsFile() || file.getSize() > 100 * 1024 * 1024)
        return juce::Result::fail ("Missing or oversized SAME bank");
    auto bank = juce::JSON::parse (file);
    if (bank["format"].toString() != "vae-native-same-1" || int (bank["schema_version"]) != schema::kSchemaVersion
        || bank["engine"].toString() != "lite" || bank["preprocessing"].toString() != "stereo-onset-peak-44100-4s-v1"
        || bank["pooling"].toString() != "mean-std-cosine-1")
        return juce::Result::fail ("Incompatible SAME bank");
    const auto modelName = bank["model_file"].toString();
    if (modelName.isEmpty() || juce::File::isAbsolutePath (modelName) || modelName.contains ("/")
        || modelName.contains ("\\") || modelName == "..")
        return juce::Result::fail ("Model must be a filename beside the bank");
    const auto modelFile = file.getSiblingFile (modelName);
    if (! modelFile.existsAsFile() || juce::SHA256 (modelFile).toHexString() != bank["model_sha256"].toString())
        return juce::Result::fail ("SAME model hash mismatch");
    auto* rows = bank["entries"].getArray();
    if (! rows || rows->isEmpty() || rows->size() > 20000)
        return juce::Result::fail ("Invalid SAME bank entries");
    std::vector<Entry> next;
    for (const auto& row : *rows)
    {
        Entry entry;
        auto r = schema::readPresetValues (row, entry.match.controls);
        if (r.failed())
            return r;
        entry.match.name = row["name"].toString();
        entry.match.note = int (row["note"]);
        entry.match.distance = 0;
        if (entry.match.note < 0 || entry.match.note > 127)
            return juce::Result::fail ("Invalid probe note");
        const auto* e = row["embedding"].getArray();
        if (! e || e->size() != dimensions)
            return juce::Result::fail ("Invalid SAME embedding");
        double norm = 0;
        for (int j = 0; j < dimensions; ++j)
        {
            const auto v = (*e)[j];
            if (! (v.isDouble() || v.isInt()))
                return juce::Result::fail ("Invalid embedding value");
            const float f = float (double (v));
            if (! std::isfinite (f))
                return juce::Result::fail ("Non-finite embedding");
            entry.embedding[(size_t) j] = f;
            norm += double (f) * f;
        }
        if (std::abs (norm - 1) > 1e-3)
            return juce::Result::fail ("Embedding is not normalised");
        next.push_back (entry);
    }
    auto candidate = std::make_unique<OnnxModel>();
    auto result = candidate->load (modelFile);
    if (result.failed())
        return result;
    const auto& in = candidate->getInputs();
    const auto& out = candidate->getOutputs();
    if (in.size() != 1 || out.size() != 1 || ! in[0].isFloat || ! out[0].isFloat
        || in[0].shape != std::vector<int64_t> { 1, 2, samples } || out[0].shape != std::vector<int64_t> { 1, 256, 44 })
        return juce::Result::fail ("Wrong SAME ONNX tensor contract");
    entries = std::move (next);
    model = std::move (candidate);
    return juce::Result::ok();
}
juce::Result SameSearch::encode (std::span<const float> stereo, Embedding& embedding)
{
    if (! model)
        return juce::Result::fail ("Load a SAME search bank first");
    if (stereo.size() != 2 * samples)
        return juce::Result::fail ("Invalid SAME input size");
    std::vector<std::vector<float>> z;
    const std::array<int64_t, 3> shape { 1, 2, samples };
    auto r = model->run (stereo, shape, z);
    if (r.failed())
        return r;
    if (z.size() != 1)
        return juce::Result::fail ("Invalid SAME output");
    return pool (z[0], embedding);
}
juce::Result SameSearch::search (const juce::File& audio, std::vector<Match>& matches)
{
    matches.clear();
    if (! model)
        return juce::Result::fail ("Load a SAME search bank first");
    std::vector<float> input;
    auto r = preprocess (audio, input);
    if (r.failed())
        return r;
    Embedding query;
    r = encode (input, query);
    if (r.failed())
        return r;
    std::vector<Match> ranked;
    for (const auto& e : entries)
    {
        auto m = e.match;
        double dot = 0;
        for (size_t i = 0; i < query.size(); ++i)
            dot += double (query[i]) * e.embedding[i];
        m.distance = float (std::max (0.0, 1 - dot));
        ranked.push_back (m);
    }
    std::stable_sort (ranked.begin(), ranked.end(),
                      [] (const auto& a, const auto& b) { return a.distance < b.distance; });
    std::set<std::array<float, schema::kControlCount>> seen;
    for (const auto& m : ranked)
        if (seen.insert (m.controls.values).second)
        {
            matches.push_back (m);
            if (matches.size() == 5)
                break;
        }
    return juce::Result::ok();
}
SameSearchWorker::~SameSearchWorker()
{
    cancel();
    worker.removeAllJobs (true, -1);
}
void SameSearchWorker::load (const juce::File& file, Callback done) { submit (file, true, std::move (done)); }
void SameSearchWorker::search (const juce::File& file, Callback done) { submit (file, false, std::move (done)); }
void SameSearchWorker::submit (juce::File file, bool loading, Callback done)
{
    enqueue ([file, loading] (State& s, uint64_t, std::vector<SameSearch::Match>& matches)
             { return loading ? s.search.load (file) : s.search.search (file, matches); }, std::move (done));
}
void SameSearchWorker::loadGenerator (const juce::File& file, Callback done)
{
    enqueue ([file] (State& s, uint64_t, auto&) { return s.generator.load (file); }, std::move (done));
}
void SameSearchWorker::generate (const juce::String& prompt, uint32_t seed, const juce::File& output, Callback done,
                                 std::function<void (juce::String)> progress)
{
    enqueue (
        [prompt, seed, output, progress, shared = state] (State& s, uint64_t id, auto& matches)
        {
            if (! s.search.isLoaded())
                return juce::Result::fail ("Load a SAME bank before generating");
            auto cancelled = [&s, id] { return id != s.request.load(); };
            auto notify = [shared, id, progress] (juce::String message)
            {
                juce::MessageManager::callAsync (
                    [shared, id, progress, message]
                    {
                        if (id == shared->request.load() && progress)
                            progress (message);
                    });
            };
            auto result = s.generator.generate (prompt, seed, output, cancelled, notify);
            if (result.failed() || cancelled())
                return result;
            notify ("Finding playable presets...");
            return s.search.search (output, matches);
        },
        std::move (done));
}
void SameSearchWorker::enqueue (Work work, Callback done)
{
    const auto id = ++state->request;
    worker.addJob (
        [shared = state, id, work = std::move (work), done = std::move (done)]
        {
            if (id != shared->request.load())
                return;
            std::vector<SameSearch::Match> matches;
            auto result = work (*shared, id, matches);
            juce::MessageManager::callAsync (
                [shared, id, done, result, matches = std::move (matches)]
                {
                    if (id == shared->request.load() && done)
                        done (result, matches);
                });
        });
}
} // namespace vae::ml
