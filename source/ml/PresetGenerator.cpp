#include "PresetGenerator.h"

#include <algorithm>
#include <cmath>

namespace vae::ml
{
namespace
{
uint64_t fnv1a (const juce::String& s)
{
    uint64_t h = 1469598103934665603ull;
    for (auto* p = s.toRawUTF8(); *p != 0; ++p)
    {
        h ^= static_cast<uint8_t> (*p);
        h *= 1099511628211ull;
    }
    return h;
}

uint64_t splitmix (uint64_t& state)
{
    uint64_t z = (state += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

void callOnMessageThread (std::weak_ptr<bool> alive, std::function<void()> fn)
{
    juce::MessageManager::callAsync (
        [alive = std::move (alive), fn = std::move (fn)]
        {
            if (alive.lock() != nullptr)
                fn();
        });
}
} // namespace

//==============================================================================
schema::Latent HashPromptEncoder::encode (const juce::String& prompt)
{
    schema::Latent latent {};
    juce::StringArray words;
    words.addTokens (prompt.toLowerCase(), " \t\r\n,.;:!?-_/()[]\"'", {});
    words.removeEmptyStrings();

    if (words.isEmpty())
        return latent;

    for (const auto& word : words)
    {
        uint64_t state = fnv1a (word);
        for (auto& l : latent)
            l += float (double (splitmix (state) >> 11) / double (1ull << 53)) * 2.0f - 1.0f;
    }

    // Mean of n uniform(-1, 1) vectors shrinks like 1/sqrt(n); undo that so
    // long prompts still reach the edges of the latent space.
    const float scale = std::sqrt (float (words.size())) / float (words.size());
    for (auto& l : latent)
        l = juce::jlimit (-1.0f, 1.0f, l * scale);
    return latent;
}

//==============================================================================
juce::Result LatentDecoder::loadModel (const juce::File& file)
{
    auto candidate = std::make_unique<OnnxModel>();
    const auto loaded = candidate->load (file);
    return install (std::move (candidate), loaded, file.getFileName());
}

juce::Result LatentDecoder::loadModel (const void* data, size_t size, const juce::String& name)
{
    auto candidate = std::make_unique<OnnxModel>();
    const auto loaded = candidate->load (data, size);
    return install (std::move (candidate), loaded, name);
}

juce::Result LatentDecoder::install (std::unique_ptr<OnnxModel> candidate, juce::Result loaded,
                                     const juce::String& name)
{
    if (loaded.failed())
        return loaded;
    if (auto r = validate (*candidate); r.failed())
        return r;
    model = std::move (candidate);
    modelName = name;
    return juce::Result::ok();
}

juce::Result LatentDecoder::validate (const OnnxModel& m)
{
    // [batch, n] with batch 1 or dynamic: what decode() feeds and reads.
    const auto fits = [] (const TensorInfo& t, int64_t n)
    { return t.isFloat && t.shape.size() == 2 && (t.shape[0] == 1 || t.shape[0] < 0) && t.shape[1] == n; };

    if (! fits (m.getInputs().front(), schema::kLatentDimensions))
        return juce::Result::fail ("Decoder input must be float [1, " + juce::String (schema::kLatentDimensions)
                                   + "] (latent_dimensions)");
    if (m.getOutputs().empty() || ! fits (m.getOutputs().front(), schema::kFeatureCount))
        return juce::Result::fail ("Decoder output must be float [1, " + juce::String (schema::kFeatureCount)
                                   + "] (feature count)");
    return juce::Result::ok();
}

juce::String LatentDecoder::describe() const
{
    if (! runtime->isAvailable())
        return "mock decoder (" + runtime->getStatus() + ")";
    if (! hasModel())
        return "mock decoder (no model loaded)";
    return "ONNX decoder" + (modelName.isEmpty() ? juce::String() : ": " + modelName);
}

schema::Snapshot LatentDecoder::decode (const schema::Latent& latent, juce::String* error)
{
    if (hasModel())
    {
        const std::array<int64_t, 2> shape { 1, schema::kLatentDimensions };
        std::vector<std::vector<float>> outs;
        const auto r = model->run (latent, shape, outs);
        juce::String problem;
        if (r.failed())
            problem = r.getErrorMessage();
        else if (outs.empty() || outs.front().size() != size_t (schema::kFeatureCount))
            problem = "decoder returned the wrong size";
        else if (! std::all_of (outs.front().begin(), outs.front().end(), [] (float v) { return std::isfinite (v); }))
            problem = "decoder returned non-finite values";
        else
            return schema::decodeFeatures (
                std::span<const float, schema::kFeatureCount> (outs.front().data(), schema::kFeatureCount));
        if (error != nullptr)
            *error = problem + " (mock decoder used)";
    }
    return schema::mockDecode (latent);
}

//==============================================================================
PresetGenerator::PresetGenerator()
    : encoder (std::make_unique<HashPromptEncoder>())
{
    const std::scoped_lock sl (lock);
    refreshStatus();
}

PresetGenerator::~PresetGenerator()
{
    alive.reset();
    pool.removeAllJobs (true, 5000);
}

void PresetGenerator::refreshStatus()
{
    auto text = encoder->describe() + " -> " + decoder.describe();
    const std::scoped_lock sl (statusLock);
    status = std::move (text);
}

juce::String PresetGenerator::describe() const
{
    const std::scoped_lock sl (statusLock);
    return status;
}

void PresetGenerator::setPromptEncoder (std::unique_ptr<PromptEncoder> e)
{
    const std::scoped_lock sl (lock);
    encoder = e != nullptr ? std::move (e) : std::make_unique<HashPromptEncoder>();
    refreshStatus();
}

void PresetGenerator::load (std::function<juce::Result (LatentDecoder&)> loadFn,
                            std::function<void (juce::Result)> done)
{
    pool.addJob (
        [this, loadFn = std::move (loadFn), done = std::move (done), weak = std::weak_ptr<bool> (alive)]
        {
            juce::Result r = juce::Result::ok();
            {
                const std::scoped_lock sl (lock);
                r = loadFn (decoder);
                refreshStatus();
            }
            if (done)
                callOnMessageThread (weak, [done, r] { done (r); });
        });
}

void PresetGenerator::loadDecoder (const juce::File& file, std::function<void (juce::Result)> done)
{
    load ([file] (LatentDecoder& d) { return d.loadModel (file); }, std::move (done));
}

void PresetGenerator::loadDecoder (juce::MemoryBlock data, juce::String name, std::function<void (juce::Result)> done)
{
    load ([data = std::move (data), name] (LatentDecoder& d)
          { return d.loadModel (data.getData(), data.getSize(), name); }, std::move (done));
}

void PresetGenerator::start (juce::String prompt, std::optional<schema::Latent> latent,
                             std::function<void (const Result&)> done)
{
    const auto request = ++latestRequest;
    pool.addJob (
        [this, request, prompt = std::move (prompt), latent, done = std::move (done),
         weak = std::weak_ptr<bool> (alive)]
        {
            if (request != latestRequest.load())
                return; // superseded by a newer request
            Result result;
            result.prompt = prompt;
            {
                const std::scoped_lock sl (lock);
                result.latent = latent.has_value() ? *latent : encoder->encode (prompt);
                result.controls = decoder.decode (result.latent, &result.note);
            }
            if (result.note.isEmpty())
                result.note = describe();
            if (done)
                callOnMessageThread (weak, [done, result] { done (result); });
        });
}

void PresetGenerator::generate (const juce::String& prompt, std::function<void (const Result&)> done)
{
    start (prompt, std::nullopt, std::move (done));
}

void PresetGenerator::generateFromLatent (const schema::Latent& latent, std::function<void (const Result&)> done)
{
    start ({}, latent, std::move (done));
}

} // namespace vae::ml
