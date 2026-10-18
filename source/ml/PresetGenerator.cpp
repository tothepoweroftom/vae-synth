#include "PresetGenerator.h"

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
    juce::MessageManager::callAsync ([alive = std::move (alive), fn = std::move (fn)]
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
    modelFile = juce::File();
    modelName = {};
    if (file == juce::File())
    {
        model.unload();
        return juce::Result::ok();
    }
    if (auto r = model.load (file); r.failed())
        return r;
    if (auto r = validate(); r.failed())
        return r;
    modelFile = file;
    modelName = file.getFileName();
    return juce::Result::ok();
}

juce::Result LatentDecoder::loadModel (const void* data, size_t size, const juce::String& name)
{
    modelFile = juce::File();
    modelName = {};
    if (auto r = model.load (data, size); r.failed())
        return r;
    if (auto r = validate(); r.failed())
        return r;
    modelName = name;
    return juce::Result::ok();
}

juce::Result LatentDecoder::validate()
{
    const auto& in = model.getInputs().front();
    const auto outs = model.getOutputs();
    const auto lastDim = [] (const TensorInfo& t) { return t.shape.empty() ? int64_t (0) : t.shape.back(); };

    if (lastDim (in) != schema::kLatentDimensions)
    {
        model.unload();
        return juce::Result::fail ("Decoder input must end in " + juce::String (schema::kLatentDimensions)
                                   + " (latent_dimensions)");
    }
    if (outs.empty() || ! outs.front().isFloat || lastDim (outs.front()) != schema::kControlCount)
    {
        model.unload();
        return juce::Result::fail ("Decoder output must be float and end in " + juce::String (schema::kControlCount)
                                   + " (control count)");
    }
    return juce::Result::ok();
}

juce::String LatentDecoder::describe() const
{
    if (! model.isRuntimeAvailable())
        return "mock decoder (" + model.getRuntimeStatus() + ")";
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
        const auto r = model.run (latent, shape, outs);
        if (r.wasOk() && ! outs.empty() && outs.front().size() == size_t (schema::kControlCount))
        {
            schema::Snapshot s;
            std::copy (outs.front().begin(), outs.front().end(), s.values.begin());
            s.sanitise();
            return s;
        }
        if (error != nullptr)
            *error = r.failed() ? r.getErrorMessage() : juce::String ("decoder returned the wrong size");
    }
    return schema::mockDecode (latent);
}

//==============================================================================
PresetGenerator::PresetGenerator() : encoder (std::make_unique<HashPromptEncoder>()) {}

PresetGenerator::~PresetGenerator()
{
    *alive = false;
    alive.reset();
    pool.removeAllJobs (true, 5000);
}

void PresetGenerator::setPromptEncoder (std::unique_ptr<PromptEncoder> e)
{
    const std::scoped_lock sl (lock);
    encoder = e != nullptr ? std::move (e) : std::make_unique<HashPromptEncoder>();
}

void PresetGenerator::enqueue (std::function<void()> job)
{
    pool.addJob (std::move (job));
}

void PresetGenerator::loadDecoder (const juce::File& file, std::function<void (juce::Result)> done)
{
    enqueue ([this, file, done = std::move (done), weak = std::weak_ptr<bool> (alive)]
    {
        juce::Result r = juce::Result::ok();
        {
            const std::scoped_lock sl (lock);
            r = decoder.loadModel (file);
        }
        if (done)
            callOnMessageThread (weak, [done, r] { done (r); });
    });
}

void PresetGenerator::loadDecoder (juce::MemoryBlock data, juce::String name, std::function<void (juce::Result)> done)
{
    enqueue ([this, data = std::move (data), name, done = std::move (done), weak = std::weak_ptr<bool> (alive)]
    {
        juce::Result r = juce::Result::ok();
        {
            const std::scoped_lock sl (lock);
            r = decoder.loadModel (data.getData(), data.getSize(), name);
        }
        if (done)
            callOnMessageThread (weak, [done, r] { done (r); });
    });
}

void PresetGenerator::generate (const juce::String& prompt, std::function<void (const Result&)> done)
{
    const auto request = ++latestRequest;
    enqueue ([this, request, prompt, done = std::move (done), weak = std::weak_ptr<bool> (alive)]
    {
        if (request != latestRequest.load())
            return;   // superseded by a newer request
        Result result;
        result.prompt = prompt;
        {
            const std::scoped_lock sl (lock);
            result.latent = encoder->encode (prompt);
            result.controls = decoder.decode (result.latent, &result.note);
            if (result.note.isEmpty())
                result.note = encoder->describe() + " -> " + decoder.describe();
        }
        if (done)
            callOnMessageThread (weak, [done, result] { done (result); });
    });
}

void PresetGenerator::generateFromLatent (const schema::Latent& latent, std::function<void (const Result&)> done)
{
    const auto request = ++latestRequest;
    enqueue ([this, request, latent, done = std::move (done), weak = std::weak_ptr<bool> (alive)]
    {
        if (request != latestRequest.load())
            return;
        Result result;
        result.latent = latent;
        {
            const std::scoped_lock sl (lock);
            result.controls = decoder.decode (latent, &result.note);
            if (result.note.isEmpty())
                result.note = decoder.describe();
        }
        if (done)
            callOnMessageThread (weak, [done, result] { done (result); });
    });
}

juce::String PresetGenerator::describe() const
{
    const std::scoped_lock sl (lock);
    return encoder->describe() + " -> " + decoder.describe();
}

} // namespace vae::ml
