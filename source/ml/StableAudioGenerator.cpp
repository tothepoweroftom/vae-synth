#include "StableAudioGenerator.h"
#include <juce_cryptography/juce_cryptography.h>
#include <random>
#include <cmath>

extern "C"
{
    void* vae_tokenizer_load (const char*);
    void vae_tokenizer_free (void*);
    bool vae_tokenizer_encode (const void*, const char*, float*, float*);
}
namespace vae::ml
{
bool matchesSha256 (const juce::File& file, const juce::String& expectedHex)
{
    // Guards against corrupt or swapped files, not deliberate tampering (mtime can be forged).
    if (! isSha256Hex (expectedHex) || ! file.existsAsFile())
        return false;
    const auto stamp = appDataDir().getChildFile ("cache/verified/" + expectedHex);
    const auto identity = file.getFullPathName() + "|" + juce::String (file.getSize()) + "|"
                          + juce::String (file.getLastModificationTime().toMilliseconds());
    if (stamp.loadFileAsString() == identity)
        return true;
    if (juce::SHA256 (file).toHexString() != expectedHex)
        return false;
    if (stamp.getParentDirectory().createDirectory().wasOk())
        stamp.replaceWithText (identity);
    return true;
}

StableAudioGenerator::~StableAudioGenerator() { vae_tokenizer_free (tokenizer); }
juce::Result StableAudioGenerator::tokenize (void* t, const juce::String& prompt, std::array<float, 256>& ids,
                                             std::array<float, 256>& mask)
{
    if (! t || ! vae_tokenizer_encode (t, prompt.toRawUTF8(), ids.data(), mask.data()))
        return juce::Result::fail ("Could not tokenize prompt");
    return juce::Result::ok();
}
juce::Result StableAudioGenerator::load (const juce::File& manifest)
{
    if (! manifest.existsAsFile() || manifest.getSize() > 1024 * 1024)
        return juce::Result::fail ("Select generator.json from the native Small-SFX package");
    auto data = juce::JSON::parse (manifest);
    if (data["format"].toString() != "vae-stable-audio-small-sfx-1")
        return juce::Result::fail ("Not a native Small-SFX generator package");
    const auto root = manifest.getParentDirectory();
    for (const auto* name : { "text.onnx", "dit.onnx", "decoder.onnx", "tokenizer.json" })
    {
        const auto file = root.getChildFile (name);
        if (! matchesSha256 (file, data["sha256"][name].toString()))
            return juce::Result::fail (juce::String ("Missing or corrupt generator file: ") + name);
    }
    auto* times = data["schedule"].getArray();
    if (! times || times->size() != 9)
        return juce::Result::fail ("Invalid diffusion schedule");
    std::array<float, 9> newSchedule;
    for (int i = 0; i < 9; ++i)
    {
        newSchedule[size_t (i)] = float ((*times)[i]);
        if (! std::isfinite (newSchedule[size_t (i)]) || newSchedule[size_t (i)] < 0 || newSchedule[size_t (i)] > 1
            || (i > 0 && newSchedule[size_t (i)] >= newSchedule[size_t (i - 1)]))
            return juce::Result::fail ("Invalid diffusion schedule");
    }
    if (newSchedule.front() != 1 || newSchedule.back() != 0)
        return juce::Result::fail ("Invalid diffusion endpoints");
    auto tx = std::make_unique<OnnxModel>(), di = std::make_unique<OnnxModel>(), de = std::make_unique<OnnxModel>();
    for (const auto& item :
         { std::pair { tx.get(), "text.onnx" }, { di.get(), "dit.onnx" }, { de.get(), "decoder.onnx" } })
    {
        // A few threads: generation is otherwise ~20 s on one core.
        const int threads = juce::jlimit (1, 4, juce::SystemStats::getNumPhysicalCpus() / 2);
        auto result =
            item.first->load (root.getChildFile (item.second), { threads, data["sha256"][item.second].toString() });
        if (result.failed())
            return result;
    }
    const std::vector<std::vector<int64_t>> expectedText { { 1, 256 }, { 1, 256 } };
    const std::vector<std::vector<int64_t>> expectedDit { { 1, 256, 108 }, { 1 }, { 1, 256, 768 } };
    auto shapesMatch = [] (const OnnxModel& model, const auto& shapes, const std::vector<int64_t>& out)
    {
        if (model.getInputs().size() != shapes.size() || model.getOutputs().size() != 1
            || ! model.getOutputs()[0].isFloat || model.getOutputs()[0].shape != out)
            return false;
        for (size_t i = 0; i < shapes.size(); ++i)
            if (model.getInputs()[i].shape != shapes[i])
                return false;
        return true;
    };
    const auto noiseCount = int (data["decoder_noise_count"]);
    const std::vector<std::vector<int64_t>> expectedDecoder { { 1, 256, 108 }, { noiseCount } };
    if (noiseCount < 1 || noiseCount > 4000000 || ! shapesMatch (*tx, expectedText, { 1, 256, 768 })
        || ! shapesMatch (*di, expectedDit, { 1, 256, 108 }) || ! shapesMatch (*de, expectedDecoder, { 1, 2, 176400 }))
        return juce::Result::fail ("Generator tensor contract mismatch");
    auto* tok = vae_tokenizer_load (root.getChildFile ("tokenizer.json").getFullPathName().toRawUTF8());
    if (! tok)
        return juce::Result::fail ("Could not load tokenizer");
    vae_tokenizer_free (tokenizer);
    tokenizer = tok;
    text = std::move (tx);
    dit = std::move (di);
    decoder = std::move (de);
    schedule = newSchedule;
    decoderNoiseCount = size_t (noiseCount);
    return juce::Result::ok();
}
juce::Result StableAudioGenerator::generate (const juce::String& prompt, uint32_t seed, const juce::File& output,
                                             std::function<bool()> cancelled,
                                             std::function<void (juce::String)> progress,
                                             std::span<const float> fixtureNoise)
{
    if (! text)
        return juce::Result::fail ("Load a Stable Audio generator package first");
    if (prompt.trim().isEmpty() || prompt.getNumBytesAsUTF8() > 16384)
        return juce::Result::fail ("Enter a prompt of at most 16384 UTF-8 bytes");
    constexpr size_t count = 256 * 108;
    if (! fixtureNoise.empty() && fixtureNoise.size() != count * 8 + decoderNoiseCount)
        return juce::Result::fail ("Invalid noise fixture");
    auto check = [&] { return cancelled && cancelled(); };
    auto report = [&] (juce::String message)
    {
        if (progress)
            progress (message);
    };
    if (check())
        return juce::Result::fail ("Cancelled");
    std::array<float, 256> ids {}, mask {};
    auto result = tokenize (tokenizer, prompt, ids, mask);
    if (result.failed())
        return result;
    report ("Encoding prompt...");
    const std::array<int64_t, 2> tokenShape { 1, 256 };
    const std::array<FloatTensorView, 2> tokenInputs { FloatTensorView { ids, tokenShape }, { mask, tokenShape } };
    std::vector<std::vector<float>> hidden;
    result = text->run (tokenInputs, hidden);
    if (result.failed())
        return result;
    if (hidden.size() != 1 || hidden[0].size() != 256 * 768)
        return juce::Result::fail ("Invalid text encoder output");
    std::mt19937 rng (seed);
    std::normal_distribution<float> normal;
    size_t noiseOffset = 0;
    auto noise = [&] { return fixtureNoise.empty() ? normal (rng) : fixtureNoise[noiseOffset++]; };
    std::vector<float> x (count);
    for (auto& value : x)
        value = noise();
    const std::array<int64_t, 3> latentShape { 1, 256, 108 }, hiddenShape { 1, 256, 768 };
    const std::array<int64_t, 1> timeShape { 1 };
    for (size_t step = 0; step < 8; ++step)
    {
        if (check())
            return juce::Result::fail ("Cancelled");
        report ("Generating audio: step " + juce::String (int (step + 1)) + "/8...");
        const std::array<FloatTensorView, 3> in { FloatTensorView { x, latentShape },
                                                  { std::span (&schedule[step], 1), timeShape },
                                                  { hidden[0], hiddenShape } };
        std::vector<std::vector<float>> v;
        result = dit->run (in, v);
        if (result.failed())
            return result;
        if (v.size() != 1 || v[0].size() != count)
            return juce::Result::fail ("Invalid diffusion output");
        for (size_t i = 0; i < count; ++i)
        {
            x[i] = (1 - schedule[step + 1]) * (x[i] - schedule[step] * v[0][i])
                   + (step == 7 ? 0.0f : schedule[step + 1] * noise());
            if (! std::isfinite (x[i]))
                return juce::Result::fail ("Non-finite diffusion output");
        }
    }
    if (check())
        return juce::Result::fail ("Cancelled");
    report ("Decoding reference audio...");
    std::vector<std::vector<float>> audio;
    std::vector<float> decoderNoise (decoderNoiseCount);
    for (auto& value : decoderNoise)
        value = noise();
    const std::array<int64_t, 1> noiseShape { int64_t (decoderNoiseCount) };
    const std::array<FloatTensorView, 2> decoderInputs { FloatTensorView { x, latentShape },
                                                         { decoderNoise, noiseShape } };
    result = decoder->run (decoderInputs, audio);
    if (result.failed())
        return result;
    if (audio.size() != 1 || audio[0].size() != 2 * 176400)
        return juce::Result::fail ("Invalid decoder output");
    for (auto value : audio[0])
        if (! std::isfinite (value))
            return juce::Result::fail ("Non-finite audio output");
    if (check())
        return juce::Result::fail ("Cancelled");
    if (output.exists())
        return juce::Result::fail ("Reference output already exists");
    std::unique_ptr<juce::OutputStream> stream = output.createOutputStream();
    if (! stream)
        return juce::Result::fail ("Could not create reference file");
    juce::WavAudioFormat format;
    auto writer = format.createWriterFor (
        stream, juce::AudioFormatWriterOptions {}.withSampleRate (44100).withNumChannels (2).withBitsPerSample (24));
    if (! writer)
        return juce::Result::fail ("Could not create WAV writer");
    const float* channels[] = { audio[0].data(), audio[0].data() + 176400 };
    if (! writer->writeFromFloatArrays (channels, 2, 176400))
        return juce::Result::fail ("Could not write reference");
    writer.reset();
    auto metadata = std::make_unique<juce::DynamicObject>();
    metadata->setProperty ("model", "stabilityai/stable-audio-3-small-sfx");
    metadata->setProperty ("prompt", prompt);
    metadata->setProperty ("seed", juce::int64 (seed));
    metadata->setProperty ("runtime", "JUCE C++ / ONNX Runtime / FP32");
    metadata->setProperty ("duration", 4);
    metadata->setProperty ("steps", 8);
    if (! output.withFileExtension ("json").replaceWithText (juce::JSON::toString (juce::var (metadata.release()))))
        return juce::Result::fail ("Reference saved, but could not save generation metadata");
    return juce::Result::ok();
}
} // namespace vae::ml
