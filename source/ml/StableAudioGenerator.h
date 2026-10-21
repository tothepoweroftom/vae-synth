#pragma once
#include "OnnxModel.h"
#include <juce_audio_formats/juce_audio_formats.h>
#include <array>
#include <functional>

namespace vae::ml
{
/** SHA-256 check that is skipped once this exact file (path, size, mtime) has passed. */
bool matchesSha256 (const juce::File&, const juce::String& expectedHex);

// Fixed four-second Small-SFX inference. All work belongs on a worker thread.
class StableAudioGenerator
{
public:
    ~StableAudioGenerator();
    juce::Result load (const juce::File& manifest);
    bool isLoaded() const noexcept { return dit != nullptr; }
    juce::Result generate (const juce::String& prompt, uint32_t seed, const juce::File& output,
                           std::function<bool()> cancelled = {}, std::function<void (juce::String)> progress = {},
                           std::span<const float> fixtureNoise = {});
    static juce::Result tokenize (void* tokenizer, const juce::String&, std::array<float, 256>& ids,
                                  std::array<float, 256>& mask);

private:
    std::unique_ptr<OnnxModel> text, dit, decoder;
    void* tokenizer = nullptr;
    std::array<float, 9> schedule {};
    size_t decoderNoiseCount = 0;
};
} // namespace vae::ml
