#pragma once
#include "OnnxModel.h"
#include <juce_audio_formats/juce_audio_formats.h>
#include <array>
#include <functional>

namespace vae::ml
{
// Fixed four-second Small-SFX inference. All work belongs on a worker thread.
class StableAudioGenerator
{
public:
    ~StableAudioGenerator();
    juce::Result load (const juce::File& manifest);
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
