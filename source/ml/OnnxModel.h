/*
  A loaded ONNX model with float tensor inputs/outputs. Generic on purpose:
  it knows nothing about latents or presets, so prompt encoders, decoders or
  audio-feature models can all sit on top of it.

  Never use from the audio thread: loading and running allocate and may take
  milliseconds. PresetGenerator runs it on a worker thread.
*/

#pragma once

#include <juce_core/juce_core.h>

#include "OrtRuntime.h"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace vae::ml
{

struct TensorInfo
{
    std::string name;
    std::vector<int64_t> shape; // -1 for dynamic dimensions
    bool isFloat = false;
    bool isInt64 = false;
};

struct FloatTensorView
{
    std::span<const float> data;
    std::span<const int64_t> shape;
};

struct Int64TensorView
{
    std::span<const int64_t> data;
    std::span<const int64_t> shape;
};

/** Per-user "VAE Synth" data folder (or $VAE_DATA_DIR): development models and derived caches. */
juce::File appDataDir();

/** Installed model asset; shared installer data, with per-user development fallback. */
juce::File installedModels (const char* relative);

bool isSha256Hex (const juce::String&);

struct ModelLoadOptions
{
    int threads = 1;
    juce::String cacheKey;    // a SHA-256: keep ORT's optimized graph under it so later loads skip optimizing
    bool int64Inputs = false; // accept int64 input tensors too (token ids); run them with the Int64TensorView overload
};

class OnnxModel
{
public:
    OnnxModel();
    ~OnnxModel();

    /** Load from disk, replacing any current model. */
    juce::Result load (const juce::File& file, ModelLoadOptions = {});
    /** Load from memory (embedded models, tests). */
    juce::Result load (const void* data, size_t size);
    void unload();

    bool isLoaded() const noexcept;
    const std::vector<TensorInfo>& getInputs() const noexcept { return inputs; }
    const std::vector<TensorInfo>& getOutputs() const noexcept { return outputs; }

    /** Run with one float input tensor; returns every float output, flattened,
        in getOutputs() order. Not thread-safe per instance. */
    juce::Result run (std::span<const float> input, std::span<const int64_t> inputShape,
                      std::vector<std::vector<float>>& outputsOut);

    juce::Result run (std::span<const FloatTensorView> tensors, std::vector<std::vector<float>>& outputsOut);
    juce::Result run (std::span<const Int64TensorView> tensors, std::vector<std::vector<float>>& outputsOut);

    /** Status of the shared runtime ("not found", version, ...). */
    juce::String getRuntimeStatus() const { return runtime->getStatus(); }
    bool isRuntimeAvailable() const { return runtime->isAvailable(); }

private:
    template <typename MakeSession>
    juce::Result loadWith (MakeSession&& makeSession, int threads = 1, const juce::File& saveOptimizedTo = {},
                           bool preOptimized = false, bool int64Inputs = false);
    juce::Result finishLoad (bool int64Inputs);
    template <typename View>
    juce::Result runViews (std::span<const View> tensors, std::vector<std::vector<float>>& outputsOut);

    juce::SharedResourcePointer<OrtRuntime> runtime;
    struct Session;
    std::unique_ptr<Session> session;
    std::vector<TensorInfo> inputs, outputs;
};

} // namespace vae::ml
