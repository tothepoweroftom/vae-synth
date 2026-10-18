/*
  Prompt -> latent -> control vector, the stubbed "prompt to preset" path.

  Two replaceable stages, both deliberately minimal until the model design is
  settled:
    PromptEncoder  text -> latent. Default: HashPromptEncoder, a deterministic
                   bag-of-words hash (no model). A learned text encoder (CLAP,
                   sentence embedding + projection, ...) goes behind the same
                   interface.
    LatentDecoder  latent -> schema vector. Uses an ONNX model when one is
                   loaded (input [1, latent_dimensions], output
                   [1, control_count] in 0..1), else schema::mockDecode.

  PresetGenerator runs the pipeline on a worker thread and reports back on the
  message thread.
*/

#pragma once

#include <juce_events/juce_events.h>

#include "OnnxModel.h"
#include "schema/ControlSchema.h"

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>

namespace vae::ml
{

class PromptEncoder
{
public:
    virtual ~PromptEncoder() = default;
    virtual juce::String describe() const = 0;
    virtual schema::Latent encode (const juce::String& prompt) = 0;
};

/** Placeholder encoder: each lower-cased word hashes to a fixed pseudo-random
    direction; the latent is their mean, scaled into -1..1. Deterministic, so
    the same prompt always gives the same preset, and prompts that share words
    land near each other. Carries no meaning: swap for a trained encoder. */
class HashPromptEncoder final : public PromptEncoder
{
public:
    juce::String describe() const override { return "hash (stub)"; }
    schema::Latent encode (const juce::String& prompt) override;
};

class LatentDecoder
{
public:
    /** Load an ONNX decoder; validates its tensor shapes. Empty file = unload. */
    juce::Result loadModel (const juce::File& file);
    juce::Result loadModel (const void* data, size_t size, const juce::String& name = {});
    bool hasModel() const noexcept { return model.isLoaded(); }
    juce::String describe() const;
    juce::String getRuntimeStatus() const { return model.getRuntimeStatus(); }

    /** Decode one latent. Falls back to mockDecode when no model is loaded or
        inference fails (the reason goes to `error` if given). */
    schema::Snapshot decode (const schema::Latent& latent, juce::String* error = nullptr);

private:
    juce::Result validate();

    OnnxModel model;
    juce::File modelFile;
    juce::String modelName;
};

class PresetGenerator
{
public:
    struct Result
    {
        juce::String prompt;
        schema::Latent latent {};
        schema::Snapshot controls;
        juce::String note;   // e.g. "mock decoder" or an inference error
    };

    PresetGenerator();
    ~PresetGenerator();

    /** Replace the prompt encoder (default: HashPromptEncoder). Message thread. */
    void setPromptEncoder (std::unique_ptr<PromptEncoder>);

    /** Load or clear the decoder model (worker thread; `done` on message thread). */
    void loadDecoder (const juce::File& file, std::function<void (juce::Result)> done);
    /** As loadDecoder, from a model held in memory (copied). */
    void loadDecoder (juce::MemoryBlock model, juce::String name, std::function<void (juce::Result)> done);

    /** Start a generation; `done` is called on the message thread. A request
        made before an earlier one starts supersedes it (the
        earlier one is skipped and never calls back). */
    void generate (const juce::String& prompt, std::function<void (const Result&)> done);
    void generateFromLatent (const schema::Latent& latent, std::function<void (const Result&)> done);

    juce::String describe() const;

private:
    void enqueue (std::function<void()> job);

    mutable std::mutex lock;   // guards encoder/decoder between worker and message thread
    std::unique_ptr<PromptEncoder> encoder;
    LatentDecoder decoder;
    juce::ThreadPool pool { juce::ThreadPoolOptions{}.withThreadName ("vae-ml").withNumberOfThreads (1) };
    std::atomic<uint64_t> latestRequest { 0 };
    std::shared_ptr<bool> alive = std::make_shared<bool> (true);
};

} // namespace vae::ml
