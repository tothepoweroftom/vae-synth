/*
  Process-wide ONNX Runtime: loads the shared library at runtime (we build
  with ORT_API_MANUAL_INIT and never link it), initialises the C++ API and
  owns the single Ort::Env. Shared by every plug-in instance in the process
  via juce::SharedResourcePointer.

  Library search order:
    1. $VAE_ORT_LIBRARY (absolute path), for development
    2. next to this plug-in/executable binary
    3. ../Frameworks and ../Resources relative to it (macOS bundles)
    4. the bare library name (system search path)
*/

#pragma once

#include <juce_core/juce_core.h>

#include <memory>

namespace Ort
{
struct Env;
}

namespace vae::ml
{

class OrtRuntime
{
public:
    OrtRuntime();
    ~OrtRuntime();

    bool isAvailable() const noexcept { return env != nullptr; }

    /** Why the runtime is unavailable, or the loaded library path. */
    const juce::String& getStatus() const noexcept { return status; }

    /** Valid only when isAvailable(). */
    Ort::Env& getEnv() const;

private:
    juce::DynamicLibrary library;
    std::unique_ptr<Ort::Env> env;
    juce::String status;
};

} // namespace vae::ml
