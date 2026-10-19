#include "OrtRuntime.h"

#include <onnxruntime_cxx_api.h>

namespace vae::ml
{
namespace
{
juce::Array<juce::File> candidateLibraries()
{
    juce::Array<juce::File> files;

    const auto fromEnv = juce::SystemStats::getEnvironmentVariable ("VAE_ORT_LIBRARY", {});
    if (fromEnv.isNotEmpty())
        files.add (juce::File (fromEnv));

    // For plug-ins JUCE returns the plug-in binary here, not the host.
    const auto dir = juce::File::getSpecialLocation (juce::File::currentExecutableFile).getParentDirectory();
    files.add (dir.getChildFile (VAE_ORT_RUNTIME_NAME));
    files.add (dir.getSiblingFile ("Frameworks").getChildFile (VAE_ORT_RUNTIME_NAME));
    files.add (dir.getSiblingFile ("Resources").getChildFile (VAE_ORT_RUNTIME_NAME));
    return files;
}
} // namespace

OrtRuntime::OrtRuntime()
{
    bool opened = false;
    for (const auto& file : candidateLibraries())
    {
        if (file.existsAsFile() && library.open (file.getFullPathName()))
        {
            status = file.getFullPathName();
            opened = true;
            break;
        }
    }

    if (! opened && library.open (VAE_ORT_RUNTIME_NAME))
    {
        status = VAE_ORT_RUNTIME_NAME;
        opened = true;
    }

    if (! opened)
    {
        status = "ONNX Runtime library (" VAE_ORT_RUNTIME_NAME ") not found";
        return;
    }

    using GetApiBaseFn = const OrtApiBase*(ORT_API_CALL*) ();
    auto getApiBase = reinterpret_cast<GetApiBaseFn> (library.getFunction ("OrtGetApiBase"));
    if (getApiBase == nullptr)
    {
        status = "OrtGetApiBase missing from " + status;
        library.close();
        return;
    }

    const OrtApi* api = getApiBase()->GetApi (ORT_API_VERSION);
    if (api == nullptr)
    {
        status = juce::String ("ONNX Runtime ") + getApiBase()->GetVersionString() + " does not provide API version "
                 + juce::String (ORT_API_VERSION);
        library.close();
        return;
    }

    Ort::InitApi (api);

    try
    {
        env = std::make_unique<Ort::Env> (ORT_LOGGING_LEVEL_WARNING, "vae-synth");
        status = juce::String ("ONNX Runtime ") + getApiBase()->GetVersionString() + " (" + status + ")";
    }
    catch (const Ort::Exception& e)
    {
        status = juce::String ("ONNX Runtime init failed: ") + e.what();
    }
}

OrtRuntime::~OrtRuntime()
{
    env.reset(); // before the library goes away
}

Ort::Env& OrtRuntime::getEnv() const
{
    jassert (env != nullptr);
    return *env;
}

} // namespace vae::ml
