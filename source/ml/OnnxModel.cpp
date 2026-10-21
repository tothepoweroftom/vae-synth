#include "OnnxModel.h"

#include <onnxruntime_cxx_api.h>

namespace vae::ml
{

struct OnnxModel::Session
{
    Ort::SessionOptions options;
    std::unique_ptr<Ort::Session> session;
    std::vector<std::string> inputNames, outputNames;
};

namespace
{
Ort::SessionOptions makeOptions (int threads)
{
    Ort::SessionOptions options;
    // Called rarely, from a worker thread next to a busy audio thread: one
    // thread by default; large models may ask for a few. Idle workers sleep
    // rather than spin so they do not steal cores from the DAW.
    options.SetIntraOpNumThreads (threads);
    options.SetInterOpNumThreads (1);
    options.AddConfigEntry ("session.intra_op.allow_spinning", "0");
    options.SetGraphOptimizationLevel (GraphOptimizationLevel::ORT_ENABLE_ALL);
    return options;
}

auto fromPath (const juce::File& file)
{
    return [file] (Ort::Env& env, const Ort::SessionOptions& options)
    {
#if JUCE_WINDOWS
        return std::make_unique<Ort::Session> (env, file.getFullPathName().toWideCharPointer(), options);
#else
        return std::make_unique<Ort::Session> (env, file.getFullPathName().toRawUTF8(), options);
#endif
    };
}

TensorInfo describe (const Ort::TypeInfo& type, std::string name)
{
    TensorInfo info;
    info.name = std::move (name);
    if (type.GetONNXType() == ONNX_TYPE_TENSOR)
    {
        const auto tensor = type.GetTensorTypeAndShapeInfo();
        info.shape = tensor.GetShape();
        info.isFloat = tensor.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        info.isInt64 = tensor.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    }
    return info;
}
} // namespace

juce::File appDataDir()
{
    if (const auto overridden = juce::SystemStats::getEnvironmentVariable ("VAE_DATA_DIR", {}); overridden.isNotEmpty())
        return juce::File (overridden); // tests and alternate installs
#if JUCE_MAC
    const char* name = "Application Support/VAE Synth"; // JUCE's app data dir is ~/Library on macOS
#else
    const char* name = "VAE Synth";
#endif
    return juce::File::getSpecialLocation (juce::File::userApplicationDataDirectory).getChildFile (name);
}

bool isSha256Hex (const juce::String& s) { return s.length() == 64 && s.containsOnly ("0123456789abcdef"); }

OnnxModel::OnnxModel() = default;
OnnxModel::~OnnxModel() = default;

bool OnnxModel::isLoaded() const noexcept { return session != nullptr && session->session != nullptr; }

void OnnxModel::unload()
{
    session.reset();
    inputs.clear();
    outputs.clear();
}

template <typename MakeSession>
juce::Result OnnxModel::loadWith (MakeSession&& makeSession, int threads, const juce::File& saveOptimizedTo,
                                  bool preOptimized, bool int64Inputs)
{
    unload();
    if (! runtime->isAvailable())
        return juce::Result::fail (runtime->getStatus());

    try
    {
        session = std::make_unique<Session>();
        session->options = makeOptions (threads);
        if (preOptimized)
            session->options.SetGraphOptimizationLevel (GraphOptimizationLevel::ORT_DISABLE_ALL);
        if (saveOptimizedTo != juce::File())
        {
            // Weights go to a sidecar so graphs over protobuf's 2 GB limit still save.
            session->options.SetOptimizedModelFilePath (
#if JUCE_WINDOWS
                saveOptimizedTo.getFullPathName().toWideCharPointer());
#else
                saveOptimizedTo.getFullPathName().toRawUTF8());
#endif
            session->options.AddConfigEntry ("session.optimized_model_external_initializers_file_name",
                                             (saveOptimizedTo.getFileNameWithoutExtension() + ".data").toRawUTF8());
            session->options.AddConfigEntry ("session.optimized_model_external_initializers_min_size_in_bytes", "1024");
        }
        session->session = makeSession (runtime->getEnv(), session->options);
    }
    catch (const Ort::Exception& e)
    {
        unload();
        return juce::Result::fail (juce::String ("Could not load model: ") + e.what());
    }
    return finishLoad (int64Inputs);
}

juce::Result OnnxModel::load (const juce::File& file, ModelLoadOptions options)
{
    if (! file.existsAsFile())
    {
        unload();
        return juce::Result::fail ("Model not found: " + file.getFullPathName());
    }
    if (! isSha256Hex (options.cacheKey) || ! runtime->isAvailable())
        return loadWith (fromPath (file), options.threads, {}, false, options.int64Inputs);

    // Content-addressed, so a changed model or runtime never picks up a stale graph.
    const auto cached =
        appDataDir().getChildFile ("cache/optimized/" + options.cacheKey + "-ort" + runtime->getVersion() + ".onnx");
    if (cached.existsAsFile())
    {
        if (loadWith (fromPath (cached), options.threads, {}, true, options.int64Inputs).wasOk())
            return juce::Result::ok();
        cached.deleteFile(); // unreadable: rebuild it below
    }

    // ORT writes while creating the session; publish under the real name only once that succeeded.
    const auto partial = cached.getSiblingFile (cached.getFileNameWithoutExtension() + ".partial");
    if (cached.getParentDirectory().createDirectory().wasOk()
        && loadWith (fromPath (file), options.threads, partial, false, options.int64Inputs).wasOk()
        && partial.moveFileTo (cached))
        return juce::Result::ok();
    return loadWith (fromPath (file), options.threads, {}, false, options.int64Inputs); // caching failed: plain load
}

juce::Result OnnxModel::load (const void* data, size_t size)
{
    return loadWith ([data, size] (Ort::Env& env, const Ort::SessionOptions& options)
                     { return std::make_unique<Ort::Session> (env, data, size, options); });
}

juce::Result OnnxModel::finishLoad (bool int64Inputs)
{
    Ort::AllocatorWithDefaultOptions allocator;
    auto& s = *session->session;

    for (size_t i = 0; i < s.GetInputCount(); ++i)
    {
        auto name = std::string (s.GetInputNameAllocated (i, allocator).get());
        inputs.push_back (describe (s.GetInputTypeInfo (i), name));
        session->inputNames.push_back (std::move (name));
    }
    for (size_t i = 0; i < s.GetOutputCount(); ++i)
    {
        auto name = std::string (s.GetOutputNameAllocated (i, allocator).get());
        outputs.push_back (describe (s.GetOutputTypeInfo (i), name));
        session->outputNames.push_back (std::move (name));
    }

    if (inputs.empty()
        || std::any_of (inputs.begin(), inputs.end(),
                        [int64Inputs] (const auto& i) { return ! i.isFloat && ! (int64Inputs && i.isInt64); }))
    {
        unload();
        return juce::Result::fail ("Expected float input tensors");
    }
    return juce::Result::ok();
}

juce::Result OnnxModel::run (std::span<const float> input, std::span<const int64_t> inputShape,
                             std::vector<std::vector<float>>& outputsOut)
{
    const FloatTensorView view { input, inputShape };
    return run (std::span (&view, 1), outputsOut);
}

juce::Result OnnxModel::run (std::span<const FloatTensorView> tensors, std::vector<std::vector<float>>& outputsOut)
{
    return runViews (tensors, outputsOut);
}

juce::Result OnnxModel::run (std::span<const Int64TensorView> tensors, std::vector<std::vector<float>>& outputsOut)
{
    return runViews (tensors, outputsOut);
}

template <typename View>
juce::Result OnnxModel::runViews (std::span<const View> tensors, std::vector<std::vector<float>>& outputsOut)
{
    using Element = std::remove_const_t<typename decltype (View::data)::element_type>;
    if (! isLoaded())
        return juce::Result::fail ("No model loaded");
    if (tensors.size() != inputs.size())
        return juce::Result::fail ("Input tensor count mismatch");

    try
    {
        const auto memory = Ort::MemoryInfo::CreateCpu (OrtArenaAllocator, OrtMemTypeDefault);
        // ORT takes a non-const pointer but does not write to inputs.
        std::vector<Ort::Value> values;
        for (const auto& tensor : tensors)
        {
            size_t count = 1;
            for (auto dim : tensor.shape)
            {
                if (dim <= 0 || size_t (dim) > tensor.data.size() / count)
                    return juce::Result::fail ("Invalid input shape");
                count *= size_t (dim);
            }
            if (count != tensor.data.size())
                return juce::Result::fail ("Input size mismatch");
            values.push_back (Ort::Value::CreateTensor<Element> (memory, const_cast<Element*> (tensor.data.data()),
                                                                 tensor.data.size(), tensor.shape.data(),
                                                                 tensor.shape.size()));
        }

        std::vector<const char*> inNames, outNames;
        for (auto& n : session->inputNames)
            inNames.push_back (n.c_str());
        for (auto& n : session->outputNames)
            outNames.push_back (n.c_str());

        auto results = session->session->Run (Ort::RunOptions { nullptr }, inNames.data(), values.data(), values.size(),
                                              outNames.data(), outNames.size());

        outputsOut.clear();
        for (auto& value : results)
        {
            auto& out = outputsOut.emplace_back();
            if (! value.IsTensor())
                continue;
            const auto info = value.GetTensorTypeAndShapeInfo();
            if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
                continue;
            const auto* data = value.GetTensorData<float>();
            out.assign (data, data + info.GetElementCount());
        }
    }
    catch (const Ort::Exception& e)
    {
        return juce::Result::fail (juce::String ("Inference failed: ") + e.what());
    }
    return juce::Result::ok();
}

} // namespace vae::ml
