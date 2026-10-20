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
Ort::SessionOptions makeOptions()
{
    Ort::SessionOptions options;
    // Small models, called rarely, from a worker thread next to a busy audio
    // thread: keep ORT to one thread so it never competes with the DAW.
    options.SetIntraOpNumThreads (1);
    options.SetInterOpNumThreads (1);
    options.SetGraphOptimizationLevel (GraphOptimizationLevel::ORT_ENABLE_ALL);
    return options;
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
    }
    return info;
}
} // namespace

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
juce::Result OnnxModel::loadWith (MakeSession&& makeSession)
{
    unload();
    if (! runtime->isAvailable())
        return juce::Result::fail (runtime->getStatus());

    try
    {
        session = std::make_unique<Session>();
        session->options = makeOptions();
        session->session = makeSession (runtime->getEnv(), session->options);
    }
    catch (const Ort::Exception& e)
    {
        unload();
        return juce::Result::fail (juce::String ("Could not load model: ") + e.what());
    }
    return finishLoad();
}

juce::Result OnnxModel::load (const juce::File& file)
{
    if (! file.existsAsFile())
    {
        unload();
        return juce::Result::fail ("Model not found: " + file.getFullPathName());
    }

    return loadWith (
        [&file] (Ort::Env& env, const Ort::SessionOptions& options)
        {
#if JUCE_WINDOWS
            return std::make_unique<Ort::Session> (env, file.getFullPathName().toWideCharPointer(), options);
#else
            return std::make_unique<Ort::Session> (env, file.getFullPathName().toRawUTF8(), options);
#endif
        });
}

juce::Result OnnxModel::load (const void* data, size_t size)
{
    return loadWith ([data, size] (Ort::Env& env, const Ort::SessionOptions& options)
                     { return std::make_unique<Ort::Session> (env, data, size, options); });
}

juce::Result OnnxModel::finishLoad()
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

    if (inputs.empty() || std::any_of (inputs.begin(), inputs.end(), [] (const auto& i) { return ! i.isFloat; }))
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
            values.push_back (Ort::Value::CreateTensor<float> (memory, const_cast<float*> (tensor.data.data()),
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
