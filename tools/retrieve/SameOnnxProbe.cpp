// Fixed-shape encoder feasibility probe using the plug-in's actual runtime wrapper.
#include "ml/OnnxModel.h"
#include <iostream>

int main (int argc, char** argv)
{
    if (argc != 4)
    {
        std::cerr << "Usage: vae_same_onnx_probe encoder.onnx input-float32.raw output-float32.raw\n";
        return 1;
    }
    auto path = [] (const char* p) { return juce::File::getCurrentWorkingDirectory().getChildFile (p); };
    vae::ml::OnnxModel model;
    auto result = model.load (path (argv[1]));
    if (result.failed())
    {
        std::cerr << result.getErrorMessage() << '\n';
        return 1;
    }
    juce::MemoryBlock input;
    constexpr size_t count = 2 * 176400;
    if (! path (argv[2]).loadFileAsData (input) || input.getSize() != count * sizeof (float))
    {
        std::cerr << "Expected [1,2,176400] float32 audio\n";
        return 1;
    }
    std::vector<float> audio (count);
    std::memcpy (audio.data(), input.getData(), input.getSize());
    const std::array<int64_t, 3> shape { 1, 2, 176400 };
    std::vector<std::vector<float>> outputs;
    result = model.run (audio, shape, outputs);
    if (result.failed() || outputs.size() != 1 || outputs[0].size() != 256 * 44)
    {
        std::cerr << "Encoder failed: " << result.getErrorMessage() << '\n';
        return 1;
    }
    if (path (argv[3]).exists()
        || ! path (argv[3]).replaceWithData (outputs[0].data(), outputs[0].size() * sizeof (float)))
    {
        std::cerr << "Choose a new, writable output path\n";
        return 1;
    }
    std::cout << "C++ encoder succeeded: [1,256,44], " << model.getRuntimeStatus() << '\n';
    return 0;
}
