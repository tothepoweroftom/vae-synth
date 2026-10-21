#pragma once
#include <juce_gui_basics/juce_gui_basics.h>
#include <array>
#include <vector>

struct StudyAxis
{
    juce::String name, low, high;
    std::array<float, 32> direction {};
    float mean = 0, sd = 1;
};

// Message-thread-only study model. No audio processing or parameter decoding.
class StudyData
{
public:
    std::vector<StudyAxis> axes;
    std::vector<std::array<float, 32>> presets;
    std::vector<int> labels;
    std::vector<juce::String> names;
    std::array<float, 32> initialLatent {};
    int initialCategory = 0;
    bool load (const void*, size_t);
    std::array<float, 3> project (const std::array<float, 32>&,
                                std::array<int, 3> assigned) const;
    void moveTo (std::array<float, 32>&, std::array<int, 3> assigned,
                 std::array<float, 3> target) const;
    juce::Image spectrogram (const std::array<float, 32>&, int category);
    juce::String soundTooltip() const { return tooltip; }
private:
    struct Layer { int inputs = 0, outputs = 0; std::vector<float> weights, biases; };
    std::vector<Layer> decoder;
    std::vector<float> components, mean, bandHz;
    int frames = 0, categories = 0, holdFrames = 0;
    float scale = 1, dt = 0, f0 = 1;
    juce::String tooltip;
};
