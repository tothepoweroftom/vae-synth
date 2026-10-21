#pragma once
#include "XyPad.h"
#include <array>
#include <juce_gui_basics/juce_gui_basics.h>

namespace vae::experimental
{
/** The Lab's axis list (page drawLatList): "Sound axes", "Latent dimensions" and the minor dims behind a toggle.
    Each row: name, X/Y/Z buttons, description, tags and its own slider. Lives in a Viewport. */
class LabAxisList final : public juce::Component
{
public:
    /** `axes` must outlive the list (the panel keeps the bundle's XyPadData). */
    void setAxes (const std::vector<LatentAxis>& axes);
    /** The knob pad's rows: one per group in file order, ids are group names, sliders hidden. */
    void setGroups (const KnobGroups&);
    /** Ids on X, Y, Z (Z may be empty). */
    void setAssigned (const std::array<juce::String, 3>&);
    /** Slider positions from z, except the row being dragged. */
    void setPositions (const JointLatent&, const LatentAxis* except = nullptr);
    /** Lays the rows out for `width` and sets this component's height. */
    void layOut (int width);
    void paint (juce::Graphics&) override;

    std::function<void (int slot /* 0 x, 1 y, 2 z */, const juce::String& id)> assign;
    std::function<void (const LatentAxis&, float pos)> slid;

private:
    struct Row final : juce::Component
    {
        Row (const LatentAxis&, int rank, float maxImpact);
        const LatentAxis& axis;
        const int rank;
        const float maxImpact;
        juce::TextButton slot[3] { juce::TextButton ("X"), juce::TextButton ("Y"), juce::TextButton ("Z") };
        juce::Slider slider;
        int heightFor (int width) const;
        void paint (juce::Graphics&) override;
        void resized() override;
    };
    juce::OwnedArray<Row> rows;
    std::vector<LatentAxis> groupAxes; // what the group rows point at
    juce::TextButton moreButton;
    bool showMinor = false;
    int soundHeaderY = -1, dimHeaderY = -1;
};
} // namespace vae::experimental
