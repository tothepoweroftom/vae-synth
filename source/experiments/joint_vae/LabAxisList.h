#pragma once
#include "PitcherControls.h"
#include "XyPad.h"
#include <array>
#include <juce_gui_basics/juce_gui_basics.h>

namespace vae::experimental
{
/** The Lab's flat axis list: one study AxisRow per axis (sound axes, then latent dimensions, in bundle
    order), with X / Y / Z buttons and, on assigned rows, a low-high slider. Lives in a Viewport. */
class LabAxisList final : public juce::Component
{
public:
    /** `axes` must outlive the list (the panel keeps the bundle's XyPadData). */
    void setAxes (const std::vector<LatentAxis>& axes);
    /** The knob pad's rows: one per group in file order; ids are group names. */
    void setGroups (const KnobGroups&);
    /** Ids on X, Y, Z (Z may be empty). */
    void setAssigned (const std::array<juce::String, 3>&);
    /** Slider positions from z, except the row being dragged. */
    void setPositions (const JointLatent&, const LatentAxis* except = nullptr);
    /** The knob pad: the assigned rows' sliders read the pad's X, Y, Z positions. */
    void setSlotPositions (const std::array<float, 3>&);
    /** Stacks the rows for `width` and sets this component's height. */
    void layOut (int width);

    std::function<void (int slot /* 0 x, 1 y, 2 z */, const juce::String& id)> assign;
    std::function<void (const LatentAxis&, float pos)> slid;

private:
    void update (size_t row);
    const std::vector<LatentAxis>* axes = nullptr;
    std::vector<LatentAxis> groupAxes; // what the group rows point at
    juce::OwnedArray<AxisRow> rows;
    std::vector<float> values;
    std::array<int, 3> assigned { -1, -1, -1 }; // row index on X, Y, Z
};
} // namespace vae::experimental
