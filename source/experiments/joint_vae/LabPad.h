#pragma once
#include "LabTheme.h"
#include <juce_gui_basics/juce_gui_basics.h>
#include <vector>

namespace vae::experimental
{
/** The knob pad: a flat XY pad over two knob groups, positions 0..1 with y up, over its category map.
    Arrow keys move the dot by 0.02 (Shift: 0.1). The latent pad is LabOrbit. */
class LabPad final : public juce::Component
{
public:
    LabPad();
    std::function<void (juce::Point<float>)> moved; // drag, click or arrow key
    void setDot (juce::Point<float>);
    /** `labels`: category names at their region centroids (0..1, y up). */
    void setMap (juce::Image, std::vector<std::pair<juce::String, juce::Point<float>>> labels = {});
    juce::Point<float> getDot() const { return dot; }
    const juce::Image& getMap() const { return map; }

    void paint (juce::Graphics&) override;
    void mouseDown (const juce::MouseEvent& e) override { mouseDrag (e); }
    void mouseDrag (const juce::MouseEvent&) override;
    bool keyPressed (const juce::KeyPress&) override;
    void focusGained (FocusChangeType) override { repaint(); }
    void focusLost (FocusChangeType) override { repaint(); }

private:
    void move (juce::Point<float>);
    juce::Point<float> dot { 0.5f, 0.5f };
    juce::Image map;
    std::vector<std::pair<juce::String, juce::Point<float>>> mapLabels;
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (LabPad)
};
} // namespace vae::experimental
