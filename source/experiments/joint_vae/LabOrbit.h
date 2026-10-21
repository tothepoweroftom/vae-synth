#pragma once
#include "OrbitView.h"
#include <juce_gui_basics/juce_gui_basics.h>
#include <vector>

namespace vae::experimental
{
/** The Lab's latent view: the design study's OrbitView (positions 0..1 on the X, Y, Z axes), with the
    camera's - / + / Reset view buttons, Find's numbered highlights, and a click on a highlight or a
    preset dot. Arrow keys move the handle in X and Y, Page Up / Down in Z (0.02; Shift: 0.1).
    Only handle drags and those keys call `moved`: orbit, zoom and Reset view never do. */
class LabOrbit final
    : public juce::Component
    , private juce::KeyListener
{
public:
    using Position = std::array<float, 3>;
    LabOrbit();
    ~LabOrbit() override;
    std::function<void (Position)> moved; // a handle drag or key
    std::function<void (int)> picked;     // a click on Find highlight i
    std::function<void (int)> jumped;     // a click on preset i, one of the active label's dots
    void setHandle (Position p) { view.setPosition (p); }
    Position getHandle() const { return view.getPosition(); }
    /** Every preset at its X, Y, Z position, with its label; clicks jump to `activeLabel`'s dots. */
    void setPresets (std::vector<Position>, std::vector<int> labels, int activeLabel);
    /** Find's candidates, numbered 1..; `selected` is drawn filled. */
    void setHighlights (std::vector<Position>, int selected);
    const std::vector<Position>& getHighlights() const { return highlights; }
    void setCategoryHighlight (int category) { view.setHighlight (category); } // visual only
    void setLabelFont (juce::Font font) { view.setLabelFont (std::move (font)); }
    juce::Point<float> project (Position) const; // local pixels, at the current camera

    void resized() override;
    void paintOverChildren (juce::Graphics&) override;
    void mouseDown (const juce::MouseEvent&) override; // also the view's presses (a mouse listener)
    void mouseUp (const juce::MouseEvent&) override;
    bool keyPressed (const juce::KeyPress&) override;

private:
    bool keyPressed (const juce::KeyPress&, juce::Component*) override; // the view's keys, before its own
    OrbitView view;
    juce::TextButton zoomOut { "-" }, zoomIn { "+" }, resetView { "Reset view" };
    std::vector<Position> presets, highlights;
    std::vector<int> labels;
    int activeLabel = -1, selectedHighlight = -1;
    bool pressOnHandle = false;
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (LabOrbit)
};
} // namespace vae::experimental
