#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include "OrbitMath.h"
#include <functional>
#include <vector>

struct PresetPoint
{
    std::array<float, 3> position;
    int category;
};

// A message-thread-only visual prototype. This component never accesses audio
// or inference state. Callbacks hand normalised coordinates to its owner.
class OrbitView final : public juce::Component
{
public:
    OrbitView();
    void setPoints(std::vector<PresetPoint>);
    void setHighlight(int category);
    void setPosition(std::array<float, 3>);
    std::array<float, 3> getPosition() const { return position; }
    void resetCamera();
    OrbitMath::Camera getCamera() const { return camera; } // hosts project their own overlays and hit tests
    void setLabelFont(juce::Font font) { labelFont=std::move(font);repaint(); }
    void zoom(float factor);
    double lastPaintMs() const { return paintMilliseconds; }
    std::function<void(std::array<float, 3>)> onPositionChanged;

    void paint(juce::Graphics&) override;
    void mouseDown(const juce::MouseEvent&) override;
    void mouseDrag(const juce::MouseEvent&) override;
    void mouseUp(const juce::MouseEvent&) override;
    void mouseWheelMove(const juce::MouseEvent&, const juce::MouseWheelDetails&) override;
    bool keyPressed(const juce::KeyPress&) override;

private:
    juce::Font labelFont {juce::FontOptions(11.0f)};
    struct Dot { OrbitMath::Projected screen; int category; };
    struct Arrow { int axis; juce::Line<float> line; };
    OrbitMath::Projection projection() const;
    std::vector<Arrow> arrows() const;
    void moveTo(OrbitMath::Position);
    std::vector<PresetPoint> points;
    std::vector<Dot> projected;
    OrbitMath::Position position {.5f, .5f, .5f}, dragStart {};
    OrbitMath::Camera camera, dragCamera;
    OrbitMath::Drag dragMode = OrbitMath::Drag::plane;
    juce::Point<float> pointerStart;
    int highlightedCategory = -1;
    bool selected = false, rotating = false, dragging = false;
    double paintMilliseconds = 0;
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(OrbitView)
};
