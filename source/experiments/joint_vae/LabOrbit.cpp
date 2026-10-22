#include "LabOrbit.h"
#include "LabTheme.h"
#include "PitcherControls.h"

namespace vae::experimental
{
namespace
{
constexpr float kHitRadius = 7.0f;     // a dot click, as the old pad's
constexpr float kHandleRadius = 14.0f; // OrbitView's handle hit radius

LabOrbit::Position clamped (LabOrbit::Position p)
{
    for (auto& v : p)
        v = juce::jlimit (0.0f, 1.0f, v);
    return p;
}
bool inside (const LabOrbit::Position& p)
{
    return std::all_of (p.begin(), p.end(), [] (float v) { return std::isfinite (v) && v >= 0.0f && v <= 1.0f; });
}
} // namespace

LabOrbit::LabOrbit()
{
    addAndMakeVisible (view);
    view.setDotScale (kDotScale);
    view.addMouseListener (this, false);
    view.addKeyListener (this);
    // The study's description says arrows rotate the camera; in the Lab they move the handle (keyPressed)
    view.setDescription ("Drag the handle to move the sound; Shift drag moves it in Z. Drag empty space to orbit the "
                         "camera, scroll to zoom. Arrow keys move the handle in X and Y, Page Up and Down in Z; "
                         "Shift takes bigger steps. Home resets the camera. The Orbit LFO in the left sidebar "
                         "circles the handle slowly around the current sound.");
    view.onPositionChanged = [this] (Position p)
    {
        if (moved)
            moved (p);
    };
    for (auto* b : { &zoomOut, &zoomIn, &resetView })
        addAndMakeVisible (b);
    zoomOut.setTooltip ("Zoom out");
    zoomIn.setTooltip ("Zoom in");
    resetView.setTooltip ("Reset the camera; the sound stays");
    zoomOut.onClick = [this] { view.zoom (1.0f / 1.15f); };
    zoomIn.onClick = [this] { view.zoom (1.15f); };
    resetView.onClick = [this] { view.resetCamera(); };
}

LabOrbit::~LabOrbit()
{
    view.removeKeyListener (this);
    view.removeMouseListener (this);
}

void LabOrbit::setPresets (std::vector<Position> p, std::vector<int> l, int active)
{
    std::vector<PresetPoint> points;
    points.reserve (p.size());
    for (size_t i = 0; i < p.size() && i < l.size(); ++i)
        points.push_back ({ p[i], l[i] });
    view.setPoints (std::move (points)); // drops points outside the volume, as the study does
    presets = std::move (p);
    labels = std::move (l);
    activeLabel = active;
}

void LabOrbit::setOrbit (std::vector<Position> path, Position centre)
{
    orbitPath = std::move (path);
    orbitCentre = centre;
    repaint();
}

void LabOrbit::setHighlights (std::vector<Position> h, int selected)
{
    highlights = std::move (h);
    selectedHighlight = selected;
    repaint();
}

juce::Point<float> LabOrbit::project (Position p) const
{
    const auto s =
        OrbitMath::Projection (view.getCamera(), (float) view.getWidth(), (float) view.getHeight()).project (p);
    return { s.x, s.y }; // the view fills this component
}

void LabOrbit::resized()
{
    view.setBounds (getLocalBounds());
    auto row = getLocalBounds().reduced (12).removeFromTop (27);
    resetView.setBounds (row.removeFromRight (74));
    row.removeFromRight (5);
    zoomIn.setBounds (row.removeFromRight (27));
    row.removeFromRight (5);
    zoomOut.setBounds (row.removeFromRight (27));
}

void LabOrbit::paintOverChildren (juce::Graphics& g)
{
    if (! orbitPath.empty())
    {
        juce::Path loop, dashed;
        loop.startNewSubPath (project (orbitPath[0]));
        for (size_t i = 1; i < orbitPath.size(); ++i)
            loop.lineTo (project (orbitPath[i]));
        loop.closeSubPath();
        const float dashes[] { 4.0f, 3.0f };
        juce::PathStrokeType (1.5f).createDashedStroke (dashed, loop, dashes, 2);
        const auto ring = juce::Rectangle<float> (12, 12).withCentre (project (orbitCentre));
        g.setColour (lab::workspace.withAlpha (0.8f)); // a dark underlay keeps both readable over the dots
        g.strokePath (loop, juce::PathStrokeType (4.0f));
        g.drawEllipse (ring, 4.0f);
        g.setColour (lab::green);
        g.fillPath (dashed);
        g.drawEllipse (ring, 1.5f);
    }
    // Find's candidates, numbered as the chips
    g.setFont (PitcherLookAndFeel::monoFontFor (*this, 9.0f));
    for (size_t i = 0; i < highlights.size(); ++i)
    {
        const auto r = juce::Rectangle<float> (15, 15).withCentre (project (clamped (highlights[i])));
        const bool on = (int) i == selectedHighlight;
        g.setColour (on ? lab::green : lab::workspace.withAlpha (0.85f));
        g.fillEllipse (r);
        g.setColour (lab::green);
        g.drawEllipse (r, 1.5f);
        g.setColour (on ? lab::onGreen : lab::ink);
        g.drawText (juce::String ((int) i + 1), r, juce::Justification::centred);
    }
}

void LabOrbit::mouseDown (const juce::MouseEvent& e)
{
    pressOnHandle = e.getEventRelativeTo (this).position.getDistanceFrom (project (getHandle())) <= kHandleRadius;
}

void LabOrbit::mouseUp (const juce::MouseEvent& e)
{
    if (e.mouseWasDraggedSinceMouseDown())
        return; // an orbit or a handle drag
    const auto at = e.getEventRelativeTo (this).position;
    const auto nearest = [&] (const std::vector<Position>& points, auto&& usable)
    {
        float best = kHitRadius * kHitRadius;
        int hit = -1;
        for (size_t i = 0; i < points.size(); ++i)
            if (usable (i))
                if (const auto d = at.getDistanceSquaredFrom (project (clamped (points[i]))); d < best)
                {
                    best = d;
                    hit = (int) i;
                }
        return hit;
    };
    if (const auto i = nearest (highlights, [] (size_t) { return true; }); i >= 0)
    {
        if (picked)
            picked (i);
        return;
    }
    if (pressOnHandle || ! jumped)
        return; // a click on the handle selects it (the view shows its arrows)
    if (const auto i = nearest (presets, [this] (size_t k) { return labels[k] == activeLabel && inside (presets[k]); });
        i >= 0)
        jumped (i);
}

bool LabOrbit::keyPressed (const juce::KeyPress& key)
{
    const float d = key.getModifiers().isShiftDown() ? 0.1f : 0.02f;
    const auto code = key.getKeyCode();
    size_t axis = 0;
    float step = 0;
    if (code == juce::KeyPress::leftKey || code == juce::KeyPress::rightKey)
        step = code == juce::KeyPress::rightKey ? d : -d;
    else if (code == juce::KeyPress::upKey || code == juce::KeyPress::downKey)
    {
        axis = 1;
        step = code == juce::KeyPress::upKey ? d : -d;
    }
    else if (code == juce::KeyPress::pageUpKey || code == juce::KeyPress::pageDownKey)
    {
        axis = 2;
        step = code == juce::KeyPress::pageUpKey ? d : -d;
    }
    else
        return false;
    auto p = getHandle();
    p[axis] = juce::jlimit (0.0f, 1.0f, p[axis] + step);
    setHandle (p);
    if (moved)
        moved (p);
    return true;
}

bool LabOrbit::keyPressed (const juce::KeyPress& key, juce::Component*) { return keyPressed (key); }
} // namespace vae::experimental
