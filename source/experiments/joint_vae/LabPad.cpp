#include "LabPad.h"

namespace vae::experimental
{
LabPad::LabPad()
{
    setWantsKeyboardFocus (true);
    setMouseCursor (juce::MouseCursor::CrosshairCursor);
}
void LabPad::setDot (juce::Point<float> p)
{
    dot = { juce::jlimit (0.0f, 1.0f, p.x), juce::jlimit (0.0f, 1.0f, p.y) };
    repaint();
}
void LabPad::setMap (juce::Image image, std::vector<std::pair<juce::String, juce::Point<float>>> labels)
{
    map = std::move (image);
    mapLabels = std::move (labels);
    repaint();
}
void LabPad::paint (juce::Graphics& g)
{
    const auto r = getLocalBounds().toFloat();
    g.setColour (lab::inset);
    g.fillRect (r);
    if (map.isValid())
    {
        g.setImageResamplingQuality (juce::Graphics::mediumResamplingQuality);
        g.drawImage (map, r, juce::RectanglePlacement::stretchToFit);
    }
    g.setColour (lab::ink.withAlpha (0.75f));
    g.setFont (juce::FontOptions (12.0f, juce::Font::bold));
    for (const auto& [text, p] : mapLabels)
        g.drawText (text,
                    juce::Rectangle<float> (120, 16).withCentre (
                        { r.getX() + p.x * r.getWidth(), r.getBottom() - p.y * r.getHeight() }),
                    juce::Justification::centred);
    g.setColour (lab::line);
    for (int i = 1; i < 4; ++i)
    {
        g.drawVerticalLine ((int) (r.getX() + r.getWidth() * (float) i / 4), r.getY(), r.getBottom());
        g.drawHorizontalLine ((int) (r.getY() + r.getHeight() * (float) i / 4), r.getX(), r.getRight());
    }
    // The handle as Orbit draws it: a green ring with a centre dot
    const juce::Point<float> c { r.getX() + dot.x * r.getWidth(), r.getBottom() - dot.y * r.getHeight() };
    g.setColour (lab::inset);
    g.fillEllipse (juce::Rectangle<float> (16, 16).withCentre (c));
    g.setColour (lab::green);
    g.drawEllipse (juce::Rectangle<float> (16, 16).withCentre (c), 2);
    g.fillEllipse (juce::Rectangle<float> (5, 5).withCentre (c));
    g.setColour (hasKeyboardFocus (false) ? lab::green : lab::line);
    g.drawRect (r, 1.0f);
}
void LabPad::move (juce::Point<float> p)
{
    setDot (p);
    if (moved)
        moved (dot);
}
void LabPad::mouseDrag (const juce::MouseEvent& e)
{
    grabKeyboardFocus();
    const auto r = getLocalBounds().toFloat();
    if (! r.isEmpty())
        move ({ (e.position.x - r.getX()) / r.getWidth(), (r.getBottom() - e.position.y) / r.getHeight() });
}
bool LabPad::keyPressed (const juce::KeyPress& key)
{
    const float d = key.getModifiers().isShiftDown() ? 0.1f : 0.02f;
    const auto code = key.getKeyCode();
    juce::Point<float> step;
    if (code == juce::KeyPress::leftKey)
        step = { -d, 0 };
    else if (code == juce::KeyPress::rightKey)
        step = { d, 0 };
    else if (code == juce::KeyPress::upKey)
        step = { 0, d };
    else if (code == juce::KeyPress::downKey)
        step = { 0, -d };
    else
        return false;
    move (dot + step);
    return true;
}
} // namespace vae::experimental
