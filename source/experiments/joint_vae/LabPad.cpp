#include "LabPad.h"
#include "plugin/ui/VaeLookAndFeel.h"

namespace vae::experimental
{
using LF = VaeLookAndFeel;

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
void LabPad::setPresetDots (std::vector<std::tuple<float, float, int>> d, int label)
{
    dots = std::move (d);
    activeLabel = label;
    repaint();
}
void LabPad::setHighlights (std::vector<juce::Point<float>> h, int selected)
{
    highlights = std::move (h);
    selectedHighlight = selected;
    repaint();
}
void LabPad::paint (juce::Graphics& g)
{
    const auto r = getLocalBounds().toFloat();
    g.setColour (labWell);
    g.fillRoundedRectangle (r, 6);
    if (map.isValid())
    {
        g.setImageResamplingQuality (juce::Graphics::mediumResamplingQuality);
        g.drawImage (map, r, juce::RectanglePlacement::stretchToFit);
    }
    g.setColour (LF::ink.withAlpha (0.75f));
    g.setFont (LF::labelFont (12.0f, true));
    for (const auto& [text, p] : mapLabels)
        g.drawText (text,
                    juce::Rectangle<float> (120, 16).withCentre (
                        { r.getX() + p.x * r.getWidth(), r.getBottom() - p.y * r.getHeight() }),
                    juce::Justification::centred);
    g.setColour (LF::faint);
    for (int i = 1; i < 4; ++i)
    {
        g.drawVerticalLine ((int) (r.getX() + r.getWidth() * (float) i / 4), r.getY(), r.getBottom());
        g.drawHorizontalLine ((int) (r.getY() + r.getHeight() * (float) i / 4), r.getX(), r.getRight());
    }
    for (const auto& [x, y, label] : dots)
    {
        if (x < -0.02f || x > 1.02f || y < -0.02f || y > 1.02f)
            continue;
        const bool same = label == activeLabel;
        const auto colour = labCategoryColours[juce::jlimit (0, 8, label)];
        g.setColour (colour.withAlpha (same ? 0.85f : 0.28f));
        const float radius = same ? 2.6f : 1.8f;
        const auto d = juce::Rectangle<float> (2 * radius, 2 * radius)
                           .withCentre ({ r.getX() + x * r.getWidth(), r.getBottom() - y * r.getHeight() });
        g.fillEllipse (d);
        if (same)
        {
            g.setColour (LF::ink.withAlpha (0.35f));
            g.drawEllipse (d, 0.8f);
        }
    }
    g.setFont (LF::labelFont (9.0f, true));
    for (size_t i = 0; i < highlights.size(); ++i)
    {
        const auto h = juce::Rectangle<float> (14, 14).withCentre (
            { r.getX() + highlights[i].x * r.getWidth(), r.getBottom() - highlights[i].y * r.getHeight() });
        const bool on = (int) i == selectedHighlight;
        g.setColour (on ? LF::accent : labWell.withAlpha (0.8f));
        g.fillEllipse (h);
        g.setColour (LF::accent);
        g.drawEllipse (h, 1.5f);
        g.setColour (on ? labWell : LF::ink);
        g.drawText (juce::String ((int) i + 1), h, juce::Justification::centred);
    }
    const juce::Point<float> c { r.getX() + dot.x * r.getWidth(), r.getBottom() - dot.y * r.getHeight() };
    g.setColour (labWell);
    g.fillEllipse (juce::Rectangle<float> (18, 18).withCentre (c));
    g.setColour (LF::ink);
    g.drawEllipse (juce::Rectangle<float> (18, 18).withCentre (c), 2);
    g.fillEllipse (juce::Rectangle<float> (5, 5).withCentre (c));
    g.setColour (hasKeyboardFocus (false) ? LF::accent : LF::faint);
    g.drawRoundedRectangle (r.reduced (0.5f), 6, 1);
}
juce::Point<float> LabPad::toPad (juce::Point<float> pixel) const
{
    const auto r = getLocalBounds().toFloat();
    if (r.isEmpty())
        return dot;
    return { juce::jlimit (0.0f, 1.0f, (pixel.x - r.getX()) / r.getWidth()),
             juce::jlimit (0.0f, 1.0f, (r.getBottom() - pixel.y) / r.getHeight()) };
}
void LabPad::move (juce::Point<float> p)
{
    setDot (p);
    if (moved)
        moved (dot);
}
int LabPad::highlightAt (juce::Point<float> pixel) const
{
    const auto r = getLocalBounds().toFloat();
    float best = 7.0f * 7.0f;
    int hit = -1;
    for (size_t i = 0; i < highlights.size(); ++i)
    {
        const auto d = pixel.getDistanceSquaredFrom (
            { r.getX() + highlights[i].x * r.getWidth(), r.getBottom() - highlights[i].y * r.getHeight() });
        if (d < best)
        {
            best = d;
            hit = (int) i;
        }
    }
    return hit;
}
void LabPad::mouseDown (const juce::MouseEvent& e)
{
    grabKeyboardFocus();
    downOn = picked ? highlightAt (e.position) : -1;
    if (downOn < 0)
        move (toPad (e.position));
}
void LabPad::mouseDrag (const juce::MouseEvent& e)
{
    if (downOn >= 0 && e.getDistanceFromDragStart() < 3)
        return;
    downOn = -1; // a drag that started on a highlight explores like any other
    move (toPad (e.position));
}
void LabPad::mouseUp (const juce::MouseEvent& e)
{
    if (e.getDistanceFromDragStart() >= 3)
        return;
    if (const auto i = std::exchange (downOn, -1); i >= 0)
        picked (i);
    else if (tapped)
        tapped (toPad (e.position));
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
