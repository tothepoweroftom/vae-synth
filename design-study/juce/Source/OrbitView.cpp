#include "OrbitView.h"

namespace
{
const juce::Colour background(0xff252525), accent(0xff70ff63), ink(0xffdcded8), wire(0xff4b4e49);
const std::array<juce::Colour, 8> categoryColours {
    juce::Colour(0xff70bcff), juce::Colour(0xffffad66), juce::Colour(0xff86eeac), juce::Colour(0xffffe078),
    juce::Colour(0xffff8fc7), juce::Colour(0xff63e0d5), juce::Colour(0xffb6a0ff), juce::Colour(0xffff918c)
};
juce::Point<float> xy(OrbitMath::Projected p) { return {p.x, p.y}; }
}

OrbitView::OrbitView()
{
    setWantsKeyboardFocus(true);
    setTitle("Preset space");
    setDescription("Drag the handle to move; Shift drag changes depth. Drag empty space to orbit. "
                   "Select the handle for X Y Z arrows. Scroll to zoom. Arrow keys rotate; Page Up and Down change depth; Home resets camera.");
    setMouseCursor(juce::MouseCursor::CrosshairCursor);
}

void OrbitView::setPoints(std::vector<PresetPoint> next)
{
    next.erase(std::remove_if(next.begin(), next.end(), [](const auto& p) {
        return std::any_of(p.position.begin(), p.position.end(), [](float v) {
            return ! std::isfinite(v) || v < 0.0f || v > 1.0f;
        });
    }), next.end());
    points = std::move(next);
    projected.reserve(points.size());
    repaint();
}

void OrbitView::setHighlight(int category) { highlightedCategory = category; repaint(); }
void OrbitView::setPosition(std::array<float, 3> next)
{
    if (std::any_of(next.begin(), next.end(), [](float v) { return ! std::isfinite(v); })) return;
    for (auto& v : next) v = std::clamp(v, 0.0f, 1.0f);
    position = next;
    repaint();
}
void OrbitView::moveTo(OrbitMath::Position next)
{
    if (next == position) return;
    setPosition(next);
    if (onPositionChanged) onPositionChanged(position);
}
void OrbitView::resetCamera() { camera = {}; repaint(); }
void OrbitView::zoom(float factor)
{
    if (! std::isfinite(factor) || factor <= 0.0f) return;
    camera.zoom = std::clamp(camera.zoom * factor, .55f, 2.5f);
    repaint();
}
OrbitMath::Projection OrbitView::projection() const
{
    return {camera, static_cast<float>(getWidth()), static_cast<float>(getHeight())};
}
std::vector<OrbitView::Arrow> OrbitView::arrows() const
{
    const auto p = projection();
    const auto origin = xy(p.project(position));
    std::vector<Arrow> result;
    for (int i = 0; i < 3; ++i)
    {
        auto end = position;
        end[i] += position[i] > .75f ? -.22f : .22f;
        juce::Line<float> line(origin, xy(p.project(end)));
        if (line.getLength() > 10.0f) result.push_back({i, line});
    }
    return result;
}

void OrbitView::paint(juce::Graphics& g)
{
    const auto started = juce::Time::getMillisecondCounterHiRes();
    g.fillAll(background);
    g.setColour(juce::Colour(0xffbcbeba));g.drawRect(getLocalBounds(),1);
    const auto p = projection();
    const auto line = [&](OrbitMath::Position a, OrbitMath::Position b) {
        g.drawLine({xy(p.project(a)), xy(p.project(b))}, 1.0f);
    };
    g.setColour(wire);
    for (float a : {0.0f, 1.0f}) for (float b : {0.0f, 1.0f})
    {
        line({a, b, 0}, {a, b, 1}); line({a, 0, b}, {a, 1, b}); line({0, a, b}, {1, a, b});
    }
    const auto z = position[2];
    juce::Path plane;
    plane.startNewSubPath(xy(p.project({0, 0, z})));
    for (auto corner : {OrbitMath::Position{1, 0, z}, OrbitMath::Position{1, 1, z}, OrbitMath::Position{0, 1, z}})
        plane.lineTo(xy(p.project(corner)));
    plane.closeSubPath();
    g.setColour(accent.withAlpha(.04f)); g.fillPath(plane);
    g.setColour(accent.withAlpha(.3f)); g.strokePath(plane, juce::PathStrokeType(1.0f));
    g.setColour(wire.withAlpha(.65f));
    for (float q : {.25f, .5f, .75f}) { line({q, 0, z}, {q, 1, z}); line({0, q, z}, {1, q, z}); }

    projected.clear();
    for (const auto& point : points) projected.push_back({p.project(point.position), point.category});
    std::sort(projected.begin(), projected.end(), [](const auto& a, const auto& b) { return a.screen.depth < b.screen.depth; });
    for (const auto& dot : projected)
    {
        const auto matches = highlightedCategory < 0 || highlightedCategory == dot.category;
        const auto radius = matches ? (highlightedCategory < 0 ? 2.5f : 3.2f) : 2.1f;
        const auto colour = dot.category >= 0 && dot.category < static_cast<int>(categoryColours.size())
                          ? categoryColours[static_cast<size_t>(dot.category)] : ink;
        g.setColour(colour.withAlpha(matches ? 1.0f : .12f));
        g.fillEllipse(dot.screen.x - radius, dot.screen.y - radius, radius * 2, radius * 2);
    }

    g.setColour(accent.withAlpha(.4f));
    const float dash[] {3.0f, 4.0f};
    g.drawDashedLine({xy(p.project({position[0], position[1], 0})), xy(p.project({position[0], position[1], 1}))}, dash, 2);
    const auto cursor = xy(p.project(position));
    g.setColour(juce::Colour(0xff121212)); g.fillEllipse(cursor.x - 8, cursor.y - 8, 16, 16);
    g.setColour(accent); g.drawEllipse(cursor.x - 8, cursor.y - 8, 16, 16, 2);
    g.fillEllipse(cursor.x - 2.5f, cursor.y - 2.5f, 5, 5);
    g.setFont(labelFont);
    if (selected)
        for (const auto& arrow : arrows())
        {
            g.drawArrow(arrow.line, 1.5f, 7, 7);
            const auto end = arrow.line.getEnd();
            g.drawText(juce::String::charToString("XYZ"[arrow.axis]), juce::Rectangle<float>(end.x + 6, end.y - 16, 18, 16), juce::Justification::centred);
        }
    g.setColour(ink.withAlpha(.65f));
    for (int i = 0; i < 3; ++i)
    {
        const std::array<OrbitMath::Position, 3> labels {{{.5f, -.07f, 0}, {-.06f, .5f, 0}, {1.07f, 0, .5f}}};
        const auto at = xy(p.project(labels[i]));
        g.drawText(juce::String::charToString("XYZ"[i]), juce::Rectangle<float>(at.x - 8, at.y - 7, 16, 14), juce::Justification::centred);
    }
    paintMilliseconds = juce::Time::getMillisecondCounterHiRes() - started;
}

void OrbitView::mouseDown(const juce::MouseEvent& e)
{
    grabKeyboardFocus();
    pointerStart = e.position; dragStart = position; dragCamera = camera;
    rotating = true; dragging = true;
    if (e.position.getDistanceFrom(xy(projection().project(position))) <= 14.0f)
    {
        rotating = false;
        dragMode = e.mods.isShiftDown() ? OrbitMath::Drag::depth : OrbitMath::Drag::plane;
    }
    else if (selected)
    {
        float nearest = 10.0f;
        for (const auto& arrow : arrows())
        {
            juce::Point<float> point;
            const auto distance = arrow.line.getDistanceFromPoint(e.position, point);
            if (distance < nearest)
            {
                nearest = distance; rotating = false;
                dragMode = static_cast<OrbitMath::Drag>(static_cast<int>(OrbitMath::Drag::x) + arrow.axis);
            }
        }
    }
    selected = ! rotating;
    repaint();
}
void OrbitView::mouseDrag(const juce::MouseEvent& e)
{
    if (! dragging) return;
    const auto delta = e.position - pointerStart;
    if (rotating)
    {
        camera.yaw = dragCamera.yaw + delta.x * .008f;
        camera.pitch = dragCamera.pitch + delta.y * .008f;
        repaint();
    }
    else
        moveTo(OrbitMath::Projection(dragCamera, static_cast<float>(getWidth()), static_cast<float>(getHeight()))
                   .drag(dragStart, delta.x, delta.y, dragMode));
}
void OrbitView::mouseUp(const juce::MouseEvent&) { dragging = false; }
void OrbitView::mouseWheelMove(const juce::MouseEvent&, const juce::MouseWheelDetails& wheel)
{
    zoom(std::exp(wheel.deltaY * .5f));
}
bool OrbitView::keyPressed(const juce::KeyPress& key)
{
    const auto code = key.getKeyCode();
    if (code == juce::KeyPress::homeKey) { resetCamera(); return true; }
    if (code == juce::KeyPress::pageUpKey || code == juce::KeyPress::pageDownKey)
    {
        moveTo(projection().drag(position, 0, (code == juce::KeyPress::pageUpKey ? -.025f : .025f) * projection().scale,
                                 OrbitMath::Drag::depth));
        return true;
    }
    if (code == '+' || code == '=') { zoom(1.1f); return true; }
    if (code == '-') { zoom(1.0f / 1.1f); return true; }
    if (code == juce::KeyPress::leftKey) camera.yaw -= .08f;
    else if (code == juce::KeyPress::rightKey) camera.yaw += .08f;
    else if (code == juce::KeyPress::upKey) camera.pitch -= .08f;
    else if (code == juce::KeyPress::downKey) camera.pitch += .08f;
    else return false;
    repaint();
    return true;
}
