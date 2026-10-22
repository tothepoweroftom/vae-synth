#include "StudyOrbit.h"

namespace
{
const juce::Colour background(0xff252525), accent(0xff70ff63), ink(0xffdcded8), wire(0xff4b4e49);
const std::array<juce::Colour, 8> categoryColours {
    juce::Colour(0xff70bcff), juce::Colour(0xffffad66), juce::Colour(0xff86eeac), juce::Colour(0xffffe078),
    juce::Colour(0xffff8fc7), juce::Colour(0xff63e0d5), juce::Colour(0xffb6a0ff), juce::Colour(0xffff918c)
};
juce::Point<float> xy(OrbitMath::Projected p) { return {p.x, p.y}; }
}

StudyOrbit::StudyOrbit()
{
    setWantsKeyboardFocus(true);
    setTitle("Preset space");
    setDescription("Drag the handle to move; Shift drag changes depth. Drag empty space to orbit. "
                   "Select the handle for X Y Z arrows. Scroll to zoom. Arrow keys rotate; Page Up and Down change depth; Home resets camera.");
    setMouseCursor(juce::MouseCursor::CrosshairCursor);
}

void StudyOrbit::setPoints(std::vector<StudyPresetPoint> next)
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

void StudyOrbit::setHighlight(int category) { highlightedCategory = category; repaint(); }
void StudyOrbit::setPosition(std::array<float, 3> next)
{
    if (std::any_of(next.begin(), next.end(), [](float v) { return ! std::isfinite(v); })) return;
    for (auto& v : next) v = std::clamp(v, 0.0f, 1.0f);
    position = next;
    repaint();
}
void StudyOrbit::moveTo(OrbitMath::Position next)
{
    if (next == position) return;
    setPosition(next);
    if (onPositionChanged) onPositionChanged(position);
}
void StudyOrbit::resetCamera()
{
    if (reducedMotion) { camera={}; repaint(); return; }
    resetFrom=camera; resetStarted=juce::Time::getMillisecondCounterHiRes();
    resetting=true; startTimerHz(60);
}
void StudyOrbit::mouseMove(const juce::MouseEvent& e)
{
    hovered=e.position.getDistanceFrom(xy(projection().project(position)))<=18.0f;
    setMouseCursor(hovered?juce::MouseCursor::PointingHandCursor:juce::MouseCursor::CrosshairCursor);
    startTimerHz(60);
}
void StudyOrbit::mouseExit(const juce::MouseEvent&) { hovered=false; startTimerHz(60); }
void StudyOrbit::timerCallback()
{
    if (!isShowing()) { stopTimer(); return; }
    const auto now=juce::Time::getMillisecondCounterHiRes();
    const auto dt=lastFrame==0?16.0:juce::jlimit(0.0,50.0,now-lastFrame); lastFrame=now;
    const float target=(hovered || (dragging&&!rotating))?1.0f:0.0f;
    emphasis=reducedMotion?target:emphasis+(target-emphasis)*(float)(1-std::exp(-dt/45.0));
    if (std::abs(target-emphasis)<.005f) emphasis=target;
    if (resetting) {
        float t=juce::jlimit(0.f,1.f,(float)((now-resetStarted)/220.0));
        float ease=1-std::pow(1-t,3.f); OrbitMath::Camera end;
        camera={resetFrom.yaw+(end.yaw-resetFrom.yaw)*ease,
                resetFrom.pitch+(end.pitch-resetFrom.pitch)*ease,
                resetFrom.zoom+(end.zoom-resetFrom.zoom)*ease};
        if(t>=1 || reducedMotion) { camera=end; resetting=false; }
    }
    repaint();
    if(emphasis==target&&!resetting) { stopTimer(); lastFrame=0; }
}
void StudyOrbit::zoom(float factor)
{
    if (! std::isfinite(factor) || factor <= 0.0f) return;
    resetting=false;
    camera.zoom = std::clamp(camera.zoom * factor, .55f, 2.5f);
    repaint();
}
OrbitMath::Projection StudyOrbit::projection() const
{
    return {camera, static_cast<float>(getWidth()), static_cast<float>(getHeight())};
}
std::vector<StudyOrbit::Arrow> StudyOrbit::arrows() const
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

void StudyOrbit::paint(juce::Graphics& g)
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
    g.setColour(accent.withAlpha(.04f + emphasis*.05f)); g.fillPath(plane);
    g.setColour(accent.withAlpha(.3f + emphasis*.3f)); g.strokePath(plane, juce::PathStrokeType(1.0f));
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
    if (selected || emphasis>.05f)
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
    if (emphasis>.01f || hasKeyboardFocus(true)) {
        g.setColour(accent.withAlpha(hasKeyboardFocus(true)?.7f:emphasis*.6f));
        g.drawEllipse(cursor.x-14,cursor.y-14,28,28,1);
    }
    g.setColour(pending?ink:accent); g.setFont(labelFont);
    g.drawText(pending?"UPDATING PREDICTION":"PREDICTION READY / NO AUDIO",18,16,320,18,juce::Justification::centredLeft);
    g.setColour(ink.withAlpha(.8f));
    const auto hint=dragging ? (rotating?"ROTATE VIEW": "MOVE SOUND / RELEASE TO KEEP")
                            : hovered?"DRAG TO EXPLORE / SHIFT: DEPTH":"DRAG SPACE: CAMERA / SCROLL: ZOOM";
    g.drawText(hint,18,getHeight()-28,getWidth()-36,18,juce::Justification::centredLeft);
    paintMilliseconds = juce::Time::getMillisecondCounterHiRes() - started;
}

void StudyOrbit::mouseDown(const juce::MouseEvent& e)
{
    resetting=false;
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
    if (!rotating && onGestureStart) onGestureStart();
    startTimerHz(60);
    repaint();
}
void StudyOrbit::mouseDrag(const juce::MouseEvent& e)
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
void StudyOrbit::mouseUp(const juce::MouseEvent&) { dragging = false; startTimerHz(60); }
void StudyOrbit::mouseWheelMove(const juce::MouseEvent&, const juce::MouseWheelDetails& wheel)
{
    zoom(std::exp(wheel.deltaY * .5f));
}
bool StudyOrbit::keyPressed(const juce::KeyPress& key)
{
    const auto code = key.getKeyCode();
    if (code == juce::KeyPress::homeKey) { resetCamera(); return true; }
    if (code == juce::KeyPress::pageUpKey || code == juce::KeyPress::pageDownKey)
    {
        if(onGestureStart) onGestureStart();
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
