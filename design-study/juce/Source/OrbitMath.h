#pragma once

#include <algorithm>
#include <array>
#include <cmath>

// Orthographic camera math shared by drawing, picking and drag gestures.
// Coordinates are normalised to the unit cube; no JUCE or model dependency.
namespace OrbitMath
{
using Position = std::array<float, 3>;
struct Camera { float yaw = -.55f, pitch = .35f, zoom = 1.0f; };
struct Projected { float x, y, depth; };
enum class Drag { plane, depth, x, y, z };

struct Projection
{
    float cy, sy, cp, sp, scale, width, height;
    Projection(Camera camera, float w, float h)
        : cy(std::cos(camera.yaw)), sy(std::sin(camera.yaw)),
          cp(std::cos(camera.pitch)), sp(std::sin(camera.pitch)),
          scale(std::min(w, h) * .57f * camera.zoom), width(w), height(h) {}

    Projected project(Position p) const
    {
        const auto x = p[0] - .5f, y = p[1] - .5f, z = p[2] - .5f;
        const auto rx = cy * x + sy * z, rz = -sy * x + cy * z;
        return {width * .5f + rx * scale, height * .5f - (cp * y - sp * rz) * scale,
                sp * y + cp * rz};
    }

    Position drag(Position start, float dx, float dy, Drag mode) const
    {
        if (scale <= 0.0f) return start;
        Position delta {};
        if (mode >= Drag::x)
        {
            const auto axis = static_cast<int>(mode) - static_cast<int>(Drag::x);
            Position unit {}; unit[axis] = 1.0f;
            const auto a = project({0, 0, 0}), b = project(unit);
            const auto vx = b.x - a.x, vy = b.y - a.y, n = vx * vx + vy * vy;
            if (n < scale * scale * .0025f) return start; // End-on axis: orbit first.
            delta[axis] = (dx * vx + dy * vy) / n;
        }
        else
        {
            const auto rx = mode == Drag::depth ? 0.0f : dx / scale;
            const auto ry = mode == Drag::depth ? 0.0f : -dy / scale;
            const auto d = mode == Drag::depth ? -dy / scale : 0.0f;
            const auto rz = -sp * ry + cp * d;
            delta = {cy * rx - sy * rz, cp * ry + sp * d, sy * rx + cy * rz};
        }
        for (int i = 0; i < 3; ++i) start[i] = std::clamp(start[i] + delta[i], 0.0f, 1.0f);
        return start;
    }
};
}
