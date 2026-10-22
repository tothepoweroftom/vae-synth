#include "LatentOrbit.h"
#include <cmath>

namespace vae::experimental
{
std::array<float, 3> orbitPosition (std::array<float, 3> centre, const OrbitSettings& s, double seconds, bool hasZ)
{
    const double w = 2.0 * juce::MathConstants<double>::pi * s.rateHz * seconds;
    const double offset[3] { std::cos (w), std::sin (w), hasZ ? 0.5 * std::sin (0.7 * w) : 0.0 };
    for (size_t i = 0; i < 3; ++i)
        centre[i] = juce::jlimit (0.0f, 1.0f, centre[i] + s.depth * (float) offset[i]);
    return centre;
}

void holdDiscrete (schema::Snapshot& lite, const schema::Snapshot& centre)
{
    for (int i = 0; i < schema::kControlCount; ++i)
        if (schema::control (i).kind != schema::Kind::Continuous)
            lite[i] = centre[i];
}
} // namespace vae::experimental
