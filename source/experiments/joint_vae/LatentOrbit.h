#pragma once
#include "schema/ControlSchema.h"
#include <array>
#include <juce_core/juce_core.h>

namespace vae::experimental
{
/** The Lab's orbit LFO (Task 19): a free-running circle around the current point, in pad units. */
struct OrbitSettings
{
    bool on = false;
    float rateHz = 0.1f, depth = 0.08f;
};

/** Pad positions (x, y, z) on the orbit at `seconds`: centre + depth * (cos wt, sin wt, 0.5 sin 0.7wt), the
    Z wobble only when `hasZ`; each clamped to 0..1. */
std::array<float, 3> orbitPosition (std::array<float, 3> centre, const OrbitSettings&, double seconds, bool hasZ);

/** Every Toggle and Categorical control of `lite` takes `centre`'s value: only continuous controls move, so a
    switch never flips (and Lite never ducks) while the orbit runs (Task 18 finding). */
void holdDiscrete (schema::Snapshot& lite, const schema::Snapshot& centre);
} // namespace vae::experimental
