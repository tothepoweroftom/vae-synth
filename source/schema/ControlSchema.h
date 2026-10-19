/*
  The model-facing control vector (schema/controls.json, generated into
  ControlSchema.generated.h). Every engine backend, the plug-in parameters and
  the ML side speak this vector.

  Snapshot values are "raw" per kind:
    continuous   0..1 along the control's curve (Surge-normalised)
    categorical  option index, 0..numOptions-1
    toggle       0 or 1
  The model sees "features" instead: one per continuous/toggle control and
  one per option of a categorical control (one-hot). encodeFeatures /
  decodeFeatures convert between the two.
*/

#pragma once

#include "ControlSchema.generated.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace vae::schema
{
using generated::Control;
using generated::kControlCount;
using generated::kCurveCount;
using generated::kFeatureCount;
using generated::kLatentDimensions;
using generated::kSchemaVersion;
using generated::Kind;
using ControlSpec = generated::Row;
using Curve = generated::Curve;
using namespace generated;   // Control enumerators (kOsc1Wave, ...)

inline constexpr std::span<const ControlSpec, kControlCount> controls() { return generated::kRows; }
inline constexpr const ControlSpec& control (int index) { return generated::kRows[index]; }

inline constexpr const Curve* curveOf (int index)
{
    const int c = generated::kRows[index].curve;
    return c >= 0 ? &generated::kCurves[c] : nullptr;
}

/** Index of a control id, or nullopt. */
inline std::optional<int> indexOf (std::string_view id)
{
    for (int i = 0; i < kControlCount; ++i)
        if (id == generated::kRows[i].id)
            return i;
    return std::nullopt;
}

/** Index of a categorical option by name, or nullopt. */
inline std::optional<int> optionIndex (int controlIndex, std::string_view option)
{
    const auto& c = control (controlIndex);
    for (int o = 0; o < c.numOptions; ++o)
        if (option == c.options[o])
            return o;
    return std::nullopt;
}

/** Number of features a control occupies. */
inline constexpr int featureWidth (const ControlSpec& c)
{
    return c.kind == Kind::Categorical ? c.numOptions : 1;
}

/** Clamp/round one raw value to what its kind allows. */
inline float sanitiseValue (int index, float v)
{
    const auto& c = control (index);
    if (! std::isfinite (v))
        return c.defaultValue;
    switch (c.kind)
    {
        case Kind::Continuous:  return std::clamp (v, 0.0f, 1.0f);
        case Kind::Toggle:      return v >= 0.5f ? 1.0f : 0.0f;
        case Kind::Categorical: return std::clamp (std::round (v), 0.0f, float (c.numOptions - 1));
    }
    return v;
}

/** Categorical/toggle value as an int (rounded, clamped). */
inline int asIndex (int index, float v) { return int (sanitiseValue (index, v)); }

/** A full control vector. */
struct Snapshot
{
    std::array<float, kControlCount> values {};

    static Snapshot defaults()
    {
        Snapshot s;
        for (int i = 0; i < kControlCount; ++i)
            s.values[(size_t) i] = generated::kRows[i].defaultValue;
        return s;
    }

    float operator[] (int i) const { return values[(size_t) i]; }
    float& operator[] (int i) { return values[(size_t) i]; }

    /** True when every value is finite and legal for its kind. */
    bool isValid() const
    {
        for (int i = 0; i < kControlCount; ++i)
        {
            const float v = values[(size_t) i];
            const float clean = sanitiseValue (i, v);
            if (! std::isfinite (v) || v < clean || v > clean)
                return false;
        }
        return true;
    }

    /** Replace non-finite values with defaults; clamp and round per kind. */
    void sanitise()
    {
        for (int i = 0; i < kControlCount; ++i)
            values[(size_t) i] = sanitiseValue (i, values[(size_t) i]);
    }
};

using Latent = std::array<float, kLatentDimensions>;
using Features = std::array<float, kFeatureCount>;

/** Snapshot -> model features (one-hot for categoricals). */
inline Features encodeFeatures (const Snapshot& s)
{
    Features f {};
    for (int i = 0; i < kControlCount; ++i)
    {
        const auto& c = control (i);
        if (c.kind == Kind::Categorical)
            f[(size_t) (c.firstFeature + asIndex (i, s[i]))] = 1.0f;
        else
            f[(size_t) c.firstFeature] = sanitiseValue (i, s[i]);
    }
    return f;
}

/** Model features -> snapshot: continuous clamped, toggles thresholded at
    0.5, categoricals by argmax over their options (probabilities or logits). */
inline Snapshot decodeFeatures (std::span<const float, kFeatureCount> f)
{
    Snapshot s;
    for (int i = 0; i < kControlCount; ++i)
    {
        const auto& c = control (i);
        if (c.kind == Kind::Categorical)
        {
            int best = 0;
            for (int o = 1; o < c.numOptions; ++o)
                if (f[(size_t) (c.firstFeature + o)] > f[(size_t) (c.firstFeature + best)])
                    best = o;
            s[i] = float (best);
        }
        else
        {
            s[i] = f[(size_t) c.firstFeature];
        }
    }
    s.sanitise();
    return s;
}

/** Deterministic stand-in decoder with no trained model: each feature is
    0.5 * default + 0.5 * a latent-seeded value. At the zero latent, toggles and
    categoricals keep their defaults and continuous values sit halfway between
    their default and 0.5. (v1 used the same rule on controls; v2 applies it
    to features.) */
inline Snapshot mockDecode (const Latent& latent)
{
    const auto defaults = encodeFeatures (Snapshot::defaults());
    Features f {};
    for (int k = 0; k < kFeatureCount; ++k)
    {
        const float l = std::clamp (latent[(size_t) (k % kLatentDimensions)], -1.0f, 1.0f);
        const float seeded = (l + 1.0f) * 0.5f;
        f[(size_t) k] = std::clamp (0.5f * defaults[(size_t) k] + 0.5f * seeded, 0.0f, 1.0f);
    }
    return decodeFeatures (f);
}

} // namespace vae::schema
