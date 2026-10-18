/*
  The model-facing control vector (schema/controls.json, generated into
  ControlSchema.generated.h). Every engine backend, the plug-in parameters and
  the ML side speak this vector: kControlCount Surge-normalised floats, 0..1.
*/

#pragma once

#include "ControlSchema.generated.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <span>
#include <string_view>

namespace vae::schema
{
using generated::kControlCount;
using generated::kLatentDimensions;
using generated::kSchemaVersion;
using ControlSpec = generated::Row;

inline constexpr std::span<const ControlSpec, kControlCount> controls() { return generated::kRows; }

/** Index of a control id, or nullopt. */
inline std::optional<int> indexOf (std::string_view id)
{
    for (int i = 0; i < kControlCount; ++i)
        if (id == generated::kRows[i].id)
            return i;
    return std::nullopt;
}

/** Stable indices for the controls the engines interpret directly. The
    static_asserts below check them against the generated table, so reordering
    the JSON without updating this enum fails to build. */
enum Control : int
{
    kOsc1Type, kOsc1Octave, kOsc1Shape, kOsc1Volume, kOsc1Voices, kOsc1Detune,
    kOsc2Type, kOsc2Octave, kOsc2Volume, kOsc2Mute,
    kOsc3Type, kOsc3Octave, kOsc3Volume,
    kNoiseVolume,
    kFilterType, kFilterSubtype, kCutoff, kResonance, kFilterFeg, kFilterKeytrack, kFilterBalance,
    kAmpAttack, kAmpDecay, kAmpSustain, kAmpRelease,
    kFegAttack, kFegDecay, kFegSustain, kFegRelease,
    kLfo1Rate, kLfo1Amplitude,
    kGlobalVolume,
    kNumControls
};
static_assert (kNumControls == kControlCount, "schema/controls.json and Control enum disagree");

namespace detail
{
constexpr bool sameId (const char* a, const char* b)
{
    while (*a != '\0' && *a == *b) { ++a; ++b; }
    return *a == *b;
}
} // namespace detail
static_assert (detail::sameId (generated::kRows[kOsc1Type].id, "osc1_type"));
static_assert (detail::sameId (generated::kRows[kNoiseVolume].id, "noise_volume"));
static_assert (detail::sameId (generated::kRows[kFilterType].id, "filter_type"));
static_assert (detail::sameId (generated::kRows[kAmpAttack].id, "amp_attack"));
static_assert (detail::sameId (generated::kRows[kLfo1Rate].id, "lfo1_rate"));
static_assert (detail::sameId (generated::kRows[kGlobalVolume].id, "global_volume"));

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

    /** True when every value is finite and within 0..1. */
    bool isValid() const
    {
        return std::all_of (values.begin(), values.end(),
                            [] (float v) { return std::isfinite (v) && v >= 0.0f && v <= 1.0f; });
    }

    /** Replace non-finite values with defaults and clamp to 0..1. */
    void sanitise()
    {
        for (int i = 0; i < kControlCount; ++i)
        {
            auto& v = values[(size_t) i];
            v = std::isfinite (v) ? std::clamp (v, 0.0f, 1.0f) : generated::kRows[i].defaultValue;
        }
    }
};

using Latent = std::array<float, kLatentDimensions>;

/** Deterministic stand-in decoder with no trained model; identical to
    VAE_SURGE's snapshot_from_latent so both projects agree before weights exist. */
inline Snapshot mockDecode (const Latent& latent)
{
    auto s = Snapshot::defaults();
    for (int i = 0; i < kControlCount; ++i)
    {
        const float l = std::clamp (latent[(size_t) (i % kLatentDimensions)], -1.0f, 1.0f);
        const float seeded = (l + 1.0f) * 0.5f;
        s[i] = std::clamp (0.5f * generated::kRows[i].defaultValue + 0.5f * seeded, 0.0f, 1.0f);
    }
    return s;
}

} // namespace vae::schema
