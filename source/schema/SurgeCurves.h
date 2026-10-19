/*
  Value curves for the schema controls. A continuous control's value v (0..1)
  maps linearly onto its curve's [min, max] in Surge units (log2 seconds,
  semitones from A440, ...); this header turns those units into physical
  values for the engines and into text for the UI.

  Re-implemented from Surge's documented parameter ranges (Parameter.cpp
  set_type); no Surge code is used here.
*/

#pragma once

#include "ControlSchema.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>

namespace vae::schema::surge
{
/** Surge's normalised -> integer mapping for an integer parameter with
    `steps` values (Parameter::set_value_f01 on vt_int). */
inline int stepFromNorm (float v, int steps)
{
    if (steps <= 1)
        return 0;
    const double scaled = (1.0 / 0.99) * (double (v) - 0.005) * double (steps - 1) + 0.5;
    return std::clamp (int (std::floor (scaled)), 0, steps - 1);
}

/** Inverse of stepFromNorm: the normalised value Surge stores for a step. */
inline float normFromStep (int step, int steps)
{
    if (steps <= 1)
        return 0.0f;
    return 0.005f + 0.99f * float (std::clamp (step, 0, steps - 1)) / float (steps - 1);
}

/** Surge treats a curve's minimum as "off" (glide, LFO fade-in); values at or
    below this count as the minimum. */
inline constexpr float kOffAtCurveMin = 1.0e-4f;

/** A curve of the schema's curve table by name, or nullptr. */
constexpr const Curve* curveNamed (std::string_view name)
{
    for (const auto& c : generated::kCurves)
        if (name == c.name)
            return &c;
    return nullptr;
}

/** Width of a named curve in its units (max - min), e.g. "cutoff" = 130 st. */
constexpr float curveSpan (std::string_view name)
{
    const auto* c = curveNamed (name);
    return c != nullptr ? c->max - c->min : 0.0f;
}

/** Continuous control value (0..1) -> its curve's units. */
inline float toUnits (int index, float v)
{
    const auto* c = curveOf (index);
    return c != nullptr ? c->min + std::clamp (v, 0.0f, 1.0f) * (c->max - c->min) : v;
}

/** Curve units -> continuous control value (0..1). */
inline float fromUnits (int index, float x)
{
    const auto* c = curveOf (index);
    return c != nullptr ? std::clamp ((x - c->min) / (c->max - c->min), 0.0f, 1.0f) : x;
}

// Physical conversions of curve units ----------------------------------------
/** ct_amplitude: shown in dB, applied as x^3. */
inline float amplitudeGain (float v)
{
    const float x = std::max (0.0f, v);
    return x * x * x;
}
/** Semitones from A440 -> Hz (cutoff, EQ frequency). */
inline float semitonesToHz (float st) { return 440.0f * std::exp2 (st / 12.0f); }
/** log2 seconds -> seconds (envelopes, glide, delay, reverb). */
inline float log2Seconds (float x) { return std::exp2 (x); }
/** log2 Hz -> Hz (LFO / chorus / mod FX rates). */
inline float log2Hz (float x) { return std::exp2 (x); }

// Common shortcuts for continuous controls by index --------------------------
inline float seconds (int index, float v) { return log2Seconds (toUnits (index, v)); }
inline float hz (int index, float v) { return log2Hz (toUnits (index, v)); }
inline float cutoffHz (int index, float v) { return semitonesToHz (toUnits (index, v)); }
inline float bipolar (float v) { return 2.0f * v - 1.0f; }

/** Glide: the curve minimum means "off". */
inline float glideSeconds (float v) { return v <= kOffAtCurveMin ? 0.0f : log2Seconds (toUnits (kGlideTime, v)); }

// Display ----------------------------------------------------------------------
inline std::string formatNumber (float x, int decimals)
{
    char buf[32];
    std::snprintf (buf, sizeof (buf), "%.*f", decimals, double (x));
    return buf;
}

inline std::string formatSeconds (float s)
{
    return s < 1.0f ? formatNumber (s * 1000.0f, s < 0.1f ? 1 : 0) + " ms" : formatNumber (s, 2) + " s";
}

inline std::string formatHz (float hzValue)
{
    return hzValue >= 1000.0f ? formatNumber (hzValue / 1000.0f, 2) + " kHz"
                              : formatNumber (hzValue, hzValue < 10.0f ? 2 : 0) + " Hz";
}

inline std::string formatGainDb (float linear)
{
    return linear <= 1.0e-5f ? std::string ("-inf dB") : formatNumber (20.0f * std::log10 (linear), 1) + " dB";
}

/** Human-readable value of control `index` at raw value `v`. */
inline std::string describe (int index, float v)
{
    const auto& c = control (index);
    if (c.kind == Kind::Categorical)
        return c.options[asIndex (index, v)];
    if (c.kind == Kind::Toggle)
        return v >= 0.5f ? "On" : "Off";

    if (index == kGlideTime && v <= kOffAtCurveMin)
        return "Off";

    const auto* curve = curveOf (index);
    const float x = toUnits (index, v);
    const std::string display = curve->display;
    if (display == "percent")
        return formatNumber (100.0f * x, 0) + " %";
    if (display == "amplitude_cubed")
        return formatGainDb (amplitudeGain (v));
    if (display == "log2_seconds")
        return formatSeconds (log2Seconds (x));
    if (display == "log2_hz")
        return formatHz (log2Hz (x));
    if (display == "semitones_a440")
        return formatHz (semitonesToHz (x));
    return formatNumber (x, 1) + " " + curve->unit;
}

} // namespace vae::schema::surge
