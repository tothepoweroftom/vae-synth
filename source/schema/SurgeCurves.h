/*
  Surge XT value curves for the schema controls: normalised 0..1 -> the
  physical value Surge would use for that parameter (pinned revision in
  schema/controls.json). The Lite engine interprets the control vector through
  these, so a vector means roughly the same sound on either backend, and the
  UI uses them for value display.

  Re-implemented from Surge's documented parameter ranges (Parameter.cpp
  set_type); no Surge code is used here.
*/

#pragma once

#include "ControlSchema.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <string>

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

enum class OscType : int
{
    Classic, Sine, Wavetable, SampleHoldNoise, AudioInput, FM3, FM2,
    Window, Modern, String, Twist, Alias,
};
inline constexpr const char* kOscTypeNames[] = {
    "Classic", "Sine", "Wavetable", "S&H Noise", "Audio In", "FM3",
    "FM2", "Window", "Modern", "String", "Twist", "Alias",
};

/** sst-filters FilterType order (num_filter_types == 36 at the pin). */
inline constexpr const char* kFilterTypeNames[] = {
    "Off", "LP 12", "LP 24", "LP Legacy Ladder", "HP 12", "HP 24", "BP 12", "N 12",
    "FX Comb +", "FX S&H", "LP Vintage Ladder", "LP OB-Xd 12", "LP OB-Xd 24", "LP K35",
    "HP K35", "LP Diode Ladder", "LP Cutoff Warp", "HP Cutoff Warp", "N Cutoff Warp",
    "BP Cutoff Warp", "HP OB-Xd 12", "N OB-Xd 12", "BP OB-Xd 12", "BP 24", "N 24",
    "FX Comb -", "FX Allpass", "FX Cutoff Warp AP", "LP Res Warp", "HP Res Warp",
    "N Res Warp", "BP Res Warp", "FX Res Warp AP", "MULTI Tri-pole", "MULTI Cytomic SVF",
    "MULTI OB-Xd Xpander",
};
static_assert (std::size (kFilterTypeNames) == 36);

enum class FilterFamily { Off, LowPass12, LowPass24, HighPass, BandPass, Notch, Other };

inline FilterFamily filterFamily (int surgeFilterType)
{
    switch (surgeFilterType)
    {
        case 0:  return FilterFamily::Off;
        case 1: case 11: return FilterFamily::LowPass12;
        case 2: case 3: case 10: case 12: case 13: case 15: case 16: case 28: return FilterFamily::LowPass24;
        case 4: case 5: case 14: case 17: case 20: case 29: return FilterFamily::HighPass;
        case 6: case 19: case 22: case 23: case 31: return FilterFamily::BandPass;
        case 7: case 18: case 21: case 24: case 30: return FilterFamily::Notch;
        default: return FilterFamily::Other;
    }
}

inline OscType oscType (float v)          { return OscType (stepFromNorm (v, 12)); }
inline int octave (float v)               { return stepFromNorm (v, 7) - 3; }
inline int unisonVoices (float v)         { return stepFromNorm (v, 16) + 1; }
inline int filterType (float v)           { return stepFromNorm (v, 36); }
inline int filterSubtype (float v)        { return stepFromNorm (v, 16); }
inline bool toggle (float v)              { return v > 0.5f; }

/** ct_amplitude: displayed in dB, applied as x^3. */
inline float amplitude (float v)          { const float x = std::max (0.0f, v); return x * x * x; }
/** ct_oscspread: 0..1 shown as 0..100 cents. */
inline float unisonDetuneCents (float v)  { return 100.0f * v; }
/** ct_percent_bipolar. */
inline float bipolar (float v)            { return 2.0f * v - 1.0f; }
/** ct_freq_audible: semitones from A440, -60..70. */
inline float cutoffHz (float v)           { return 440.0f * std::exp2 ((130.0f * v - 60.0f) / 12.0f); }
/** ct_freq_mod: -96..96 semitones. */
inline float fegSemitones (float v)       { return 192.0f * v - 96.0f; }
/** ct_envtime: log2 seconds, -8..5. */
inline float envSeconds (float v)         { return std::exp2 (13.0f * v - 8.0f); }
/** ct_lforate: log2 Hz, -7..9. */
inline float lfoRateHz (float v)          { return std::exp2 (16.0f * v - 7.0f); }
/** ct_decibel_attenuation_clipper: -48..0 dB. */
inline float globalVolumeDb (float v)     { return 48.0f * v - 48.0f; }

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

inline std::string formatGainDb (float linear)
{
    return linear <= 1.0e-5f ? std::string ("-inf dB") : formatNumber (20.0f * std::log10 (linear), 1) + " dB";
}

/** Human-readable value of control `index` at normalised `v`, in Surge's units. */
inline std::string describe (int index, float v)
{
    switch (index)
    {
        case kOsc1Type: case kOsc2Type: case kOsc3Type:
            return kOscTypeNames[int (oscType (v))];
        case kOsc1Octave: case kOsc2Octave: case kOsc3Octave:
        {
            const int o = octave (v);
            return (o > 0 ? "+" : "") + std::to_string (o) + " oct";
        }
        case kOsc1Shape: case kFilterKeytrack:
            return formatNumber (100.0f * bipolar (v), 0) + " %";
        case kOsc1Volume: case kOsc2Volume: case kOsc3Volume: case kNoiseVolume:
            return formatGainDb (amplitude (v));
        case kOsc1Voices:   return std::to_string (unisonVoices (v)) + " voices";
        case kOsc1Detune:   return formatNumber (unisonDetuneCents (v), 1) + " ct";
        case kOsc2Mute:     return toggle (v) ? "Muted" : "On";
        case kFilterType:   return kFilterTypeNames[filterType (v)];
        case kFilterSubtype: return std::to_string (filterSubtype (v) + 1);
        case kCutoff:
        {
            const float hz = cutoffHz (v);
            return hz >= 1000.0f ? formatNumber (hz / 1000.0f, 2) + " kHz" : formatNumber (hz, 0) + " Hz";
        }
        case kFilterFeg:    return formatNumber (fegSemitones (v), 1) + " st";
        case kAmpAttack: case kAmpDecay: case kAmpRelease:
        case kFegAttack: case kFegDecay: case kFegRelease:
            return formatSeconds (envSeconds (v));
        case kLfo1Rate:     return formatNumber (lfoRateHz (v), 2) + " Hz";
        case kGlobalVolume: return formatNumber (globalVolumeDb (v), 1) + " dB";
        default:            return formatNumber (100.0f * v, 0) + " %";
    }
}

} // namespace vae::schema::surge
