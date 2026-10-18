/*
  Parameter table for the instrument — the single source of truth shared by
  the WASM module (inst_param_*), the JUCE plugin (its APVTS is generated
  from this table) and the JSON preset format (keys are the ids below, values
  are plain values in the ranges below, exactly what inst_set_param takes).

  Keep ids stable once shipped — they end up in saved projects and presets.
  Append new rows at the end of a section; the host discovers indices at
  runtime, so reordering is safe for the host but ids must never change.
*/

#pragma once

namespace ablaze
{

enum class ParamKind : int
{
    Continuous,   // smoothed (~20 ms)
    Toggle,       // 0 / 1, smoothed as a gain so switching never clicks
    Choice,       // integer index into `choices`, snaps
    Integer,      // integer steps, snaps
};

struct ParamInfo
{
    const char* id;
    const char* name;      // display name (plugin UI)
    const char* unit;      // display unit, may be ""
    float min, max, def;
    ParamKind kind;
    const char* choices;   // '|' separated labels for Choice, else nullptr
    float skewCentre;      // plugin UI skew (value at slider centre), 0 = linear
};

namespace detail
{
constexpr ParamInfo cont (const char* id, const char* name, const char* unit,
                          float mn, float mx, float def, float centre = 0.0f)
{
    return { id, name, unit, mn, mx, def, ParamKind::Continuous, nullptr, centre };
}

constexpr ParamInfo toggle (const char* id, const char* name, bool def)
{
    return { id, name, "", 0.0f, 1.0f, def ? 1.0f : 0.0f, ParamKind::Toggle, nullptr, 0.0f };
}

constexpr ParamInfo choice (const char* id, const char* name, const char* choices,
                            int numChoices, int def)
{
    return { id, name, "", 0.0f, float (numChoices - 1), float (def), ParamKind::Choice, choices, 0.0f };
}

constexpr ParamInfo integer (const char* id, const char* name, const char* unit,
                             int mn, int mx, int def)
{
    return { id, name, unit, float (mn), float (mx), float (def), ParamKind::Integer, nullptr, 0.0f };
}
} // namespace detail

enum ParamIndex : int
{
    kParamGain = 0,

    // Oscillator A — wavetable
    kParamOscAOn,
    kParamOscATable,
    kParamOscAPos,
    kParamOscAOctave,
    kParamOscASemi,
    kParamOscAFine,
    kParamOscALevel,
    kParamOscAUnison,
    kParamOscADetune,
    kParamOscASpread,

    // Oscillator B — analog (PolyBLEP)
    kParamOscBOn,
    kParamOscBWave,
    kParamOscBPw,
    kParamOscBOctave,
    kParamOscBSemi,
    kParamOscBFine,
    kParamOscBLevel,

    // Sub oscillator + noise
    kParamSubOn,
    kParamSubWave,
    kParamSubOctave,
    kParamSubLevel,
    kParamNoiseLevel,

    // Filter (per voice, shared by all sources)
    kParamFilterOn,
    kParamFilterType,
    kParamFilterCutoff,
    kParamFilterReso,
    kParamFilterDrive,
    kParamFilterKeytrack,
    kParamFilterEnvAmt,
    kParamFilterVel,

    // Filter envelope
    kParamFenvAttack,
    kParamFenvDecay,
    kParamFenvSustain,
    kParamFenvRelease,

    // Amp envelope
    kParamAmpAttack,
    kParamAmpDecay,
    kParamAmpSustain,
    kParamAmpRelease,
    kParamAmpVel,

    // FX: waveshaper
    kParamShaperOn,
    kParamShaperType,
    kParamShaperDrive,
    kParamShaperMix,

    // FX: 3-band EQ
    kParamEqOn,
    kParamEqLowFreq,
    kParamEqLowGain,
    kParamEqMidFreq,
    kParamEqMidGain,
    kParamEqMidQ,
    kParamEqHighFreq,
    kParamEqHighGain,

    // FX: chorus
    kParamChorusOn,
    kParamChorusRate,
    kParamChorusDepth,
    kParamChorusMix,

    // FX: delay
    kParamDelayOn,
    kParamDelayTime,
    kParamDelayFeedback,
    kParamDelayDamp,
    kParamDelayPingPong,
    kParamDelayMix,

    // FX: reverb
    kParamReverbOn,
    kParamReverbSize,
    kParamReverbDamp,
    kParamReverbWidth,
    kParamReverbMix,

    kNumParams,
};

// Choice labels, shared with the voice/FX enums below. The built-in tables
// come first; "User" is the table loaded from a .wav (plugin) or embedded at
// build time (WASM). Append only: presets store the index.
inline constexpr const char* kWavetableNames = "Basic|PWM|Harmonic|Formant|Sync|Fold|User";
inline constexpr int kNumBuiltinWavetables = 6;
inline constexpr int kUserWavetableIndex = kNumBuiltinWavetables;
inline constexpr int kNumWavetables = kNumBuiltinWavetables + 1;

enum class AnalogWave : int { Sine, Triangle, Saw, Square, Pulse };
enum class SubWave : int { Sine, Triangle, Square, Pulse };
enum class FilterType : int { LowPass12, LowPass24, BandPass, HighPass, Notch };
enum class ShaperType : int { Soft, Hard, Fold, Asym };

inline constexpr ParamInfo kParams[kNumParams] = {
    detail::cont    ("gain",             "Master",       "",    0.0f,   1.0f,     0.8f),

    detail::toggle  ("osca_on",          "On",                  true),
    detail::choice  ("osca_table",       "Table",        kWavetableNames, kNumWavetables, 0),
    detail::cont    ("osca_pos",         "Position",     "",    0.0f,   1.0f,     0.0f),
    detail::integer ("osca_octave",      "Octave",       "oct", -3,     3,        0),
    detail::integer ("osca_semi",        "Semi",         "st",  -12,    12,       0),
    detail::cont    ("osca_fine",        "Fine",         "ct",  -100.0f, 100.0f,  0.0f),
    detail::cont    ("osca_level",       "Level",        "",    0.0f,   1.0f,     0.8f),
    detail::integer ("osca_unison",      "Unison",       "",    1,      7,        1),
    detail::cont    ("osca_detune",      "Detune",       "ct",  0.0f,   100.0f,   20.0f),
    detail::cont    ("osca_spread",      "Spread",       "",    0.0f,   1.0f,     0.6f),

    detail::toggle  ("oscb_on",          "On",                  false),
    detail::choice  ("oscb_wave",        "Wave",         "Sine|Triangle|Saw|Square|Pulse", 5, 2),
    detail::cont    ("oscb_pw",          "Width",        "",    0.05f,  0.95f,    0.5f),
    detail::integer ("oscb_octave",      "Octave",       "oct", -3,     3,        0),
    detail::integer ("oscb_semi",        "Semi",         "st",  -12,    12,       0),
    detail::cont    ("oscb_fine",        "Fine",         "ct",  -100.0f, 100.0f,  0.0f),
    detail::cont    ("oscb_level",       "Level",        "",    0.0f,   1.0f,     0.6f),

    detail::toggle  ("sub_on",           "On",                  false),
    detail::choice  ("sub_wave",         "Wave",         "Sine|Triangle|Square|Pulse", 4, 0),
    detail::choice  ("sub_octave",       "Octave",       "-1 oct|-2 oct", 2, 0),
    detail::cont    ("sub_level",        "Sub",          "",    0.0f,   1.0f,     0.5f),
    detail::cont    ("noise_level",      "Noise",        "",    0.0f,   1.0f,     0.0f),

    detail::toggle  ("filter_on",        "On",                  true),
    detail::choice  ("filter_type",      "Type",         "LP 12|LP 24|Band-pass|High-pass|Notch", 5, 1),
    detail::cont    ("filter_cutoff",    "Cutoff",       "Hz",  20.0f,  20000.0f, 6000.0f, 1000.0f),
    detail::cont    ("filter_reso",      "Reso",         "",    0.0f,   1.0f,     0.15f),
    detail::cont    ("filter_drive",     "Drive",        "",    0.0f,   1.0f,     0.0f),
    detail::cont    ("filter_keytrack",  "Keytrack",     "",    0.0f,   1.0f,     0.5f),
    detail::cont    ("filter_env_amt",   "Env Amt",      "",    -1.0f,  1.0f,     0.35f),
    detail::cont    ("filter_vel",       "Velocity",     "",    0.0f,   1.0f,     0.2f),

    detail::cont    ("fenv_attack",      "Attack",       "ms",  0.5f,   10000.0f, 5.0f,   500.0f),
    detail::cont    ("fenv_decay",       "Decay",        "ms",  1.0f,   10000.0f, 400.0f, 500.0f),
    detail::cont    ("fenv_sustain",     "Sustain",      "",    0.0f,   1.0f,     0.3f),
    detail::cont    ("fenv_release",     "Release",      "ms",  1.0f,   10000.0f, 400.0f, 500.0f),

    detail::cont    ("amp_attack",       "Attack",       "ms",  0.5f,   10000.0f, 3.0f,   500.0f),
    detail::cont    ("amp_decay",        "Decay",        "ms",  1.0f,   10000.0f, 300.0f, 500.0f),
    detail::cont    ("amp_sustain",      "Sustain",      "",    0.0f,   1.0f,     0.8f),
    detail::cont    ("amp_release",      "Release",      "ms",  1.0f,   10000.0f, 250.0f, 500.0f),
    detail::cont    ("amp_vel",          "Velocity",     "",    0.0f,   1.0f,     0.6f),

    detail::toggle  ("shaper_on",        "On",                  false),
    detail::choice  ("shaper_type",      "Type",         "Soft|Hard|Fold|Asym", 4, 0),
    detail::cont    ("shaper_drive",     "Drive",        "",    0.0f,   1.0f,     0.3f),
    detail::cont    ("shaper_mix",       "Mix",          "",    0.0f,   1.0f,     1.0f),

    detail::toggle  ("eq_on",            "On",                  false),
    detail::cont    ("eq_low_freq",      "Low Freq",     "Hz",  20.0f,  1000.0f,  120.0f,  150.0f),
    detail::cont    ("eq_low_gain",      "Low",          "dB",  -18.0f, 18.0f,    0.0f),
    detail::cont    ("eq_mid_freq",      "Mid Freq",     "Hz",  100.0f, 10000.0f, 1000.0f, 1000.0f),
    detail::cont    ("eq_mid_gain",      "Mid",          "dB",  -18.0f, 18.0f,    0.0f),
    detail::cont    ("eq_mid_q",         "Mid Q",        "",    0.1f,   10.0f,    0.7f,    1.0f),
    detail::cont    ("eq_high_freq",     "High Freq",    "Hz",  1000.0f, 20000.0f, 8000.0f, 5000.0f),
    detail::cont    ("eq_high_gain",     "High",         "dB",  -18.0f, 18.0f,    0.0f),

    detail::toggle  ("chorus_on",        "On",                  false),
    detail::cont    ("chorus_rate",      "Rate",         "Hz",  0.05f,  5.0f,     0.6f,    1.0f),
    detail::cont    ("chorus_depth",     "Depth",        "",    0.0f,   1.0f,     0.5f),
    detail::cont    ("chorus_mix",       "Mix",          "",    0.0f,   1.0f,     0.5f),

    detail::toggle  ("delay_on",         "On",                  false),
    detail::cont    ("delay_time",       "Time",         "ms",  1.0f,   2000.0f,  375.0f,  400.0f),
    detail::cont    ("delay_feedback",   "Feedback",     "",    0.0f,   0.95f,    0.35f),
    detail::cont    ("delay_damp",       "Damp",         "",    0.0f,   1.0f,     0.3f),
    detail::toggle  ("delay_pingpong",   "Ping-Pong",           true),
    detail::cont    ("delay_mix",        "Wet",          "",    0.0f,   1.0f,     0.25f),

    detail::toggle  ("reverb_on",        "On",                  false),
    detail::cont    ("reverb_size",      "Size",         "",    0.0f,   1.0f,     0.6f),
    detail::cont    ("reverb_damp",      "Damp",         "",    0.0f,   1.0f,     0.5f),
    detail::cont    ("reverb_width",     "Width",        "",    0.0f,   1.0f,     1.0f),
    detail::cont    ("reverb_mix",       "Wet",          "",    0.0f,   1.0f,     0.25f),
};

/** Index of a parameter id, or -1. Linear scan: init / preset-load time only. */
inline int findParamIndex (const char* id)
{
    for (int i = 0; i < kNumParams; ++i)
    {
        const char* a = kParams[i].id;
        const char* b = id;
        while (*a != '\0' && *a == *b) { ++a; ++b; }
        if (*a == '\0' && *b == '\0')
            return i;
    }
    return -1;
}

inline constexpr bool isSmoothed (ParamKind k)
{
    return k == ParamKind::Continuous || k == ParamKind::Toggle;
}

} // namespace ablaze
