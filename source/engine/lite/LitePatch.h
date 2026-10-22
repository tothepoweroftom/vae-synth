/*
  The Lite patch: schema v2 controls converted to physical values (seconds,
  semitones, Hz, linear gains). Rebuilt per control change on the audio
  thread by LiteEngine::setControl; voices and FX read it at control rate.
*/

#pragma once

#include "schema/SurgeCurves.h"

#include <array>

namespace vae::lite
{

enum class PlayMode
{
    Poly,
    Mono,
    Legato
};
enum class Wave
{
    Sine,
    Triangle,
    Saw,
    Square,
    Noise,
    Wavetable,
    Fm
};
enum class Route
{
    Filter1,
    Both,
    Filter2
};
enum class FilterType
{
    Off,
    Lp12,
    Lp24,
    Hp12,
    Hp24,
    Bp,
    Notch
};
enum class LfoShape
{
    Sine,
    Triangle,
    Square,
    Saw,
    Random,
    SmoothRandom
};
enum class ModDest
{
    None,
    Pitch,
    Filter1Cutoff,
    Filter2Cutoff,
    Filter1Resonance,
    Volume,
    Pan
};
enum class ModFxType
{
    Phaser,
    Flanger
};

/** Envelope stage curve as an exponent applied in the shaped domain (Surge
    style): attack convex/linear/concave, decay/release linear/quad/cubic. */
struct EnvSettings
{
    float attack = 0.004f, decay = 0.25f, sustain = 1.0f, release = 0.03f; // seconds / level
    float attackPower = 1.0f, decayPower = 2.0f, releasePower = 3.0f;
};

struct OscSettings
{
    bool on = false;
    Wave wave = Wave::Saw;
    float gain = 1.0f;      // linear (Surge amplitude x^3)
    float semitones = 0.0f; // octave * 12 + pitch
    Route route = Route::Both;
};

struct FilterSettings
{
    FilterType type = FilterType::Off;
    float cutoffSemis = 3.0f; // from A440
    float resonance = 0.0f;
    float envSemis = 0.0f; // at full filter EG
    float keytrack = 0.0f; // -1..1, 1 = 100 %
};

struct ModRoute
{
    ModDest dest = ModDest::None;
    float depth = 0.0f; // -1..1
};

struct LfoSettings
{
    LfoShape shape = LfoShape::Sine;
    float rateHz = 1.0f;
    float amount = 1.0f;
    bool unipolar = false;
    bool retrigger = true;
    float phase = 0.0f;
    float fadeIn = 0.0f; // seconds, 0 = none
    ModRoute route;
};

struct FxSettings
{
    bool distOn = false;
    float distDriveDb = 4.0f;
    float distToneHz = 440.0f;
    bool eqOn = false;
    float eqLowDb = 0.0f, eqMidHz = 622.0f, eqMidDb = 0.0f, eqHighDb = 0.0f;
    bool chorusOn = false;
    float chorusRateHz = 0.25f, chorusDepth = 0.3f, chorusMix = 1.0f;
    bool modfxOn = false;
    ModFxType modfxType = ModFxType::Phaser;
    float modfxRateHz = 1.0f, modfxDepth = 0.5f, modfxFeedback = 0.0f, modfxMix = 0.5f;
    bool delayOn = false;
    float delaySeconds = 0.25f, delayFeedback = 0.5f, delayMix = 0.5f;
    bool reverbOn = false;
    float reverbSize = 0.0f;    // Surge room-size units (bipolar)
    float reverbDecay = 0.75f;  // Surge log2-seconds (0.75 -> 1.68 s)
    float reverbDamping = 0.2f, reverbMix = 0.33f;
};

struct Patch
{
    PlayMode playMode = PlayMode::Poly;
    float glideSeconds = 0.0f;
    std::array<OscSettings, 3> osc {};
    bool parallel = true;
    std::array<FilterSettings, 2> filter {};
    EnvSettings ampEnv, filterEnv;
    std::array<LfoSettings, 2> lfo {};
    ModRoute fegRoute;
    FxSettings fx;
    float masterGain = 1.0f;

    /** Rebuild every field from a full control vector. */
    void setAll (const schema::Snapshot& s)
    {
        for (int i = 0; i < schema::kControlCount; ++i)
            set (i, s);
    }

    /** Rebuild the field(s) that depend on control `index`. */
    void set (int index, const schema::Snapshot& s);
};

} // namespace vae::lite
