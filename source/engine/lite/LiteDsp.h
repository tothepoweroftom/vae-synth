/*
  Per-voice DSP for the Lite engine: shaped ADSR, LFO, dual-SVF filter unit
  and a multi-wave oscillator. Header-only, allocation-free. Oscillator,
  filter core and noise are the building blocks in dsp/VoiceDsp.h.
*/

#pragma once

#include "LitePatch.h"
#include "dsp/VoiceDsp.h"
#include "dsp/Wavetables.h"

#include <algorithm>
#include <cmath>

namespace vae::lite
{

inline constexpr float kTwoPi = 6.28318530718f;

/** x^p for the exponents the envelope uses (0.5, 1, 2, 3), cheaply. */
inline float shapePow (float x, float p) noexcept
{
    if (p < 0.75f)
        return std::sqrt (x);
    if (p < 1.5f)
        return x;
    if (p < 2.5f)
        return x * x;
    return x * x * x;
}

/** x^(1/p): inverse of shapePow. */
inline float shapeRoot (float y, float p) noexcept
{
    if (p < 0.75f)
        return y * y;
    if (p < 1.5f)
        return y;
    if (p < 2.5f)
        return std::sqrt (y);
    return std::cbrt (y);
}

/** Surge-style ADSR: each stage moves a shaped-domain value x linearly at
    1/time per second across the full 0..1 range, and the output is x^power
    (attack convex/linear/concave = 0.5/1/2, decay/release linear/quadratic/
    cubic = 1/2/3). noteOn() restarts the attack from the current level. */
class ShapedEnvelope
{
public:
    enum class Stage
    {
        Idle,
        Attack,
        Decay,
        Release
    };

    void noteOn (const EnvSettings& e) noexcept
    {
        stage = Stage::Attack;
        x = shapeRoot (level, e.attackPower);
    }

    void noteOff (const EnvSettings& e) noexcept
    {
        if (stage == Stage::Idle)
            return;
        stage = Stage::Release;
        x = shapeRoot (level, e.releasePower);
    }

    void reset() noexcept
    {
        stage = Stage::Idle;
        level = 0.0f;
        x = 0.0f;
    }
    bool isActive() const noexcept { return stage != Stage::Idle; }
    bool isReleasing() const noexcept { return stage == Stage::Release; }
    float getLevel() const noexcept { return level; }

    float next (const EnvSettings& e, float dt) noexcept
    {
        switch (stage)
        {
            case Stage::Attack:
                x += dt / std::max (e.attack, 1.0e-4f);
                if (x >= 1.0f)
                {
                    level = 1.0f;
                    stage = Stage::Decay;
                    x = 1.0f;
                    break;
                }
                level = shapePow (x, e.attackPower);
                break;
            case Stage::Decay:
            {
                const float target = shapeRoot (e.sustain, e.decayPower);
                if (x > target)
                    x = std::max (target, x - dt / std::max (e.decay, 1.0e-4f));
                else
                    x = target; // sustain moved up: follow it
                level = shapePow (x, e.decayPower);
                break;
            }
            case Stage::Release:
                x -= dt / std::max (e.release, 1.0e-4f);
                if (x <= 0.0f)
                {
                    reset();
                    break;
                }
                level = shapePow (x, e.releasePower);
                break;
            case Stage::Idle: break;
        }
        return level;
    }

private:
    Stage stage = Stage::Idle;
    float level = 0.0f;
    float x = 0.0f;
};

/** LFO core: phase in cycles, bipolar output -1..1. */
class Lfo
{
public:
    void seed (uint32_t s) noexcept
    {
        rng.seed (s);
        current = rng.next();
        target = rng.next();
    }

    void restart (float startPhase) noexcept
    {
        phase = startPhase - std::floor (startPhase);
        current = rng.next();
        target = rng.next();
    }

    /** Advance by `seconds` at `rateHz`; returns the value at the new phase. */
    float advance (const LfoSettings& s, float seconds) noexcept
    {
        phase += s.rateHz * seconds;
        while (phase >= 1.0f)
        {
            phase -= 1.0f;
            current = target;
            target = rng.next();
        }
        return value (s.shape);
    }

    float value (LfoShape shape) const noexcept
    {
        switch (shape)
        {
            case LfoShape::Sine: return std::sin (kTwoPi * phase);
            case LfoShape::Triangle:
                return phase < 0.25f ? 4.0f * phase : phase < 0.75f ? 2.0f - 4.0f * phase : 4.0f * phase - 4.0f;
            case LfoShape::Square: return phase < 0.5f ? 1.0f : -1.0f;
            case LfoShape::Saw:    return 2.0f * phase - 1.0f;
            case LfoShape::Random: return current;
            case LfoShape::SmoothRandom:
            {
                const float t = 0.5f - 0.5f * std::cos (kTwoPi * 0.5f * phase);
                return current + t * (target - current);
            }
        }
        return 0.0f;
    }

private:
    float phase = 0.0f;
    float current = 0.0f, target = 0.0f;
    dsp::Noise rng;
};

/** One filter slot (mono): SVF, two cascaded for the 24 dB types. Damping is
    recomputed per sample so it can rise with the band signal; that bounds the
    self-oscillation once resonance pushes the damping below zero. */
class FilterUnit
{
public:
    void reset() noexcept
    {
        a.reset();
        b.reset();
        bandA = bandB = 0.0f;
    }

    /** Set coefficients for this control block. */
    void configure (FilterType t, float cutoffHz, float resonance, float sampleRate) noexcept
    {
        type = t;
        if (t == FilterType::Off)
            return;
        const float hz = std::clamp (cutoffHz, 13.75f, 0.45f * sampleRate);
        g = std::tan (3.14159265f * hz / sampleRate);
        const float r = std::clamp (resonance, 0.0f, 1.0f);
        const bool fourPole = t == FilterType::Lp24 || t == FilterType::Hp24;
        // Damping vs resonance, fitted to Surge renders: 2 at r = 0, crossing 0
        // (self-oscillation) near r = 0.87 for 2-pole and r = 0.9 for 4-pole.
        k = 2.0f - 2.15f * std::pow ((fourPole ? 0.97f : 1.0f) * r, 0.54f);
        // Output level falls with resonance (fitted to Surge renders: -1.4 dB at 0.05, -9 dB at 1).
        gain = (2.0f - 2.15f * std::pow (r, 0.54f) + 1.27f) / 3.27f;
        // Extra damping per |band|^3; sets the ringing level, fitted to Surge renders.
        limit = fourPole ? 0.012f : 0.005f;
        if (t == FilterType::Notch)
        {
            // Notch width vs resonance, fitted to Surge renders; unity gain, never rings.
            k = 5.1f * (1.0f - r) * (1.0f - r) + 0.02f;
            gain = 1.0f;
        }
    }

    float process (float x) noexcept
    {
        switch (type)
        {
            case FilterType::Off:  return x;
            case FilterType::Lp12: return gain * stage (a, bandA, x, k).low;
            case FilterType::Lp24:
                return gain * stage (b, bandB, stage (a, bandA, x, std::max (k, kFirstStageMinDamping)).low, k).low;
            case FilterType::Hp12: return gain * stage (a, bandA, x, k).high;
            case FilterType::Hp24:
                return gain * stage (b, bandB, stage (a, bandA, x, std::max (k, kFirstStageMinDamping)).high, k).high;
            case FilterType::Bp: return gain * stage (a, bandA, x, k).band;
            case FilterType::Notch:
            {
                const auto o = stage (a, bandA, x, k);
                return o.low + o.high;
            }
        }
        return x;
    }

private:
    // Only the last stage of a 24 dB pair self-oscillates (best match to Surge renders).
    dsp::Svf::Out stage (dsp::Svf& s, float& band, float x, float kStage) noexcept
    {
        coefs.set (g, kStage + limit * band * band * std::abs (band));
        const auto o = s.process (x, coefs);
        band = o.band;
        return o;
    }

    static constexpr float kFirstStageMinDamping = 0.3f; // 24 dB first stage never rings; fitted to Surge renders

    FilterType type = FilterType::Off;
    float g = 0.0f, k = 2.0f, gain = 1.0f, limit = 0.0f, bandA = 0.0f, bandB = 0.0f;
    dsp::SvfCoefs coefs;
    dsp::Svf a, b;
};

/** One oscillator slot: every schema wave, band-limited where it matters. */
class Oscillator
{
public:
    void seed (uint32_t s) noexcept { noise.seed (s); }

    void resetPhase() noexcept
    {
        analog.setPhase (0.0f);
        wtPhase = carrier = modulator = 0.0f;
    }

    /** Like Surge without osc retrigger: a random start phase, so stacked
        oscillators sum incoherently (+3 dB on average, not +6 dB). */
    void randomisePhase() noexcept
    {
        resetPhase();
        wtPhase = carrier = 0.5f + 0.5f * noise.next();
        analog.setPhase (wtPhase);
    }

    /** Per control block: frequency as a phase increment (cycles/sample). */
    void setIncrement (float inc, Wave wave, const dsp::Wavetable* table) noexcept
    {
        increment = std::min (inc, 0.49f);
        analog.setIncrement (increment);
        if (wave == Wave::Wavetable && table != nullptr)
        {
            const int level = dsp::Wavetable::levelForMaxHarmonics (0.5f / std::max (1.0e-6f, increment));
            frame = table->getFrame (level, 0);
            frameSize = float (dsp::Wavetable::levelSize (level));
        }
    }

    float next (Wave wave) noexcept
    {
        switch (wave)
        {
            case Wave::Sine:     return kHotOscGain * analog.next (dsp::AnalogWave::Sine);
            case Wave::Triangle: return analog.next (dsp::AnalogWave::Triangle);
            case Wave::Saw:      return analog.next (dsp::AnalogWave::Saw);
            case Wave::Square:   return analog.next (dsp::AnalogWave::Square);
            case Wave::Noise:
            {
                holdPhase += kNoiseHoldRate * increment;
                if (holdPhase >= 1.0f)
                {
                    holdPhase -= std::floor (holdPhase);
                    held = noise.next();
                }
                return kNoiseGain * held;
            }
            case Wave::Wavetable:
            {
                if (frame == nullptr)
                    return 0.0f;
                const float idx = wtPhase * frameSize;
                const int i0 = int (idx);
                const float x = frame[i0] + (idx - float (i0)) * (frame[i0 + 1] - frame[i0]);
                wtPhase += increment;
                if (wtPhase >= 1.0f)
                    wtPhase -= 1.0f;
                return kHotOscGain * x;
            }
            case Wave::Fm:
            {
                // 2-operator phase modulation, modulator at 2:1.
                const float y = std::sin (kTwoPi * carrier + kFmIndex * std::sin (kTwoPi * modulator));
                carrier += increment;
                modulator += kFmRatio * increment;
                carrier -= std::floor (carrier);
                modulator -= std::floor (modulator);
                return kHotOscGain * y;
            }
        }
        return 0.0f;
    }

    static constexpr float kFmRatio = 2.0f;
    static constexpr float kFmIndex = 0.68f; // the Surge backend's FM2 M1 amount 0.3: 0.3^3 * 8 pi rad
    static constexpr float kNoiseGain = 1.4f;
    // Surge holds 32 values per cycle but band-limits the steps; naive S&H at 24 matches its brightness.
    static constexpr float kNoiseHoldRate = 24.0f;
    static constexpr float kHotOscGain = 2.0f; // Surge's Sine, Wavetable and FM2 oscs run ~6 dB above its Classic saw

private:
    dsp::AnalogOsc analog;
    dsp::Noise noise;
    float increment = 0.0f;
    float wtPhase = 0.0f, carrier = 0.0f, modulator = 0.0f;
    float holdPhase = 0.0f, held = 0.0f;
    const float* frame = nullptr;
    float frameSize = 256.0f;
};

} // namespace vae::lite
