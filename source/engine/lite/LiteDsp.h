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

/** One filter slot (mono): Surge's two-step SVF, cascaded for the 24 dB types.
    The state-scaling limiter bounds self-oscillation without the old
    band-dependent damping fit. */
class FilterUnit
{
public:
    void reset() noexcept
    {
        a.reset();
        b.reset();
        rState = 1.0f;
    }

    /** Set coefficients for this control block. */
    void configure (FilterType t, float cutoffHz, float resonance, float sampleRate) noexcept
    {
        type = t;
        if (t == FilterType::Off)
            return;

        const float reso = std::clamp (resonance, 0.0f, 1.0f);
        const float rs = std::sqrt (reso);
        const bool fourPole = t == FilterType::Lp24 || t == FilterType::Hp24;
        const float hz = std::clamp (cutoffHz, 13.75f, 0.44f * sampleRate);

        // Surge's coefficient at 2x: F1 = 2 sin(pi f / (2 fs_os)), fs_os = 2 fs.
        // Each base sample is processed as two OS samples (zero-order hold in,
        // box average out), which is the filter Surge runs at 2x.
        F = 2.0f * std::sin (3.14159265f * std::min (0.11f, hz / (4.0f * sampleRate)));
        const float overshoot = fourPole ? 0.10f : 0.15f;
        Q = 2.0f - rs * (2.0f + overshoot) + F * F * overshoot * 0.9f;
        Q = std::min (Q, std::min (2.0f, 2.0f - 1.52f * F));
        clipDamp = 0.1f * rs * F;
        gain = 1.0f - 0.65f * rs;

        // The zero-order-hold 2x input loses a little resonance level at low
        // cutoffs; fit a small makeup that vanishes above ~1 kHz. (The 2x
        // filter already matches the high-cutoff levels without it.)
        if (t != FilterType::Notch)
        {
            const float lowCutWeight = std::max (0.0f, 1.0f - F / 0.12f);
            const float resoWeight = std::pow (reso, fourPole ? 12.0f : 8.0f);
            gain *= 1.0f + (fourPole ? 0.0f : 0.65f) * resoWeight * lowCutWeight;
        }

        if (t == FilterType::Notch)
        {
            // Surge's notch damping; no self-oscillation, unity gain.
            Q = 2.5f - 2.49f * (1.0f - (1.0f - reso) * (1.0f - reso));
            clipDamp = 0.0f;
            gain = 1.0f;
        }
        rState = 1.0f;
    }

    float process (float x) noexcept
    {
        float r = rState;
        switch (type)
        {
            case FilterType::Off:  return x;
            case FilterType::Lp12:
            {
                const auto o1 = a.process (x, F, Q, r);
                r = limit (o1.band);
                const auto o2 = a.process (x, F, Q, r);
                rState = limit (o2.band);
                return gain * 0.5f * (o1.low + o2.low);
            }
            case FilterType::Hp12:
            {
                const auto o1 = a.process (x, F, Q, r);
                r = limit (o1.band);
                const auto o2 = a.process (x, F, Q, r);
                rState = limit (o2.band);
                return gain * 0.5f * (o1.high + o2.high);
            }
            case FilterType::Bp:
            {
                const auto o1 = a.process (x, F, Q, r);
                r = limit (o1.band);
                const auto o2 = a.process (x, F, Q, r);
                rState = limit (o2.band);
                return gain * 0.5f * (o1.band + o2.band);
            }
            case FilterType::Notch:
            {
                const auto o1 = a.process (x, F, Q, 1.0f);
                const auto o2 = a.process (x, F, Q, 1.0f);
                return 0.5f * (o1.low + o1.high + o2.low + o2.high);
            }
            case FilterType::Lp24:
            {
                const auto o1 = a.process (x, F, Q, r);
                r = limit (o1.band);
                const auto o2 = a.process (x, F, Q, r);
                r = limit (o2.band);
                const auto p1 = b.process (o1.low, F, Q, r);
                r = limit (p1.band);
                const auto p2 = b.process (o2.low, F, Q, r);
                rState = limit (p2.band);
                return gain * 0.5f * (p1.low + p2.low);
            }
            case FilterType::Hp24:
            {
                const auto o1 = a.process (x, F, Q, r);
                r = limit (o1.band);
                const auto o2 = a.process (x, F, Q, r);
                r = limit (o2.band);
                const auto p1 = b.process (o1.high, F, Q, r);
                r = limit (p1.band);
                const auto p2 = b.process (o2.high, F, Q, r);
                rState = limit (p2.band);
                return gain * 0.5f * (p1.high + p2.high);
            }
        }
        return x;
    }

private:
    float limit (float band) const noexcept
    {
        return std::max (0.1f, 1.0f - clipDamp * band * band);
    }

    FilterType type = FilterType::Off;
    float F = 0.0f, Q = 2.0f, clipDamp = 0.0f, gain = 1.0f, rState = 1.0f;
    dsp::Svf2 a, b;
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
                const float t = idx - float (i0);
                const int n = int (frameSize);
                auto sampleAt = [this, n] (int i)
                {
                    if (i < 0)
                        i += n;
                    else if (i > n)
                        i -= n;
                    return frame[i];
                };
                // Catmull-Rom: Surge reads the table with a windowed-sinc FIR;
                // cubic is a cheap step in the same direction.
                const float y0 = sampleAt (i0 - 1), y1 = sampleAt (i0);
                const float y2 = sampleAt (i0 + 1), y3 = sampleAt (i0 + 2);
                const float a0 = -0.5f * y0 + 1.5f * y1 - 1.5f * y2 + 0.5f * y3;
                const float a1 = y0 - 2.5f * y1 + 2.0f * y2 - 0.5f * y3;
                const float a2 = -0.5f * y0 + 0.5f * y2;
                const float x = ((a0 * t + a1) * t + a2) * t + y1;
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
    // Surge's S&H holds 32 values per note cycle; 32 matches its spectrum.
    static constexpr float kNoiseHoldRate = 32.0f;
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
