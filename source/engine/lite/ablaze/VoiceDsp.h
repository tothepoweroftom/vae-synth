/*
  Per-voice DSP building blocks: PolyBLEP analog oscillator, TPT
  state-variable filter, noise. Header-only, allocation-free.
  Derived from Ablaze.WASMSynth@947be83 (source/engine/VoiceDsp.h), trimmed
  to what the Lite engine uses.
*/

#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

namespace ablaze
{

enum class AnalogWave : int { Sine, Triangle, Saw, Square };

/** NaN/Inf test that survives -ffast-math (which lets the compiler assume
    std::isfinite is always true): checks the exponent bits directly. */
inline bool isFiniteBits (float x) noexcept
{
    uint32_t u;
    std::memcpy (&u, &x, sizeof (u));
    return (u & 0x7f800000u) != 0x7f800000u;
}

//==============================================================================
/** Band-limited "analog" oscillator: PolyBLEP saw/square, PolyBLAMP
    triangle, pure sine. Phase in cycles [0, 1). */
class AnalogOsc
{
public:
    void setPhase (float p) noexcept      { phase = p; }
    void setIncrement (float inc) noexcept { dt = inc < 0.49f ? inc : 0.49f; }

    float next (AnalogWave wave) noexcept
    {
        const float p = phase;
        float y = 0.0f;

        switch (wave)
        {
            case AnalogWave::Sine:
                y = std::sin (6.2831853f * p);
                break;

            case AnalogWave::Triangle:
            {
                // Valley at p = 0, peak at p = 0.5; round both corners.
                y = 1.0f - 4.0f * std::abs (p - 0.5f);
                y += 4.0f * dt * (blamp (p) - blamp (wrap (p + 0.5f)));
                break;
            }

            case AnalogWave::Saw:
                y = 2.0f * p - 1.0f - blep (p);
                break;

            case AnalogWave::Square:
                y = (p < 0.5f ? 1.0f : -1.0f) + blep (p) - blep (wrap (p + 0.5f));
                break;
        }

        phase += dt;
        if (phase >= 1.0f)
            phase -= 1.0f;

        return y;
    }

private:
    static float wrap (float x) noexcept { return x >= 1.0f ? x - 1.0f : x; }

    float blep (float t) const noexcept
    {
        if (t < dt)
        {
            t /= dt;
            return t + t - t * t - 1.0f;
        }
        if (t > 1.0f - dt)
        {
            t = (t - 1.0f) / dt;
            return t * t + t + t + 1.0f;
        }
        return 0.0f;
    }

    float blamp (float t) const noexcept
    {
        if (t < dt)
        {
            t = t / dt - 1.0f;
            return -(1.0f / 3.0f) * t * t * t;
        }
        if (t > 1.0f - dt)
        {
            t = (t - 1.0f) / dt + 1.0f;
            return (1.0f / 3.0f) * t * t * t;
        }
        return 0.0f;
    }

    float phase = 0.0f;
    float dt = 0.0f;
};

//==============================================================================
/** Zavalishin/Simper TPT state-variable filter, one channel. Coefficients are
    shared across channels via SvfCoefs so stereo costs one tan(). */
struct SvfCoefs
{
    float a1 = 0.0f, a2 = 0.0f, a3 = 0.0f, k = 2.0f;

    /** g = tan(pi * cutoff / sampleRate), k = damping (2 = no resonance). */
    void set (float g, float k_) noexcept
    {
        k = k_;
        a1 = 1.0f / (1.0f + g * (g + k));
        a2 = g * a1;
        a3 = g * a2;
    }
};

class Svf
{
public:
    struct Out { float low, band, high; };

    Out process (float v0, const SvfCoefs& c) noexcept
    {
        const float v3 = v0 - ic2;
        const float v1 = c.a1 * ic1 + c.a2 * v3;
        const float v2 = ic2 + c.a2 * ic1 + c.a3 * v3;
        ic1 = 2.0f * v1 - ic1;
        ic2 = 2.0f * v2 - ic2;
        return { v2, v1, v0 - c.k * v1 - v2 };
    }

    void reset() noexcept { ic1 = ic2 = 0.0f; }

private:
    float ic1 = 0.0f, ic2 = 0.0f;
};

//==============================================================================
/** Cheap per-voice white noise (xorshift32). */
class Noise
{
public:
    void seed (uint32_t s) noexcept { state = s != 0 ? s : 0x9e3779b9u; }

    float next() noexcept
    {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return static_cast<float> (static_cast<int32_t> (state)) * (1.0f / 2147483648.0f);
    }

private:
    uint32_t state = 0x9e3779b9u;
};

} // namespace ablaze
