/*
  FX building blocks for the Lite engine: an RBJ biquad and a circular delay
  line. Derived from Ablaze.WASMSynth@947be83 (source/engine/FxChain.h),
  trimmed to these two classes; DelayLine gained restart() (vae-synth).
*/

#pragma once

#include <juce_core/juce_core.h>

#include <cmath>
#include <vector>

namespace ablaze::fx
{

/** RBJ-cookbook biquad, transposed direct form II, stereo. */
class Biquad
{
public:
    enum class Shape { LowShelf, Peak, HighShelf };

    void set (Shape shape, float freq, float gainDb, float q, float sr) noexcept
    {
        const double A = std::pow (10.0, gainDb / 40.0);
        const double w0 = 2.0 * juce::MathConstants<double>::pi
                          * juce::jlimit (10.0, 0.49 * sr, static_cast<double> (freq)) / sr;
        const double cw = std::cos (w0), sw = std::sin (w0);
        double b0, b1, b2, a0, a1, a2;

        if (shape == Shape::Peak)
        {
            const double alpha = sw / (2.0 * q);
            b0 = 1.0 + alpha * A;  b1 = -2.0 * cw;  b2 = 1.0 - alpha * A;
            a0 = 1.0 + alpha / A;  a1 = -2.0 * cw;  a2 = 1.0 - alpha / A;
        }
        else
        {
            const double alpha = sw / 2.0 * std::sqrt (2.0);   // shelf slope S = 1
            const double sA = 2.0 * std::sqrt (A) * alpha;
            const double s = shape == Shape::LowShelf ? 1.0 : -1.0;
            b0 =     A * ((A + 1.0) - s * (A - 1.0) * cw + sA);
            b1 = s * 2.0 * A * ((A - 1.0) - s * (A + 1.0) * cw);
            b2 =     A * ((A + 1.0) - s * (A - 1.0) * cw - sA);
            a0 =          (A + 1.0) + s * (A - 1.0) * cw + sA;
            a1 = -s * 2.0 * ((A - 1.0) + s * (A + 1.0) * cw);
            a2 =          (A + 1.0) + s * (A - 1.0) * cw - sA;
        }

        c0 = static_cast<float> (b0 / a0);
        c1 = static_cast<float> (b1 / a0);
        c2 = static_cast<float> (b2 / a0);
        c3 = static_cast<float> (a1 / a0);
        c4 = static_cast<float> (a2 / a0);
    }

    float process (int ch, float x) noexcept
    {
        auto& z = state[ch];
        const float y = c0 * x + z[0];
        z[0] = c1 * x - c3 * y + z[1];
        z[1] = c2 * x - c4 * y;
        return y;
    }

    void reset() noexcept { state[0][0] = state[0][1] = state[1][0] = state[1][1] = 0.0f; }

private:
    float c0 = 1.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f, c4 = 0.0f;
    float state[2][2] = {};
};

/** Power-of-two circular delay line with fractional (linear) reads. */
class DelayLine
{
public:
    void prepare (int maxDelaySamples)
    {
        int size = 1;
        while (size < maxDelaySamples + 4)
            size <<= 1;
        buffer.assign (static_cast<size_t> (size), 0.0f);
        mask = size - 1;
        writePos = 0;
        written = 0;
    }

    /** Forget the history in O(1): taps older than what is pushed from now on
        read as silence, so no buffer has to be cleared on the audio thread. */
    void restart() noexcept { written = 0; }

    void push (float x) noexcept
    {
        buffer[static_cast<size_t> (writePos)] = x;
        writePos = (writePos + 1) & mask;
        if (written <= mask)
            ++written;
    }

    /** Sample written `delay` samples before the most recent push (delay >= 1). */
    float read (float delay) const noexcept
    {
        if (std::ceil (delay) > static_cast<float> (written))
            return 0.0f;
        const float pos = static_cast<float> (writePos) - delay;
        const int i0 = static_cast<int> (std::floor (pos));
        const float fr = pos - static_cast<float> (i0);
        const float a = buffer[static_cast<size_t> (i0 & mask)];
        const float b = buffer[static_cast<size_t> ((i0 + 1) & mask)];
        return a + fr * (b - a);
    }

private:
    std::vector<float> buffer;
    int mask = 0, writePos = 0, written = 0;
};

} // namespace ablaze::fx
