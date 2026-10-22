/*
  FX building blocks for the Lite engine: an RBJ biquad and a circular delay
  line with an O(1) restart().
*/

#pragma once

#include <juce_core/juce_core.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace vae::dsp::fx
{

inline float softClip (float x) noexcept { return std::tanh (x); }
inline float hardClip (float x) noexcept { return std::clamp (x, -1.0f, 1.0f); }

/** The cubic soft clip Surge uses in several feedback paths. */
inline float cubicSoftClip (float x) noexcept
{
    x = std::clamp (x, -1.0f, 1.0f);
    return 1.5f * x - 0.5f * x * x * x;
}

/** RBJ-cookbook biquad, transposed direct form II, stereo. */
class Biquad
{
public:
    enum class Shape
    {
        LowShelf,
        Peak,
        HighShelf
    };

    void set (Shape shape, float freq, float gainDb, float q, float sr) noexcept
    {
        const double A = std::pow (10.0, gainDb / 40.0);
        const double w0 =
            2.0 * juce::MathConstants<double>::pi * juce::jlimit (10.0, 0.49 * sr, static_cast<double> (freq)) / sr;
        const double cw = std::cos (w0), sw = std::sin (w0);
        double b0, b1, b2, a0, a1, a2;

        if (shape == Shape::Peak)
        {
            const double alpha = sw / (2.0 * q);
            b0 = 1.0 + alpha * A;
            b1 = -2.0 * cw;
            b2 = 1.0 - alpha * A;
            a0 = 1.0 + alpha / A;
            a1 = -2.0 * cw;
            a2 = 1.0 - alpha / A;
        }
        else
        {
            const double alpha = sw / 2.0 * std::sqrt (2.0); // shelf slope S = 1
            const double sA = 2.0 * std::sqrt (A) * alpha;
            const double s = shape == Shape::LowShelf ? 1.0 : -1.0;
            b0 = A * ((A + 1.0) - s * (A - 1.0) * cw + sA);
            b1 = s * 2.0 * A * ((A - 1.0) - s * (A + 1.0) * cw);
            b2 = A * ((A + 1.0) - s * (A - 1.0) * cw - sA);
            a0 = (A + 1.0) + s * (A - 1.0) * cw + sA;
            a1 = -s * 2.0 * ((A - 1.0) + s * (A + 1.0) * cw);
            a2 = (A + 1.0) + s * (A - 1.0) * cw - sA;
        }

        setCoefficients (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0);
    }

    /** RBJ 2-pole low-pass with Q. */
    void setLowPass (float freq, float q, float sr) noexcept { set2Pole (false, freq, q, sr); }

    /** RBJ 2-pole high-pass with Q. */
    void setHighPass (float freq, float q, float sr) noexcept { set2Pole (true, freq, q, sr); }

    /** Second-order all-pass with Q. */
    void setAllPass (float freq, float q, float sr) noexcept
    {
        const double w0 =
            2.0 * juce::MathConstants<double>::pi * juce::jlimit (10.0, 0.49 * sr, static_cast<double> (freq)) / sr;
        const double cw = std::cos (w0), sw = std::sin (w0);
        const double alpha = sw / (2.0 * juce::jlimit (0.1, 10.0, static_cast<double> (q)));
        const double a0 = 1.0 + alpha;
        setCoefficients ((1.0 - alpha) / a0, (-2.0 * cw) / a0, (1.0 + alpha) / a0, (-2.0 * cw) / a0,
                         (1.0 - alpha) / a0);
    }

    /** RBJ peaking EQ with Q. */
    void setPeak (float freq, float q, float gainDb, float sr) noexcept
    {
        set (Shape::Peak, freq, gainDb, q, sr);
    }

    /** Orfanidis high-order peaking EQ, bandwidth in octaves. */
    void setOrfanidisPeak (float freq, float bwOctaves, float gainDb, float sr) noexcept
    {
        const double w0 =
            2.0 * juce::MathConstants<double>::pi * juce::jlimit (10.0, 0.49 * sr, static_cast<double> (freq)) / sr;
        const double bw = std::max (0.1, static_cast<double> (bwOctaves));
        const double G = std::pow (10.0, gainDb / 20.0);
        const double GB = std::pow (10.0, gainDb * 0.5 / 20.0);
        const double G0 = 1.0;
        const double Dww = 2.0 * w0 * std::sinh ((std::log (2.0) / 2.0) * bw);
        const double F = std::abs (G * G - GB * GB);
        const double G00 = std::abs (G * G - G0 * G0);
        const double F00 = std::abs (GB * GB - G0 * G0);

        if (F < 1.0e-12 || std::abs (G - G0) < 1.0e-5)
        {
            setCoefficients (1.0, 0.0, 0.0, 0.0, 0.0);
            return;
        }

        const double pi2 = juce::MathConstants<double>::pi * juce::MathConstants<double>::pi;
        const double num = G0 * G0 * std::pow (w0 * w0 - pi2, 2.0) + G * G * F00 * pi2 * Dww * Dww / F;
        const double den = std::pow (w0 * w0 - pi2, 2.0) + F00 * pi2 * Dww * Dww / F;
        const double G1 = std::sqrt (num / den);
        const double G01 = std::abs (G * G - G0 * G1);
        const double G11 = std::abs (G * G - G1 * G1);
        const double F01 = std::abs (GB * GB - G0 * G1);
        const double F11 = std::abs (GB * GB - G1 * G1);
        const double W2 = std::sqrt (G11 / G00) * std::pow (std::tan (w0 / 2.0), 2.0);
        const double wLower = w0 * std::pow (2.0, -0.5 * bw);
        const double wUpper = 2.0 * std::atan (std::sqrt (F00 / F11) * std::sqrt (G11 / G00)
                                              * std::pow (std::tan (w0 / 2.0), 2.0) / std::tan (wLower / 2.0));
        const double Dw = std::abs (wUpper - wLower);
        const double DW = (1.0 + std::sqrt (F00 / F11) * W2) * std::tan (Dw / 2.0);
        const double C = F11 * DW * DW - 2.0 * W2 * (F01 - std::sqrt (F00 * F11));
        const double D = 2.0 * W2 * (G01 - std::sqrt (G00 * G11));
        const double A = std::sqrt (std::max (0.0, (C + D) / F));
        const double B = std::sqrt (std::max (0.0, (G * G * C + GB * GB * D) / F));
        const double a0 = 1.0 + W2 + A;
        const double a1 = -2.0 * (1.0 - W2);
        const double a2 = 1.0 + W2 - A;
        const double b0 = G1 + G0 * W2 + B;
        const double b1 = -2.0 * (G1 - G0 * W2);
        const double b2 = G1 - B + G0 * W2;
        setCoefficients (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0);
    }

    float process (int ch, float x) noexcept
    {
        auto& z = state[ch];
        const float y = c0 * x + z[0];
        z[0] = c1 * x - c3 * y + z[1];
        z[1] = c2 * x - c4 * y;
        return y;
    }

    void reset() noexcept
    {
        state[0][0] = state[0][1] = state[1][0] = state[1][1] = 0.0f;
    }

private:
    void set2Pole (bool highPass, float freq, float q, float sr) noexcept
    {
        const double w0 =
            2.0 * juce::MathConstants<double>::pi * juce::jlimit (10.0, 0.49 * sr, static_cast<double> (freq)) / sr;
        const double cw = std::cos (w0), sw = std::sin (w0);
        const double alpha = sw / (2.0 * juce::jlimit (0.1, 10.0, static_cast<double> (q)));
        const double b0 = highPass ? (1.0 + cw) * 0.5 : (1.0 - cw) * 0.5;
        const double b1 = highPass ? -(1.0 + cw) : (1.0 - cw);
        const double b2 = b0;
        const double a0 = 1.0 + alpha;
        const double a1 = -2.0 * cw;
        const double a2 = 1.0 - alpha;
        setCoefficients (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0);
    }

    void setCoefficients (double b0, double b1, double b2, double a1, double a2) noexcept
    {
        c0 = static_cast<float> (b0);
        c1 = static_cast<float> (b1);
        c2 = static_cast<float> (b2);
        c3 = static_cast<float> (a1);
        c4 = static_cast<float> (a2);
    }

    float c0 = 1.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f, c4 = 0.0f;
    float state[2][2] = {};
};

/** Power-of-two circular delay line with fractional reads. */
class DelayLine
{
public:
    void prepare (int maxDelaySamples)
    {
        int size = 1;
        while (size < maxDelaySamples + 8)
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

    /** Catmull-Rom read, for short modulated delays (flanger, reverb taps). */
    float readCubic (float delay) const noexcept
    {
        if (delay + 2.0f > static_cast<float> (written))
            return 0.0f;
        const float pos = static_cast<float> (writePos) - delay;
        const int i1 = static_cast<int> (std::floor (pos));
        const float t = pos - static_cast<float> (i1);
        const float y0 = buffer[static_cast<size_t> ((i1 - 1) & mask)];
        const float y1 = buffer[static_cast<size_t> (i1 & mask)];
        const float y2 = buffer[static_cast<size_t> ((i1 + 1) & mask)];
        const float y3 = buffer[static_cast<size_t> ((i1 + 2) & mask)];
        const float a0 = -0.5f * y0 + 1.5f * y1 - 1.5f * y2 + 0.5f * y3;
        const float a1 = y0 - 2.5f * y1 + 2.0f * y2 - 0.5f * y3;
        const float a2 = -0.5f * y0 + 0.5f * y2;
        return ((a0 * t + a1) * t + a2) * t + y1;
    }

private:
    std::vector<float> buffer;
    int mask = 0, writePos = 0, written = 0;
};

} // namespace vae::dsp::fx
