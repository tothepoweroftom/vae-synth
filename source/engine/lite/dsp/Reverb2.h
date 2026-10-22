/*
  A clean-room four-line feedback delay network in the spirit of Surge's
  Reverb 2. The topology and the fixed parameter defaults are behaviour facts
  taken from renders/specs; the implementation is original.

  Mono input, pre-delay, four input all-passes, four series blocks (two
  all-passes, an HF low-pass, an LF high-pass and a delay line with two output
  taps and a modulated feedback tap), then a linear dry/wet mix.
*/

#pragma once

#include "FxBlocks.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace vae::lite
{

class Reverb2
{
public:
    void prepare (double sampleRate)
    {
        sr = float (sampleRate);
        const int lineSize = int (0.5f * sr) + 8;
        predelay.prepare (int (0.12f * sr) + 8);
        for (auto& p : inputAllpass)
            p.prepare (int (0.10f * sr) + 8);
        for (auto& b : blockAllpass)
            for (auto& p : b)
                p.prepare (int (0.25f * sr) + 8);
        for (auto& d : delay)
            d.prepare (lineSize);
        reset();
    }

    void reset() noexcept
    {
        predelay.restart();
        for (auto& p : inputAllpass)
            p.restart();
        for (auto& b : blockAllpass)
            for (auto& p : b)
                p.restart();
        for (auto& d : delay)
            d.restart();
        hfState.fill (0.0f);
        lfState.fill (0.0f);
        state = 0.0f;
        lfoPhase = 0.0f;
    }

    /** size/decay are in Surge units: room size is bipolar, decay is log2 seconds. */
    void process (float* left, float* right, int n, float size, float decayLog2, float damping, float mix) noexcept
    {
        if (n <= 0)
            return;

        const float scale = std::exp2 (size);
        const float loopTime = 0.5508f * scale;
        const float decay = std::pow (0.001f, loopTime / (4.0f * std::exp2 (decayLog2)));
        const float diffusion = 0.7f;
        const float buildup = 0.7f;
        const float hfCoef = std::clamp (0.8f * damping, 0.01f, 0.99f);
        const float lfCoef = std::clamp (0.04f, 0.01f, 0.99f); // 0.2 * fixed LF damping 0.2
        const float modSamples = 0.5f * sr * 0.001f * 5.0f;
        const float lfoInc = juce::MathConstants<float>::twoPi * std::exp2 (-2.0f) / sr;
        const int pd = std::clamp (int (sr * std::exp2 (-4.0f)), 1, int (0.11f * sr));

        const int tapL[4] = { samples (80.3f, scale), samples (59.3f, scale), samples (97.7f, scale),
                              samples (122.6f, scale) };
        const int tapR[4] = { samples (35.5f, scale), samples (101.6f, scale), samples (73.9f, scale),
                              samples (80.3f, scale) };
        const int fbLen[4] = { samples (178.8f, scale), samples (126.5f, scale), samples (106.1f, scale),
                               samples (139.4f, scale) };
        const int apIn[4] = { samples (4.76f, scale), samples (6.81f, scale), samples (10.13f, scale),
                              samples (16.72f, scale) };
        const int apA[4] = { samples (38.2f, scale), samples (44.0f, scale), samples (48.3f, scale),
                             samples (38.9f, scale) };
        const int apB[4] = { samples (53.4f, scale), samples (41.0f, scale), samples (60.5f, scale),
                             samples (42.2f, scale) };
        const float tapGain[4] = { 1.5f / 4.0f, 1.2f / 4.0f, 1.0f / 4.0f, 0.8f / 4.0f };

        for (int i = 0; i < n; ++i)
        {
            lfoPhase += lfoInc;
            if (lfoPhase >= juce::MathConstants<float>::twoPi)
                lfoPhase -= juce::MathConstants<float>::twoPi;
            const float c = std::cos (lfoPhase), s = std::sin (lfoPhase);
            const std::array<float, 4> lfo { c, s, -c, -s };

            const float dryIn = 0.5f * (left[i] + right[i]);
            float in = predelay.read (float (pd));
            predelay.push (dryIn);

            for (int a = 0; a < 4; ++a)
                in = allpass (inputAllpass[(size_t) a], apIn[a], diffusion, in);

            float x = state;
            float outL = 0.0f, outR = 0.0f;
            for (int b = 0; b < 4; ++b)
            {
                x += in;
                x = allpass (blockAllpass[(size_t) b][0], apA[b], buildup, x);
                x = allpass (blockAllpass[(size_t) b][1], apB[b], buildup, x);

                hfState[(size_t) b] = hfState[(size_t) b] * hfCoef + x * (1.0f - hfCoef);
                x = hfState[(size_t) b];
                lfState[(size_t) b] = lfState[(size_t) b] * (1.0f - lfCoef) + x * lfCoef;
                x -= lfState[(size_t) b];

                const float tL = delay[(size_t) b].read (float (tapL[b]));
                const float tR = delay[(size_t) b].read (float (tapR[b]));
                const float fb = delay[(size_t) b].readCubic (float (fbLen[b]) + modSamples * lfo[(size_t) b]);
                delay[(size_t) b].push (x);

                outL += tL * tapGain[b];
                outR += tR * tapGain[b];
                x = fb * decay;
            }
            state = x;

            left[i] += mix * (outL - left[i]);
            right[i] += mix * (outR - right[i]);
        }
    }

private:
    int samples (float ms, float scale) const noexcept
    {
        return std::max (1, int (sr * ms * 0.001f * scale));
    }

    static float allpass (vae::dsp::fx::DelayLine& line, int len, float g, float x) noexcept
    {
        const float delayed = line.read (float (len));
        const float delayIn = x - g * delayed;
        const float y = delayed + g * delayIn;
        line.push (delayIn);
        return y;
    }

    float sr = 48000.0f;
    vae::dsp::fx::DelayLine predelay;
    std::array<vae::dsp::fx::DelayLine, 4> inputAllpass;
    std::array<std::array<vae::dsp::fx::DelayLine, 2>, 4> blockAllpass;
    std::array<vae::dsp::fx::DelayLine, 4> delay;
    std::array<float, 4> hfState {}, lfState {};
    float state = 0.0f;
    float lfoPhase = 0.0f;
};

} // namespace vae::lite
