/*
  Post-engine stereo FX chain: waveshaper -> 3-band EQ -> chorus -> delay ->
  reverb. Processes in place, any block length. All buffers are allocated in
  prepare(); process() never allocates.

  Every stage has an "on" gain (the smoothed 0/1 toggle) that crossfades it
  in and out, so switching an effect never clicks. A stage whose gain has
  settled at 0 is skipped entirely (no CPU), and its state is cleared so a
  stale tail never reappears when it is switched back on.
*/

#pragma once

#include <juce_audio_basics/juce_audio_basics.h>

#include <cmath>
#include <vector>

#include "Params.h"

namespace ablaze
{

/** FX parameter values for one control block (already smoothed). */
struct FxParams
{
    float shaperOn = 0.0f;
    ShaperType shaperType = ShaperType::Soft;
    float shaperDrive = 0.3f, shaperMix = 1.0f;

    float eqOn = 0.0f;
    float eqLowFreq = 120.0f, eqLowGain = 0.0f;
    float eqMidFreq = 1000.0f, eqMidGain = 0.0f, eqMidQ = 0.7f;
    float eqHighFreq = 8000.0f, eqHighGain = 0.0f;

    float chorusOn = 0.0f;
    float chorusRate = 0.6f, chorusDepth = 0.5f, chorusMix = 0.5f;

    float delayOn = 0.0f;
    float delayTimeMs = 375.0f, delayFeedback = 0.35f, delayDamp = 0.3f, delayMix = 0.25f;
    bool delayPingPong = true;

    float reverbOn = 0.0f;
    float reverbSize = 0.6f, reverbDamp = 0.5f, reverbWidth = 1.0f, reverbMix = 0.25f;
};

//==============================================================================
namespace fx
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
    }

    void clear() noexcept { std::fill (buffer.begin(), buffer.end(), 0.0f); }

    void push (float x) noexcept
    {
        buffer[static_cast<size_t> (writePos)] = x;
        writePos = (writePos + 1) & mask;
    }

    /** Sample written `delay` samples before the most recent push (delay >= 1). */
    float read (float delay) const noexcept
    {
        const float pos = static_cast<float> (writePos) - delay;
        const int i0 = static_cast<int> (std::floor (pos));
        const float fr = pos - static_cast<float> (i0);
        const float a = buffer[static_cast<size_t> (i0 & mask)];
        const float b = buffer[static_cast<size_t> ((i0 + 1) & mask)];
        return a + fr * (b - a);
    }

private:
    std::vector<float> buffer;
    int mask = 0, writePos = 0;
};

} // namespace fx

//==============================================================================
class FxChain
{
public:
    void prepare (double sampleRate)
    {
        sr = static_cast<float> (sampleRate);
        // Same function, same input: silence through the Asym curve is exactly 0.
        asymOffset = std::tanh (kAsymBias);

        chorusL.prepare (static_cast<int> (0.05f * sr));
        chorusR.prepare (static_cast<int> (0.05f * sr));
        delayL.prepare (static_cast<int> (2.05f * sr));
        delayR.prepare (static_cast<int> (2.05f * sr));

        reverb.setSampleRate (sampleRate);
        reset();
    }

    void reset() noexcept
    {
        shaperDcL = shaperDcR = shaperDcInL = shaperDcInR = 0.0f;
        for (auto& b : eq)
            b.reset();
        chorusL.clear();
        chorusR.clear();
        delayL.clear();
        delayR.clear();
        delayLpL = delayLpR = 0.0f;
        reverb.reset();
        chorusActive = delayActive = reverbActive = eqActive = false;
    }

    void process (float* l, float* r, int n, const FxParams& p) noexcept
    {
        if (p.shaperOn > 0.0f)
            processShaper (l, r, n, p);

        if (p.eqOn > 0.0f)
            processEq (l, r, n, p);
        else if (eqActive)
        {
            for (auto& b : eq)
                b.reset();
            eqActive = false;
        }

        if (p.chorusOn > 0.0f)
            processChorus (l, r, n, p);
        else if (chorusActive)
        {
            chorusL.clear();
            chorusR.clear();
            chorusActive = false;
        }

        if (p.delayOn > 0.0f)
            processDelay (l, r, n, p);
        else if (delayActive)
        {
            delayL.clear();
            delayR.clear();
            delayLpL = delayLpR = 0.0f;
            delayActive = false;
        }

        if (p.reverbOn > 0.0f)
            processReverb (l, r, n, p);
        else if (reverbActive)
        {
            reverb.reset();
            reverbActive = false;
        }
    }

private:
    //--------------------------------------------------------------------------
    static constexpr float kAsymBias = 0.35f;

    float shape (ShaperType t, float x) const noexcept
    {
        switch (t)
        {
            case ShaperType::Soft: return std::tanh (x);
            case ShaperType::Hard: return juce::jlimit (-1.0f, 1.0f, x);
            case ShaperType::Fold: return std::sin (x * juce::MathConstants<float>::halfPi);
            case ShaperType::Asym: return std::tanh (x + kAsymBias) - asymOffset;
        }
        return x;
    }

    void processShaper (float* l, float* r, int n, const FxParams& p) noexcept
    {
        const float drive = 1.0f + p.shaperDrive * p.shaperDrive * 24.0f;
        const float comp = 1.0f / (1.0f + p.shaperDrive);
        const float wet = p.shaperMix * p.shaperOn;
        const float dcCoef = 1.0f - (2.0f * juce::MathConstants<float>::pi * 10.0f / sr);

        for (int i = 0; i < n; ++i)
        {
            float yl = shape (p.shaperType, l[i] * drive) * comp;
            float yr = shape (p.shaperType, r[i] * drive) * comp;

            // DC blocker (the asymmetric curve generates DC).
            const float ol = yl - shaperDcInL + dcCoef * shaperDcL;
            const float orr = yr - shaperDcInR + dcCoef * shaperDcR;
            shaperDcInL = yl;  shaperDcL = ol;
            shaperDcInR = yr;  shaperDcR = orr;

            l[i] += wet * (ol - l[i]);
            r[i] += wet * (orr - r[i]);
        }
    }

    //--------------------------------------------------------------------------
    void processEq (float* l, float* r, int n, const FxParams& p) noexcept
    {
        eq[0].set (fx::Biquad::Shape::LowShelf,  p.eqLowFreq,  p.eqLowGain,  0.707f,  sr);
        eq[1].set (fx::Biquad::Shape::Peak,      p.eqMidFreq,  p.eqMidGain,  p.eqMidQ, sr);
        eq[2].set (fx::Biquad::Shape::HighShelf, p.eqHighFreq, p.eqHighGain, 0.707f,  sr);
        eqActive = true;

        for (int i = 0; i < n; ++i)
        {
            float yl = l[i], yr = r[i];
            for (auto& b : eq)
            {
                yl = b.process (0, yl);
                yr = b.process (1, yr);
            }
            l[i] += p.eqOn * (yl - l[i]);
            r[i] += p.eqOn * (yr - r[i]);
        }
    }

    //--------------------------------------------------------------------------
    void processChorus (float* l, float* r, int n, const FxParams& p) noexcept
    {
        chorusActive = true;
        const float inc = p.chorusRate / sr;
        const float base = 0.012f * sr;                  // 12 ms centre
        const float depth = p.chorusDepth * 0.006f * sr; // up to +/- 6 ms
        const float wet = p.chorusMix * p.chorusOn;

        for (int i = 0; i < n; ++i)
        {
            chorusPhase += inc;
            if (chorusPhase >= 1.0f)
                chorusPhase -= 1.0f;

            const float twoPi = juce::MathConstants<float>::twoPi;
            const float modL = std::sin (twoPi * chorusPhase);
            const float modR = std::cos (twoPi * chorusPhase);   // 90 degrees apart

            chorusL.push (l[i]);
            chorusR.push (r[i]);
            const float wl = chorusL.read (base + depth * modL);
            const float wr = chorusR.read (base + depth * modR);

            l[i] += wet * (wl - l[i]);
            r[i] += wet * (wr - r[i]);
        }
    }

    //--------------------------------------------------------------------------
    void processDelay (float* l, float* r, int n, const FxParams& p) noexcept
    {
        delayActive = true;
        const float d = juce::jlimit (1.0f, 2.0f * sr, p.delayTimeMs * 0.001f * sr);
        const float dampHz = 18000.0f * (1.0f - p.delayDamp) * (1.0f - p.delayDamp) + 400.0f;
        const float lp = std::exp (-2.0f * juce::MathConstants<float>::pi * dampHz / sr);
        const float fb = p.delayFeedback;
        const float wet = p.delayMix * p.delayOn;

        for (int i = 0; i < n; ++i)
        {
            const float wl = delayL.read (d);
            const float wr = delayR.read (d);

            // One-pole low-pass in the feedback path (tape-ish darkening).
            delayLpL = wl + lp * (delayLpL - wl);
            delayLpR = wr + lp * (delayLpR - wr);

            if (p.delayPingPong)
            {
                delayL.push (0.5f * (l[i] + r[i]) + fb * delayLpR);
                delayR.push (fb * delayLpL);
            }
            else
            {
                delayL.push (l[i] + fb * delayLpL);
                delayR.push (r[i] + fb * delayLpR);
            }

            l[i] += wet * wl;
            r[i] += wet * wr;
        }
    }

    //--------------------------------------------------------------------------
    void processReverb (float* l, float* r, int n, const FxParams& p) noexcept
    {
        reverbActive = true;

        juce::Reverb::Parameters rp;
        rp.roomSize = p.reverbSize;
        rp.damping = p.reverbDamp;
        rp.width = p.reverbWidth;
        rp.wetLevel = p.reverbMix * p.reverbOn / 3.0f;   // juce::Reverb scales wet by 3
        rp.dryLevel = 0.5f;                              // ... and dry by 2: unity dry
        rp.freezeMode = 0.0f;
        reverb.setParameters (rp);

        reverb.processStereo (l, r, n);
    }

    float sr = 48000.0f;

    float asymOffset = 0.0f;
    float shaperDcL = 0.0f, shaperDcR = 0.0f, shaperDcInL = 0.0f, shaperDcInR = 0.0f;

    fx::Biquad eq[3];
    bool eqActive = false;

    fx::DelayLine chorusL, chorusR;
    float chorusPhase = 0.0f;
    bool chorusActive = false;

    fx::DelayLine delayL, delayR;
    float delayLpL = 0.0f, delayLpR = 0.0f;
    bool delayActive = false;

    juce::Reverb reverb;
    bool reverbActive = false;
};

} // namespace ablaze
