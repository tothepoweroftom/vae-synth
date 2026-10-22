/*
  The Lite FX chain, in the schema's fixed order:
    distortion -> EQ -> chorus -> phaser/flanger -> delay -> reverb
  Each block is skipped when off; its state is forgotten when it switches off
  so a stale tail never reappears (delay lines restart in O(1) rather than
  clearing their buffers on the audio thread). Buffers are allocated in
  prepare() only.
*/

#pragma once

#include "LitePatch.h"
#include "dsp/FxBlocks.h"
#include "dsp/Reverb2.h"

#include <juce_audio_basics/juce_audio_basics.h>

#include <array>
#include <cmath>

namespace vae::lite
{

class FxRack
{
public:
    static constexpr float kMaxDelaySeconds = 4.0f;
    static constexpr float kChorusTimeSeconds = 0.015625f; // Surge's -6 -> 2^-6 s
    static constexpr float kDistPreHighCutHz = 440.0f; // pre high cut, 4x-effective
    static constexpr float kFlangerBasePitchHz = 261.625565f; // MIDI 60
    static constexpr float kFlangerDamping = 0.1f;

    void prepare (double sampleRate)
    {
        sr = float (sampleRate);
        chorus[0].prepare (int (0.05f * sr));
        chorus[1].prepare (int (0.05f * sr));
        flanger[0].prepare (int (0.02f * sr));
        flanger[1].prepare (int (0.02f * sr));
        delay[0].prepare (int ((kMaxDelaySeconds + 0.05f) * sr));
        delay[1].prepare (int ((kMaxDelaySeconds + 0.05f) * sr));
        reverb.prepare (sampleRate);

        for (int i = 0; i < 4; ++i)
        {
            const float x = 2.0f * float (i) / 3.0f - 1.0f;
            chorusPanL[(size_t) i] = std::sqrt (0.5f - 0.5f * x) * 0.5f;
            chorusPanR[(size_t) i] = std::sqrt (0.5f + 0.5f * x) * 0.5f;
        }
        reset();
    }

    void reset() noexcept
    {
        resetDist();
        resetEq();
        resetChorus();
        resetModFx();
        resetDelay();
        reverb.reset();
        wasOn = {};
    }

    void process (float* l, float* r, int n, const FxSettings& p) noexcept
    {
        run (0, p.distOn, [&] { processDist (l, r, n, p); }, [&] { resetDist(); });
        run (1, p.eqOn, [&] { processEq (l, r, n, p); }, [&] { resetEq(); });
        run (2, p.chorusOn, [&] { processChorus (l, r, n, p); }, [&] { resetChorus(); });
        run (3, p.modfxOn, [&] { processModFx (l, r, n, p); }, [&] { resetModFx(); });
        run (4, p.delayOn, [&] { processDelay (l, r, n, p); }, [&] { resetDelay(); });
        run (5, p.reverbOn, [&] { processReverb (l, r, n, p); }, [&] { reverb.reset(); });
    }

private:
    template <typename Process, typename Reset>
    void run (int slot, bool on, Process&& process, Reset&& reset) noexcept
    {
        if (on)
            process();
        else if (wasOn[(size_t) slot])
            reset();
        wasOn[(size_t) slot] = on;
    }

    static float dbToGain (float db) noexcept { return std::pow (10.0f, db / 20.0f); }

    // Distortion: 2-pole pre-LP, rational-ish tanh, 2-pole post-LP at the tone.
    void processDist (float* l, float* r, int n, const FxSettings& p) noexcept
    {
        const float drive = dbToGain (p.distDriveDb);
        const float postHz = std::max (10.0f, p.distToneHz);
        distPre.setLowPass (kDistPreHighCutHz, 0.707f, sr);
        distPost.setLowPass (postHz, 0.707f, sr);

        for (int ch = 0; ch < 2; ++ch)
        {
            float* x = ch == 0 ? l : r;
            for (int i = 0; i < n; ++i)
            {
                float y = distPre.process (ch, x[i]);
                y = std::tanh (y * drive);
                x[i] = distPost.process (ch, y);
            }
        }
    }
    void resetDist() noexcept
    {
        distPre.reset();
        distPost.reset();
    }

    // EQ: three peaking bands at Surge's fixed outer frequencies, BW = 2 oct.
    // The settings ramp across the span in 16-sample steps: new coefficients
    // only at block edges click (Task 18).
    void processEq (float* l, float* r, int n, const FxSettings& p) noexcept
    {
        const std::array<float, 4> target { p.eqLowDb, p.eqMidHz, p.eqMidDb, p.eqHighDb };
        const auto from = eqPrimed ? eqNow : target;
        for (int start = 0; start < n; start += kEqStep)
        {
            const int len = std::min (kEqStep, n - start);
            if (start == 0 || from != target)
            {
                const float t = from == target ? 1.0f : float (start + len) / float (n);
                for (size_t k = 0; k < 4; ++k)
                    eqNow[k] = from[k] + (target[k] - from[k]) * t;
                eq[0].setOrfanidisPeak (77.8f, 2.0f, eqNow[0], sr);
                eq[1].setOrfanidisPeak (eqNow[1], 2.0f, eqNow[2], sr);
                eq[2].setOrfanidisPeak (13400.0f, 2.0f, eqNow[3], sr);
            }
            for (int i = start; i < start + len; ++i)
                for (auto& b : eq)
                {
                    l[i] = b.process (0, l[i]);
                    r[i] = b.process (1, r[i]);
                }
        }
        eqNow = target;
        eqPrimed = true;
    }
    void resetEq() noexcept
    {
        for (auto& b : eq)
            b.reset();
        eqPrimed = false;
    }

    // Chorus: 4 voices on one delay line per channel, 1/sqrt(4) gain,
    // equal-power pans, triangle LFOs 1/3 apart, filtered feedback.
    void processChorus (float* l, float* r, int n, const FxSettings& p) noexcept
    {
        const float inc = p.chorusRateHz / sr;
        const float depth = std::clamp (p.chorusDepth, 0.0f, 1.0f);
        const float base = kChorusTimeSeconds * sr;
        const float mix = std::clamp (p.chorusMix, 0.0f, 1.0f);
        chorusHp.setHighPass (55.0f, 0.707f, sr);
        chorusLp.setLowPass (3520.0f, 0.707f, sr);

        for (int i = 0; i < n; ++i)
        {
            chorusPhase += inc;
            chorusPhase -= std::floor (chorusPhase);
            float wetL = 0.0f, wetR = 0.0f;
            for (int v = 0; v < 4; ++v)
            {
                float ph = chorusPhase + float (v) / 3.0f;
                ph -= std::floor (ph);
                const float tri = 2.0f * std::abs (2.0f * ph - 1.0f) - 1.0f;
                const float d = base * (1.0f + depth * tri);
                wetL += chorus[0].readCubic (d) * chorusPanL[(size_t) v];
                wetR += chorus[1].readCubic (d) * chorusPanR[(size_t) v];
            }
            wetL = chorusLp.process (0, chorusHp.process (0, wetL));
            wetR = chorusLp.process (1, chorusHp.process (1, wetR));

            const float fbL = dsp::fx::hardClip (kChorusFeedback * wetL);
            const float fbR = dsp::fx::hardClip (kChorusFeedback * wetR);
            chorus[0].push (l[i] + fbL);
            chorus[1].push (r[i] + fbR);

            l[i] += mix * (wetL - l[i]);
            r[i] += mix * (wetR - r[i]);
        }
    }
    void resetChorus() noexcept
    {
        chorus[0].restart();
        chorus[1].restart();
        chorusPhase = 0.0f;
        chorusHp.reset();
        chorusLp.reset();
    }

    // Phaser: four second-order all-pass biquads at 440 Hz with 2/(i+1) octave
    // spans, or flanger: four combs at MIDI 60 with an additive mix.
    void processModFx (float* l, float* r, int n, const FxSettings& p) noexcept
    {
        if (p.modfxType == ModFxType::Phaser)
            processPhaser (l, r, n, p);
        else
            processFlanger (l, r, n, p);
    }

    void processPhaser (float* l, float* r, int n, const FxSettings& p) noexcept
    {
        const float inc = p.modfxRateHz / sr;
        const float depth = std::clamp (p.modfxDepth, 0.0f, 1.0f);
        const float fb = 0.95f * std::clamp (p.modfxFeedback, 0.0f, 1.0f);
        const float mix = std::clamp (p.modfxMix, 0.0f, 1.0f);
        float* ch[2] = { l, r };

        for (int i = 0; i < n; ++i)
        {
            modPhase += inc;
            modPhase -= std::floor (modPhase);
            const float lfo = std::sin (juce::MathConstants<float>::twoPi * modPhase);

            // Surge's phaser stages are second-order all-pass sections.
            for (int st = 0; st < 4; ++st)
            {
                const float hz = 440.0f * std::exp2 ((2.0f / float (st + 1)) * depth * lfo);
                phAllpass[st].setAllPass (hz, 1.0f, sr);
            }

            for (int c = 0; c < 2; ++c)
            {
                const float x = ch[c][i];
                float y = std::clamp (x + fb * modState[c], -32.0f, 32.0f);
                for (int st = 0; st < 4; ++st)
                    y = phAllpass[st].process (c, y);
                modState[c] = y;
                ch[c][i] = x + mix * (y - x);
            }
        }
    }

    void processFlanger (float* l, float* r, int n, const FxSettings& p) noexcept
    {
        const float inc = p.modfxRateHz / sr;
        const float depth = std::clamp (p.modfxDepth, 0.0f, 1.0f);
        const float mix = std::clamp (p.modfxMix, 0.0f, 1.0f);
        const float base = sr / kFlangerBasePitchHz;
        const float fbControl = std::clamp (p.modfxFeedback, 0.0f, 1.0f);
        float fbv = fbControl;
        if (fbv > 0.0f && fbv < 1.0f)
            fbv = std::sqrt (fbv);
        const float dApprox = p.modfxRateHz * base * depth;
        const float feedbackScale = 0.4f * std::sqrt ((std::clamp (dApprox, 2.0f, 60.0f) + 30.0f) / 100.0f);
        const float fb = feedbackScale * fbv;
        const float df = std::clamp (0.4f * kFlangerDamping, 0.01f, 0.99f);
        const float gainAdj = -1.0f / std::sqrt (3.0f) - 0.07f * mix; // 4 voices
        float* ch[2] = { l, r };

        for (int i = 0; i < n; ++i)
        {
            for (int c = 0; c < 2; ++c)
            {
                float combs = 0.0f;
                for (int v = 0; v < 4; ++v)
                {
                    flangerPhase[c][(size_t) v] += inc;
                    flangerPhase[c][(size_t) v] -= std::floor (flangerPhase[c][(size_t) v]);
                    const float lfo = std::sin (juce::MathConstants<float>::twoPi
                                                * flangerPhase[c][(size_t) v]);
                    const float tap = base * (1.0f + depth * lfo) + 1.0f;
                    combs += flanger[c].readCubic (tap);
                }

                float fbl = 0.0f;
                if (fb > 0.0f)
                {
                    fbl = dsp::fx::cubicSoftClip (std::clamp (fb * combs, -1.0f, 1.0f));
                    flangerDamp[c] = flangerDamp[c] * (1.0f - df) + fbl * df;
                    fbl -= flangerDamp[c];
                }

                flanger[c].push (ch[c][i] - fbl);
                float out = ch[c][i] + mix * combs;
                out = dsp::fx::cubicSoftClip (dsp::fx::hardClip ((1.0f + gainAdj) * out));
                ch[c][i] = out;
            }
        }
    }

    void resetModFx() noexcept
    {
        for (auto& b : phAllpass)
            b.reset();
        modState[0] = modState[1] = 0.0f;
        modPhase = 0.0f;
        flanger[0].restart();
        flanger[1].restart();
        for (int c = 0; c < 2; ++c)
            for (int i = 0; i < 4; ++i)
                flangerPhase[c][i] = (float (i) + 0.5f * float (c)) / 4.0f;
        flangerDamp[0] = flangerDamp[1] = 0.0f;
    }

    // Delay: feedback = control^3, 110 Hz / 2.49 kHz feedback filters, soft
    // clipped feedback, linear dry/wet mix, time ramps across the span.
    void processDelay (float* l, float* r, int n, const FxSettings& p) noexcept
    {
        const float target = std::clamp (p.delaySeconds, 0.001f, kMaxDelaySeconds) * sr;
        if (delayRead < 0.0f)
            delayRead = target;
        const float step = (target - delayRead) / float (std::max (n, 1));
        const float fb = std::pow (std::clamp (p.delayFeedback, 0.0f, 1.0f), 3.0f);
        const float mix = std::clamp (p.delayMix, 0.0f, 1.0f);
        delayHp.setHighPass (110.0f, 0.707f, sr);
        delayLp.setLowPass (2489.0f, 0.707f, sr);
        float* ch[2] = { l, r };

        for (int i = 0; i < n; ++i)
            for (int k = 0; k < 2; ++k)
            {
                const float wet = delay[k].readCubic (delayRead + step * float (i + 1));
                float filtered = delayLp.process (k, delayHp.process (k, wet));
                const float fbSig = dsp::fx::softClip (filtered) * fb;
                delay[k].push (ch[k][i] + fbSig);
                ch[k][i] += mix * (filtered - ch[k][i]);
            }
        delayRead = target;
    }
    void resetDelay() noexcept
    {
        delay[0].restart();
        delay[1].restart();
        delayHp.reset();
        delayLp.reset();
        delayRead = -1.0f;
    }

    // Reverb: four-line FDN in the Reverb 2 topology.
    void processReverb (float* l, float* r, int n, const FxSettings& p) noexcept
    {
        reverb.process (l, r, n, p.reverbSize, p.reverbDecay, p.reverbDamping, p.reverbMix);
    }

    static constexpr float kChorusFeedback = 0.0625f;

    float sr = 48000.0f;
    std::array<bool, 6> wasOn {};

    // Distortion
    dsp::fx::Biquad distPre, distPost;

    // EQ
    dsp::fx::Biquad eq[3];
    static constexpr int kEqStep = 16;
    std::array<float, 4> eqNow {}; // low dB, mid Hz, mid dB, high dB at the end of the last span
    bool eqPrimed = false;         // false after a reset: the next span starts at its target

    // Chorus
    dsp::fx::DelayLine chorus[2];
    dsp::fx::Biquad chorusHp, chorusLp;
    float chorusPhase = 0.0f;
    std::array<float, 4> chorusPanL {}, chorusPanR {};

    // Phaser / flanger
    dsp::fx::Biquad phAllpass[4];
    float modState[2] {};
    float modPhase = 0.0f;
    dsp::fx::DelayLine flanger[2];
    float flangerPhase[2][4] {};
    float flangerDamp[2] {};

    // Delay
    dsp::fx::DelayLine delay[2];
    dsp::fx::Biquad delayHp, delayLp;
    float delayRead = -1.0f; // samples; < 0 after a reset

    // Reverb
    Reverb2 reverb;
};

} // namespace vae::lite
