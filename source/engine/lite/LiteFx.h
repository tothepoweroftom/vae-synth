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

#include <juce_audio_basics/juce_audio_basics.h>

namespace vae::lite
{

class FxRack
{
public:
    static constexpr float kMaxDelaySeconds = 4.0f;

    void prepare (double sampleRate)
    {
        sr = float (sampleRate);
        chorus[0].prepare (int (0.05f * sr));
        chorus[1].prepare (int (0.05f * sr));
        flanger[0].prepare (int (0.02f * sr));
        flanger[1].prepare (int (0.02f * sr));
        delay[0].prepare (int ((kMaxDelaySeconds + 0.05f) * sr));
        delay[1].prepare (int ((kMaxDelaySeconds + 0.05f) * sr));
        reverb.setSampleRate (sampleRate);
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

    static float onePoleCoef (float hz, float sampleRate) noexcept
    {
        return std::exp (-2.0f * juce::MathConstants<float>::pi * std::min (hz, 0.45f * sampleRate) / sampleRate);
    }

    // Distortion: drive into tanh, then a 2-pole low-pass "tone" (Surge's post high cut).
    void processDist (float* l, float* r, int n, const FxSettings& p) noexcept
    {
        const float drive = juce::Decibels::decibelsToGain (p.distDriveDb);
        const float c = onePoleCoef (p.distToneHz, sr);
        float* ch[] = { l, r };
        for (int c2 = 0; c2 < 2; ++c2)
            for (int i = 0; i < n; ++i)
            {
                const float y = std::tanh (ch[c2][i] * drive);
                tone[c2][0] = y + c * (tone[c2][0] - y);
                tone[c2][1] = tone[c2][0] + c * (tone[c2][1] - tone[c2][0]);
                ch[c2][i] = tone[c2][1];
            }
    }
    void resetDist() noexcept { tone[0][0] = tone[0][1] = tone[1][0] = tone[1][1] = 0.0f; }

    // EQ: Surge's 3-band EQ with bands 1 and 3 at their default frequencies,
    // modelled as low shelf / peak / high shelf.
    void processEq (float* l, float* r, int n, const FxSettings& p) noexcept
    {
        using B = dsp::fx::Biquad;
        eq[0].set (B::Shape::LowShelf, schema::surge::semitonesToHz (-30.0f), p.eqLowDb, 0.707f, sr);
        eq[1].set (B::Shape::Peak, p.eqMidHz, p.eqMidDb, 0.67f, sr);
        eq[2].set (B::Shape::HighShelf, schema::surge::semitonesToHz (54.0f), p.eqHighDb, 0.707f, sr);
        for (int i = 0; i < n; ++i)
            for (auto& b : eq)
            {
                l[i] = b.process (0, l[i]);
                r[i] = b.process (1, r[i]);
            }
    }
    void resetEq() noexcept
    {
        for (auto& b : eq)
            b.reset();
    }

    // Chorus: two modulated delay lines, 90 degrees apart.
    void processChorus (float* l, float* r, int n, const FxSettings& p) noexcept
    {
        const float inc = p.chorusRateHz / sr;
        const float base = 0.012f * sr, depth = p.chorusDepth * 0.008f * sr;
        for (int i = 0; i < n; ++i)
        {
            chorusPhase += inc;
            chorusPhase -= std::floor (chorusPhase);
            const float a = juce::MathConstants<float>::twoPi * chorusPhase;
            chorus[0].push (l[i]);
            chorus[1].push (r[i]);
            const float wl = chorus[0].read (base + depth * std::sin (a));
            const float wr = chorus[1].read (base + depth * std::cos (a));
            l[i] += p.chorusMix * (wl - l[i]);
            r[i] += p.chorusMix * (wr - r[i]);
        }
    }
    void resetChorus() noexcept
    {
        chorus[0].restart();
        chorus[1].restart();
    }

    // Phaser (4 first-order all-passes, swept) or flanger (short modulated delay).
    void processModFx (float* l, float* r, int n, const FxSettings& p) noexcept
    {
        const float inc = p.modfxRateHz / sr;
        const float fb = 0.9f * p.modfxFeedback;
        float* ch[] = { l, r };
        for (int i = 0; i < n; ++i)
        {
            modPhase += inc;
            modPhase -= std::floor (modPhase);
            for (int c = 0; c < 2; ++c)
            {
                const float lfo = std::sin (juce::MathConstants<float>::twoPi * (modPhase + 0.25f * float (c)));
                const float x = ch[c][i];
                float wet;
                if (p.modfxType == ModFxType::Phaser)
                {
                    const float hz = 800.0f * std::exp2 (2.5f * p.modfxDepth * lfo);
                    const float t = std::tan (juce::MathConstants<float>::pi * std::min (hz, 0.45f * sr) / sr);
                    const float a = (t - 1.0f) / (t + 1.0f);
                    float y = x + fb * modState[c];
                    for (auto& z : allpass[c])
                    {
                        const float out = a * y + z;
                        z = y - a * out;
                        y = out;
                    }
                    modState[c] = y;
                    wet = y;
                }
                else
                {
                    const float d = (0.0008f + p.modfxDepth * 0.004f * (0.5f + 0.5f * lfo)) * sr;
                    wet = flanger[c].read (std::max (1.0f, d));
                    flanger[c].push (x + fb * wet);
                }
                ch[c][i] = x + p.modfxMix * (wet - x);
            }
        }
    }
    void resetModFx() noexcept
    {
        for (auto& c : allpass)
            for (auto& z : c)
                z = 0.0f;
        modState[0] = modState[1] = 0.0f;
        flanger[0].restart();
        flanger[1].restart();
    }

    // Delay: unsynced stereo delay with gentle damping; mix crossfades dry/wet.
    void processDelay (float* l, float* r, int n, const FxSettings& p) noexcept
    {
        // The time ramps across the span: a read head that jumps each block crackles while it moves (Task 18).
        const float target = std::clamp (p.delaySeconds, 0.001f, kMaxDelaySeconds) * sr;
        if (delayRead < 0.0f)
            delayRead = target;
        const float step = (target - delayRead) / float (std::max (n, 1));
        const float fb = std::clamp (p.delayFeedback, 0.0f, 0.99f);
        const float c = onePoleCoef (9000.0f, sr);
        float* ch[] = { l, r };
        for (int i = 0; i < n; ++i)
            for (int k = 0; k < 2; ++k)
            {
                const float wet = delay[k].read (delayRead + step * float (i + 1));
                delayLp[k] = wet + c * (delayLp[k] - wet);
                delay[k].push (ch[k][i] + fb * delayLp[k]);
                ch[k][i] += p.delayMix * (wet - ch[k][i]);
            }
        delayRead = target;
    }
    void resetDelay() noexcept
    {
        delay[0].restart();
        delay[1].restart();
        delayLp[0] = delayLp[1] = 0.0f;
        delayRead = -1.0f;
    }

    // Reverb: juce::Reverb (Freeverb); decay time and size both drive room size.
    void processReverb (float* l, float* r, int n, const FxSettings& p) noexcept
    {
        const float t = p.reverbDecay; // the 0..1 reverb_decay control
        juce::Reverb::Parameters rp;
        rp.roomSize = std::clamp (0.25f + 0.7f * t + 0.1f * p.reverbSize, 0.0f, 1.0f);
        rp.damping = p.reverbDamping;
        rp.width = 1.0f;
        rp.wetLevel = p.reverbMix / 3.0f;          // juce::Reverb scales wet by 3
        rp.dryLevel = (1.0f - p.reverbMix) / 2.0f; // ... and dry by 2
        rp.freezeMode = 0.0f;
        reverb.setParameters (rp);
        reverb.processStereo (l, r, n);
    }

    float sr = 48000.0f;
    std::array<bool, 6> wasOn {};
    float tone[2][2] {};
    dsp::fx::Biquad eq[3];
    dsp::fx::DelayLine chorus[2];
    float chorusPhase = 0.0f;
    float allpass[2][4] {};
    float modState[2] {};
    dsp::fx::DelayLine flanger[2];
    float modPhase = 0.0f;
    dsp::fx::DelayLine delay[2];
    float delayLp[2] {};
    float delayRead = -1.0f; // samples; < 0 after a reset
    juce::Reverb reverb;
};

} // namespace vae::lite
