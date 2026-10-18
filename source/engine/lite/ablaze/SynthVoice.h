/*
  The synth voice: wavetable oscillator (up to 7-voice unison, stereo spread)
  + analog oscillator + sub + noise -> drive -> filter -> amp envelope.

  All voices read one shared VoiceParams block that SynthEngine refreshes at
  control rate (every kControlBlock samples) from the smoothed parameters.

  Behaviours carried over from the reference sine voice:
    - oscillator phase and envelope level are NOT reset when a sounding voice
      is retriggered, so same-note retrigger / voice steal never clicks,
    - stopNote(allowTailOff = false) fades over ~3 ms instead of hard-cutting,
    - kill() is the hard cut used only by inst_reset,
    - the voice frees itself (clearCurrentNote) when its envelope ends.
*/

#pragma once

#include <juce_audio_basics/juce_audio_basics.h>

#include <cmath>

#include "Params.h"
#include "VoiceDsp.h"
#include "Wavetables.h"

namespace ablaze
{

/** Parameter values as the voices consume them (already smoothed). */
struct VoiceParams
{
    float oscAGain = 0.0f;         // level * on-toggle
    const Wavetable* oscATable = nullptr;   // resolved by the engine, never null

    float oscAPos = 0.0f;
    float oscASemis = 0.0f;        // octave*12 + semi + fine/100
    int unison = 1;
    float detuneCents = 0.0f;
    float spread = 0.0f;

    float oscBGain = 0.0f;
    AnalogWave oscBWave = AnalogWave::Saw;
    float oscBPw = 0.5f;
    float oscBSemis = 0.0f;

    float subGain = 0.0f;
    SubWave subWave = SubWave::Sine;
    float subRatio = 0.5f;

    float noiseGain = 0.0f;

    float filterMix = 1.0f;        // on-toggle, smoothed
    FilterType filterType = FilterType::LowPass24;
    float cutoffHz = 6000.0f;
    float filterK = 2.0f;          // SVF damping from resonance
    float driveGain = 1.0f;
    float driveComp = 1.0f;
    float keytrack = 0.5f;
    float envOctaves = 0.0f;       // filter env amount, in octaves
    float velOctaves = 0.0f;

    EnvCoefs ampEnv, filterEnv;
    float ampVel = 0.6f;
};

struct WavetableSound final : juce::SynthesiserSound
{
    bool appliesToNote (int) override    { return true; }
    bool appliesToChannel (int) override { return true; }
};

class WavetableVoice final : public juce::SynthesiserVoice
{
public:
    static constexpr int kMaxUnison = 7;

    WavetableVoice (const VoiceParams& paramsRef, const float* noteHzTable, int voiceIndex)
        : vp (paramsRef), noteHz (noteHzTable)
    {
        noise.seed (0x12345u + 7919u * static_cast<uint32_t> (voiceIndex));
        random.setSeed (0x5eedLL + voiceIndex);
    }

    bool canPlaySound (juce::SynthesiserSound* s) override
    {
        return dynamic_cast<WavetableSound*> (s) != nullptr;
    }

    void startNote (int midiNoteNumber, float velocity, juce::SynthesiserSound*, int) override
    {
        const bool wasSilent = ampEnv.getLevel() <= 0.0f && ! fading;

        note = midiNoteNumber;
        baseHz = noteHz[juce::jlimit (0, 127, midiNoteNumber)];
        vel = velocity;

        if (wasSilent)
        {
            // Fresh voice: deterministic phase for a single oscillator (tight
            // attacks), random phases for unison (the classic supersaw).
            for (int u = 0; u < kMaxUnison; ++u)
                wtPhase[u] = (u == 0) ? 0.0f : random.nextFloat();
            oscB.setPhase (0.0f);
            sub.setPhase (0.0f);
            for (auto& f : filters)
                f.reset();
            filterEnv.reset();
            table = vp.oscATable;   // nothing sounding: no table crossfade needed
            prevTable = nullptr;
            tableFade = 0.0f;
        }

        ampEnv.noteOn();
        filterEnv.noteOn();
        fading = false;
        fadeGain = 1.0f;
    }

    void stopNote (float, bool allowTailOff) override
    {
        if (allowTailOff)
        {
            ampEnv.noteOff();
            filterEnv.noteOff();
        }
        else
        {
            // Voice stealing / hard stop: never cut — fade out over ~3 ms.
            fading = true;
            fadeStep = 1.0f / (0.003f * static_cast<float> (getSampleRate()));
        }
    }

    /** Immediate hard stop, used by SynthEngine::reset only. */
    void kill()
    {
        ampEnv.reset();
        filterEnv.reset();
        fading = false;
        fadeGain = 1.0f;
        prevTable = nullptr;
        tableFade = 0.0f;
        for (auto& f : filters)
            f.reset();
        clearCurrentNote();
    }

    void pitchWheelMoved (int) override {}
    void controllerMoved (int, int) override {}

    using SynthesiserVoice::renderNextBlock;
    void renderNextBlock (juce::AudioBuffer<float>& output, int startSample, int numSamples) override
    {
        if (! isVoiceActive())
            return;

        const float sr = static_cast<float> (getSampleRate());
        auto* left = output.getWritePointer (0);
        auto* right = output.getWritePointer (1);

        // ---- per-call setup (control rate) --------------------------------
        const int numUnison = juce::jlimit (1, kMaxUnison, vp.unison);
        const float hzA = baseHz * semisToRatio (vp.oscASemis);
        float gainL[kMaxUnison], gainR[kMaxUnison], incA[kMaxUnison];
        const float unisonNorm = 1.0f / std::sqrt (static_cast<float> (numUnison));

        for (int u = 0; u < numUnison; ++u)
        {
            const float pos = numUnison > 1 ? (2.0f * static_cast<float> (u) / static_cast<float> (numUnison - 1) - 1.0f) : 0.0f;
            incA[u] = juce::jmin (0.49f, hzA * centsToRatio (vp.detuneCents * pos) / sr);
            // Equal-power pan, normalised so centre == unity.
            const float theta = (pos * vp.spread + 1.0f) * 0.25f * juce::MathConstants<float>::pi;
            gainL[u] = std::cos (theta) * juce::MathConstants<float>::sqrt2 * unisonNorm;
            gainR[u] = std::sin (theta) * juce::MathConstants<float>::sqrt2 * unisonNorm;
        }

        // Mip level for the highest-pitched unison voice.
        const float maxInc = incA[numUnison - 1] > incA[0] ? incA[numUnison - 1] : incA[0];
        const int level = Wavetable::levelForMaxHarmonics (0.5f / juce::jmax (1.0e-6f, maxInc));

        // Table switched (menu change, or a new user table loaded) while
        // sounding: crossfade from the old table over ~10 ms instead of
        // jumping to a different waveform mid-cycle.
        if (vp.oscATable != table)
        {
            prevTable = table;
            table = vp.oscATable;
            tableFade = prevTable != nullptr ? 1.0f : 0.0f;
        }
        const float tableFadeStep = 1.0f / (0.010f * sr);
        const TableTap cur = makeTap (table, level, vp.oscAPos);
        const TableTap prev = tableFade > 0.0f ? makeTap (prevTable, level, vp.oscAPos) : cur;

        oscB.setIncrement (baseHz * semisToRatio (vp.oscBSemis) / sr);
        sub.setIncrement (baseHz * vp.subRatio / sr);
        const AnalogWave subShape = subWaveToAnalog (vp.subWave);

        const float keyOct = vp.keytrack * (static_cast<float> (note) - 60.0f) / 12.0f;
        const float velOct = vp.velOctaves * (vel - 1.0f);
        const float nyquistGuard = 0.45f * sr;
        const float velGain = 1.0f - vp.ampVel * (1.0f - vel);
        const bool filterActive = vp.filterMix > 0.0f;

        // ---- per-sample loop ----------------------------------------------
        for (int i = startSample; i < startSample + numSamples; ++i)
        {
            const float fenv = filterEnv.next (vp.filterEnv);
            const float aenv = ampEnv.next (vp.ampEnv);

            float l = 0.0f, r = 0.0f;

            if (tableFade > 0.0f)
            {
                tableFade -= tableFadeStep;
                if (tableFade <= 0.0f)
                {
                    tableFade = 0.0f;
                    prevTable = nullptr;
                }
            }

            if (vp.oscAGain > 0.0f)
            {
                if (tableFade > 0.0f)
                {
                    for (int u = 0; u < numUnison; ++u)
                    {
                        float s = cur.read (wtPhase[u]);
                        s += tableFade * (prev.read (wtPhase[u]) - s);
                        advanceAndPan (u, s, incA, gainL, gainR, l, r);
                    }
                }
                else
                {
                    // Hot path: kept free of the crossfade branch.
                    const float* a = cur.a;
                    const float* b = cur.b;
                    const float size = cur.size, mix = cur.mix;
                    for (int u = 0; u < numUnison; ++u)
                    {
                        const float idx = wtPhase[u] * size;
                        const int i0 = static_cast<int> (idx);
                        const float fr = idx - static_cast<float> (i0);
                        const float x = a[i0] + fr * (a[i0 + 1] - a[i0]);
                        const float y = b[i0] + fr * (b[i0 + 1] - b[i0]);
                        advanceAndPan (u, x + mix * (y - x), incA, gainL, gainR, l, r);
                    }
                }
                l *= vp.oscAGain;
                r *= vp.oscAGain;
            }

            float mono = 0.0f;
            if (vp.oscBGain > 0.0f)
                mono += vp.oscBGain * oscB.next (vp.oscBWave, vp.oscBPw);
            if (vp.subGain > 0.0f)
                mono += vp.subGain * sub.next (subShape, 0.25f);
            if (vp.noiseGain > 0.0f)
                mono += vp.noiseGain * noise.next();

            l += mono;
            r += mono;

            if (filterActive)
            {
                const float cutoff = juce::jlimit (20.0f, nyquistGuard,
                                                   vp.cutoffHz * std::exp2 (keyOct + velOct + fenv * vp.envOctaves));
                float dl = l, dr = r;
                if (vp.driveGain > 1.0001f)
                {
                    dl = std::tanh (l * vp.driveGain) * vp.driveComp;
                    dr = std::tanh (r * vp.driveGain) * vp.driveComp;
                }
                const float fl = runFilter (0, dl, cutoff, sr);
                const float fr = runFilter (1, dr, cutoff, sr);
                l += vp.filterMix * (fl - l);
                r += vp.filterMix * (fr - r);
            }

            float g = aenv * velGain * kVoiceHeadroom;

            if (fading)
            {
                fadeGain -= fadeStep;
                if (fadeGain <= 0.0f)
                {
                    kill();
                    return;
                }
                g *= fadeGain;
            }

            left[i] += l * g;
            right[i] += r * g;

            if (! ampEnv.isActive())
            {
                filterEnv.reset();
                clearCurrentNote();
                return;
            }
        }
    }

private:
    static constexpr float kVoiceHeadroom = 0.5f;

    /** The two frames around the current position, at one mip level. */
    struct TableTap
    {
        const float* a;
        const float* b;
        float size;
        float mix;

        float read (float phase) const noexcept
        {
            const float idx = phase * size;
            const int i0 = static_cast<int> (idx);
            const float fr = idx - static_cast<float> (i0);
            const float x = a[i0] + fr * (a[i0 + 1] - a[i0]);
            const float y = b[i0] + fr * (b[i0 + 1] - b[i0]);
            return x + mix * (y - x);
        }
    };

    void advanceAndPan (int u, float s, const float* inc, const float* gl, const float* gr,
                        float& l, float& r) noexcept
    {
        wtPhase[u] += inc[u];
        if (wtPhase[u] >= 1.0f)
            wtPhase[u] -= 1.0f;
        l += s * gl[u];
        r += s * gr[u];
    }

    static TableTap makeTap (const Wavetable* t, int level, float position) noexcept
    {
        const int frames = t->getNumFrames();
        const float framePos = juce::jlimit (0.0f, 1.0f, position) * static_cast<float> (frames - 1);
        const int f0 = juce::jmin (static_cast<int> (framePos), juce::jmax (0, frames - 2));
        const int f1 = juce::jmin (f0 + 1, frames - 1);
        return { t->getFrame (level, f0), t->getFrame (level, f1),
                 static_cast<float> (Wavetable::levelSize (level)),
                 framePos - static_cast<float> (f0) };
    }

    static float semisToRatio (float s) noexcept { return std::exp2 (s * (1.0f / 12.0f)); }
    static float centsToRatio (float c) noexcept { return std::exp2 (c * (1.0f / 1200.0f)); }

    static AnalogWave subWaveToAnalog (SubWave w) noexcept
    {
        switch (w)
        {
            case SubWave::Sine:     return AnalogWave::Sine;
            case SubWave::Triangle: return AnalogWave::Triangle;
            case SubWave::Square:   return AnalogWave::Square;
            case SubWave::Pulse:    return AnalogWave::Pulse;
        }
        return AnalogWave::Sine;
    }

    /** Filter one channel. Channel 0 computes the (shared) coefficients. */
    float runFilter (int ch, float x, float cutoff, float sr) noexcept
    {
        auto& s1 = filters[ch * 2];
        auto& s2 = filters[ch * 2 + 1];

        if (ch == 0)
        {
            const float g = std::tan (juce::MathConstants<float>::pi * cutoff / sr);
            coefs2.set (g, vp.filterK);
            if (vp.filterType == FilterType::LowPass24)
                coefs1.set (g, 1.414f);   // Butterworth first stage, resonance on the second
        }

        switch (vp.filterType)
        {
            case FilterType::LowPass12: return s2.process (x, coefs2).low;
            case FilterType::LowPass24: return s2.process (s1.process (x, coefs1).low, coefs2).low;
            case FilterType::BandPass:  return s2.process (x, coefs2).band;
            case FilterType::HighPass:  return s2.process (x, coefs2).high;
            case FilterType::Notch:
            {
                const auto o = s2.process (x, coefs2);
                return o.low + o.high;
            }
        }
        return x;
    }

    const VoiceParams& vp;
    const float* noteHz;

    const Wavetable* table = nullptr;       // what this voice is playing
    const Wavetable* prevTable = nullptr;   // fading out after a switch
    float tableFade = 0.0f;

    int note = 60;
    float baseHz = 261.63f;
    float vel = 1.0f;

    float wtPhase[kMaxUnison] = {};
    AnalogOsc oscB, sub;
    Noise noise;
    juce::Random random;

    Svf filters[4];   // [L stage1, L stage2, R stage1, R stage2]
    SvfCoefs coefs1, coefs2;

    Envelope ampEnv, filterEnv;
    bool fading = false;
    float fadeGain = 1.0f, fadeStep = 0.0f;
};

} // namespace ablaze
