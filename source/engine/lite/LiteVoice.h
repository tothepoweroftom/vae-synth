/*
  One Lite voice: 3 oscillators -> filter routing (serial / parallel, per-osc
  route) -> 2 filters -> amp envelope -> pan. Owns its filter EG and two
  key-triggered LFOs; free-running LFO values and pitch bend come from the
  engine.

  render() is called per control block (<= LiteEngine::kControlBlock
  samples): pitch, LFOs, filter coefficients and modulation are updated per
  block, envelopes and gains per sample.
*/

#pragma once

#include "LiteDsp.h"

#include <array>

namespace vae::lite
{

/** Engine-wide values for the current control block. */
struct BlockValues
{
    std::array<float, 2> freeLfo {}; // free-running LFO outputs
    float pitchBendSemis = 0.0f;     // MIDI pitch wheel
};

class Voice
{
    static constexpr const schema::Curve* kCutoffCurve = schema::surge::curveNamed ("cutoff");

public:
    // Scene volume of Surge's default patch (0.8909 amplitude -> x^3 = -3 dB):
    // the base the "volume" mod destination moves around.
    static constexpr float kSceneVolume = 0.890899f;
    // Calibrates the default patch to Surge's level (single saw, note 60).
    static constexpr float kVoiceGain = 0.25f;

    void prepare (float sampleRate, const dsp::Wavetable* wavetables, uint32_t seed)
    {
        sr = sampleRate;
        table = wavetables;
        for (size_t i = 0; i < osc.size(); ++i)
            osc[i].seed (seed * 3u + uint32_t (i) + 1u);
        for (size_t i = 0; i < lfo.size(); ++i)
            lfo[i].seed (seed * 7u + uint32_t (i) + 11u);
        kill();
    }

    bool isActive() const noexcept { return amp.isActive(); }
    bool isReleasing() const noexcept { return amp.isReleasing(); }
    int getNote() const noexcept { return note; }
    uint64_t getStartOrder() const noexcept { return startedAt; }

    void kill() noexcept
    {
        amp.reset();
        feg.reset();
        for (auto& f : filter)
            f.reset();
        for (auto& o : osc)
            o.resetPhase();
        note = -1;
    }

    /** Start (or retrigger) the voice. `fromSemis` < 0 means no glide. */
    void start (const Patch& p, int midiNote, float fromSemis, uint64_t order) noexcept
    {
        if (! amp.isActive())
        {
            // Fresh voice: seeded random phases and clean filters.
            for (auto& o : osc)
                o.randomisePhase();
            for (auto& f : filter)
                f.reset();
            firstBlock = true;
        }
        note = midiNote;
        startedAt = order;
        glideTo (p, midiNote, fromSemis);
        amp.noteOn (p.ampEnv);
        feg.noteOn (p.filterEnv);
        secondsSinceStart = 0.0f;
        for (size_t i = 0; i < lfo.size(); ++i)
            if (p.lfo[i].retrigger)
                lfo[i].restart (p.lfo[i].phase);
    }

    /** Change pitch without retriggering envelopes (legato, mono returns). */
    void glideTo (const Patch& p, int midiNote, float fromSemis) noexcept
    {
        note = midiNote;
        target = float (midiNote);
        if (fromSemis >= 0.0f && p.glideSeconds > 0.0f)
        {
            current = fromSemis;
            glideRate = std::abs (target - current) / p.glideSeconds;
        }
        else
        {
            current = target;
            glideRate = 0.0f;
        }
    }

    void retriggerEnvelopes (const Patch& p) noexcept
    {
        amp.noteOn (p.ampEnv);
        feg.noteOn (p.filterEnv);
    }

    void release (const Patch& p) noexcept
    {
        amp.noteOff (p.ampEnv);
        feg.noteOff (p.filterEnv);
    }

    float getCurrentSemis() const noexcept { return current; }

    void render (float* left, float* right, int n, const Patch& p, const BlockValues& shared) noexcept
    {
        if (! amp.isActive() || n <= 0)
            return;

        const float blockSeconds = float (n) / sr;

        // Glide (constant time: rate fixed at note start).
        if (current < target)
            current = std::min (target, current + glideRate * blockSeconds);
        else if (current > target)
            current = std::max (target, current - glideRate * blockSeconds);

        // Modulators for this block.
        secondsSinceStart += blockSeconds;
        std::array<float, 2> lfoOut {};
        for (size_t i = 0; i < lfo.size(); ++i)
        {
            const auto& s = p.lfo[i];
            float v = s.retrigger ? lfo[i].advance (s, blockSeconds) : shared.freeLfo[i];
            if (s.unipolar)
                v = 0.5f * (v + 1.0f);
            const float fade = s.fadeIn > 0.0f ? std::min (1.0f, secondsSinceStart / s.fadeIn) : 1.0f;
            lfoOut[i] = v * s.amount * fade;
        }
        const float fegLevel = feg.getLevel();

        Mods mods;
        mods.add (p.fegRoute.dest, p.fegRoute.depth * fegLevel);
        for (size_t i = 0; i < lfo.size(); ++i)
            mods.add (p.lfo[i].route.dest, p.lfo[i].route.depth * lfoOut[i]);

        // Oscillator increments.
        for (size_t i = 0; i < osc.size(); ++i)
        {
            const auto& o = p.osc[i];
            const float semis = current + o.semitones + mods.pitch + shared.pitchBendSemis;
            const float hz = 440.0f * std::exp2 ((semis - 69.0f) / 12.0f);
            osc[i].setIncrement (hz / sr, o.wave, table);
        }

        // Filter coefficients. Keytrack follows the gliding pitch and pitch
        // bend, as Surge's state.pitch does, not the target note.
        const float keyOffset = current + shared.pitchBendSemis - 60.0f;
        for (size_t f = 0; f < filter.size(); ++f)
        {
            const auto& fs = p.filter[f];
            const float cutMod = f == 0 ? mods.cutoff1 : mods.cutoff2;
            const float semis = fs.cutoffSemis + fs.envSemis * fegLevel + fs.keytrack * keyOffset + cutMod;
            const float reso = fs.resonance + (f == 0 ? mods.resonance1 : 0.0f);
            filter[f].configure (
                fs.type, schema::surge::semitonesToHz (std::clamp (semis, kCutoffCurve->min, kCutoffCurve->max)), reso,
                sr);
        }

        // Output gains, ramped across the block.
        const float vol = std::clamp (kSceneVolume + mods.volume, 0.0f, 1.0f);
        const float volGain = vol * vol * vol * kVoiceGain;
        const float pan = std::clamp (mods.pan, -1.0f, 1.0f);
        const float theta = (pan + 1.0f) * 0.25f * 3.14159265f;
        const float targetL = volGain * std::cos (theta) * 1.41421356f;
        const float targetR = volGain * std::sin (theta) * 1.41421356f;
        if (firstBlock)
        {
            gainL = targetL;
            gainR = targetR;
            firstBlock = false;
        }
        const float stepL = (targetL - gainL) / float (n);
        const float stepR = (targetR - gainR) / float (n);

        // Per-osc contributions to each filter input.
        std::array<float, 3> toF1 {}, toF2 {}, direct2 {};
        for (size_t i = 0; i < osc.size(); ++i)
        {
            const auto& o = p.osc[i];
            const float g = o.on ? o.gain : 0.0f;
            if (p.parallel)
            {
                // As Surge's Dual 1 (measured): each filter path carries the
                // oscs routed to it at -6 dB, and the two paths are summed.
                toF1[i] = o.route != Route::Filter2 ? 0.5f * g : 0.0f;
                toF2[i] = o.route != Route::Filter1 ? 0.5f * g : 0.0f;
            }
            else
            {
                // Serial: F1 -> F2; an osc routed to filter 2 skips filter 1.
                toF1[i] = o.route == Route::Filter2 ? 0.0f : g;
                direct2[i] = o.route == Route::Filter2 ? g : 0.0f;
            }
        }

        const float dt = 1.0f / sr;
        for (int s = 0; s < n; ++s)
        {
            float in1 = 0.0f, in2 = 0.0f;
            for (size_t i = 0; i < osc.size(); ++i)
            {
                if (! p.osc[i].on)
                    continue;
                const float y = osc[i].next (p.osc[i].wave);
                in1 += toF1[i] * y;
                in2 += (p.parallel ? toF2[i] : direct2[i]) * y;
            }

            float y;
            if (p.parallel)
                y = filter[0].process (in1) + filter[1].process (in2);
            else
                y = filter[1].process (filter[0].process (in1) + in2);

            feg.next (p.filterEnv, dt);
            const float a = amp.next (p.ampEnv, dt);
            gainL += stepL;
            gainR += stepR;
            left[s] += y * a * gainL;
            right[s] += y * a * gainR;

            if (! amp.isActive())
            {
                feg.reset();
                note = -1;
                return;
            }
        }
    }

private:
    /** Summed provisional-route modulation, in destination units. */
    struct Mods
    {
        float pitch = 0.0f, cutoff1 = 0.0f, cutoff2 = 0.0f, resonance1 = 0.0f, volume = 0.0f, pan = 0.0f;

        // A route moves its destination by depth x the full range of the
        // destination's Surge parameter, i.e. the width of its schema curve.
        static constexpr float kPitchSpan = schema::surge::curveSpan ("pitch7");
        static constexpr float kCutoffSpan = schema::surge::curveSpan ("cutoff");
        static constexpr float kResonanceSpan = schema::surge::curveSpan ("unit");
        static constexpr float kVolumeSpan = schema::surge::curveSpan ("amplitude");
        static constexpr float kPanSpan = schema::surge::curveSpan ("bipolar");

        void add (ModDest d, float amount) noexcept
        {
            switch (d)
            {
                case ModDest::None:             break;
                case ModDest::Pitch:            pitch += kPitchSpan * amount; break;
                case ModDest::Filter1Cutoff:    cutoff1 += kCutoffSpan * amount; break;
                case ModDest::Filter2Cutoff:    cutoff2 += kCutoffSpan * amount; break;
                case ModDest::Filter1Resonance: resonance1 += kResonanceSpan * amount; break;
                case ModDest::Volume:           volume += kVolumeSpan * amount; break;
                case ModDest::Pan:              pan += kPanSpan * amount; break;
            }
        }
    };

    float sr = 48000.0f;
    const dsp::Wavetable* table = nullptr;
    std::array<Oscillator, 3> osc;
    std::array<FilterUnit, 2> filter;
    ShapedEnvelope amp, feg;
    std::array<Lfo, 2> lfo;

    int note = -1;
    uint64_t startedAt = 0;
    float current = 60.0f, target = 60.0f, glideRate = 0.0f;
    float secondsSinceStart = 0.0f;
    float gainL = 0.0f, gainR = 0.0f;
    bool firstBlock = true;
};

} // namespace vae::lite
