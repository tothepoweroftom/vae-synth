/*
  The Lite backend: a permissive, Surge-shaped synth for schema v2.

    voice   poly (16 voices) / mono (retrigger) / legato, constant-time glide
    osc x3  sine, triangle, saw, square, noise, wavetable, fm; level, octave,
            pitch, route to filter 1 / both / filter 2
    filters 2 x (off, LP/HP 12/24, BP, notch), serial or parallel, cutoff,
            resonance, filter-EG amount, keytrack
    envs    amp + filter ADSR with Surge times and per-stage curves
    LFOs    2 x shape, rate, amount, unipolar, retrigger/phase, fade-in
    mod     provisional routes: filter EG / LFO 1 / LFO 2 -> one destination each
    FX      distortion -> EQ -> chorus -> phaser/flanger -> delay -> reverb
  Controls are read through the schema's Surge curves (LitePatch), so a vector
  means roughly the same sound on the Surge backend. See LitePatch.cpp and
  docs/SCHEMA.md for the mapping and known differences.
*/

#pragma once

#include "engine/SynthEngine.h"
#include "LiteFx.h"
#include "LitePatch.h"
#include "LiteVoice.h"

#include <array>
#include <memory>

namespace vae
{

class LiteEngine final : public SynthEngine
{
public:
    static constexpr int kMaxVoices = 16;
    static constexpr int kControlBlock = 16;
    static constexpr float kPitchBendRange = 2.0f; // semitones, Surge's default

    LiteEngine();
    ~LiteEngine() override;

    EngineKind kind() const noexcept override { return EngineKind::Lite; }
    const char* displayName() const noexcept override { return "Lite"; }

    void prepare (double sampleRate, int maxBlockSize) override;
    void reset() override;
    void setControl (int index, float value) override;
    void process (juce::AudioBuffer<float>& buffer, const juce::MidiBuffer& midi) override;

    /** Voices currently sounding (for tests). */
    int numActiveVoices() const noexcept;

    /** The smoothed value in effect right now (for tests). */
    float currentControl (int index) const noexcept { return controls[index]; }

    /** Click diagnostics. Written on the audio thread; only tests read them (between process() calls). */
    struct Diagnostics
    {
        int ducksStarted = 0;
        int64_t duckedSamples = 0; // output samples at a duck gain below 1
    };
    Diagnostics diagnostics;

private:
    void noteOn (int note);
    void noteOff (int note);
    void releaseNote (int note);
    void setSustain (bool down);
    void allNotesOff();
    void renderVoices (float* left, float* right, int numSamples);
    float glideSource() const noexcept;
    void snapControls();
    void smoothControls (int numSamples);
    void applyDiscrete();
    float finishSpan (float* left, float* right, int len);

    std::shared_ptr<const dsp::Wavetable> wavetable;
    // `controls` are the values in effect (smoothed); `targets` what was set.
    schema::Snapshot controls = schema::Snapshot::defaults();
    schema::Snapshot targets = schema::Snapshot::defaults();
    float smoothCoeff = 1.0f; // one-pole step per kControlBlock
    float smoothRate = 0.0f;  // -1 / (tau * sampleRate)
    // Toggle/Categorical changes duck the output: Down, swap at gain 0, Up.
    enum class Duck
    {
        Idle,
        Down,
        Up
    };
    Duck duck = Duck::Idle;
    float duckGain = 1.0f;
    float duckStep = 1.0f;
    float gainNow = -1.0f; // master gain at the end of the last span; < 0 after a snap
    int duckCountdown = 0; // samples until a Down duck reaches gain 0
    lite::Patch patch;
    std::array<lite::Voice, kMaxVoices> voices;
    lite::FxRack fx;
    std::array<lite::Lfo, 2> freeLfo;
    lite::BlockValues blockValues;

    // Held keys, oldest first (mono/legato note priority and glide).
    std::array<int, 128> held {};
    int numHeld = 0;
    bool sustainDown = false; // CC64: note-offs wait for pedal up
    float lastNoteSemis = -1.0f;
    uint64_t noteCounter = 0;
    float sampleRate = 48000.0f;
    bool prepared = false;
};

} // namespace vae
