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

private:
    void noteOn (int note);
    void noteOff (int note);
    void allNotesOff();
    void renderVoices (float* left, float* right, int numSamples);
    float glideSource() const noexcept;

    std::shared_ptr<const ablaze::WavetableBank> bank;
    schema::Snapshot controls = schema::Snapshot::defaults();
    lite::Patch patch;
    std::array<lite::Voice, kMaxVoices> voices;
    lite::FxRack fx;
    std::array<lite::Lfo, 2> freeLfo;
    lite::FreeLfoValues freeLfoValues;

    // Held keys, oldest first (mono/legato note priority and glide).
    std::array<int, 128> held {};
    int numHeld = 0;
    float lastNoteSemis = -1.0f;
    uint64_t noteCounter = 0;
    float sampleRate = 48000.0f;
    bool prepared = false;
};

} // namespace vae
