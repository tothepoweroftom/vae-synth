/*
  The Surge XT backend (VAE_WITH_SURGE, GPL-3.0-or-later): headless
  SurgeSynthesizer from surge-common, driven by the schema vector. Each
  control is resolved once to the Surge parameter whose full name matches
  `surge_param` in schema/controls.json, then written with setParameter01.

  Surge renders fixed BLOCK_SIZE (32) quanta; arbitrary host blocks are served
  from a one-quantum buffer, with MIDI applied at the quantum it falls in (up
  to 31 samples early; no added latency). Mirrors VAE_SURGE's SurgeBackend.
*/

#pragma once

#include "engine/SynthEngine.h"

#include <memory>

namespace vae
{

class SurgeEngine final : public SynthEngine
{
public:
    SurgeEngine();
    ~SurgeEngine() override;

    EngineKind kind() const noexcept override { return EngineKind::Surge; }
    const char* displayName() const noexcept override { return "Surge XT"; }

    void prepare (double sampleRate, int maxBlockSize) override;
    void reset() override;
    void setControl (int index, float normalised) override;
    void process (juce::AudioBuffer<float>& buffer, const juce::MidiBuffer& midi) override;

    /** Number of schema controls that resolved to a Surge parameter (for tests). */
    int numResolvedControls() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace vae
