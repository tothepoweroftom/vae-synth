/*
  The Lite backend: a Surge-shaped subset synth on the permissive DSP from
  Ablaze.WASMSynth (source/engine/lite/ablaze). It reads the schema vector
  through Surge's value curves (schema/SurgeCurves.h), so a vector designed or
  generated for Surge sounds broadly similar here.

  Mapping (Surge scene A -> Lite):
    Osc 1        -> wavetable osc with unison (type picks a table, shape
                    morphs the table position)
    Osc 2        -> PolyBLEP analog osc (type picks a waveform)
    Osc 3        -> sub oscillator, -3..+3 oct (type picks a waveform)
    Noise        -> noise
    Filter 1     -> TPT SVF (Surge filter type folded to LP12/LP24/HP/BP/Notch;
                    "Off" bypasses it; subtype ignored)
    Amp/Filt EG  -> exponential ADSRs, Surge envelope times
    Global Vol   -> master gain
  Not modelled: filter balance (Lite has a single filter) and LFO 1 (Surge's
  default patch routes it nowhere, so it is inaudible there too).
*/

#pragma once

#include "engine/SynthEngine.h"

#include <memory>

namespace ablaze
{
class SynthEngine;
class WavetableBank;
} // namespace ablaze

namespace vae
{

class LiteEngine final : public SynthEngine
{
public:
    LiteEngine();
    ~LiteEngine() override;

    EngineKind kind() const noexcept override { return EngineKind::Lite; }
    const char* displayName() const noexcept override { return "Lite"; }

    void prepare (double sampleRate, int maxBlockSize) override;
    void reset() override;
    void setControl (int index, float normalised) override;
    void process (juce::AudioBuffer<float>& buffer, const juce::MidiBuffer& midi) override;

private:
    void apply (int index);
    void setNative (int nativeIndex, float value);

    std::shared_ptr<const ablaze::WavetableBank> bank;
    std::unique_ptr<ablaze::SynthEngine> engine;
    schema::Snapshot controls = schema::Snapshot::defaults();
};

} // namespace vae
