/*
  The Surge XT backend (VAE_WITH_SURGE, GPL-3.0-or-later): headless
  SurgeSynthesizer from surge-common, driven by the schema v2 vector
  (schema/controls.json). Scene A only. Surge parameters are resolved by
  storage name (Parameter::get_storage_name), which is stable across osc/FX
  types, unlike display names.

  Mapping (schema option -> Surge value): the tables at the top of
  SurgeEngine.cpp. Render rules and measured Surge behaviour (osc waves, filter
  routing, the route depth convention, the fixed FX slots): docs/SCHEMA.md.
  FX *_on sets that slot's bit in Surge's fx_disable mask; all six effects are
  loaded once in prepare(), so toggling never allocates.

  Osc and FX type changes are applied lazily by Surge (loadOscalgos/loadFx
  inside process()) and reset that unit's parameters to the type defaults.
  The engine applies them itself before rendering and then re-writes the
  parameters it owns, so no voice ever starts with a stale layout.

  Audio-thread allocation: steady rendering, FX on/off and every continuous,
  filter-type, envelope or LFO change allocate nothing. Inside Surge, an osc
  wave change (loadOscalgos), a mod FX type switch (loadFx) and adding a new
  modulation route still allocate; those are discrete patch changes.

  Surge renders fixed BLOCK_SIZE (32) quanta; arbitrary host blocks are served
  from a one-quantum buffer, with MIDI applied at the quantum it falls in, so
  an event lands within ±31 samples (one 32-sample quantum) of its offset; no
  added latency.
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
    void setControl (int index, float value) override;
    void process (juce::AudioBuffer<float>& buffer, const juce::MidiBuffer& midi) override;

    //==========================================================================
    // Inspection, for tests and diagnostics. Valid after prepare().

    /** Number of schema controls whose Surge parameters all resolved. */
    int numResolvedControls() const noexcept;

    /** Current value of a scene A / global Surge parameter by storage name:
        ints and bools as their integer value, floats in Surge units. NaN if
        the name is unknown. */
    float surgeValue (const char* storageName) const;

    /** Surge osc type (ot_*) of oscillator 0..2, and its p[k] in Surge units. */
    int oscType (int osc) const;
    float oscParam (int osc, int k) const;

    /** Surge FX type (fxt_*) loaded in an FX slot, and its p[k] in Surge units. */
    int fxType (int slot) const;
    /** False when the slot's fx_disable bit is set (the *_on control is off). */
    bool fxEnabled (int slot) const;
    float fxParam (int slot, int k) const;

    /** Voice modulation routings in scene A: count, and the depth (Surge
        units) of the route from source (0 FEG, 1 LFO 1, 2 LFO 2) to the
        parameter with this storage name, or NaN when there is none. */
    int numModRoutes() const;
    float modRouteDepth (int source, const char* destinationStorageName) const;

    /** Voices that are not yet released, scene A. */
    int numHeldVoices() const;

    /** Surge FX slot indices (0-based) used by the fixed chain. Surge XT appended
        A3/A4 as slots 8/9 after the original eight (A1 A2 B1 B2 S1 S2 G1 G2);
        patch XML numbers the same slots fx1..fx16 (1-based). */
    enum FxSlot : int
    {
        kDistSlot = 0,
        kEqSlot = 1,
        kDelaySlot = 6,
        kReverbSlot = 7,
        kChorusSlot = 8,
        kModFxSlot = 9
    };

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace vae
