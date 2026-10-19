/*
  The Surge XT backend (VAE_WITH_SURGE, GPL-3.0-or-later): headless
  SurgeSynthesizer from surge-common, driven by the schema v2 vector
  (schema/controls.json). Scene A only. Surge parameters are resolved by
  storage name (Parameter::get_storage_name), which is stable across osc/FX
  types, unlike display names.

  Mapping notes (verified against the pinned Surge revision):
    play_mode      a_polymode: poly 0, mono 1, legato 2 (Mono Single Trigger)
    oscN_wave      sine -> Sine osc; triangle -> Modern osc (triangle mix 1,
                   saw/pulse 0); saw -> Classic shape 0; square -> Classic
                   shape -1, width 0.5; noise -> S&H Noise (correlation 0,
                   sync +48 st: Surge's S&H noise holds at the note rate, the
                   sync raises that so it is broadband); wavetable ->
                   Wavetable osc on a built-in 8-frame additive table, morph 0
                   (headless Surge has no factory tables and its fallback is a
                   sine); fm -> FM2 with M1 amount 0.3, ratio 2 (FM2's
                   defaults are a pure sine).
    filter_routing serial -> Serial 1: routes "filter1"/"both" pass F1 then
                   F2, "filter2" skips F1. parallel -> Dual 1: each filter
                   path carries the oscs routed to it at -6 dB (x0.5) and the
                   paths are summed. In both configs a filter set to Off is a
                   pass-through, so in parallel an osc routed to "both" with
                   filter 2 Off leaks through dry at half level. (Wide, Surge's
                   template default, ignores per-osc routing.)
    filter types   off 0, lp12 1, lp24 2, hp12 4, hp24 5, bp 6, notch 7; subtype 0.
    LFO shapes     sine 0, triangle 1, square 2, saw 3, random 5 (S&H),
                   smooth_random 4 (Noise); retrigger -> trigmode 1, else 0.
    routes         *_destination and *_depth -> one modulation routing from
                   ms_filtereg / ms_lfo1 / ms_lfo2 with setModDepth01(d),
                   d = 2v - 1. Surge stores depth as d x (param max - min) in
                   the parameter's units and adds source output x depth, so at
                   full modulator output the destination moves by
                   d x (its full Surge range): the shared routing convention.
    FX slots       A insert 1 Distortion, A2 EQ, A3 Chorus, A4 Phaser/Flanger,
                   Global 1 Delay, Global 2 Reverb 2; *_on = type set / Off.

  Osc and FX type changes are applied lazily by Surge (loadOscalgos/loadFx
  inside process()) and reset that unit's parameters to the type defaults.
  The engine applies them itself before rendering and then re-writes the
  parameters it owns, so no voice ever starts with a stale layout.

  Surge renders fixed BLOCK_SIZE (32) quanta; arbitrary host blocks are served
  from a one-quantum buffer, with MIDI applied at the quantum it falls in (up
  to 31 samples early; no added latency).
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
    float fxParam (int slot, int k) const;

    /** Voice modulation routings in scene A: count, and the depth (Surge
        units) of the route from source (0 FEG, 1 LFO 1, 2 LFO 2) to the
        parameter with this storage name, or NaN when there is none. */
    int numModRoutes() const;
    float modRouteDepth (int source, const char* destinationStorageName) const;

    /** Voices that are not yet released, scene A. */
    int numHeldVoices() const;

    /** Surge FX slot indices used by the fixed chain. */
    enum FxSlot : int { kDistSlot = 0, kEqSlot = 1, kDelaySlot = 6, kReverbSlot = 7, kChorusSlot = 8, kModFxSlot = 9 };

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace vae
