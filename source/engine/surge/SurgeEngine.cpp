#include "SurgeEngine.h"

#include "engine/lite/dsp/Wavetables.h"
#include "schema/SurgeCurves.h"

// Surge's headers are not written for JUCE's recommended warning set.
JUCE_BEGIN_IGNORE_WARNINGS_GCC_LIKE ("-Wsign-conversion", "-Wfloat-equal", "-Wpedantic", "-Wshadow", "-Wconversion",
                                     "-Wfloat-conversion", "-Wold-style-cast", "-Wzero-as-null-pointer-constant",
                                     "-Wunused-parameter", "-Wswitch-enum", "-Wimplicit-int-float-conversion",
                                     "-Wshorten-64-to-32", "-Wextra-semi", "-Wcast-align", "-Wunused-value",
                                     "-Wunused-variable", "-Wunused-local-typedefs", "-Wstrict-aliasing",
                                     "-Wsign-compare")
JUCE_BEGIN_IGNORE_WARNINGS_MSVC (4244 4267 4100 4456 4458)
#include "SurgeSynthesizer.h"
JUCE_END_IGNORE_WARNINGS_MSVC
JUCE_END_IGNORE_WARNINGS_GCC_LIKE

#include <array>
#include <cstring>
#include <limits>
#include <unordered_map>

namespace vae
{
namespace
{
using namespace schema;

struct SilentPluginLayer final : SurgeSynthesizer::PluginLayer
{
    void surgeParameterUpdated (const SurgeSynthesizer::ID&, float) override {}
    void surgeMacroUpdated (long, float) override {}
};

constexpr int kQuantum = BLOCK_SIZE;
constexpr char kChannel = 0;
constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

// Schema option index -> Surge integer value.
constexpr int kPlayModes[] = { pm_poly, pm_mono, pm_mono_st };
constexpr int kFilterTypes[] = { sst::filters::fut_none,   sst::filters::fut_lp12, sst::filters::fut_lp24,
                                 sst::filters::fut_hp12,   sst::filters::fut_hp24, sst::filters::fut_bp12,
                                 sst::filters::fut_notch12 };
constexpr int kLfoShapes[] = { lt_sine, lt_tri, lt_square, lt_ramp, lt_snh, lt_noise };
constexpr int kFilterConfigs[] = { fc_serial1, fc_dual1 };
constexpr modsources kRouteSources[] = { ms_filtereg, ms_lfo1, ms_lfo2 };

enum Wave
{
    kSine,
    kTriangle,
    kSaw,
    kSquare,
    kNoise,
    kWavetable,
    kFm,
    kNumWaves
};
constexpr int kWaveOscType[] = { ot_sine, ot_modern, ot_classic, ot_classic, ot_shnoise, ot_wavetable, ot_FM2 };
static_assert (std::size (kWaveOscType) == kNumWaves);

struct OscControls
{
    int on, wave, level, octave, pitch, route;
};
constexpr OscControls kOscControls[] = {
    { kOsc1On, kOsc1Wave, kOsc1Level, kOsc1Octave, kOsc1Pitch, kOsc1Filter },
    { kOsc2On, kOsc2Wave, kOsc2Level, kOsc2Octave, kOsc2Pitch, kOsc2Filter },
    { kOsc3On, kOsc3Wave, kOsc3Level, kOsc3Octave, kOsc3Pitch, kOsc3Filter },
};

struct RouteControls
{
    int destination, depth;
};
constexpr RouteControls kRouteControls[] = {
    { kFegDestination, kFegDepth },
    { kLfo1Destination, kLfo1Depth },
    { kLfo2Destination, kLfo2Depth },
};

// Fixed FX chain: slot, Surge type, and (control, p index) for its parameters.
struct FxParamMap
{
    int control;
    int k;
};
struct FxUnit
{
    int slot;
    int fxType; // the mod FX unit's type depends on modfx_type, see fxTypeFor
    int onControl;
    std::array<FxParamMap, 5> params; // control < 0 = unused
};
constexpr FxUnit kFxUnits[] = {
    { SurgeEngine::kDistSlot,
      fxt_distortion,
      kDistOn,
      { { { kDistDrive, 4 }, { kDistTone, 9 }, { -1, 0 }, { -1, 0 }, { -1, 0 } } } },
    { SurgeEngine::kEqSlot,
      fxt_eq,
      kEqOn,
      { { { kEqLowGain, 0 }, { kEqMidFreq, 4 }, { kEqMidGain, 3 }, { kEqHighGain, 6 }, { -1, 0 } } } },
    { SurgeEngine::kChorusSlot,
      fxt_chorus4,
      kChorusOn,
      { { { kChorusRate, 1 }, { kChorusDepth, 2 }, { kChorusMix, 6 }, { -1, 0 }, { -1, 0 } } } },
    { SurgeEngine::kModFxSlot,
      fxt_phaser,
      kModfxOn,
      { { { -1, 0 }, { -1, 0 }, { -1, 0 }, { -1, 0 }, { -1, 0 } } } }, // type-dependent, see modFxParam
    { SurgeEngine::kDelaySlot,
      fxt_delay,
      kDelayOn,
      { { { kDelayTime, 0 }, { kDelayTime, 1 }, { kDelayFeedback, 2 }, { kDelayMix, 10 }, { -1, 0 } } } },
    { SurgeEngine::kReverbSlot,
      fxt_reverb2,
      kReverbOn,
      { { { kReverbSize, 1 }, { kReverbDecay, 2 }, { kReverbDamping, 7 }, { kReverbMix, 9 }, { -1, 0 } } } },
};
constexpr int kModFxUnit = 3;

int fxTypeFor (int unit, const Snapshot& c)
{
    if (unit == kModFxUnit)
        return asIndex (kModfxType, c[kModfxType]) == 0 ? fxt_phaser : fxt_flanger;
    return kFxUnits[(size_t) unit].fxType;
}

/** Phaser / flanger parameter index for a modfx control, or -1. */
int modFxParam (int fxType, int control)
{
    const bool phaser = fxType == fxt_phaser;
    switch (control)
    {
        case kModfxRate:     return phaser ? 3 : 2;
        case kModfxDepth:    return phaser ? 4 : 3;
        case kModfxFeedback: return phaser ? 1 : 7;
        case kModfxMix:      return phaser ? 6 : 10;
        default:             return -1;
    }
}

/** Which FX unit a control belongs to, or -1. */
int fxUnitOf (int control)
{
    for (int u = 0; u < int (std::size (kFxUnits)); ++u)
    {
        if (control == kFxUnits[u].onControl)
            return u;
        for (const auto& p : kFxUnits[u].params)
            if (p.control == control)
                return u;
    }
    if (control == kModfxType || control == kModfxRate || control == kModfxDepth || control == kModfxFeedback
        || control == kModfxMix)
        return kModFxUnit;
    return -1;
}

/** Without a data path Surge has no factory wavetables and its default table
    is a single sine frame, which would make "wavetable" sound like "sine".
    Build the one-frame table Lite plays instead (dsp::defaultTableHarmonic). */
void buildDefaultWavetable (Wavetable& wt)
{
    constexpr int size = 1024;
    std::vector<float> frame ((size_t) size);
    float peak = 0.0f;
    for (int i = 0; i < size; ++i)
    {
        const float phase = juce::MathConstants<float>::twoPi * float (i) / float (size);
        float s = 0.0f;
        for (int h = 1; h <= vae::dsp::kDefaultTableHarmonics; ++h)
            s += vae::dsp::defaultTableHarmonic (h) * std::sin (phase * float (h));
        frame[(size_t) i] = s;
        peak = juce::jmax (peak, std::abs (s));
    }
    for (auto& s : frame)
        s /= peak;

    wt_header header {};
    std::memcpy (header.tag, "vawt", 4);
    header.n_samples = size;
    header.n_tables = 1;
    header.flags = 0; // float data
    wt.BuildWT (frame.data(), header, false);
}

float unitsOf (const Parameter& p)
{
    switch (p.valtype)
    {
        case vt_int:   return float (p.val.i);
        case vt_bool:  return p.val.b ? 1.0f : 0.0f;
        case vt_float: return p.val.f;
    }
    return kNaN;
}
} // namespace

//==============================================================================
struct SurgeEngine::Impl
{
    SilentPluginLayer layer;
    std::unique_ptr<SurgeSynthesizer> synth;

    Snapshot controls = Snapshot::defaults(); // desired
    Snapshot written;                         // last written to Surge (NaN = never)

    // Scene A / global parameters, by role.
    std::unordered_map<std::string, Parameter*> byName;
    std::array<Parameter*, kControlCount> direct {}; // continuous / int controls with one Surge param
    std::array<Parameter*, 7> destinations {};       // route destination option -> param (none = null)
    std::array<Parameter*, 2> filterSubtype {};      // a_filter1_subtype, a_filter2_subtype

    std::array<bool, 3> oscDirty {};
    std::array<bool, std::size (kFxUnits)> fxDirty {};
    std::array<int, 3> routeDestination { 0, 0, 0 }; // option currently routed per source

    std::array<float, kQuantum> quantumL {}, quantumR {};
    int quantumPos = kQuantum;

    // Full patch mode (loadPatchData). Data blocks: message thread only.
    std::atomic<bool> external { false }, leaveExternal { false };
    juce::MemoryBlock externalPatch, defaultPatch;

    //--------------------------------------------------------------------------
    Parameter* find (const char* name) const
    {
        const auto it = byName.find (name);
        return it != byName.end() ? it->second : nullptr;
    }

    void resolve()
    {
        byName.clear();
        for (auto* p : synth->storage.getPatch().param_ptr)
            if (p->scene <= 1) // global (0) and scene A (1)
                byName.emplace (p->get_storage_name(), p);

        static const std::pair<int, const char*> kDirect[] = {
            { kPlayMode, "a_polymode" },
            { kGlideTime, "a_portamento" },
            { kFilterRouting, "a_fb_config" },
            { kOsc1Level, "a_level_o1" },
            { kOsc1Octave, "a_osc1_octave" },
            { kOsc1Pitch, "a_osc1_pitch" },
            { kOsc1On, "a_mute_o1" },
            { kOsc1Filter, "a_route_o1" },
            { kOsc1Wave, "a_osc1_type" },
            { kOsc2Level, "a_level_o2" },
            { kOsc2Octave, "a_osc2_octave" },
            { kOsc2Pitch, "a_osc2_pitch" },
            { kOsc2On, "a_mute_o2" },
            { kOsc2Filter, "a_route_o2" },
            { kOsc2Wave, "a_osc2_type" },
            { kOsc3Level, "a_level_o3" },
            { kOsc3Octave, "a_osc3_octave" },
            { kOsc3Pitch, "a_osc3_pitch" },
            { kOsc3On, "a_mute_o3" },
            { kOsc3Filter, "a_route_o3" },
            { kOsc3Wave, "a_osc3_type" },
            { kFilter1Type, "a_filter1_type" },
            { kFilter1Cutoff, "a_filter1_cutoff" },
            { kFilter1Resonance, "a_filter1_resonance" },
            { kFilter1Env, "a_filter1_envmod" },
            { kFilter1Keytrack, "a_filter1_keytrack" },
            { kFilter2Type, "a_filter2_type" },
            { kFilter2Cutoff, "a_filter2_cutoff" },
            { kFilter2Resonance, "a_filter2_resonance" },
            { kFilter2Env, "a_filter2_envmod" },
            { kFilter2Keytrack, "a_filter2_keytrack" },
            { kAmpAttack, "a_env1_attack" },
            { kAmpAttackCurve, "a_env1_attack_shape" },
            { kAmpDecay, "a_env1_decay" },
            { kAmpDecayCurve, "a_env1_decay_shape" },
            { kAmpSustain, "a_env1_sustain" },
            { kAmpRelease, "a_env1_release" },
            { kAmpReleaseCurve, "a_env1_release_shape" },
            { kFegAttack, "a_env2_attack" },
            { kFegAttackCurve, "a_env2_attack_shape" },
            { kFegDecay, "a_env2_decay" },
            { kFegDecayCurve, "a_env2_decay_shape" },
            { kFegSustain, "a_env2_sustain" },
            { kFegRelease, "a_env2_release" },
            { kFegReleaseCurve, "a_env2_release_shape" },
            { kLfo1Shape, "a_lfo0_shape" },
            { kLfo1Rate, "a_lfo0_rate" },
            { kLfo1Amount, "a_lfo0_magnitude" },
            { kLfo1Unipolar, "a_lfo0_unipolar" },
            { kLfo1Retrigger, "a_lfo0_trigmode" },
            { kLfo1Phase, "a_lfo0_phase" },
            { kLfo1FadeIn, "a_lfo0_attack" },
            { kLfo2Shape, "a_lfo1_shape" },
            { kLfo2Rate, "a_lfo1_rate" },
            { kLfo2Amount, "a_lfo1_magnitude" },
            { kLfo2Unipolar, "a_lfo1_unipolar" },
            { kLfo2Retrigger, "a_lfo1_trigmode" },
            { kLfo2Phase, "a_lfo1_phase" },
            { kLfo2FadeIn, "a_lfo1_attack" },
            { kMasterVolume, "volume" },
        };
        direct.fill (nullptr);
        for (const auto& [index, name] : kDirect)
            direct[(size_t) index] = find (name);

        destinations = { nullptr,
                         find ("a_pitch"),
                         find ("a_filter1_cutoff"),
                         find ("a_filter2_cutoff"),
                         find ("a_filter1_resonance"),
                         find ("a_volume"),
                         find ("a_pan") };
        filterSubtype = { find ("a_filter1_subtype"), find ("a_filter2_subtype") };
    }

    int numResolved() const
    {
        int n = 0;
        for (int i = 0; i < kControlCount; ++i)
        {
            bool ok = direct[(size_t) i] != nullptr || fxUnitOf (i) >= 0;
            for (const auto& r : kRouteControls)
                if (i == r.destination || i == r.depth)
                    ok = std::all_of (destinations.begin() + 1, destinations.end(),
                                      [] (auto* p) { return p != nullptr; });
            n += ok ? 1 : 0;
        }
        return n;
    }

    //--------------------------------------------------------------------------
    void setNorm (Parameter& p, float norm) { synth->setParameter01 (synth->idForParameter (&p), norm, false, false); }

    void setUnits (Parameter& p, float units)
    {
        if (p.valtype == vt_float)
            setNorm (p, juce::jlimit (0.0f, 1.0f, (units - p.val_min.f) / (p.val_max.f - p.val_min.f)));
        else if (p.valtype == vt_int)
            setNorm (p, Parameter::intScaledToFloat (juce::roundToInt (units), p.val_max.i, p.val_min.i));
        else
            setNorm (p, units >= 0.5f ? 1.0f : 0.0f);
    }

    /** Write one control to Surge (skips unchanged values). */
    void write (int index)
    {
        if (external)
            return;
        const float v = controls[index];
        if (juce::exactlyEqual (written[index], v))
            return;
        written[index] = v;

        if (const int unit = fxUnitOf (index); unit >= 0)
        {
            writeFx (unit, index);
            return;
        }

        for (int r = 0; r < 3; ++r)
            if (index == kRouteControls[r].destination || index == kRouteControls[r].depth)
            {
                writeRoute (r);
                return;
            }

        auto* p = direct[(size_t) index];
        if (p == nullptr)
            return;
        const int k = asIndex (index, v);

        switch (index)
        {
            case kPlayMode:      setUnits (*p, float (kPlayModes[k])); return;
            case kFilterRouting: setUnits (*p, float (kFilterConfigs[k])); return;
            case kOsc1On:
            case kOsc2On:
            case kOsc3On:
                setUnits (*p, k != 0 ? 0.0f : 1.0f); // mute is inverted
                return;
            case kOsc1Octave:
            case kOsc2Octave:
            case kOsc3Octave: setUnits (*p, float (k - 3)); return;
            case kOsc1Filter:
            case kOsc2Filter:
            case kOsc3Filter: setUnits (*p, float (k)); return;
            case kOsc1Wave:
            case kOsc2Wave:
            case kOsc3Wave:
            {
                const int n = index == kOsc1Wave ? 0 : index == kOsc2Wave ? 1 : 2;
                setUnits (*p, float (kWaveOscType[k]));
                oscDirty[(size_t) n] = true; // shape params are written once the type is live
                return;
            }
            case kFilter1Type:
            case kFilter2Type:
            {
                setUnits (*p, float (kFilterTypes[k]));
                if (auto* subtype = filterSubtype[index == kFilter1Type ? 0 : 1])
                    setUnits (*subtype, 0.0f);
                return;
            }
            case kLfo1Shape:
            case kLfo2Shape: setUnits (*p, float (kLfoShapes[k])); return;
            case kLfo1Retrigger:
            case kLfo2Retrigger:
                setUnits (*p, k != 0 ? 1.0f : 0.0f); // keytrigger / freerun
                return;
            default: break;
        }

        // Remaining categoricals/toggles store their index directly (envelope curves, unipolar).
        if (control (index).kind != Kind::Continuous)
            setUnits (*p, float (k));
        else
            setUnits (*p, surge::toUnits (index, v));
    }

    //--------------------------------------------------------------------------
    void writeRoute (int r)
    {
        const auto& rc = kRouteControls[r];
        const int destination = asIndex (rc.destination, controls[rc.destination]);
        const float depth = surge::bipolar (controls[rc.depth]);
        const auto source = kRouteSources[r];

        // Clear the previous destination when it changes.
        if (routeDestination[(size_t) r] != destination)
        {
            if (auto* old = destinations[(size_t) routeDestination[(size_t) r]])
                synth->clearModulation (synth->idForParameter (old).getSynthSideId(), source, 0, 0, true);
            routeDestination[(size_t) r] = destination;
        }

        if (auto* p = destinations[(size_t) destination])
            synth->setModDepth01 (synth->idForParameter (p).getSynthSideId(), source, 0, 0, depth);
    }

    //--------------------------------------------------------------------------
    Parameter& fxTypeParam (int unit) { return synth->storage.getPatch().fx[kFxUnits[unit].slot].type; }

    /** Load a unit's FX type (lazy: settle() makes it live, which allocates).
        Done for every unit in prepare(), and again only when modfx_type changes. */
    void stageFxType (int unit)
    {
        // setParameter01 on an FX type stages it in fxsync; loadFx makes it live.
        setUnits (fxTypeParam (unit), float (fxTypeFor (unit, controls)));
        fxDirty[(size_t) unit] = true;
    }

    void writeFx (int unit, int index)
    {
        if (index == kFxUnits[unit].onControl)
        {
            // On/off goes through Surge's per-slot fx_disable bits: the effect
            // stays loaded, so toggling never allocates on the audio thread.
            const int slot = kFxUnits[unit].slot;
            const bool on = controls[index] >= 0.5f;
            auto& mask = synth->storage.getPatch().fx_disable.val.i;
            const bool wasOn = (mask & (1 << slot)) == 0;
            mask = on ? (mask & ~(1 << slot)) : (mask | (1 << slot));
            if (wasOn && ! on && synth->fx[slot] != nullptr)
                synth->fx[slot]->suspend(); // no stale tail when re-enabled
            return;
        }
        if (index == kModfxType)
        {
            stageFxType (unit);
            return;
        }
        if (! fxDirty[(size_t) unit])
            writeFxParam (unit, index);
    }

    void writeFxParam (int unit, int index)
    {
        auto& fx = synth->storage.getPatch().fx[kFxUnits[unit].slot];
        if (fx.type.val.i == fxt_off)
            return;

        const float units = surge::toUnits (index, controls[index]);
        if (unit == kModFxUnit)
        {
            if (const int k = modFxParam (fx.type.val.i, index); k >= 0)
                setUnits (fx.p[k], units);
            return;
        }
        for (const auto& m : kFxUnits[unit].params)
            if (m.control == index)
                setUnits (fx.p[m.k], units);
    }

    //--------------------------------------------------------------------------
    /** Apply staged osc/FX type changes and re-write the parameters they reset. */
    void settle()
    {
        if (std::any_of (oscDirty.begin(), oscDirty.end(), [] (bool b) { return b; }))
        {
            synth->loadOscalgos();
            for (int n = 0; n < 3; ++n)
                if (oscDirty[(size_t) n])
                {
                    writeOscShape (n);
                    oscDirty[(size_t) n] = false;
                }
        }

        if (std::any_of (fxDirty.begin(), fxDirty.end(), [] (bool b) { return b; }))
        {
            if (synth->load_fx_needed)
                synth->loadFx (false, false);
            for (int u = 0; u < int (fxDirty.size()); ++u)
                if (fxDirty[(size_t) u])
                {
                    fxDirty[(size_t) u] = false;
                    for (int i = 0; i < kControlCount; ++i)
                        if (fxUnitOf (i) == u && i != kFxUnits[u].onControl && i != kModfxType)
                            writeFxParam (u, i);
                }
        }
    }

    void writeOscShape (int n)
    {
        auto& data = synth->storage.getPatch().scene[0].osc[n];
        const int wave = asIndex (kOscControls[n].wave, controls[kOscControls[n].wave]);
        switch (wave)
        {
            case kTriangle: // Modern: saw, pulse, triangle mixes
                setUnits (data.p[0], 0.0f);
                setUnits (data.p[1], 0.0f);
                setUnits (data.p[2], 1.0f);
                break;
            case kSaw:
            case kSquare: // Classic: shape, width 1, width 2, sub mix, sync
                setUnits (data.p[0], wave == kSaw ? 0.0f : -1.0f);
                setUnits (data.p[1], 0.5f);
                setUnits (data.p[2], 0.5f);
                setUnits (data.p[3], 0.0f);
                setUnits (data.p[4], 0.0f);
                break;
            case kNoise:                    // S&H Noise: correlation 0; sync +48 st so the hold rate
                setUnits (data.p[0], 0.0f); // sits well above the note (broadband)
                setUnits (data.p[4], 48.0f);
                break;
            case kFm: // FM2: M1 amount, M1 ratio
                setUnits (data.p[0], 0.3f);
                setUnits (data.p[1], 2.0f);
                break;
            default: // Sine, Wavetable: type defaults
                break;
        }
    }

    //--------------------------------------------------------------------------
    void applyMidi (const juce::MidiMessage& msg)
    {
        if (msg.isNoteOn())
            synth->playNote (kChannel, (char) msg.getNoteNumber(), (char) msg.getVelocity(), 0);
        else if (msg.isNoteOff())
            synth->releaseNote (kChannel, (char) msg.getNoteNumber(), (char) msg.getVelocity());
        else if (msg.isPitchWheel())
            synth->pitchBend (kChannel, msg.getPitchWheelValue() - 8192);
        else if (msg.isAllNotesOff() || msg.isAllSoundOff())
            synth->allNotesOff();
        else if (msg.isController())
            synth->channelController (kChannel, msg.getControllerNumber(), msg.getControllerValue());
    }

    void applySchema();

    void fillQuantum()
    {
        synth->process();
        std::copy (synth->output[0], synth->output[0] + kQuantum, quantumL.begin());
        std::copy (synth->output[1], synth->output[1] + kQuantum, quantumR.begin());
        quantumPos = 0;
    }
};

//==============================================================================
SurgeEngine::SurgeEngine()
    : impl (std::make_unique<Impl>())
{
}
SurgeEngine::~SurgeEngine() = default;

void SurgeEngine::prepare (double sampleRate, int)
{
    // A fresh synth per prepare also resets voices, FX tails and routings.
    impl->synth = std::make_unique<SurgeSynthesizer> (&impl->layer, SurgeStorage::skipPatchLoadDataPathSentinel);
    impl->synth->setSamplerate (static_cast<float> (sampleRate));
    impl->synth->time_data.tempo = 120.0;
    impl->synth->process(); // let the default patch settle

    void* data = nullptr;
    const auto size = impl->synth->saveRaw (&data); // the patch owns the buffer
    impl->defaultPatch.replaceAll (data, size);

    impl->resolve();
    impl->external = false;
    impl->leaveExternal = false;
    impl->applySchema();
    impl->quantumPos = kQuantum;

    if (! impl->externalPatch.isEmpty())
        loadPatchData (juce::MemoryBlock (impl->externalPatch));
}

void SurgeEngine::Impl::applySchema()
{
    for (auto& osc : synth->storage.getPatch().scene[0].osc)
    {
        buildDefaultWavetable (osc.wt);
        osc.wt.queue_id = -1;
        osc.wt.current_id = -1;
    }

    written.values.fill (kNaN);
    oscDirty.fill (false);
    fxDirty.fill (false);
    routeDestination.fill (0);

    // Load the fixed FX chain once; *_on only flips fx_disable bits afterwards.
    for (int u = 0; u < int (std::size (kFxUnits)); ++u)
        stageFxType (u);

    // Make every wave's parameters explicit, even when the osc type is the default.
    for (int n = 0; n < 3; ++n)
        oscDirty[(size_t) n] = true;
    for (int i = 0; i < kControlCount; ++i)
        write (i);
    settle();
}

void SurgeEngine::loadPatchData (const juce::MemoryBlock& data)
{
    impl->externalPatch = data;
    impl->leaveExternal = false;
    impl->external = true;
    if (impl->synth != nullptr)
        impl->synth->enqueuePatchForLoad (data.getData(), (int) data.getSize());
}

void SurgeEngine::useSchemaControls()
{
    if (! impl->external || impl->synth == nullptr)
        return;
    impl->externalPatch.reset();
    impl->synth->enqueuePatchForLoad (impl->defaultPatch.getData(), (int) impl->defaultPatch.getSize());
    impl->leaveExternal = true;
}

bool SurgeEngine::hasPatchData() const noexcept { return impl->external; }

void SurgeEngine::reset()
{
    if (impl->synth != nullptr)
    {
        // allNotesOff clears Surge's held-key bookkeeping; stopSound then
        // frees every voice and suspends the FX (the hard cut reset() promises).
        impl->synth->allNotesOff();
        impl->synth->stopSound();
        impl->quantumL.fill (0.0f);
        impl->quantumR.fill (0.0f);
        impl->quantumPos = kQuantum;
    }
}

void SurgeEngine::setControl (int index, float value)
{
    if (index < 0 || index >= kControlCount || ! std::isfinite (value))
        return;

    impl->controls[index] = sanitiseValue (index, value);
    if (impl->synth != nullptr)
        impl->write (index);
}

void SurgeEngine::process (juce::AudioBuffer<float>& buffer, const juce::MidiBuffer& midi)
{
    const int numSamples = buffer.getNumSamples();
    if (impl->synth == nullptr || buffer.getNumChannels() < 2)
    {
        buffer.clear();
        return;
    }

    if (impl->leaveExternal.exchange (false))
    {
        // Load the default patch now, so the schema rewrite lands on top of it.
        impl->synth->processEnqueuedPatchIfNeeded();
        impl->external = false;
        impl->applySchema();
    }
    impl->settle();

    auto* left = buffer.getWritePointer (0);
    auto* right = buffer.getWritePointer (1);
    auto nextEvent = midi.cbegin();

    int done = 0;
    while (done < numSamples)
    {
        if (impl->quantumPos >= kQuantum)
        {
            // Events that fall inside the quantum about to be rendered.
            const int quantumEnd = done + kQuantum;
            for (; nextEvent != midi.cend() && (*nextEvent).samplePosition < quantumEnd; ++nextEvent)
                impl->applyMidi ((*nextEvent).getMessage());
            impl->fillQuantum();
        }

        const int take = juce::jmin (kQuantum - impl->quantumPos, numSamples - done);
        std::copy_n (impl->quantumL.begin() + impl->quantumPos, take, left + done);
        std::copy_n (impl->quantumR.begin() + impl->quantumPos, take, right + done);
        impl->quantumPos += take;
        done += take;
    }

    // Anything left (offsets past the block end) applies now.
    for (; nextEvent != midi.cend(); ++nextEvent)
        impl->applyMidi ((*nextEvent).getMessage());
}

//==============================================================================
int SurgeEngine::numResolvedControls() const noexcept { return impl->synth != nullptr ? impl->numResolved() : 0; }

float SurgeEngine::surgeValue (const char* storageName) const
{
    const auto* p = impl->synth != nullptr ? impl->find (storageName) : nullptr;
    return p != nullptr ? unitsOf (*p) : kNaN;
}

int SurgeEngine::oscType (int osc) const
{
    return impl->synth->storage.getPatch().scene[0].osc[juce::jlimit (0, 2, osc)].type.val.i;
}

float SurgeEngine::oscParam (int osc, int k) const
{
    return unitsOf (impl->synth->storage.getPatch().scene[0].osc[juce::jlimit (0, 2, osc)].p[juce::jlimit (
        0, n_osc_params - 1, k)]);
}

int SurgeEngine::fxType (int slot) const
{
    return impl->synth->storage.getPatch().fx[juce::jlimit (0, n_fx_slots - 1, slot)].type.val.i;
}

bool SurgeEngine::fxEnabled (int slot) const
{
    return (impl->synth->storage.getPatch().fx_disable.val.i & (1 << juce::jlimit (0, n_fx_slots - 1, slot))) == 0;
}

float SurgeEngine::fxParam (int slot, int k) const
{
    return unitsOf (impl->synth->storage.getPatch()
                        .fx[juce::jlimit (0, n_fx_slots - 1, slot)]
                        .p[juce::jlimit (0, n_fx_params - 1, k)]);
}

int SurgeEngine::numModRoutes() const { return int (impl->synth->storage.getPatch().scene[0].modulation_voice.size()); }

float SurgeEngine::modRouteDepth (int source, const char* destinationStorageName) const
{
    const auto* p = impl->find (destinationStorageName);
    if (p == nullptr || source < 0 || source > 2)
        return kNaN;
    for (const auto& m : impl->synth->storage.getPatch().scene[0].modulation_voice)
        if (m.destination_id == p->param_id_in_scene && m.source_id == kRouteSources[source] && m.source_index == 0)
            return m.depth;
    return kNaN;
}

int SurgeEngine::numHeldVoices() const { return impl->synth->getNonReleasedVoices (0); }

} // namespace vae
