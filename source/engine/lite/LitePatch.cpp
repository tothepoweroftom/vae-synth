#include "LitePatch.h"

#include <cmath>
#include <string_view>

namespace vae::lite
{
namespace
{
using namespace schema;
namespace sc = schema::surge;

float dbToGain (float db) { return std::pow (10.0f, db / 20.0f); }

// Shaped-domain exponents (Surge's envelope shapes).
constexpr float kAttackPowers[] = { 0.5f, 1.0f, 2.0f };   // convex, linear, concave
constexpr float kDecayPowers[] = { 1.0f, 2.0f, 3.0f };    // linear, quadratic, cubic

// The enums in LitePatch.h are read straight from option indices, so they must
// mirror the option lists in schema/controls.json; these fail to compile when
// a list changes length or its last option moves.
constexpr bool endsWith (int control, int lastEnumValue, std::string_view lastOption)
{
    const auto& c = generated::kRows[control];
    return c.numOptions == lastEnumValue + 1 && lastOption == c.options[lastEnumValue];
}
static_assert (endsWith (kPlayMode, int (PlayMode::Legato), "legato"));
static_assert (endsWith (kOsc1Wave, int (Wave::Fm), "fm"));
static_assert (endsWith (kOsc1Filter, int (Route::Filter2), "filter2"));
static_assert (endsWith (kFilter1Type, int (FilterType::Notch), "notch"));
static_assert (endsWith (kLfo1Shape, int (LfoShape::SmoothRandom), "smooth_random"));
static_assert (endsWith (kLfo1Destination, int (ModDest::Pan), "pan"));
static_assert (endsWith (kModfxType, int (ModFxType::Flanger), "flanger"));

int idx (const Snapshot& s, int index) { return asIndex (index, s[index]); }
bool on (const Snapshot& s, int index) { return s[index] >= 0.5f; }

void setOsc (OscSettings& o, const Snapshot& s, int onI, int waveI, int levelI, int octI, int pitchI, int routeI)
{
    o.on = on (s, onI);
    o.wave = Wave (idx (s, waveI));
    o.gain = sc::amplitudeGain (s[levelI]);
    o.semitones = 12.0f * float (idx (s, octI) - 3) + sc::toUnits (pitchI, s[pitchI]);
    o.route = Route (idx (s, routeI));
}

void setFilter (FilterSettings& f, const Snapshot& s, int typeI, int cutI, int resI, int envI, int keyI)
{
    f.type = FilterType (idx (s, typeI));
    f.cutoffSemis = sc::toUnits (cutI, s[cutI]);
    f.resonance = s[resI];
    f.envSemis = sc::toUnits (envI, s[envI]);
    f.keytrack = sc::toUnits (keyI, s[keyI]);
}

void setEnv (EnvSettings& e, const Snapshot& s, int a, int ac, int d, int dc, int su, int r, int rc)
{
    e.attack = sc::seconds (a, s[a]);
    e.attackPower = kAttackPowers[idx (s, ac)];
    e.decay = sc::seconds (d, s[d]);
    e.decayPower = kDecayPowers[idx (s, dc)];
    e.sustain = s[su];
    e.release = sc::seconds (r, s[r]);
    e.releasePower = kDecayPowers[idx (s, rc)];
}

void setLfo (LfoSettings& l, const Snapshot& s, int base)
{
    // Controls are laid out shape, rate, amount, unipolar, retrigger, phase, fade_in, destination, depth.
    l.shape = LfoShape (idx (s, base));
    l.rateHz = sc::hz (base + 1, s[base + 1]);
    l.amount = s[base + 2];
    l.unipolar = on (s, base + 3);
    l.retrigger = on (s, base + 4);
    l.phase = s[base + 5];
    // The curve minimum (2^-8 s, ~4 ms) is Surge's "no fade".
    l.fadeIn = s[base + 6] <= sc::kOffAtCurveMin ? 0.0f : sc::seconds (base + 6, s[base + 6]);
    l.route.dest = ModDest (idx (s, base + 7));
    l.route.depth = sc::bipolar (s[base + 8]);
}

static_assert (kLfo1Rate == kLfo1Shape + 1 && kLfo1Depth == kLfo1Shape + 8, "LFO 1 control layout");
static_assert (kLfo2Rate == kLfo2Shape + 1 && kLfo2Depth == kLfo2Shape + 8, "LFO 2 control layout");
} // namespace

void Patch::set (int index, const Snapshot& s)
{
    switch (index)
    {
        case kPlayMode:  playMode = PlayMode (idx (s, index)); break;
        case kGlideTime: glideSeconds = sc::glideSeconds (s[index]); break;

        case kOsc1On: case kOsc1Wave: case kOsc1Level: case kOsc1Octave: case kOsc1Pitch: case kOsc1Filter:
            setOsc (osc[0], s, kOsc1On, kOsc1Wave, kOsc1Level, kOsc1Octave, kOsc1Pitch, kOsc1Filter);
            break;
        case kOsc2On: case kOsc2Wave: case kOsc2Level: case kOsc2Octave: case kOsc2Pitch: case kOsc2Filter:
            setOsc (osc[1], s, kOsc2On, kOsc2Wave, kOsc2Level, kOsc2Octave, kOsc2Pitch, kOsc2Filter);
            break;
        case kOsc3On: case kOsc3Wave: case kOsc3Level: case kOsc3Octave: case kOsc3Pitch: case kOsc3Filter:
            setOsc (osc[2], s, kOsc3On, kOsc3Wave, kOsc3Level, kOsc3Octave, kOsc3Pitch, kOsc3Filter);
            break;

        case kFilterRouting: parallel = idx (s, index) == 1; break;
        case kFilter1Type: case kFilter1Cutoff: case kFilter1Resonance: case kFilter1Env: case kFilter1Keytrack:
            setFilter (filter[0], s, kFilter1Type, kFilter1Cutoff, kFilter1Resonance, kFilter1Env, kFilter1Keytrack);
            break;
        case kFilter2Type: case kFilter2Cutoff: case kFilter2Resonance: case kFilter2Env: case kFilter2Keytrack:
            setFilter (filter[1], s, kFilter2Type, kFilter2Cutoff, kFilter2Resonance, kFilter2Env, kFilter2Keytrack);
            break;

        case kAmpAttack: case kAmpAttackCurve: case kAmpDecay: case kAmpDecayCurve:
        case kAmpSustain: case kAmpRelease: case kAmpReleaseCurve:
            setEnv (ampEnv, s, kAmpAttack, kAmpAttackCurve, kAmpDecay, kAmpDecayCurve, kAmpSustain, kAmpRelease, kAmpReleaseCurve);
            break;
        case kFegAttack: case kFegAttackCurve: case kFegDecay: case kFegDecayCurve:
        case kFegSustain: case kFegRelease: case kFegReleaseCurve:
            setEnv (filterEnv, s, kFegAttack, kFegAttackCurve, kFegDecay, kFegDecayCurve, kFegSustain, kFegRelease, kFegReleaseCurve);
            break;
        case kFegDestination: fegRoute.dest = ModDest (idx (s, index)); break;
        case kFegDepth:       fegRoute.depth = sc::bipolar (s[index]); break;

        case kLfo1Shape: case kLfo1Rate: case kLfo1Amount: case kLfo1Unipolar: case kLfo1Retrigger:
        case kLfo1Phase: case kLfo1FadeIn: case kLfo1Destination: case kLfo1Depth:
            setLfo (lfo[0], s, kLfo1Shape);
            break;
        case kLfo2Shape: case kLfo2Rate: case kLfo2Amount: case kLfo2Unipolar: case kLfo2Retrigger:
        case kLfo2Phase: case kLfo2FadeIn: case kLfo2Destination: case kLfo2Depth:
            setLfo (lfo[1], s, kLfo2Shape);
            break;

        case kDistOn:     fx.distOn = on (s, index); break;
        case kDistDrive:  fx.distDriveDb = sc::toUnits (index, s[index]); break;
        case kDistTone:   fx.distToneHz = sc::cutoffHz (index, s[index]); break;

        case kEqOn:       fx.eqOn = on (s, index); break;
        case kEqLowGain:  fx.eqLowDb = sc::toUnits (index, s[index]); break;
        case kEqMidFreq:  fx.eqMidHz = sc::cutoffHz (index, s[index]); break;
        case kEqMidGain:  fx.eqMidDb = sc::toUnits (index, s[index]); break;
        case kEqHighGain: fx.eqHighDb = sc::toUnits (index, s[index]); break;

        case kChorusOn:    fx.chorusOn = on (s, index); break;
        case kChorusRate:  fx.chorusRateHz = sc::hz (index, s[index]); break;
        case kChorusDepth: fx.chorusDepth = s[index]; break;
        case kChorusMix:   fx.chorusMix = s[index]; break;

        case kModfxOn:       fx.modfxOn = on (s, index); break;
        case kModfxType:     fx.modfxType = ModFxType (idx (s, index)); break;
        case kModfxRate:     fx.modfxRateHz = sc::hz (index, s[index]); break;
        case kModfxDepth:    fx.modfxDepth = s[index]; break;
        case kModfxFeedback: fx.modfxFeedback = s[index]; break;
        case kModfxMix:      fx.modfxMix = s[index]; break;

        case kDelayOn:       fx.delayOn = on (s, index); break;
        case kDelayTime:     fx.delaySeconds = sc::seconds (index, s[index]); break;
        case kDelayFeedback: fx.delayFeedback = s[index]; break;
        case kDelayMix:      fx.delayMix = s[index]; break;

        case kReverbOn:      fx.reverbOn = on (s, index); break;
        case kReverbSize:    fx.reverbSize = sc::toUnits (index, s[index]); break;
        case kReverbDecay:   fx.reverbDecay = s[index]; break;
        case kReverbDamping: fx.reverbDamping = s[index]; break;
        case kReverbMix:     fx.reverbMix = s[index]; break;

        case kMasterVolume:  masterGain = dbToGain (sc::toUnits (index, s[index])); break;

        default: break;
    }
}

} // namespace vae::lite
