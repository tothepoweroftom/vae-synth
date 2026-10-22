#include "LiteTranslation.h"
#include "schema/SurgeCurves.h"
#include <cmath>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>

namespace vae::experimental
{
namespace
{
// Python ints and floats throughout; doubles here, floats only in the returned Snapshot.
// The `lossy` notes of the Python script are not ported: only the values are returned.

const char* const oscTypes[] = { "classic", "sine",   "wavetable", "sh_noise", "audio_in", "fm3",
                                 "fm2",     "window", "modern",    "string",   "twist",    "alias" };
constexpr int msFilterEg = 16, msLfo1 = 17, msLfo2 = 18;
constexpr int fxDelay = 1, fxReverb1 = 2, fxPhaser = 3, fxDist = 5, fxEq = 6, fxChorus = 9, fxReverb2 = 11,
              fxFlanger = 12;
constexpr int fxSlotsA[] = { 1, 2, 9, 10 }, fxSlotsB[] = { 3, 4, 11, 12 }, sendSlots[] = { 5, 6, 13, 14 },
              globalSlots[] = { 7, 8, 15, 16 };

const char* filterFamily (int t)
{
    switch (t)
    {
        case 0:  return "off";
        case 1:
        case 11:
        case 34: return "lp12";
        case 2:
        case 3:
        case 10:
        case 12:
        case 13:
        case 15:
        case 16:
        case 28:
        case 33:
        case 35: return "lp24";
        case 4:
        case 20: return "hp12";
        case 5:
        case 14:
        case 17:
        case 29: return "hp24";
        case 6:
        case 19:
        case 22:
        case 23:
        case 31: return "bp";
        case 7:
        case 18:
        case 21:
        case 24:
        case 30: return "notch";
        default: return "off";
    }
}
const char* const lfoShapes[] = { "sine", "triangle", "square", "saw", "smooth_random", "random" };
const char* const playModes[] = { "poly", "mono", "legato", "mono", "legato", "poly" };
struct Destination
{
    const char *dest, *suffix, *curve;
};
const Destination destinations[] = {
    { "pitch", "pitch", "pitch7" },
    { "filter1_cutoff", "filter1_cutoff", "cutoff" },
    { "filter2_cutoff", "filter2_cutoff", "cutoff" },
    { "filter1_resonance", "filter1_resonance", "unit" },
    { "volume", "volume", "amplitude" },
    { "pan", "pan", "bipolar" },
};

/** Parameter values and modulation routes from a Surge patch XML. */
struct Patch
{
    std::map<juce::String, double> values;
    std::map<juce::String, std::vector<std::pair<int, double>>> routes;

    static juce::Result parse (const juce::String& xml, Patch& out)
    {
        juce::XmlDocument doc (xml);
        const auto root = doc.getDocumentElement();
        if (root == nullptr)
            return juce::Result::fail ("patch XML: " + doc.getLastParseError());
        const auto* params = root->getChildByName ("parameters");
        if (params == nullptr)
            return juce::Result::fail ("no <parameters> in patch");
        // Surge inserted a modulation source after ms_ctrl7 (13) at patch
        // revision 9 and renumbers older patches on load (SurgePatch.cpp).
        const int revision = root->getIntAttribute ("revision", 999);
        for (const auto* p : params->getChildIterator())
        {
            if (p->hasAttribute ("value"))
                out.values[p->getTagName()] = p->getDoubleAttribute ("value");
            for (const auto* r : p->getChildWithTagNameIterator ("modrouting"))
            {
                if (r->getStringAttribute ("muted", "0") == "1")
                    continue;
                int source = r->getIntAttribute ("source");
                if (revision < 9 && source > 13)
                    source += 1;
                out.routes[p->getTagName()].emplace_back (source, r->getDoubleAttribute ("depth"));
            }
        }
        return juce::Result::ok();
    }

    double get (const juce::String& name, double fallback = 0.0) const
    {
        const auto it = values.find (name);
        return it != values.end() ? it->second : fallback;
    }
    const std::vector<std::pair<int, double>>& routesOf (const juce::String& name) const
    {
        static const std::vector<std::pair<int, double>> none;
        const auto it = routes.find (name);
        return it != routes.end() ? it->second : none;
    }
};

int indexOf (const juce::String& id)
{
    if (const auto i = schema::indexOf (id.toStdString()))
        return *i;
    throw std::invalid_argument (("no schema control " + id).toStdString()); // Python: KeyError
}

/** Maps one Surge patch onto the schema vector, one section at a time (Python Converter). */
struct Converter
{
    const Patch& patch;
    std::array<double, schema::kControlCount> values {};
    juce::String scene;

    explicit Converter (const Patch& p)
        : patch (p)
    {
        for (int i = 0; i < schema::kControlCount; ++i)
            values[(size_t) i] = schema::control (i).defaultValue;
        const int scenemode = (int) patch.get ("scenemode");
        const int active = (int) patch.get ("scene_active");
        scene = scenemode == 0 && active == 1 ? "b" : "a";
    }

    juce::String p (const juce::String& name) const { return scene + "_" + name; }
    double get (const juce::String& name, double fallback = 0.0) const { return patch.get (p (name), fallback); }
    double& at (const juce::String& id) { return values[(size_t) indexOf (id)]; }

    static double sanitise (int i, double v)
    {
        const auto& c = schema::control (i);
        if (! std::isfinite (v))
            return c.defaultValue;
        switch (c.kind)
        {
            case schema::Kind::Continuous:  return juce::jlimit (0.0, 1.0, v);
            case schema::Kind::Toggle:      return v >= 0.5 ? 1.0 : 0.0;
            case schema::Kind::Categorical: return juce::jlimit (0.0, double (c.numOptions - 1), std::floor (v + 0.5));
        }
        return v;
    }
    void put (const juce::String& id, double v)
    {
        const int i = indexOf (id);
        values[(size_t) i] = sanitise (i, v);
    }
    void putOption (const juce::String& id, const char* option)
    {
        const auto o = schema::optionIndex (indexOf (id), option);
        if (! o)
            throw std::invalid_argument (("no option " + juce::String (option) + " in " + id).toStdString());
        put (id, *o);
    }
    void putUnits (const juce::String& id, double units)
    {
        const auto* c = schema::curveOf (indexOf (id));
        if (c == nullptr)
            throw std::invalid_argument (("no curve for " + id).toStdString());
        put (id, juce::jlimit (0.0, 1.0, (units - c->min) / (double (c->max) - c->min)));
    }

    void run()
    {
        voice();
        oscillators();
        filters();
        envelopes();
        lfos();
        routes();
        fx();
        putUnits ("master_volume", patch.get ("volume", 0.0));
    }

    /** A 0..1 level plus its strongest positive modulation: presets that sit at 0
        and are opened by an LFO, aftertouch or controller would otherwise be silent. */
    double modulated (const juce::String& name, double fallback) const
    {
        const double value = get (name, fallback);
        std::optional<double> best;
        for (const auto& [src, d] : patch.routesOf (p (name)))
            if (d > 0 && (! best || d > *best))
                best = d;
        return best ? juce::jmin (1.0, value + *best) : value;
    }

    void voice()
    {
        const int mode = (int) get ("polymode");
        putOption ("play_mode", mode >= 0 && mode < 6 ? playModes[mode] : "poly");
        putUnits ("glide_time", get ("portamento", -8.0));
    }

    void oscillators()
    {
        for (int n = 1; n <= 3; ++n)
        {
            const juce::String o = "osc" + juce::String (n), N (n);
            put (o + "_on", get ("mute_o" + N) >= 0.5 ? 0.0 : 1.0);
            put (o + "_level", modulated ("level_o" + N, 1.0));
            put (o + "_octave", (int) get ("osc" + N + "_octave") + 3);
            putUnits (o + "_pitch", get ("osc" + N + "_pitch"));
            put (o + "_filter", (int) get ("route_o" + N, 1));
            putOption (o + "_wave", wave (n));
        }

        const double noise = get ("mute_noise", 1.0) < 0.5 ? modulated ("level_noise", 0.0) : 0.0;
        if (noise > 0.0)
            for (int n = 1; n <= 3; ++n)
            {
                const juce::String o = "osc" + juce::String (n);
                if (at (o + "_on") >= 0.5)
                    continue;
                // The schema has no noise channel; a muted osc slot plays it instead.
                put (o + "_on", 1.0);
                put (o + "_level", noise);
                put (o + "_octave", 3);
                putUnits (o + "_pitch", 0.0);
                put (o + "_filter", (int) get ("route_noise", 1));
                putOption (o + "_wave", "noise");
                break;
            }
    }

    /** Schema wave for Surge osc n (may also switch the osc off). */
    const char* wave (int n)
    {
        const juce::String o = "osc" + juce::String (n), N (n);
        const int t = (int) get ("osc" + N + "_type");
        const juce::String name = t >= 0 && t < 12 ? oscTypes[t] : "classic";
        const double shape = get ("osc" + N + "_param0");
        if (name == "classic")
            return shape <= -0.5 ? "square" : "saw";
        if (name == "sine")
            return "sine";
        if (name == "modern")
        {
            // max() keeps the first of equal entries: saw, square, triangle
            const double saw = std::abs (shape), square = std::abs (get ("osc" + N + "_param1")),
                         triangle = std::abs (get ("osc" + N + "_param2"));
            if (square > saw && square >= triangle)
                return "square";
            if (triangle > saw && triangle > square)
                return "triangle";
            return "saw";
        }
        if (name == "fm2" || name == "fm3")
            return "fm";
        if (name == "sh_noise")
            return "noise";
        if (name == "audio_in")
        {
            put (o + "_on", 0.0);
            return "sine";
        }
        // wavetable, window, string, twist, alias
        return name == "wavetable" || name == "window" ? "wavetable" : "saw";
    }

    void filters()
    {
        const int config = (int) get ("fb_config", 7);
        putOption ("filter_routing", config == 0 || config == 1 || config == 2 ? "serial" : "parallel");
        const double f1Cutoff = get ("filter1_cutoff", 3.0);
        for (int n = 1; n <= 2; ++n)
        {
            const juce::String f = "filter" + juce::String (n);
            putOption (f + "_type", filterFamily ((int) get (f + "_type")));
            double cutoff = get (f + "_cutoff", 3.0);
            if (n == 2 && get ("f2_cf_is_offset") >= 0.5)
                cutoff += f1Cutoff;
            putUnits (f + "_cutoff", cutoff);
            put (f + "_resonance", get (f + "_resonance"));
            putUnits (f + "_env", get (f + "_envmod"));
            putUnits (f + "_keytrack", get (f + "_keytrack"));
        }
    }

    void envelopes()
    {
        for (const auto& [prefix, env] : { std::pair { "amp", "env1" }, std::pair { "feg", "env2" } })
        {
            const juce::String pre (prefix), e (env);
            for (const auto* stage : { "attack", "decay", "release" })
            {
                putUnits (pre + "_" + stage, get (e + "_" + stage, -8.0));
                put (pre + "_" + stage + "_curve", (int) get (e + "_" + stage + "_shape", 1));
            }
            put (pre + "_sustain", modulated (e + "_sustain", 1.0));
        }
    }

    void lfos()
    {
        for (int n = 1; n <= 2; ++n)
        {
            const juce::String l = "lfo" + juce::String (n), lfo = "lfo" + juce::String (n - 1);
            const int shape = (int) get (lfo + "_shape");
            putOption (l + "_shape", shape >= 0 && shape < 6 ? lfoShapes[shape] : "random");
            putUnits (l + "_rate", get (lfo + "_rate"));
            put (l + "_amount", get (lfo + "_magnitude", 1.0));
            put (l + "_unipolar", get (lfo + "_unipolar"));
            put (l + "_retrigger", (int) get (lfo + "_trigmode", 1) == 1 ? 1.0 : 0.0);
            put (l + "_phase", get (lfo + "_phase"));
            putUnits (l + "_fade_in", get (lfo + "_attack", -8.0));
        }
    }

    /** PROVISIONAL routing: the strongest supported route per source. */
    void routes()
    {
        for (const auto& [source, prefix] :
             { std::pair { msFilterEg, "feg" }, std::pair { msLfo1, "lfo1" }, std::pair { msLfo2, "lfo2" } })
        {
            std::optional<std::pair<const char*, double>> best;
            for (const auto& d : destinations)
            {
                const auto* curve = schema::surge::curveNamed (d.curve);
                const double span = double (curve->max) - curve->min;
                for (const auto& [src, depth] : patch.routesOf (p (d.suffix)))
                    if (src == source && (! best || std::abs (depth / span) > std::abs (best->second)))
                        best = std::pair { d.dest, depth / span };
            }
            if (best)
            {
                putOption (juce::String (prefix) + "_destination", best->first);
                put (juce::String (prefix) + "_depth", (juce::jlimit (-1.0, 1.0, best->second) + 1.0) / 2.0);
            }
        }
    }

    /** First audible effect of each type along the chain (scene inserts,
        sends, globals). fx_disable has one bit per slot (bit slot-1). */
    void fx()
    {
        const int disabled = (int) patch.get ("fx_disable");
        std::set<int> seen;
        std::vector<int> chain (std::begin (fxSlotsA), std::end (fxSlotsA));
        if (scene == "b")
            chain.assign (std::begin (fxSlotsB), std::end (fxSlotsB));
        chain.insert (chain.end(), std::begin (sendSlots), std::end (sendSlots));
        chain.insert (chain.end(), std::begin (globalSlots), std::end (globalSlots));
        for (const int slot : chain)
        {
            const int t = (int) patch.get ("fx" + juce::String (slot) + "_type");
            if (t == 0 || (disabled >> (slot - 1)) & 1)
                continue;
            const double wet = sendWetness (slot);
            if (wet <= 0.0)
                continue; // nothing is sent to it: inaudible, so treat as off
            const int family = t == fxReverb1 ? fxReverb2 : t == fxFlanger ? fxPhaser : t;
            if (seen.count (family) != 0)
                continue;
            const auto mixed = fxUnit (slot, t, wet);
            if (! mixed)
                continue;
            if (mixed->isNotEmpty() && at (*mixed + "_mix") <= 0.0)
            {
                // Mix 0 is inaudible (Surge loads it that way too): label it off so
                // "on, silent" and "off" don't become two labels for one sound.
                const juce::String section (schema::control (indexOf (*mixed + "_on")).section);
                for (int i = 0; i < schema::kControlCount; ++i)
                    if (section == schema::control (i).section)
                        values[(size_t) i] = schema::control (i).defaultValue;
                continue;
            }
            seen.insert (family);
        }
    }

    /** How much of the signal reaches an effect slot: 1 for inserts and
        globals; for a send, the scene send level times the send's return level,
        both applied by Surge as amplitude^3 (send N = slot 5, 6, 13, 14). */
    double sendWetness (int slot) const
    {
        const int n = slot == 5 ? 1 : slot == 6 ? 2 : slot == 13 ? 3 : slot == 14 ? 4 : 0;
        if (n == 0)
            return 1.0;
        return std::pow (get ("send_fx_" + juce::String (n), 0.0), 3.0)
               * std::pow (patch.get ("volume_FX" + juce::String (n), 1.0), 3.0);
    }

    /** Writes one effect's controls. Returns the schema prefix whose mix
        decides audibility ("" when the effect has no mix), or nullopt when the
        type is not in the schema. */
    std::optional<juce::String> fxUnit (int slot, int t, double wet)
    {
        auto fx = [&] (int k) { return patch.get ("fx" + juce::String (slot) + "_p" + juce::String (k)); };
        if (t == fxDist)
        {
            put ("dist_on", 1);
            putUnits ("dist_drive", fx (4));
            putUnits ("dist_tone", fx (9));
            return juce::String();
        }
        if (t == fxEq)
        {
            put ("eq_on", 1);
            putUnits ("eq_low_gain", fx (0));
            putUnits ("eq_mid_freq", fx (4));
            putUnits ("eq_mid_gain", fx (3));
            putUnits ("eq_high_gain", fx (6));
            return juce::String();
        }
        if (t == fxChorus)
        {
            put ("chorus_on", 1);
            putUnits ("chorus_rate", fx (1));
            put ("chorus_depth", fx (2));
            put ("chorus_mix", fx (6) * wet);
            return juce::String ("chorus");
        }
        if (t == fxPhaser)
        {
            put ("modfx_on", 1);
            putOption ("modfx_type", "phaser");
            putUnits ("modfx_rate", fx (3));
            put ("modfx_depth", fx (4));
            put ("modfx_feedback", juce::jmax (0.0, fx (1)));
            put ("modfx_mix", fx (6) * wet);
            return juce::String ("modfx");
        }
        if (t == fxFlanger)
        {
            put ("modfx_on", 1);
            putOption ("modfx_type", "flanger");
            putUnits ("modfx_rate", fx (2));
            put ("modfx_depth", fx (3));
            put ("modfx_feedback", fx (7));
            put ("modfx_mix", juce::jmax (0.0, fx (10)) * wet);
            return juce::String ("modfx");
        }
        if (t == fxDelay)
        {
            put ("delay_on", 1);
            putUnits ("delay_time", fx (0));
            put ("delay_feedback", fx (2));
            put ("delay_mix", fx (10) * wet);
            return juce::String ("delay");
        }
        if (t == fxReverb2)
        {
            put ("reverb_on", 1);
            put ("reverb_size", (fx (1) + 1.0) / 2.0);
            putUnits ("reverb_decay", fx (2));
            put ("reverb_damping", fx (7));
            put ("reverb_mix", fx (9) * wet);
            return juce::String ("reverb");
        }
        if (t == fxReverb1)
        {
            put ("reverb_on", 1);
            put ("reverb_size", fx (2));
            putUnits ("reverb_decay", fx (3));
            put ("reverb_damping", fx (4));
            put ("reverb_mix", fx (9) * wet);
            return juce::String ("reverb");
        }
        return std::nullopt;
    }
};
} // namespace

juce::Result liteControls (const juce::MemoryBlock& data, schema::Snapshot& out)
{
    const std::string_view bytes (static_cast<const char*> (data.getData()), data.getSize());
    const auto start = bytes.find ("<?xml");
    const auto end = bytes.find ("</patch>");
    if (start == std::string_view::npos || end == std::string_view::npos || end < start)
        return juce::Result::fail ("no patch XML found");
    std::string text (bytes.substr (start, end + 8 - start));
    // Python decodes with errors="replace"; non-ASCII only occurs in names, never in values.
    for (auto& c : text)
        if ((unsigned char) c >= 0x80)
            c = '?';
    Patch patch;
    if (auto r = Patch::parse (juce::String (text), patch); r.failed())
        return r;
    try
    {
        Converter converter (patch);
        converter.run();
        for (int i = 0; i < schema::kControlCount; ++i)
            out[i] = (float) converter.values[(size_t) i];
    }
    catch (const std::exception& e) // a schema mismatch: Python skips the preset
    {
        return juce::Result::fail (e.what());
    }
    return juce::Result::ok();
}
} // namespace vae::experimental
