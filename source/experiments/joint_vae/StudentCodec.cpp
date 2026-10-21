#include "StudentCodec.h"
#include <cmath>
#include <limits>

namespace vae::experimental
{
namespace
{
// pitcher/core/schema.py and codec.py
const juce::StringArray oscTypes { "classic", "sine",   "wavetable", "shnoise", "audioinput", "fm3",
                                   "fm2",     "window", "modern",    "string",  "twist",      "alias" };
const juce::StringArray fxTypes { "off",         "delay",      "reverb",      "phaser",     "rotary",  "distortion",
                                  "eq",          "freqshift",  "conditioner", "chorus",     "vocoder", "reverb2",
                                  "flanger",     "ringmod",    "airwindows",  "neuron",     "geq11",   "resonator",
                                  "chow",        "exciter",    "ensemble",    "combulator", "nimbus",  "tape",
                                  "treemonster", "waveshaper", "mstool",      "spring",     "bonsai",  "audioinput",
                                  "floatydelay", "convolution" };
const juce::StringArray placements { "insert", "send", "global" };
constexpr int slots[3][4] = { { 0, 1, 8, 9 }, { 4, 5, 12, 13 }, { 6, 7, 14, 15 } };
constexpr const char* fallbackWavetable = "Sine Octaves";

juce::String flagMethod (const juce::String& suffix)
{
    if (suffix == "off")
        return "Deactivated";
    if (suffix == "sync")
        return "TempoSync";
    if (suffix == "ext")
        return "Extend";
    if (suffix == "abs")
        return "Absolute";
    if (suffix == "deform")
        return "Deform";
    return {};
}

int sendIndex (int slot)
{
    switch (slot)
    {
        case 4:  return 0;
        case 5:  return 1;
        case 12: return 2;
        case 13: return 3;
        default: return -1;
    }
}

int optionIndex (float c) { return (int) std::lround (c); }
} // namespace

juce::Result StudentCodec::load (const juce::File& dir, StudentCodec& output)
{
    juce::var schema, pruning;
    if (auto r = juce::JSON::parse (dir.getChildFile ("schema.json").loadFileAsString(), schema); r.failed())
        return juce::Result::fail ("schema.json: " + r.getErrorMessage());
    if (auto r = juce::JSON::parse (dir.getChildFile ("pruning.json").loadFileAsString(), pruning); r.failed())
        return juce::Result::fail ("pruning.json: " + r.getErrorMessage());
    auto* list = schema["features"].getArray();
    auto* frozen = pruning["frozen"].getDynamicObject();
    if (list == nullptr || frozen == nullptr)
        return juce::Result::fail ("Bundle lacks the student's schema.json / pruning.json");

    StudentCodec next;
    for (const auto& f : *list)
    {
        Feature feature { f["name"].toString(), f["kind"].toString(), {}, -1, {}, {} };
        if (auto* options = f["options"].getArray())
            feature.options = *options;
        if (const auto& route = f["route"]; route.isObject() && ! route["source_id"].isVoid())
        {
            feature.routeSource = (int) route["source_id"];
            feature.routeDest = route["dest"].toString();
            feature.routeOscType = route["osc_type"].isString() ? route["osc_type"].toString() : juce::String();
        }
        next.index[feature.name] = (int) next.features.size();
        next.features.push_back (std::move (feature));
    }
    next.frozen.assign (next.features.size(), std::numeric_limits<float>::quiet_NaN());
    for (const auto& [name, how] : frozen->getProperties())
    {
        const auto it = next.index.find (name.toString());
        if (it == next.index.end())
            continue;
        if (how["how"] == "mode" && ! how["code"].isVoid())
            next.frozen[(size_t) it->second] = (float) (double) how["code"];
        else if (how["how"] == "absent" && name.toString().endsWith ("/present"))
            next.frozen[(size_t) it->second] = 0.0f;
    }
    output = std::move (next);
    return juce::Result::ok();
}

juce::Result StudentCodec::plan (const juce::StringArray& names, std::span<const float> codes,
                                 StudentPlan& output) const
{
    if (features.empty() || (size_t) names.size() != codes.size())
        return juce::Result::fail ("Codec not loaded, or codes do not match the feature names");
    auto row = frozen;
    for (int i = 0; i < names.size(); ++i)
    {
        const auto it = index.find (names[i]);
        if (it == index.end())
            return juce::Result::fail ("Decoder feature missing from schema.json: " + names[i]);
        row[(size_t) it->second] = codes[(size_t) i];
    }
    auto code = [&] (const juce::String& name)
    {
        const auto it = index.find (name);
        return it == index.end() ? std::numeric_limits<float>::quiet_NaN() : row[(size_t) it->second];
    };
    auto opt = [&] (const juce::String& name) -> juce::var
    {
        const float c = code (name);
        if (std::isnan (c))
            return {};
        const auto& options = features[(size_t) index.at (name)].options;
        const int k = optionIndex (c);
        return k >= 0 && k < options.size() ? options[k] : juce::var();
    };

    StudentPlan plan;
    plan.types = { { "scenemode", 0 }, { "scene_active", 0 }, { "fx_disable", 0 } };

    // FX slot assignment
    std::map<juce::String, int> fxSlot;
    std::array<std::vector<int>, 3> used;
    for (int t = 0; t < fxTypes.size(); ++t)
    {
        const auto& name = fxTypes[t];
        if (code ("fx/" + name + "/present") != 1.0f)
            continue;
        const auto where = opt ("fx/" + name + "/placement");
        const int place = juce::jmax (0, placements.indexOf (where.isVoid() ? "insert" : where.toString()));
        std::vector<int> free;
        for (int s : slots[place])
            if (std::find (used[(size_t) place].begin(), used[(size_t) place].end(), s) == used[(size_t) place].end())
                free.push_back (s);
        if (free.empty())
            continue;
        const auto pos = opt ("fx/" + name + "/position");
        const int wanted = pos.isVoid() ? -1 : slots[place][juce::jlimit (0, 3, (int) pos)];
        const int slot = std::find (free.begin(), free.end(), wanted) != free.end() ? wanted : free[0];
        used[(size_t) place].push_back (slot);
        fxSlot[name] = slot;
        plan.types.push_back ({ "fx/" + juce::String (slot) + "/type", t });
    }
    for (int s = 0; s < 16; ++s)
        if (std::none_of (fxSlot.begin(), fxSlot.end(), [s] (const auto& kv) { return kv.second == s; }))
            plan.types.push_back ({ "fx/" + juce::String (s) + "/type", 0 });

    std::map<int, int> oscType;
    for (int o = 0; o < 3; ++o)
        if (const auto t = opt ("scene/osc/" + juce::String (o) + "/type"); ! t.isVoid())
            oscType[o] = (int) t;
    auto oscIs = [&] (int o, const juce::String& type)
    { return oscType.count (o) != 0 && oscTypes[oscType.at (o)] == type; };

    // codec._path_for
    auto pathFor = [&] (const juce::String& name) -> juce::String
    {
        const auto part = juce::StringArray::fromTokens (name, "/", "");
        if (name == "character")
            return name;
        if (part.size() == 4 && part[0] == "scene" && part[1] == "filterunit" && part[3] == "model")
            return "scene/0/filterunit/" + part[2] + "/type";
        if (part.size() == 3 && part[0] == "fx")
        {
            const auto it = fxSlot.find (part[1]);
            const bool knob =
                part[2].startsWith ("p") && part[2].substring (1).containsOnly ("0123456789") && part[2].length() > 1;
            if (! knob && part[2] != "send_level" && part[2] != "return_level")
                return {};
            if (it == fxSlot.end())
                return {};
            if (knob)
                return "fx/" + juce::String (it->second) + "/p/" + part[2].substring (1);
            if (sendIndex (it->second) < 0)
                return {};
            return part[2] == "send_level" ? "scene/0/send_level/" + juce::String (sendIndex (it->second))
                                           : "fx/" + juce::String (it->second) + "/return_level";
        }
        if (part.size() == 5 && part[0] == "scene" && part[1] == "osc" && part[4].length() == 2 && part[4][0] == 'p'
            && juce::CharacterFunctions::isDigit (part[4][1]))
            return oscIs (part[2].getIntValue(), part[3]) ? "scene/0/osc/" + part[2] + "/p/" + part[4].substring (1)
                                                          : juce::String();
        if (name.startsWith ("scene/"))
            return "scene/0/" + name.substring (6);
        return {};
    };

    for (size_t i = 0; i < features.size(); ++i)
    {
        const float c = row[i];
        const auto& f = features[i];
        if (std::isnan (c))
            continue;
        const auto& name = f.name;
        const int dot = name.lastIndexOfChar ('.');
        if (dot > 0)
            if (const auto method = flagMethod (name.substring (dot + 1)); method.isNotEmpty())
            {
                if (const auto path = pathFor (name.substring (0, dot)); path.isNotEmpty())
                    plan.flags.push_back (
                        { path, method, f.kind == "bool" ? (c >= 0.5f ? 1 : 0) : (int) f.options[optionIndex (c)] });
                continue;
            }
        if (name.startsWith ("route/"))
        {
            if (f.routeSource < 0)
                continue;
            if (f.routeOscType.isNotEmpty()
                && ! oscIs (juce::StringArray::fromTokens (f.routeDest, "/", "")[1].getIntValue(), f.routeOscType))
                continue;
            plan.routings.push_back ({ f.routeSource, "scene/0/" + f.routeDest, 2.0f * c - 1.0f });
            continue;
        }
        const auto part = juce::StringArray::fromTokens (name, "/", "");
        if (part.size() == 4 && part[0] == "scene" && part[1] == "filterunit" && part[3] == "model")
        {
            const auto model = f.options[optionIndex (c)].toString();
            plan.types.push_back ({ "scene/0/filterunit/" + part[2] + "/type",
                                    model.upToFirstOccurrenceOf (":", false, false).getIntValue() });
            plan.subtypes.push_back ({ "scene/0/filterunit/" + part[2] + "/subtype",
                                       model.fromFirstOccurrenceOf (":", false, false).getIntValue() });
            continue;
        }
        if (part.size() == 5 && part[0] == "scene" && part[1] == "osc" && part[4] == "wt")
        {
            const int o = part[2].getIntValue();
            if (oscIs (o, part[3]))
            {
                const auto table = f.options[optionIndex (c)].toString();
                plan.wavetables.push_back ({ o, table != "other" ? table : juce::String (fallbackWavetable) });
            }
            continue;
        }
        if (part.size() == 3 && part[0] == "fx"
            && (part[2] == "present" || part[2] == "placement" || part[2] == "position"))
            continue;

        const auto path = pathFor (name);
        if (path.isEmpty())
            continue;
        const int natural = f.kind == "cat" ? (int) f.options[optionIndex (c)] : 0;
        const auto field = path.fromLastOccurrenceOf ("/", false, false);
        if (f.kind == "cat"
            && (field == "type" || field == "shape" || field == "fm_switch" || field == "filterblock_configuration"))
            plan.types.push_back ({ path, natural });
        else
            plan.values.push_back ({ path, f.kind, c, natural });
    }
    for (const auto& [o, t] : oscType)
        if ((t == 2 || t == 7)
            && std::none_of (plan.wavetables.begin(), plan.wavetables.end(),
                             [o] (const auto& w) { return w.osc == o; }))
            plan.wavetables.push_back ({ o, fallbackWavetable });
    output = std::move (plan);
    return juce::Result::ok();
}
} // namespace vae::experimental
