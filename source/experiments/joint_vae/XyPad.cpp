#include "XyPad.h"
#include <algorithm>
#include <cmath>
#include <regex>

namespace vae::experimental
{
namespace
{
constexpr float range = 5.0f; // the pad spans +-2.5 standard deviations
constexpr double mainDb = 2.0;

struct Eff
{
    const char* key;
    double min;
};
constexpr Eff effMin[] = {
    { "brightness", 0.15 }, { "attack", 0.04 }, { "length", 0.15 }, { "noise", 0.02 }, { "movement", 0.04 }
};

juce::String u8 (const char* s) { return juce::String::fromUTF8 (s); }

double minOf (const juce::String& key)
{
    for (auto& e : effMin)
        if (key == e.key)
            return e.min;
    return 1;
}

juce::String fmtEff (const juce::String& k, double v)
{
    const bool up = v >= 0;
    const auto sign = up ? juce::String ("+") : u8 ("\xe2\x88\x92");
    const auto a = std::abs (v);
    if (k == "brightness")
        return juce::String (up ? "brighter " : "darker ") + sign + juce::String (a, 1) + " oct";
    if (k == "attack")
        return juce::String (up ? "slower" : "faster") + " attack " + sign
               + juce::String ((int) std::floor (a * 1000 + 0.5)) + " ms";
    if (k == "length")
        return juce::String (up ? "longer " : "shorter ") + sign + juce::String (a, 1) + " s";
    if (k == "noise")
        return up ? "noisier" : "more tonal";
    return up ? "more movement" : "more static";
}

std::string replaceFirst (std::string s, const std::string& from, const std::string& to)
{
    if (auto p = s.find (from); p != std::string::npos)
        s.replace (p, from.size(), to);
    return s;
}

// JS `String.replace(str, str)` semantics for the page's Osc/Noise rewrites.
std::string oscName (const std::string& m) { return replaceFirst (replaceFirst (m, "o", "Osc "), "noise", "Noise"); }

juce::String pname (const juce::String& idStr, const std::map<juce::String, juce::String>& group)
{
    if (auto it = group.find (idStr); it != group.end())
        return it->second;
    const std::string id = idStr.toStdString();
    std::smatch m;
    const auto rx = [&] (const char* p) { return std::regex_match (id, m, std::regex (p)); };
    const auto env = [] (const std::string& c) -> std::string
    {
        return c == "a" ? "attack" : c == "d" ? "decay" : c == "s" ? "sustain" : c == "r" ? "release" : "";
    };
    const auto s = [] (const std::string& x) { return juce::String::fromUTF8 (x.c_str()); };
    if (rx (R"(scene/adsr/([01])/([adsr]))"))
        return s ((m[1] == "0" ? "Amp" : "Filter") + std::string (" env ") + env (m[2]));
    if (rx (R"(scene/adsr/([01])/(\w)_s)"))
        return s ((m[1] == "0" ? "Amp" : "Filter") + std::string (" env ")
                  + (env (m[2]).empty() ? m[2].str() : env (m[2])) + " shape");
    if (rx (R"(scene/filterunit/(\d)/(\w+))"))
        return s ("Filter " + std::to_string (std::stoi (m[1]) + 1) + " "
                  + (m[2] == "envmod"  ? "env amount"
                     : m[2] == "model" ? "type"
                                       : m[2].str()));
    if (rx (R"(scene/osc/(\d)/type)"))
        return s ("Osc " + std::to_string (std::stoi (m[1]) + 1) + " type");
    if (std::regex_search (id, m, std::regex (R"(^scene/osc/(\d)/(\w+)/p(\d+))")))
        return s ("Osc " + std::to_string (std::stoi (m[1]) + 1) + " " + m[2].str() + " param "
                  + std::to_string (std::stoi (m[3]) + 1));
    if (std::regex_search (id, m, std::regex (R"(^scene/osc/(\d)/(\w+))")))
        return s ("Osc " + std::to_string (std::stoi (m[1]) + 1) + " " + m[2].str());
    if (std::regex_search (id, m, std::regex (R"(^scene/lfo/(\d+)/(\w+))")))
        return s ("LFO " + std::to_string (std::stoi (m[1]) + 1) + " " + m[2].str());
    if (rx (R"(scene/level_(\w+))"))
        return s (oscName (m[1]) + " level");
    if (rx (R"(scene/mute_(\w+))"))
        return s (oscName (m[1]) + " mute");
    if (rx (R"(fx/(\w+)/present)"))
        return s (m[1].str() + " on");
    if (rx (R"(fx/(\w+)/(placement|position))"))
        return s (m[1].str() + " " + m[2].str());
    if (std::regex_search (id, m, std::regex (R"(^fx/(\w+)/p(\d+))")))
        return s (m[1].str() + " param " + std::to_string (std::stoi (m[2]) + 1));
    if (rx (R"(route/(\w+)/(.+))"))
    {
        auto a = m[1].str();
        std::replace (a.begin(), a.end(), '_', ' ');
        return s ("mod " + a + " \xe2\x86\x92 " + replaceFirst (m[2], "scene/", ""));
    }
    return s (replaceFirst (id, "scene/", ""));
}

bool readLatent (const juce::var& v, JointLatent& z)
{
    auto* a = v.getArray();
    if (a == nullptr || a->size() != (int) z.size())
        return false;
    for (size_t i = 0; i < z.size(); ++i)
    {
        if (! (*a)[(int) i].isDouble() && ! (*a)[(int) i].isInt())
            return false;
        z[i] = (float) (double) (*a)[(int) i];
    }
    return true;
}

float half (juce::uint16 h)
{
    const float s = (h & 0x8000) ? -1.0f : 1.0f;
    const int e = (h >> 10) & 0x1f, f = h & 0x3ff;
    if (e == 0)
        return s * (float) f * 5.960464477539063e-8f;
    if (e == 31)
        return f ? std::nanf ("") : s * INFINITY;
    return s * (1.0f + (float) f / 1024.0f) * std::ldexp (1.0f, e - 15);
}

float dot (const JointLatent& a, const JointLatent& b)
{
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i)
        s += (double) a[i] * b[i];
    return (float) s;
}
} // namespace

const LatentAxis* XyPadData::find (const juce::String& id) const
{
    for (auto& a : axes)
        if (a.id == id)
            return &a;
    return nullptr;
}

juce::Result XyPadData::load (const juce::File& file, XyPadData& out)
{
    const auto root = juce::JSON::parse (file);
    if (! root.isObject())
        return juce::Result::fail ("xypad.json: cannot read " + file.getFullPathName());
    const auto bad = [] (const char* what)
    { return juce::Result::fail (juce::String ("xypad.json: bad or missing ") + what); };
    XyPadData d;

    std::map<juce::String, juce::String> group;
    if (auto* gs = root["knobs"]["groups"].getArray())
        for (auto& g : *gs)
            if (auto* ks = g["knobs"].getArray())
                for (auto& k : *ks)
                    group[k.toString()] = g["name"].toString();
    if (group.empty())
        return bad ("knobs.groups");

    auto* dirs = root["directions"].getArray();
    auto* dims = root["dims"].getArray();
    if (dirs == nullptr || dims == nullptr || dirs->isEmpty() || dims->isEmpty())
        return bad ("directions/dims");
    for (auto& v : *dirs)
    {
        LatentAxis a;
        const auto key = v["key"].toString();
        if (key.isEmpty() || ! readLatent (v["u"], a.u) || ! v["mean"].isDouble() || ! (double (v["sd"]) > 0)
            || ! v["effect"].isDouble() || ! v["r2"].isDouble())
            return bad ("directions entry");
        const double effect = v["effect"];
        a.id = "s:" + key;
        a.kind = "sound";
        a.key = key;
        a.name = v["name"].toString();
        a.low = v["low"].toString();
        a.high = v["high"].toString();
        a.mean = (float) (double) v["mean"];
        a.sd = (float) (double) v["sd"];
        a.desc = "Follows the measured " + key + " of the training sounds (R" + u8 ("\xc2\xb2") + " "
                 + juce::String ((double) v["r2"], 2) + " from z). Two standard deviations along it: "
                 + (std::abs (effect) >= minOf (key) ? fmtEff (key, effect) : juce::String ("a small change"))
                 + " in the decoded sound.";
        d.axes.push_back (std::move (a));
    }
    for (auto& v : *dims)
    {
        LatentAxis a;
        const int dim = v["d"];
        auto* eff = v["effect"].getDynamicObject();
        auto* sec = v["sections"].getDynamicObject();
        auto* top = v["top"].getArray();
        if (dim < 0 || dim >= 32 || eff == nullptr || sec == nullptr || top == nullptr || ! (double (v["sd"]) > 0)
            || ! v["mean"].isDouble() || ! v["impact"].isDouble())
            return bad ("dims entry");
        for (auto& p : sec->getProperties())
            a.sections.emplace_back (p.name.toString(), (float) (double) p.value);
        std::stable_sort (a.sections.begin(), a.sections.end(), [] (auto& x, auto& y) { return x.second > y.second; });
        if (a.sections.size() < 2)
            return bad ("dims.sections");
        std::vector<std::pair<juce::String, double>> effs;
        for (auto& p : eff->getProperties())
            if (const auto k = p.name.toString(); std::abs ((double) p.value) >= minOf (k))
                effs.emplace_back (k, (double) p.value);
        std::stable_sort (effs.begin(), effs.end(), [] (auto& x, auto& y)
                          { return std::abs (x.second) / minOf (x.first) > std::abs (y.second) / minOf (y.first); });
        const bool named = a.sections[0].second > 0.17f;
        const auto name = named ? a.sections[0].first
                                      + (a.sections[1].second > 0.15f ? " + " + a.sections[1].first.toLowerCase() : "")
                                : juce::String ("Mixed");
        juce::StringArray up, feat;
        for (size_t i = 0; i < effs.size() && i < 3; ++i)
            up.add (fmtEff (effs[i].first, effs[i].second));
        for (int i = 0; i < top->size() && i < 3; ++i)
            feat.add (pname ((*top)[i].toString(), group));
        a.u = {};
        a.u[(size_t) dim] = 1;
        a.id = "z:" + juce::String (dim);
        a.kind = "dim";
        a.name = "z" + juce::String (dim) + " " + u8 ("\xc2\xb7") + " " + name;
        a.low = u8 ("\xe2\x88\x92");
        a.high = "+";
        a.mean = (float) (double) v["mean"];
        a.sd = (float) (double) v["sd"];
        a.impact = (float) (double) v["impact"];
        a.minor = (double) v["impact"] < mainDb;
        a.desc = "Up: " + (up.isEmpty() ? juce::String ("small change in the sound") : up.joinIntoString (", "))
                 + ". Most specific changes: " + feat.joinIntoString (", ") + ".";
        d.axes.push_back (std::move (a));
    }

    if (auto* mz = root["mean_z"].getDynamicObject())
        for (auto& p : mz->getProperties())
            if (! readLatent (p.value, d.typical[p.name.toString()]))
                return bad ("mean_z");
    if (d.typical.empty())
        return bad ("mean_z");

    const auto& pre = root["presets"];
    auto* labels = pre["label"].getArray();
    auto* names = pre["name"].getArray();
    juce::MemoryOutputStream bytes;
    if (labels == nullptr || names == nullptr || labels->size() != names->size()
        || ! juce::Base64::convertFromBase64 (bytes, pre["z"].toString())
        || bytes.getDataSize() != (size_t) labels->size() * 32 * 2)
        return bad ("presets");
    const auto* raw = static_cast<const juce::uint8*> (bytes.getData());
    d.presetZ.resize ((size_t) labels->size());
    for (size_t i = 0; i < d.presetZ.size(); ++i)
    {
        for (size_t k = 0; k < 32; ++k)
        {
            const auto* p = raw + (i * 32 + k) * 2;
            d.presetZ[i][k] = half ((juce::uint16) (p[0] | (p[1] << 8)));
        }
        d.presetLabel.push_back ((int) (*labels)[(int) i]);
        d.presetName.add ((*names)[(int) i].toString());
    }

    auto* bands = root["band_hz"].getArray();
    if (bands == nullptr || bands->isEmpty() || ! root["dt"].isDouble() || ! root["f0"].isDouble()
        || ! root["hold_frames"].isInt())
        return bad ("band_hz/dt/f0/hold_frames");
    for (auto& b : *bands)
        d.bandHz.push_back ((float) (double) b);
    d.dt = (float) (double) root["dt"];
    d.f0 = (float) (double) root["f0"];
    d.holdFrames = root["hold_frames"];

    out = std::move (d);
    return juce::Result::ok();
}

float axisTarget (const LatentAxis& a, float pos) { return a.mean + (pos - 0.5f) * range * a.sd; }
float axisPosition (const LatentAxis& a, const JointLatent& z)
{
    return (dot (a.u, z) - a.mean) / (range * a.sd) + 0.5f;
}

JointLatent constrain (const JointLatent& z0, std::span<const LatentAxis* const> axes, std::span<const float> targets)
{
    JointLatent z = z0;
    const size_t n = std::min (axes.size(), targets.size());
    if (n == 0)
        return z;
    std::vector<std::vector<double>> g (n, std::vector<double> (n));
    std::vector<double> r (n), a (n, 0.0);
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < n; ++j)
            g[i][j] = (double) dot (axes[i]->u, axes[j]->u) + (i == j ? 1e-6 : 0);
        r[i] = (double) targets[i] - dot (axes[i]->u, z0);
    }
    std::vector<bool> drop (n, false);
    for (size_t c = 0; c < n; ++c)
    {
        size_t p = c;
        for (size_t k = c + 1; k < n; ++k)
            if (std::abs (g[k][c]) > std::abs (g[p][c]))
                p = k;
        std::swap (g[c], g[p]);
        std::swap (r[c], r[p]);
        if (std::abs (g[c][c]) < 1e-9)
        {
            drop[c] = true;
            continue;
        }
        for (size_t k = c + 1; k < n; ++k)
        {
            const double f = g[k][c] / g[c][c];
            for (size_t j = c; j < n; ++j)
                g[k][j] -= f * g[c][j];
            r[k] -= f * r[c];
        }
    }
    for (size_t c = n; c-- > 0;)
    {
        if (drop[c])
            continue;
        double s = r[c];
        for (size_t j = c + 1; j < n; ++j)
            s -= g[c][j] * a[j];
        a[c] = s / g[c][c];
    }
    // Rows were swapped, but a[] indexes columns, which are never permuted.
    for (size_t i = 0; i < n; ++i)
        for (size_t k = 0; k < z.size(); ++k)
            z[k] += (float) (a[i] * axes[i]->u[k]);
    return z;
}
} // namespace vae::experimental
