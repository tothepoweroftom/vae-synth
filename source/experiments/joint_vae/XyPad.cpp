#include "XyPad.h"
#include <algorithm>
#include <cmath>
#include <cstring>
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

bool isNum (const juce::var& v) { return v.isDouble() || v.isInt() || v.isInt64(); }

bool readLatent (const juce::var& v, JointLatent& z)
{
    auto* a = v.getArray();
    if (a == nullptr || a->size() != (int) z.size())
        return false;
    for (size_t i = 0; i < z.size(); ++i)
    {
        if (! isNum ((*a)[(int) i]))
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
        if (key.isEmpty() || ! readLatent (v["u"], a.u) || ! isNum (v["mean"]) || ! (double (v["sd"]) > 0)
            || ! isNum (v["effect"]) || ! isNum (v["r2"]))
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
            || ! isNum (v["mean"]) || ! isNum (v["impact"]))
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
    if (bands == nullptr || bands->isEmpty() || ! isNum (root["dt"]) || ! isNum (root["f0"])
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
    for (size_t k = 0; k < z.size(); ++k)
    {
        double s = z0[k]; // the ~1e5 terms cancel: sum in double, round once
        for (size_t i = 0; i < n; ++i)
            s += a[i] * axes[i]->u[k];
        z[k] = (float) s;
    }
    return z;
}

namespace
{
// Minimal .npy reader: v1/v2 header, '<f4' or '<f8', C order. Returns values as floats plus the first double.
juce::Result readNpy (juce::ZipFile& zip, const char* name, std::vector<float>& out, const juce::String& shape,
                      double* scalar = nullptr)
{
    const auto fail = [&] (const char* why)
    { return juce::Result::fail (juce::String ("sound_pca.npz ") + name + ": " + why); };
    const int idx = zip.getIndexOfFileName (name);
    if (idx < 0)
        return fail ("missing");
    std::unique_ptr<juce::InputStream> in (zip.createStreamForEntry (idx));
    juce::MemoryBlock mb;
    if (in == nullptr || in->readIntoMemoryBlock (mb) == 0)
        return fail ("unreadable");
    const auto* p = static_cast<const juce::uint8*> (mb.getData());
    const size_t n = mb.getSize();
    if (n < 12 || std::memcmp (p, "\x93NUMPY", 6) != 0 || p[6] > 2)
        return fail ("not an npy file");
    const size_t lenBytes = p[6] == 1 ? 2 : 4;
    size_t hl = (size_t) p[8] | ((size_t) p[9] << 8);
    if (lenBytes == 4)
        hl |= ((size_t) p[10] << 16) | ((size_t) p[11] << 24);
    const size_t start = 8 + lenBytes + hl;
    if (start > n)
        return fail ("truncated header");
    const juce::String head (juce::CharPointer_UTF8 ((const char*) p + 8 + lenBytes),
                             juce::CharPointer_UTF8 ((const char*) p + start));
    const bool f8 = head.contains ("<f8"), f4 = head.contains ("<f4");
    if ((! f4 && ! f8) || ! head.contains ("'fortran_order': False"))
        return fail ("expected little-endian float, C order");
    if (! head.contains ("'shape': " + shape))
        return fail ("unexpected shape");
    const size_t w = f8 ? 8 : 4;
    if ((n - start) % w != 0)
        return fail ("size mismatch");
    out.resize ((n - start) / w);
    for (size_t i = 0; i < out.size(); ++i)
    {
        if (f8)
        {
            double d;
            std::memcpy (&d, p + start + i * 8, 8);
            out[i] = (float) d;
            if (i == 0 && scalar != nullptr)
                *scalar = d;
        }
        else
            std::memcpy (&out[i], p + start + i * 4, 4);
    }
    if (scalar != nullptr)
    {
        if (out.empty())
            return fail ("empty scalar");
        if (! f8)
            *scalar = out[0];
        if (! std::isfinite (*scalar))
            return fail ("non-finite scalar");
    }
    return juce::Result::ok();
}
} // namespace

juce::Result loadPca (const juce::File& file, SoundPca& out)
{
    juce::ZipFile zip (file);
    SoundPca pca;
    std::vector<float> s;
    double scale = 1;
    if (auto r = readNpy (zip, "mean.npy", pca.mean, "("); r.failed())
        return r;
    const auto n = juce::String (pca.mean.size());
    if (auto r = readNpy (zip, "mean.npy", pca.mean, "(" + n + ",)"); r.failed())
        return r;
    if (auto r = readNpy (zip, "components.npy", pca.components, "(48, " + n + ")"); r.failed())
        return r;
    if (auto r = readNpy (zip, "scale.npy", s, "()", &scale); r.failed())
        return r;
    if (pca.mean.empty() || pca.mean.size() % 64 != 0 || pca.components.size() != 48 * pca.mean.size())
        return juce::Result::fail ("sound_pca.npz: expected components 48 x (frames*64) and matching mean");
    pca.scale = (float) scale;
    pca.frames = (int) pca.mean.size() / 64;
    out = std::move (pca);
    return juce::Result::ok();
}

std::vector<float> melFromSound (const SoundPca& pca, const SoundCode& code)
{
    const size_t F = pca.mean.size();
    if (pca.components.size() != 48 * F)
        return {};
    std::vector<double> x (pca.mean.begin(), pca.mean.end());
    for (size_t k = 0; k < 48; ++k)
    {
        const double a = (double) code[k] * pca.scale;
        for (size_t j = 0; j < F; ++j)
            x[j] += a * pca.components[k * F + j];
    }
    const double mx = *std::max_element (x.begin(), x.end());
    std::vector<float> mel (F);
    for (size_t j = 0; j < F; ++j)
        mel[j] = (float) std::max (x[j] - mx, -80.0);
    return mel;
}

SoundMeasures measures (const XyPadData& d, std::span<const float> mel)
{
    const size_t NM = d.bandHz.size();
    if (NM == 0 || mel.size() < NM || mel.size() % NM != 0)
        return {};
    const size_t Tn = mel.size() / NM;
    std::vector<double> P (mel.size()), e (Tn);
    for (size_t j = 0; j < mel.size(); ++j)
        P[j] = std::pow (10.0, (double) mel[j] / 10);
    double emax = -1e9;
    for (size_t t = 0; t < Tn; ++t)
    {
        double s = 0;
        for (size_t b = 0; b < NM; ++b)
            s += P[t * NM + b];
        e[t] = 10 * std::log10 (s + 1e-12);
        emax = std::max (emax, e[t]);
    }
    std::vector<double> Pb (NM, 0.0);
    int na = 0;
    for (size_t t = 0; t < Tn; ++t)
        if (e[t] >= emax - 30)
        {
            ++na;
            for (size_t b = 0; b < NM; ++b)
                Pb[b] += P[t * NM + b];
        }
    double sp = 0, sf = 0, sl = 0;
    for (size_t b = 0; b < NM; ++b)
    {
        Pb[b] /= std::max (na, 1);
        sp += Pb[b];
        sf += Pb[b] * d.bandHz[b];
        sl += std::log (Pb[b] + 1e-12);
    }
    size_t att = 0;
    while (att < Tn && e[att] < emax - 3)
        ++att;
    size_t last = Tn - 1;
    while (last > 0 && e[last] < emax - 40)
        --last;
    double dsum = 0;
    int nl = 0;
    for (size_t t = 1; t < (size_t) d.holdFrames && t < Tn; ++t)
        for (size_t b = 0; b < NM; ++b)
        {
            const double a = mel[t * NM + b], p = mel[(t - 1) * NM + b];
            if (std::max (a, p) > -50)
            {
                dsum += std::abs (a - p);
                ++nl;
            }
        }
    return { (float) std::log2 (sf / sp / d.f0), (float) ((double) att * d.dt), (float) ((double) (last + 1) * d.dt),
             (float) (std::exp (sl / (double) NM) / (sp / (double) NM)), nl ? (float) (dsum / nl) : 0.0f };
}

juce::Result CategoryForest::load (const juce::var& forest, CategoryForest& out)
{
    auto* trees = forest["trees"].getArray();
    if (trees == nullptr || trees->isEmpty())
        return juce::Result::fail ("xypad.json: bad or missing knobs.forest");
    CategoryForest f;
    for (auto& tv : *trees)
    {
        Tree t;
        auto *fa = tv["f"].getArray(), *ta = tv["t"].getArray(), *la = tv["l"].getArray(), *ra = tv["r"].getArray(),
             *va = tv["v"].getArray();
        const int n = fa != nullptr ? fa->size() : 0;
        if (n == 0 || ta == nullptr || la == nullptr || ra == nullptr || va == nullptr || ta->size() != n
            || la->size() != n || ra->size() != n || va->size() != n)
            return juce::Result::fail ("xypad.json: bad forest tree");
        for (int i = 0; i < n; ++i)
        {
            t.f.push_back ((*fa)[i]);
            t.t.push_back ((float) (double) (*ta)[i]);
            t.l.push_back ((*la)[i]);
            t.r.push_back ((*ra)[i]);
            auto* v = (*va)[i].getArray(); // internal nodes carry a scalar 0, leaves the 8 class fractions
            if (v == nullptr && t.l.back() == -1)
                return juce::Result::fail ("xypad.json: bad forest tree");
            if (v != nullptr && v->size() != 8)
                return juce::Result::fail ("xypad.json: bad forest tree");
            std::array<float, 8> row {};
            for (int c = 0; v != nullptr && c < 8; ++c)
                row[(size_t) c] = (float) (double) (*v)[c];
            t.v.push_back (row);
        }
        for (int i = 0; i < n; ++i) // reject out-of-range links so predict can't walk off the tree
            if (t.l[(size_t) i] != -1
                && (t.l[(size_t) i] < 0 || t.l[(size_t) i] >= n || t.r[(size_t) i] < 0 || t.r[(size_t) i] >= n))
                return juce::Result::fail ("xypad.json: bad forest tree");
        f.trees.push_back (std::move (t));
    }
    out = std::move (f);
    return juce::Result::ok();
}

std::array<float, 8> CategoryForest::predict (std::span<const float> knobs) const
{
    std::array<double, 8> p {};
    for (auto& t : trees)
    {
        size_t n = 0;
        for (size_t guard = 0; guard <= t.l.size() && t.l[n] != -1; ++guard)
        {
            const int f = t.f[n];
            const float x =
                f >= 0 && (size_t) f < knobs.size() && ! std::isnan (knobs[(size_t) f]) ? knobs[(size_t) f] : -1.0f;
            n = (size_t) (x <= t.t[n] ? t.l[n] : t.r[n]);
        }
        for (size_t c = 0; c < 8; ++c)
            p[c] += t.v[n][c];
    }
    std::array<float, 8> r {};
    for (size_t c = 0; c < 8; ++c)
        r[c] = trees.empty() ? 0.0f : (float) (p[c] / (double) trees.size());
    return r;
}

juce::Result KnobGroups::load (const juce::var& knobs, KnobGroups& out)
{
    auto* ks = knobs["knobs"].getArray();
    auto* gs = knobs["groups"].getArray();
    if (ks == nullptr || gs == nullptr || ks->isEmpty() || gs->isEmpty())
        return juce::Result::fail ("xypad.json: bad or missing knobs/groups");
    KnobGroups k;
    std::map<juce::String, int> index;
    for (auto& v : *ks)
    {
        if (! isNum (v["p5"]) || ! isNum (v["p95"]))
            return juce::Result::fail ("xypad.json: bad knob p5/p95");
        index[v["name"].toString()] = (int) k.p5.size();
        k.names.add (v["name"].toString());
        k.p5.push_back ((float) (double) v["p5"]);
        k.p95.push_back ((float) (double) v["p95"]);
        k.ctrltype.add (v["ctrltype"].toString());
        const auto natRange = v["natural_range"];
        const bool ranged = natRange.size() == 2 && isNum (natRange[0]) && isNum (natRange[1]);
        k.natLo.push_back (ranged ? (float) (double) natRange[0] : 0.0f);
        k.natHi.push_back (ranged ? (float) (double) natRange[1] : 1.0f);
    }
    for (auto& v : *gs)
    {
        Group g;
        g.name = v["name"].toString();
        g.section = v["section"].toString();
        g.desc = v["desc"].toString();
        g.importance = (float) (double) v["importance"];
        g.se = (float) (double) v["se"];
        g.active = (float) (double) v["active"];
        if (isNum (v["mean_abs_rho"]))
            g.meanAbsRho = (float) (double) v["mean_abs_rho"];
        auto* names = v["knobs"].getArray();
        if (names == nullptr)
            return juce::Result::fail ("xypad.json: bad group");
        for (auto& n : *names)
        {
            const auto it = index.find (n.toString());
            if (it == index.end() || ! isNum (v["signs"][juce::Identifier (n.toString())]))
                return juce::Result::fail ("xypad.json: bad group knob " + n.toString());
            g.knob.push_back (it->second);
            g.sign.push_back ((double) v["signs"][juce::Identifier (n.toString())] > 0 ? 1.0f : -1.0f);
        }
        k.groups.push_back (std::move (g));
    }
    out = std::move (k);
    return juce::Result::ok();
}

juce::String KnobGroups::format (int knob, float v) const
{
    if (knob < 0 || knob >= ctrltype.size() || (size_t) knob >= natLo.size() || std::isnan (v))
        return "off";
    const auto x = natLo[(size_t) knob] + v * (natHi[(size_t) knob] - natLo[(size_t) knob]);
    const auto& t = ctrltype[knob];
    const auto plus = [] (float y) { return y >= 0 ? juce::String ("+") : juce::String(); };
    if (t.startsWith ("ct_envtime") || t == "ct_portatime")
        return std::exp2 (x) < 1 ? juce::String (juce::roundToInt (std::exp2 (x) * 1000)) + " ms"
                                 : juce::String (std::exp2 (x), 2) + " s";
    if (t == "ct_reverbtime")
        return juce::String (std::exp2 (x), 2) + " s";
    if (t.startsWith ("ct_lforate"))
        return juce::String (std::exp2 (x), 2) + " Hz";
    if (t.startsWith ("ct_freq_audible"))
        return juce::String (juce::roundToInt (440 * std::exp2 (x / 12))) + " Hz";
    if (t.startsWith ("ct_pitch") || t == "ct_freq_mod")
        return plus (x) + juce::String (x, 1) + " st";
    if (t.startsWith ("ct_decibel"))
        return plus (x) + juce::String (x, 1) + " dB";
    if (t.contains ("bipolar"))
        return plus (x) + juce::String (juce::roundToInt (x * 100)) + "%";
    return juce::String (juce::roundToInt (v * 100)) + "%";
}

int KnobGroups::find (const juce::String& name) const
{
    for (size_t i = 0; i < groups.size(); ++i)
        if (groups[i].name == name)
            return (int) i;
    return -1;
}

float KnobGroups::position (int group, std::span<const float> v) const
{
    if (group < 0 || (size_t) group >= groups.size())
        return std::nanf ("");
    double s = 0;
    int n = 0;
    const auto& g = groups[(size_t) group];
    for (size_t j = 0; j < g.knob.size(); ++j)
    {
        const auto i = (size_t) g.knob[j];
        if (i >= v.size() || std::isnan (v[i]))
            continue;
        const double span = (double) p95[i] - p5[i];
        const double u = std::clamp (((double) v[i] - p5[i]) / (std::abs (span) > 0 ? span : 1e-6), 0.0, 1.0);
        s += g.sign[j] > 0 ? u : 1 - u;
        ++n;
    }
    return n ? (float) (s / n) : std::nanf ("");
}

void KnobGroups::set (int group, float pos, std::span<float> v) const
{
    if (group < 0 || (size_t) group >= groups.size())
        return;
    const auto& g = groups[(size_t) group];
    for (size_t j = 0; j < g.knob.size(); ++j)
        if ((size_t) g.knob[j] < v.size())
        {
            const auto i = (size_t) g.knob[j];
            v[i] = p5[i] + (g.sign[j] > 0 ? pos : 1 - pos) * (p95[i] - p5[i]);
        }
}
} // namespace vae::experimental
