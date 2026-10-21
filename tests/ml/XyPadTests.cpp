#include "experiments/joint_vae/JointVae.h"
#include <cmath>

namespace vae::experimental
{
class XyPadTests final : public juce::UnitTest
{
public:
    XyPadTests()
        : juce::UnitTest ("XY pad maths", "vae_ml")
    {
    }

    static LatentAxis axis (JointLatent u, float mean, float sd)
    {
        LatentAxis a;
        a.u = u;
        a.mean = mean;
        a.sd = sd;
        return a;
    }

    void runTest() override
    {
        beginTest ("constrain hits targets and keeps the rest");
        JointLatent z0, ua {}, ub {};
        for (size_t i = 0; i < 32; ++i)
        {
            z0[i] = std::sin ((float) i);
            ua[i] = std::cos ((float) i * 0.7f) / 4;
            ub[i] = i == 5 ? 1.0f : 0.0f;
        }
        auto a = axis (ua, 0.2f, 1.3f), b = axis (ub, -0.1f, 0.9f);
        const LatentAxis* axes[] = { &a, &b };
        const float pos[] = { 0.8f, 0.25f };
        const float ts[] = { axisTarget (a, pos[0]), axisTarget (b, pos[1]) };
        const auto z = constrain (z0, axes, ts);
        expectWithinAbsoluteError (axisPosition (a, z), pos[0], 1e-4f);
        expectWithinAbsoluteError (axisPosition (b, z), pos[1], 1e-4f);
        // Residual of (z - z0) after projecting onto span{ua, ub} (Gram-Schmidt).
        std::array<double, 32> d {}, q1 {}, q2 {};
        double n1 = 0;
        for (size_t i = 0; i < 32; ++i)
        {
            d[i] = z[i] - z0[i];
            q1[i] = ua[i];
            n1 += q1[i] * q1[i];
        }
        double pb = 0, n2 = 0;
        for (size_t i = 0; i < 32; ++i)
            pb += ub[i] * q1[i] / n1 * 1.0;
        for (size_t i = 0; i < 32; ++i)
        {
            q2[i] = ub[i] - pb * q1[i];
            n2 += q2[i] * q2[i];
        }
        double c1 = 0, c2 = 0;
        for (size_t i = 0; i < 32; ++i)
        {
            c1 += d[i] * q1[i] / n1;
            c2 += d[i] * q2[i] / n2;
        }
        double res = 0;
        for (size_t i = 0; i < 32; ++i)
            res = std::max (res, std::abs (d[i] - c1 * q1[i] - c2 * q2[i]));
        expect (res < 1e-5, "residual " + juce::String (res));

        beginTest ("degenerate axes stay finite");
        const LatentAxis* twice[] = { &a, &a };
        const float different[] = { 0.1f, 0.9f };
        const auto zd = constrain (z0, twice, different);
        for (auto v : zd)
            expect (std::isfinite (v));
        auto near = a;
        near.u[0] += 1e-4f;
        const LatentAxis* close[] = { &a, &near };
        for (auto v : constrain (z0, close, different))
            expect (std::isfinite (v));

        beginTest ("bad file leaves output unchanged");
        XyPadData out;
        out.dt = 7;
        const auto tmp = juce::File::createTempFile (".json");
        tmp.replaceWithText ("{\"directions\":[],\"dims\":[]}");
        expect (XyPadData::load (tmp, out).failed());
        expect (XyPadData::load (tmp.getSiblingFile ("nope.json"), out).failed());
        expectEquals (out.dt, 7.0f);
        expect (out.axes.empty());
        tmp.deleteFile();

        beginTest ("pca npz accepts float32 and float64 scale");
        for (const bool f8 : { false, true })
        {
            const auto npy = [] (const juce::String& shape, const std::vector<double>& v, bool wide)
            {
                juce::String h = "{'descr': '" + juce::String (wide ? "<f8" : "<f4")
                                 + "', 'fortran_order': False, 'shape': " + shape + ", }";
                while ((10 + h.length() + 1) % 64 != 0)
                    h += " ";
                h += "\n";
                juce::MemoryOutputStream o;
                o.write ("\x93NUMPY\x01\x00", 8);
                o.writeShort ((short) h.length());
                o.write (h.toRawUTF8(), (size_t) h.length());
                for (auto x : v)
                    wide ? o.writeDouble (x) : o.writeFloat ((float) x);
                return o.getMemoryBlock();
            };
            juce::ZipFile::Builder zb;
            const auto comps = npy ("(48, 64)", std::vector<double> (48 * 64, 0.0), false);
            const auto mean = npy ("(64,)", std::vector<double> (64, -3.0), false);
            const auto scale = npy ("()", { 2.5 }, f8);
            // addEntry takes ownership of the streams
            zb.addEntry (new juce::MemoryInputStream (comps, true), 0, "components.npy", juce::Time::getCurrentTime());
            zb.addEntry (new juce::MemoryInputStream (mean, true), 0, "mean.npy", juce::Time::getCurrentTime());
            zb.addEntry (new juce::MemoryInputStream (scale, true), 0, "scale.npy", juce::Time::getCurrentTime());
            const auto f = juce::File::createTempFile (".npz");
            {
                juce::FileOutputStream fo (f);
                expect (zb.writeToStream (fo, nullptr));
            }
            SoundPca sp;
            const auto sr = loadPca (f, sp);
            expect (sr.wasOk(), sr.getErrorMessage());
            expectEquals (sp.scale, 2.5f);
            expectEquals (sp.frames, 1);
            f.deleteFile();
        }

        const auto path = juce::SystemStats::getEnvironmentVariable ("VAE_JOINT_BUNDLE", "");
        if (path.isEmpty())
        {
            logMessage ("Set VAE_JOINT_BUNDLE to run XY pad bundle tests.");
            return;
        }
        const auto dir = juce::File (path).getParentDirectory();
        XyPadData data;
        beginTest ("load");
        const auto r = XyPadData::load (dir.getChildFile ("xypad.json"), data);
        expect (r.wasOk(), r.getErrorMessage());
        if (r.failed())
            return;
        int sound = 0;
        for (auto& ax : data.axes)
            sound += ax.kind == "sound";
        expectEquals (sound, 5);
        expectEquals ((int) data.axes.size() - sound, 31);
        expectEquals (data.axes[0].id, juce::String ("s:brightness"));
        expectEquals ((int) data.presetZ.size(), 1375);
        expectEquals ((int) data.typical.size(), 8);
        logMessage (data.axes[5].name + " | " + data.axes[5].desc);
        logMessage (data.axes[0].desc);

        beginTest ("constrain parity");
        const auto parity = juce::JSON::parse (dir.getChildFile ("xypad-parity.json"));
        auto* cases = parity["constrain"].getArray();
        expect (cases != nullptr && ! cases->isEmpty());
        double worst = 0;
        if (cases != nullptr)
            for (auto& c : *cases)
            {
                JointLatent z00 {};
                std::vector<const LatentAxis*> ax;
                std::vector<float> t;
                for (int i = 0; i < 32; ++i)
                    z00[(size_t) i] = (float) (double) c["z0"][i];
                for (int i = 0; i < c["axes"].size(); ++i)
                {
                    ax.push_back (data.find (c["axes"][i].toString()));
                    t.push_back ((float) (double) c["targets"][i]);
                    expect (ax.back() != nullptr);
                }
                if (std::find (ax.begin(), ax.end(), nullptr) != ax.end())
                    continue;
                const auto got = constrain (z00, ax, t);
                for (int i = 0; i < 32; ++i)
                    worst = std::max (worst, std::abs ((double) got[(size_t) i] - (double) c["z"][i]));
            }
        logMessage ("max constrain parity error " + juce::String (worst, 8));
        expect (worst < 2e-4);

        const auto none = [] (const juce::var& v) { return v.isVoid() || v.isUndefined(); };
        const auto knobs = juce::JSON::parse (dir.getChildFile ("xypad.json"))["knobs"];

        beginTest ("measures parity");
        SoundPca pca;
        const auto pr = loadPca (dir.getChildFile ("sound_pca.npz"), pca);
        expect (pr.wasOk(), pr.getErrorMessage());
        expectEquals (pca.frames, 139);
        auto* ms = parity["measures"].getArray();
        expect (ms != nullptr && ! ms->isEmpty());
        double worstM = 0;
        if (pr.wasOk() && ms != nullptr)
            for (auto& m : *ms)
            {
                SoundCode s {};
                for (int i = 0; i < 48; ++i)
                    s[(size_t) i] = (float) (double) m["sound"][i];
                const auto mel = melFromSound (pca, s);
                expectEquals ((int) mel.size(), 139 * 64);
                expectEquals (*std::max_element (mel.begin(), mel.end()), 0.0f);
                const auto got = measures (data, mel);
                const float g[] = { got.brightness, got.attack, got.length, got.noise, got.movement };
                const char* keys[] = { "brightness", "attack", "length", "noise", "movement" };
                for (int i = 0; i < 5; ++i)
                    worstM = std::max (worstM, std::abs ((double) g[i] - (double) m["measures"][keys[i]]));
            }
        logMessage ("max measures parity error " + juce::String (worstM, 8));
        expect (worstM < 1e-3);

        beginTest ("sound decoder parity");
        JointVae vae;
        const auto lr = vae.load (juce::File (path));
        expect (lr.wasOk(), lr.getErrorMessage());
        double worstS = 0;
        if (lr.wasOk() && ms != nullptr)
            for (auto& m : *ms)
            {
                JointLatent lat {};
                for (int i = 0; i < 32; ++i)
                    lat[(size_t) i] = (float) (double) m["latent"][i];
                SoundCode snd {};
                expect (vae.decodeSound (lat, (int) m["category"], snd).wasOk());
                for (int i = 0; i < 48; ++i)
                    worstS = std::max (worstS, std::abs ((double) snd[(size_t) i] - (double) m["sound"][i]));
            }
        logMessage ("max sound decoder parity error " + juce::String (worstS, 8));
        expect (worstS < 1e-4);

        beginTest ("bundle without sound_decoder fails and keeps the model");
        auto manifest = juce::JSON::parse (juce::File (path));
        manifest.getDynamicObject()->removeProperty ("sound_decoder");
        const auto old = dir.getChildFile ("bundle-no-sound-decoder.tmp.json");
        old.replaceWithText (juce::JSON::toString (manifest));
        const auto nr = vae.load (old);
        old.deleteFile();
        expect (nr.failed() && nr.getErrorMessage().contains ("sound_decoder"), nr.getErrorMessage());
        SoundCode kept {};
        expect (vae.decodeSound ({}, 0, kept).wasOk());

        beginTest ("forest parity");
        CategoryForest forest;
        const auto fr = CategoryForest::load (knobs["forest"], forest);
        expect (fr.wasOk(), fr.getErrorMessage());
        double worstF = 0;
        int nulls = 0;
        if (fr.wasOk())
            for (auto& c : *parity["forest"].getArray())
            {
                std::vector<float> k (123);
                for (int i = 0; i < 123; ++i)
                {
                    nulls += none (c["knobs"][i]);
                    k[(size_t) i] = none (c["knobs"][i]) ? std::nanf ("") : (float) (double) c["knobs"][i];
                }
                const auto p = forest.predict (k);
                for (int i = 0; i < 8; ++i)
                    worstF = std::max (worstF, std::abs ((double) p[(size_t) i] - (double) c["p"][i]));
            }
        logMessage ("max forest parity error " + juce::String (worstF, 9));
        expect (nulls > 0);
        expect (worstF < 1e-6);

        beginTest ("group parity");
        KnobGroups kg;
        const auto gr = KnobGroups::load (knobs, kg);
        expect (gr.wasOk(), gr.getErrorMessage());
        double worstG = 0;
        int nanCases = 0;
        if (gr.wasOk())
            for (auto& c : *parity["groups"].getArray())
            {
                const int g = kg.find (c["group"].toString());
                expect (g >= 0);
                std::vector<float> k (123);
                for (int i = 0; i < 123; ++i)
                    k[(size_t) i] = none (c["vals"][i]) ? std::nanf ("") : (float) (double) c["vals"][i];
                const float at = kg.position (g, k);
                if (none (c["pos"]))
                {
                    expect (std::isnan (at));
                    ++nanCases;
                    continue;
                }
                worstG = std::max (worstG, std::abs ((double) at - (double) c["pos"]));
                kg.set (g, 0.3f, k);
                if (c["group"].toString()
                    != "EQ output") // its only knob has p5 == p95, so the position is not recoverable
                    expectWithinAbsoluteError (kg.position (g, k), 0.3f, 1e-6f);
            }
        logMessage ("max group parity error " + juce::String (worstG, 9));
        expect (worstG < 1e-6);
        expect (nanCases > 0);
    }
};
static XyPadTests xyPadTests;
} // namespace vae::experimental
