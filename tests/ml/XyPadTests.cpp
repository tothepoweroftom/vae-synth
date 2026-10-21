#include "experiments/joint_vae/XyPad.h"
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
    }
};
static XyPadTests xyPadTests;
} // namespace vae::experimental
