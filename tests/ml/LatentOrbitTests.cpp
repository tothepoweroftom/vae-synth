#include "experiments/joint_vae/LatentOrbit.h"
#include <cmath>
#include <juce_core/juce_core.h>

namespace vae::experimental
{
class LatentOrbitTests final : public juce::UnitTest
{
public:
    LatentOrbitTests()
        : juce::UnitTest ("Latent orbit LFO", "vae_ml")
    {
    }

    void runTest() override
    {
        const std::array<float, 3> centre { 0.4f, 0.6f, 0.5f };
        OrbitSettings s;
        s.on = true;
        s.rateHz = 0.25f;
        s.depth = 0.1f;

        beginTest ("t = 0 and one period: the centre plus (depth, 0)");
        for (const double t : { 0.0, 1.0 / s.rateHz })
            for (const bool hasZ : { false, true })
            {
                const auto p = orbitPosition (centre, s, t, hasZ);
                expectWithinAbsoluteError (p[0], centre[0] + s.depth, 1e-5f);
                expectWithinAbsoluteError (p[1], centre[1], 1e-5f);
                if (! hasZ || t == 0.0)
                    expectWithinAbsoluteError (p[2], centre[2], 1e-5f);
            }
        expectWithinAbsoluteError (orbitPosition (centre, s, 1.0, false)[1], centre[1] + s.depth, 1e-5f,
                                   "a quarter period is the top of the circle");

        beginTest ("stays within depth of the centre and inside 0..1");
        float zMove = 0;
        for (const auto c : { centre, std::array<float, 3> { 0.02f, 0.99f, 0.0f } })
            for (int k = 0; k < 400; ++k)
            {
                const auto p = orbitPosition (c, s, k * 0.037, true);
                float d2 = 0;
                for (size_t i = 0; i < 3; ++i)
                {
                    expect (p[i] >= 0.0f && p[i] <= 1.0f);
                    d2 += (p[i] - c[i]) * (p[i] - c[i]);
                }
                expect (std::sqrt (d2) <= s.depth * 1.12f + 1e-5f); // the Z wobble is half the depth
                zMove = std::max (zMove, std::abs (p[2] - c[2]));
            }
        expect (zMove > 0.03f, "Z wobbles when Z is assigned");

        beginTest ("discrete controls hold the centre's values, continuous ones move");
        auto moving = schema::Snapshot::defaults(), centreLite = schema::Snapshot::defaults();
        int discrete = 0;
        for (int i = 0; i < schema::kControlCount; ++i)
        {
            moving[i] = schema::control (i).kind == schema::Kind::Continuous ? 0.25f : 1.0f;
            centreLite[i] = 0.0f;
        }
        holdDiscrete (moving, centreLite);
        for (int i = 0; i < schema::kControlCount; ++i)
        {
            const bool continuous = schema::control (i).kind == schema::Kind::Continuous;
            discrete += ! continuous;
            expectEquals (moving[i], continuous ? 0.25f : 0.0f, schema::control (i).id);
        }
        expect (discrete > 10);
    }
};
static LatentOrbitTests latentOrbitTests;
} // namespace vae::experimental
