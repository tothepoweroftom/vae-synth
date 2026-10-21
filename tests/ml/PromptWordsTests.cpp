#include "experiments/joint_vae/PromptWords.h"

namespace vae::experimental
{
class PromptWordsTests final : public juce::UnitTest
{
public:
    PromptWordsTests()
        : juce::UnitTest ("Prompt words", "vae_ml")
    {
    }

    static XyPadData makeData()
    {
        XyPadData d;
        const char* ids[] = { "s:brightness", "s:length", "s:attack", "s:noise", "s:movement" };
        const char* names[] = { "Dark ↔ Bright", "Short ↔ Long", "Snappy ↔ Slow", "Tonal ↔ Noisy", "Static ↔ Moving" };
        for (int i = 0; i < 5; ++i)
        {
            LatentAxis a;
            a.id = ids[i];
            a.name = juce::String (juce::CharPointer_UTF8 (names[i]));
            d.axes.push_back (a);
        }
        for (int c = 0; c < 9; ++c)
        {
            JointLatent z {};
            z[0] = (float) c;
            d.typical[jointLabels[c]] = z;
        }
        JointLatent a {}, b {};
        a[1] = 2;
        b[1] = 4;
        d.presetZ = { a, b, JointLatent {} };
        d.presetLabel = { 8, 8, 0 };
        return d;
    }

    void runTest() override
    {
        const auto d = makeData();

        beginTest ("noun picks a start, adjective nudges");
        auto r = interpretPrompt ("dark reese bass", false, d);
        expect (r.start.has_value() && r.start->second == 0 && r.start->first == d.typical.at ("Bass"));
        expectEquals ((int) r.nudges.size(), 1);
        expectEquals (r.nudges[0].first, juce::String ("s:brightness"));
        expectWithinAbsoluteError (r.nudges[0].second, -0.1f, 1e-6f);
        expect (r.message.startsWith ("Bass"));
        expectEquals (r.ignored.size(), 0);

        beginTest ("multiplier without a noun when a start exists");
        r = interpretPrompt ("a lot brighter", true, d);
        expect (! r.start.has_value());
        expectEquals ((int) r.nudges.size(), 1);
        expectWithinAbsoluteError (r.nudges[0].second, 0.2f, 1e-6f);
        r = interpretPrompt ("slightly less bright", true, d);
        expectWithinAbsoluteError (r.nudges[0].second, -0.05f, 1e-6f);

        beginTest ("new noun re-starts");
        r = interpretPrompt ("now make it a pad", true, d);
        expect (r.start.has_value() && r.start->first == d.typical.at ("Pad"));

        beginTest ("no noun, no start: Unknown mean");
        r = interpretPrompt ("something dark", false, d);
        expect (r.start.has_value() && r.start->second == 8);
        expectWithinAbsoluteError (r.start->first[1], 3.0f, 1e-6f);
        expect (r.message.contains ("no sound type found"));

        beginTest ("nothing to change");
        for (auto* t : { "", "zorp blarg", "the" })
            for (bool hasStart : { true, false })
            {
                r = interpretPrompt (t, hasStart, d);
                expect (! r.start.has_value() && r.nudges.empty());
                expect (r.message.startsWith ("Nothing to change"));
            }
        expectEquals (interpretPrompt ("zorp blarg", false, d).ignored.joinIntoString (","),
                      juce::String ("zorp,blarg"));

        beginTest ("half step message and filler words");
        expect (interpretPrompt ("slightly less bright", true, d)
                    .message.contains (juce::String (juce::CharPointer_UTF8 ("\xe2\x88\x92")) + "0.5 steps"));
        expectEquals (interpretPrompt ("now make it a pad", true, d).ignored.size(), 0);

        beginTest ("closed word list: noisier is ignored");
        expectEquals (interpretPrompt ("noisier", true, d).ignored.joinIntoString (","), juce::String ("noisier"));

        beginTest ("multi-word noun words are not ignored; leftovers are reported");
        r = interpretPrompt ("synth strings gritty zorp", true, d);
        expect (r.nudges.size() == 1 && r.message.contains ("(small effect)"));
        expectEquals (r.ignored.joinIntoString (","), juce::String ("zorp"));
        expect (r.message.contains ("didn't understand 'zorp'"));
    }
};
static PromptWordsTests promptWordsTests;
} // namespace vae::experimental
