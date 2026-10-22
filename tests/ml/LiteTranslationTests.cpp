#include "experiments/joint_vae/LiteTranslation.h"
#include <cmath>

namespace vae::experimental
{
class LiteTranslationTests final : public juce::UnitTest
{
public:
    LiteTranslationTests()
        : juce::UnitTest ("Lite translation", "vae_ml")
    {
    }
    void runTest() override
    {
        beginTest ("data without a patch fails and leaves the output alone");
        auto out = schema::Snapshot::defaults();
        out[0] = 0.25f;
        const auto before = out;
        expect (liteControls (juce::MemoryBlock ("no xml here", 11), out).failed());
        const juce::String broken ("<?xml version=\"1.0\"?><patch><nothing/></patch>");
        expect (liteControls (juce::MemoryBlock (broken.toRawUTF8(), broken.getNumBytesAsUTF8()), out).failed());
        expect (out.values == before.values);

#if VAE_WITH_SURGE
        // Fixture: in VAE_SURGE_DATA_DIR, `python3 tools/surge_preset_to_vector.py patches_factory -o full.jsonl`
        // (relative paths), then every other line (awk 'NR%2==1') to stay under 1 MB: 321 of 641 presets.
        beginTest ("factory parity");
        const juce::File data (VAE_SURGE_DATA_DIR);
        juce::StringArray lines;
        juce::File (VAE_ML_FIXTURES).getChildFile ("surge-factory-vectors.jsonl").readLines (lines);
        lines.removeEmptyStrings();
        expectGreaterThan (lines.size(), 300);
        int checked = 0;
        double worst = 0;
        for (const auto& line : lines)
        {
            const auto row = juce::JSON::parse (line);
            const auto file = row["file"].toString();
            juce::MemoryBlock fxp;
            expect (data.getChildFile (file).loadFileAsData (fxp), file);
            schema::Snapshot lite;
            const auto r = liteControls (fxp, lite);
            expect (r.wasOk(), file + ": " + r.getErrorMessage());
            const auto* values = row["values"].getArray();
            expect (values != nullptr && values->size() == schema::kControlCount, file);
            if (r.failed() || values == nullptr || values->size() != schema::kControlCount)
                continue;
            for (int i = 0; i < schema::kControlCount; ++i)
            {
                const double e = std::abs ((double) lite[i] - (double) (*values)[i]);
                worst = juce::jmax (worst, e);
                if (e > 1e-5)
                    expect (false, file + ": " + schema::control (i).id + " " + juce::String (lite[i]) + " vs "
                                       + (*values)[i].toString());
            }
            ++checked;
        }
        logMessage ("factory parity: " + juce::String (checked) + " presets, max error " + juce::String (worst, 9));
#endif
    }
};
static LiteTranslationTests liteTranslationTests;
} // namespace vae::experimental
