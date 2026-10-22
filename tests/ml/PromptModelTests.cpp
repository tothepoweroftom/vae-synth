#include "experiments/joint_vae/PromptModel.h"
#include <cmath>

extern "C"
{
    void* vae_tokenizer_load (const char*);
    void vae_tokenizer_free (void*);
    bool vae_tokenizer_encode (const void*, const char*, float*, float*);
}

namespace vae::experimental
{
class PromptModelTests final : public juce::UnitTest
{
public:
    PromptModelTests()
        : juce::UnitTest ("Prompt model", "vae_ml")
    {
    }

    void runTest() override
    {
        existingTokenizer();
        tieBreaks();
        bundle();
    }

private:
    static double ms (double start) { return juce::Time::getMillisecondCounterHiRes() - start; }

    template <typename T>
    void expectClose (const juce::var& ref, const std::vector<T>& got, double tol, const juce::String& what,
                      double& worstOverall)
    {
        expectEquals ((int) got.size(), ref.size(), what + " size");
        double worst = 0;
        for (int i = 0; i < juce::jmin (ref.size(), (int) got.size()); ++i)
            worst = juce::jmax (worst, std::abs ((double) ref[i] - (double) got[size_t (i)]));
        expect (worst <= tol, what + " max diff " + juce::String (worst));
        worstOverall = juce::jmax (worstOverall, worst);
    }

    struct Bank
    {
        std::span<const float> z;
        std::span<const int> category;
        float minWeight;
        int maxResults;
    };
    /** R15: native rows equal the reference, or, where adjacent reference components a > b are near-tied (< 1e-4),
        equal the selection recomputed from the native weights and mu with b forced ahead of a. */
    static bool rowsAccepted (const std::vector<int>& rows, const std::vector<int>& ref,
                              const std::vector<float>& refWeights, const std::vector<int>& refComponents,
                              const std::vector<float>& weights, const std::vector<float>& mu, const Bank& bank,
                              int category)
    {
        if (rows == ref)
            return true;
        for (size_t i = 0; i + 1 < refComponents.size(); ++i)
        {
            const auto a = size_t (refComponents[i]), b = size_t (refComponents[i + 1]);
            if (refWeights[a] - refWeights[b] >= 1e-4f)
                continue;
            auto flipped = weights;
            const auto lo = juce::jmin (weights[a], weights[b]), hi = juce::jmax (weights[a], weights[b]);
            flipped[b] = std::nextafter (hi, 2.0f);
            flipped[a] = lo;
            if (rows
                == PromptModel::select (flipped, mu, bank.z, bank.category, category, bank.minWeight, bank.maxResults))
                return true;
        }
        return false;
    }

    void existingTokenizer()
    {
        beginTest ("existing tokenizer unchanged");
        const auto path = juce::SystemStats::getEnvironmentVariable ("VAE_T5_TOKENIZER", "");
        if (path.isEmpty())
        {
            logMessage ("Set VAE_T5_TOKENIZER to the Small-SFX t5gemma tokenizer.json to check Prompt-tab tokens.");
            return;
        }
        auto* t = vae_tokenizer_load (path.toRawUTF8());
        expect (t != nullptr);
        const auto fixture = juce::JSON::parse (juce::File (VAE_ML_FIXTURES).getChildFile ("t5-tokens.json"));
        for (const auto& c : *fixture["cases"].getArray())
        {
            std::array<float, 256> ids {}, mask {};
            expect (vae_tokenizer_encode (t, c["text"].toString().toRawUTF8(), ids.data(), mask.data()));
            for (int i = 0; i < 256; ++i)
            {
                expectEquals ((int) ids[size_t (i)], (int) c["ids"][i], c["text"].toString().substring (0, 40));
                expectEquals ((int) mask[size_t (i)], (int) c["mask"][i]);
            }
        }
        vae_tokenizer_free (t);
    }

    void tieBreaks()
    {
        beginTest ("selection tie-breaks");
        // Two components, three rows in category 1 (rows 1, 2, 3); row 0 is category 0 and nearest to both means.
        std::vector<float> z (4 * 32, 0.0f), mu (3 * 32, 0.0f);
        z[1 * 32] = 1; // rows 1 and 2 are equally far from mean 0 (at 0)
        z[2 * 32] = -1;
        z[3 * 32] = 5;
        const std::vector<int> category { 0, 1, 1, 1 };
        mu[2 * 32] = 5; // component 2 sits on row 3
        // Equal weights for 0 and 2: component 0 goes first and takes row 1 (lower index than row 2).
        auto rows = PromptModel::select (std::vector<float> { 0.45f, 0.1f, 0.45f }, mu, z, category, 1, 0.02f, 8);
        expect (rows == std::vector<int> { 1, 3, 2 });
        // Below min weight: dropped. Max results caps the list.
        rows = PromptModel::select (std::vector<float> { 0.45f, 0.01f, 0.54f }, mu, z, category, 1, 0.02f, 8);
        expect (rows == std::vector<int> { 3, 1 });
        rows = PromptModel::select (std::vector<float> { 0.45f, 0.1f, 0.45f }, mu, z, category, 1, 0.02f, 1);
        expect (rows == std::vector<int> { 1 });
        expect (PromptModel::select (std::vector<float> { 1, 0, 0 }, mu, z, category, 2, 0.02f, 8).empty());

        beginTest ("R15 near-tie acceptance");
        // Component 0 sits on row 1, component 1 on row 2: the weight order decides the row order.
        std::vector<float> mu2 (2 * 32, 0.0f);
        mu2[0] = 1;
        mu2[32] = -1;
        const Bank bank { z, category, 0.02f, 8 };
        const std::vector<float> refW { 0.50002f, 0.49998f }, flippedW { 0.49998f, 0.50002f };
        const std::vector<int> comps { 0, 1 };
        const auto ref = PromptModel::select (refW, mu2, z, category, 1, 0.02f, 8);
        const auto flippedRows = PromptModel::select (flippedW, mu2, z, category, 1, 0.02f, 8);
        expect (ref == std::vector<int> { 1, 2 } && flippedRows == std::vector<int> { 2, 1 });
        expect (rowsAccepted (ref, ref, refW, comps, flippedW, mu2, bank, 1), "reference order");
        expect (rowsAccepted (flippedRows, ref, refW, comps, flippedW, mu2, bank, 1), "native flipped near-tie");
        expect (! rowsAccepted ({ 3, 1 }, ref, refW, comps, flippedW, mu2, bank, 1), "different rows");
        const std::vector<float> farW { 0.6f, 0.4f }; // not near-tied: a flip is a real mismatch
        expect (! rowsAccepted (flippedRows, ref, farW, comps, flippedW, mu2, bank, 1), "flip without a near-tie");
    }

    void bundle()
    {
        const auto promptPath = juce::SystemStats::getEnvironmentVariable ("VAE_PROMPT_BUNDLE", "");
        const auto labPath = juce::SystemStats::getEnvironmentVariable ("VAE_JOINT_BUNDLE", "");
        beginTest ("prompt parity");
        if (promptPath.isEmpty() || labPath.isEmpty())
        {
            logMessage ("Set VAE_PROMPT_BUNDLE and VAE_JOINT_BUNDLE to run prompt model parity.");
            return;
        }
        const juce::File promptJson (promptPath);
        JointVae lab;
        expect (lab.load (juce::File (labPath)).wasOk());
        PromptModel model;
        std::vector<PromptCandidate> out;
        int category = -1;
        expect (model.find ("bass", out, category).failed(), "find before load");

        auto start = juce::Time::getMillisecondCounterHiRes();
        const auto loaded = model.load (promptJson, lab);
        logMessage ("cold load ms " + juce::String (ms (start), 1));
        expect (loaded.wasOk(), loaded.getErrorMessage());
        if (loaded.failed())
            return;

        const auto meta = juce::JSON::parse (promptJson);
        const auto parity = juce::JSON::parse (promptJson.getSiblingFile ("prompt-parity.json"));
        expectEquals (parity["bundle_id"].toString(), meta["id"].toString());
        const auto rowsTotal = (int) meta["bank"]["rows"], zDim = 32, numCodes = (int) meta["bank"]["codes"];
        juce::MemoryBlock bank;
        expect (promptJson.getSiblingFile ("bank.bin").loadFileAsData (bank));
        const std::span<const float> bankZ ((const float*) bank.getData(), size_t (rowsTotal * zDim));
        const auto bankJson = juce::JSON::parse (promptJson.getSiblingFile ("bank.json"));
        std::vector<int> bankCategory;
        for (const auto& c : *bankJson["category"].getArray())
            bankCategory.push_back ((int) c);
        const Bank bankRef { bankZ, bankCategory, (float) (double) meta["selection"]["min_weight"],
                             (int) meta["selection"]["max_results"] };

        double worstEmb = 0, worstLogits = 0, worstWeights = 0, worstMu = 0;
        bool first = true;
        for (const auto& c : *parity["cases"].getArray())
        {
            const auto name = c["name"].toString();
            if (! c["forced_category"].isVoid() && ! c["forced_category"].isUndefined())
                continue; // keys_forced: the no-bank test below uses an empty category
            PromptModel::Trace trace;
            start = juce::Time::getMillisecondCounterHiRes();
            const auto r = model.find (c["text"].toString(), out, category, -1, &trace);
            logMessage (name + (first ? " cold find ms " : " warm find ms ") + juce::String (ms (start), 1));
            first = false;
            expect (r.wasOk(), name + ": " + r.getErrorMessage());

            expectEquals ((int) trace.ids.size(), c["ids"].size(), name + " tokens");
            for (int i = 0; i < juce::jmin (c["ids"].size(), (int) trace.ids.size()); ++i)
            {
                expect (trace.ids[size_t (i)] == (int64_t) c["ids"][i], name + " id " + juce::String (i));
                expect (trace.mask[size_t (i)] == (int64_t) c["mask"][i], name + " mask " + juce::String (i));
            }
            expectClose (c["embedding"], trace.embedding, 1e-4, name + " embedding", worstEmb);
            expectClose (c["category_logits"], trace.logits, 1e-4, name + " logits", worstLogits);
            expectClose (c["mix_weights"], trace.weights, 1e-4, name + " weights", worstWeights);
            juce::Array<juce::var> mu;
            for (const auto& m : *c["mu"].getArray())
                mu.addArray (*m.getArray());
            expectClose (juce::var (mu), trace.mu, 1e-4, name + " mu", worstMu);
            expectEquals (category, (int) c["category"], name + " category");

            std::vector<int> rows;
            for (const auto& p : out)
                rows.push_back (p.row);
            std::vector<int> ref;
            for (const auto& row : *c["rows"].getArray())
                ref.push_back ((int) row);
            std::vector<float> refWeights;
            for (const auto& w : *c["mix_weights"].getArray())
                refWeights.push_back ((float) (double) w);
            std::vector<int> refComponents;
            for (const auto& k : *c["components"].getArray())
                refComponents.push_back ((int) k);
            const auto accepted =
                rowsAccepted (rows, ref, refWeights, refComponents, trace.weights, trace.mu, bankRef, category);
            if (accepted && rows != ref)
                logMessage (name + ": accepted a flipped near-tie");
            expect (accepted, name + " rows");
            if (! accepted)
                continue;

            for (size_t k = 0; k < out.size(); ++k)
            {
                const auto& p = out[k];
                expectEquals (p.bundleId, meta["id"].toString());
                expectEquals (p.category, category);
                expectEquals (p.name, bankJson["name"][p.row].toString());
                for (int d = 0; d < zDim; ++d)
                    expect (juce::exactlyEqual (p.z[size_t (d)], bankZ[size_t (p.row * zDim + d)]));
                expectEquals ((int) p.codes.size(), numCodes);
                const auto* codes = static_cast<const float*> (bank.getData()) + rowsTotal * zDim + p.row * numCodes;
                bool same = (int) p.codes.size() == numCodes;
                for (int i = 0; same && i < numCodes; ++i)
                    same = std::isnan (codes[i]) ? std::isnan (p.codes[size_t (i)])
                                                 : juce::exactlyEqual (codes[i], p.codes[size_t (i)]);
                expect (same, name + " codes");
                if (rows == ref)
                {
                    const auto& refCodes = c["codes"][(int) k];
                    for (int i = 0; same && i < numCodes; ++i)
                        same = refCodes[i].isVoid()
                                   ? std::isnan (p.codes[size_t (i)])
                                   : juce::exactlyEqual ((float) (double) refCodes[i], p.codes[size_t (i)]);
                    expect (same, name + " codes vs fixture");
                }
            }
        }
        logMessage ("max abs diff: embedding " + juce::String (worstEmb) + ", logits " + juce::String (worstLogits)
                    + ", weights " + juce::String (worstWeights) + ", mu " + juce::String (worstMu));

        beginTest ("no bank category");
        // A category with no bank rows (p0: Keys..Atmosphere; p1 covers all 8, then this is covered by the plugin test)
        int emptyCat = -1;
        for (int k = 0; k < (int) meta["selection"]["categories"] && emptyCat < 0; ++k)
            if (std::find (bankCategory.begin(), bankCategory.end(), k) == bankCategory.end())
                emptyCat = k;
        if (emptyCat < 0)
            logMessage ("every category has bank rows: no-bank case left to the plugin test");
        else
        {
            out.assign (1, {}); // stale candidates must not survive a failure
            auto r0 = model.find ("electric piano keys", out, category, emptyCat, nullptr);
            expect (r0.failed());
            expectEquals (r0.getErrorMessage(),
                          "This bundle has no " + meta["labels"][emptyCat].toString() + " presets");
            expect (out.empty());
        }

        beginTest ("bad bundle keeps previous");
        juce::TemporaryFile tmp ("prompt.json");
        auto bad = juce::JSON::parse (juce::JSON::toString (meta));
        bad.getDynamicObject()->setProperty ("lab_bundle_id", "wrong-lab");
        expect (tmp.getFile().replaceWithText (juce::JSON::toString (bad)));
        auto r = model.find ("deep sub bass", out, category);
        expect (r.wasOk() && ! out.empty(), r.getErrorMessage());
        const auto goodRow = out.empty() ? -1 : out[0].row; // the bundle's own answer, p0 and p1 differ
        r = model.load (tmp.getFile(), lab);
        expect (r.failed());
        expect (r.getErrorMessage().contains ("wrong-lab"), r.getErrorMessage());
        r = model.find ("deep sub bass", out, category);
        expect (r.wasOk(), r.getErrorMessage());
        expect (! out.empty() && out[0].row == goodRow && out[0].bundleId == meta["id"].toString());

        bad = juce::JSON::parse (juce::JSON::toString (meta));
        bad.getDynamicObject()->removeProperty ("features");
        expect (tmp.getFile().replaceWithText (juce::JSON::toString (bad)));
        r = model.load (tmp.getFile(), lab);
        expect (r.failed());
        expect (r.getErrorMessage().contains ("features"), r.getErrorMessage());
        bad.getDynamicObject()->setProperty ("features", 7);
        bad.getDynamicObject()->removeProperty ("labels");
        expect (tmp.getFile().replaceWithText (juce::JSON::toString (bad)));
        expect (model.load (tmp.getFile(), lab).failed());
        expect (model.find ("deep sub bass", out, category).wasOk() && ! out.empty() && out[0].row == goodRow);
    }
};

static PromptModelTests promptModelTests;
} // namespace vae::experimental
