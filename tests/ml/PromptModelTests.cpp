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
                      double& worst)
    {
        expectEquals ((int) got.size(), ref.size(), what + " size");
        for (int i = 0; i < juce::jmin (ref.size(), (int) got.size()); ++i)
            worst = juce::jmax (worst, std::abs ((double) ref[i] - (double) got[size_t (i)]));
        expect (worst <= tol, what + " max diff " + juce::String (worst));
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

        double worstEmb = 0, worstLogits = 0, worstWeights = 0, worstMu = 0;
        bool first = true;
        for (const auto& c : *parity["cases"].getArray())
        {
            const auto name = c["name"].toString();
            if (! c["forced_category"].isVoid() && ! c["forced_category"].isUndefined())
                continue; // keys_forced: the "no bank category" test below
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
            // R15: where reference weights are near-tied (< 1e-4), also accept the order with those two swapped,
            // recomputed from the native weights and mu against bank.bin.
            auto accepted = rows == ref;
            const auto& comps = *c["components"].getArray();
            for (int i = 0; ! accepted && i + 1 < comps.size(); ++i)
            {
                const int a = comps[i], b = comps[i + 1];
                if ((double) c["mix_weights"][a] - (double) c["mix_weights"][b] >= 1e-4)
                    continue;
                auto swapped = trace.weights;
                std::swap (swapped[size_t (a)], swapped[size_t (b)]);
                const auto alt = PromptModel::select (swapped, trace.mu, bankZ, bankCategory, category, 0.02f, 8);
                accepted =
                    rows == alt
                    && rows == PromptModel::select (trace.weights, trace.mu, bankZ, bankCategory, category, 0.02f, 8);
                if (accepted)
                    logMessage (name + ": near-tied components " + juce::String (a) + "/" + juce::String (b)
                                + " swapped");
            }
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
        out.assign (1, {}); // stale candidates must not survive a failure
        auto r = model.find ("electric piano keys", out, category, 3, nullptr);
        expect (r.failed());
        expectEquals (r.getErrorMessage(), juce::String ("This bundle has no Keys presets"));
        expect (out.empty());

        beginTest ("bad bundle keeps previous");
        juce::TemporaryFile tmp ("prompt.json");
        auto bad = juce::JSON::parse (juce::JSON::toString (meta));
        bad.getDynamicObject()->setProperty ("lab_bundle_id", "wrong-lab");
        expect (tmp.getFile().replaceWithText (juce::JSON::toString (bad)));
        r = model.load (tmp.getFile(), lab);
        expect (r.failed());
        expect (r.getErrorMessage().contains ("wrong-lab"), r.getErrorMessage());
        r = model.find ("deep sub bass", out, category);
        expect (r.wasOk(), r.getErrorMessage());
        expect (! out.empty() && out[0].row == 43 && out[0].bundleId == meta["id"].toString());
    }
};

static PromptModelTests promptModelTests;
} // namespace vae::experimental
