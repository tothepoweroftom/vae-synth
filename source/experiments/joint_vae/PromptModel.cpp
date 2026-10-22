#include "PromptModel.h"
#include <algorithm>
#include <cmath>
#include <numeric>

extern "C"
{
    void* vae_tokenizer_load_config (const char*, size_t maxLength, size_t padTo, uint32_t padId);
    void vae_tokenizer_free (void*);
    ptrdiff_t vae_tokenizer_encode_i64 (const void*, const char*, int64_t* ids, int64_t* mask, size_t capacity);
}

namespace vae::experimental
{
namespace
{
constexpr int zDim = std::tuple_size_v<JointLatent>;

int outputIndex (const ml::OnnxModel& m, const char* name)
{
    const auto& outs = m.getOutputs();
    for (size_t i = 0; i < outs.size(); ++i)
        if (outs[i].name == name)
            return (int) i;
    return -1;
}

bool allFinite (const std::vector<float>& v)
{
    return std::all_of (v.begin(), v.end(), [] (float x) { return std::isfinite (x); });
}
} // namespace

struct PromptModel::Model
{
    struct FreeTokenizer
    {
        void operator() (void* t) const { vae_tokenizer_free (t); }
    };
    std::unique_ptr<void, FreeTokenizer> tokenizer;
    ml::OnnxModel encoder, prior;
    juce::String id;
    juce::StringArray labels, features, presetNames;
    int maxLength = 0, dim = 0, categories = 0, k = 0, maxResults = 0, numCodes = 0;
    float minWeight = 0;
    std::vector<float> z, codes; // rows x 32, rows x numCodes
    std::vector<int> category;
    int idsInput = 0, textOut = -1, logitsOut = -1, mixOut = -1, muOut = -1, logsdOut = -1;
};

PromptModel::PromptModel() = default;
PromptModel::~PromptModel() = default;

juce::Result PromptModel::load (const juce::File& file, const JointLayout& lab)
{
    auto fail = [] (const juce::String& m) { return juce::Result::fail ("Prompt bundle: " + m); };
    juce::var meta;
    if (! file.existsAsFile() || file.getSize() > 1024 * 1024
        || juce::JSON::parse (file.loadFileAsString(), meta).failed() || meta["format"].toString() != "pitcher.prompt"
        || (int) meta["version"] != 1)
        return fail ("select a pitcher.prompt version 1 prompt.json");
    const auto labId = meta["lab_bundle_id"].toString();
    if (labId != lab.id)
        return fail ("made for Lab bundle " + labId + ", but " + (lab.id.isEmpty() ? juce::String ("none") : lab.id)
                     + " is loaded");

    auto m = std::make_unique<Model>();
    m->id = meta["id"].toString();
    const auto* features = meta["features"].getArray();
    const auto* labels = meta["labels"].getArray();
    if (m->id.isEmpty() || features == nullptr || labels == nullptr)
        return fail ("prompt.json needs an id and \"features\" and \"labels\" arrays");
    for (const auto& f : *features)
        m->features.add (f.toString());
    if (m->features != lab.names)
        return fail ("feature order differs from the Lab bundle");
    for (const auto& l : *labels)
        m->labels.add (l.toString());

    const auto tok = meta["tokenizer"], sel = meta["selection"], bank = meta["bank"];
    m->maxLength = tok["max_length"];
    m->dim = meta["embedding"]["dim"];
    m->categories = sel["categories"];
    m->k = sel["k"];
    m->minWeight = (float) (double) sel["min_weight"];
    m->maxResults = sel["max_results"];
    m->numCodes = bank["codes"];
    const int rows = bank["rows"];
    // shortcut: one prompt per call, so "longest" padding pads nothing; other recipes need a new tokenizer mode.
    if (tok["padding"].toString() != "longest" || tok["truncation"].toString() != "longest_first" || m->maxLength < 2
        || m->maxLength > 4096 || m->dim <= 0 || ! meta["embedding"]["l2_normalise"])
        return fail ("unsupported tokenizer or embedding settings");
    if (m->categories <= 0 || m->labels.size() < m->categories || m->k <= 0 || m->maxResults <= 0 || m->maxResults > 8
        || ! std::isfinite (m->minWeight) || (int) bank["z_dim"] != zDim || m->numCodes != m->features.size()
        || rows <= 0)
        return fail ("invalid selection or bank settings");

    juce::MemoryBlock data;
    const auto rowFloats = size_t (zDim + m->numCodes);
    if (! file.getSiblingFile ("bank.bin").loadFileAsData (data) || data.getSize() != size_t (rows) * rowFloats * 4)
        return fail ("bank.bin is missing or does not match prompt.json");
    const auto* f = static_cast<const float*> (data.getData()); // little-endian f32, like every supported target
    m->z.assign (f, f + rows * zDim);
    m->codes.assign (f + rows * zDim, f + size_t (rows) * rowFloats);
    if (! allFinite (m->z) || std::any_of (m->codes.begin(), m->codes.end(), [] (float x) { return std::isinf (x); }))
        return fail ("bank.bin has non-finite values"); // NaN codes are inactive and kept
    const auto bankJson = juce::JSON::parse (file.getSiblingFile ("bank.json"));
    const auto* cats = bankJson["category"].getArray();
    const auto* names = bankJson["name"].getArray();
    if (cats == nullptr || names == nullptr || cats->size() != rows || names->size() != rows)
        return fail ("bank.json does not match bank.bin");
    for (const auto& c : *cats)
    {
        if (! c.isInt() && ! c.isInt64())
            return fail ("bank.json has an invalid category");
        m->category.push_back (c);
        if (m->category.back() < 0 || m->category.back() >= m->categories)
            return fail ("bank.json has an invalid category");
    }
    for (const auto& name : *names)
        m->presetNames.add (name.toString());

    m->tokenizer.reset (vae_tokenizer_load_config (file.getSiblingFile ("tokenizer.json").getFullPathName().toRawUTF8(),
                                                   size_t (m->maxLength), 0, (uint32_t) (int) tok["pad_id"]));
    if (m->tokenizer == nullptr)
        return fail ("could not load tokenizer.json");
    ml::ModelLoadOptions options;
    options.threads = 2; // bounded: runs next to a busy audio thread
    options.int64Inputs = true;
    if (auto r = m->encoder.load (file.getSiblingFile ("text_encoder.onnx"), options); r.failed())
        return fail (r.getErrorMessage());
    options.int64Inputs = false;
    if (auto r = m->prior.load (file.getSiblingFile ("prior.onnx"), options); r.failed())
        return fail (r.getErrorMessage());
    const auto& in = m->encoder.getInputs();
    m->idsInput = in.size() == 2 && in[0].name == "input_ids" ? 0 : 1;
    m->textOut = outputIndex (m->encoder, "text");
    m->logitsOut = outputIndex (m->prior, "category_logits");
    m->mixOut = outputIndex (m->prior, "mix_logits");
    m->muOut = outputIndex (m->prior, "mu");
    m->logsdOut = outputIndex (m->prior, "logsd");
    if (in.size() != 2 || ! in[0].isInt64 || ! in[1].isInt64 || in[size_t (m->idsInput)].name != "input_ids"
        || in[size_t (1 - m->idsInput)].name != "attention_mask" || m->textOut < 0 || m->prior.getInputs().size() != 1
        || m->prior.getInputs()[0].name != "text" || std::min ({ m->logitsOut, m->mixOut, m->muOut, m->logsdOut }) < 0)
        return fail ("text encoder or prior tensor contract mismatch");
    model = std::move (m);
    return juce::Result::ok();
}

juce::Result PromptModel::find (const juce::String& text, std::vector<PromptCandidate>& out, int& category)
{
    return find (text, out, category, -1, nullptr);
}

juce::Result PromptModel::find (const juce::String& text, std::vector<PromptCandidate>& out, int& category,
                                int forceCategory, Trace* trace)
{
    out.clear();
    if (model == nullptr)
        return juce::Result::fail ("Load a prompt bundle first");
    auto& m = *model;

    std::vector<int64_t> ids (size_t (m.maxLength)), mask (ids.size());
    const auto n = vae_tokenizer_encode_i64 (m.tokenizer.get(), text.toRawUTF8(), ids.data(), mask.data(), ids.size());
    if (n <= 0)
        return juce::Result::fail ("Could not tokenize prompt");
    ids.resize (size_t (n));
    mask.resize (size_t (n));
    const int64_t tokenShape[] { 1, n };
    ml::Int64TensorView views[2];
    views[m.idsInput] = { ids, tokenShape };
    views[1 - m.idsInput] = { mask, tokenShape };
    std::vector<std::vector<float>> outs;
    if (auto r = m.encoder.run (std::span<const ml::Int64TensorView> (views), outs); r.failed())
        return r;
    auto embedding = std::move (outs[size_t (m.textOut)]);
    const auto norm = std::sqrt (std::inner_product (embedding.begin(), embedding.end(), embedding.begin(), 0.0));
    if ((int) embedding.size() != m.dim || ! allFinite (embedding) || ! std::isfinite (norm) || ! (norm > 0))
        return juce::Result::fail ("The text encoder returned an invalid embedding");
    for (auto& x : embedding)
        x = float (x / norm);

    const int64_t textShape[] { 1, m.dim };
    if (auto r = m.prior.run (embedding, textShape, outs); r.failed())
        return r;
    auto &logits = outs[size_t (m.logitsOut)], &mix = outs[size_t (m.mixOut)], &mu = outs[size_t (m.muOut)];
    if ((int) logits.size() < m.categories || (int) mix.size() != m.k || (int) mu.size() != m.k * zDim
        || ! allFinite (logits) || ! allFinite (mix) || ! allFinite (mu) || ! allFinite (outs[size_t (m.logsdOut)]))
        return juce::Result::fail ("The prompt prior returned invalid output");

    const auto best = int (std::max_element (logits.begin(), logits.begin() + m.categories) - logits.begin());
    if (forceCategory >= m.categories)
        return juce::Result::fail ("Invalid category");
    const auto chosen = forceCategory >= 0 ? forceCategory : best;
    const auto top = *std::max_element (mix.begin(), mix.end());
    std::vector<float> weights;
    for (auto x : mix)
        weights.push_back (std::exp (x - top));
    const auto sum = std::accumulate (weights.begin(), weights.end(), 0.0f);
    for (auto& w : weights)
        w /= sum;
    if (trace != nullptr)
        *trace = { ids, mask, embedding, logits, weights, mu };

    category = chosen;
    if (std::find (m.category.begin(), m.category.end(), chosen) == m.category.end())
        return juce::Result::fail ("This bundle has no " + m.labels[chosen] + " presets");
    for (auto row : select (weights, mu, m.z, m.category, chosen, m.minWeight, m.maxResults))
    {
        auto& c = out.emplace_back();
        c.bundleId = m.id;
        c.row = row;
        c.category = chosen;
        std::copy_n (m.z.begin() + row * zDim, zDim, c.z.begin());
        const auto codes = m.codes.begin() + row * m.numCodes;
        c.codes.assign (codes, codes + m.numCodes);
        c.name = m.presetNames[row];
    }
    return juce::Result::ok();
}

std::vector<int> PromptModel::select (std::span<const float> weights, std::span<const float> mu,
                                      std::span<const float> bankZ, std::span<const int> bankCategory, int category,
                                      float minWeight, int maxResults)
{
    std::vector<int> order, rows;
    for (int k = 0; k < (int) weights.size(); ++k)
        if (weights[size_t (k)] >= minWeight)
            order.push_back (k);
    std::stable_sort (order.begin(), order.end(),
                      [&] (int a, int b) { return weights[size_t (a)] > weights[size_t (b)]; }); // ties: index
    std::vector<bool> used (bankCategory.size());
    for (auto k : order)
    {
        if ((int) rows.size() >= maxResults)
            break;
        int nearest = -1;
        double nearestDistance = 0;
        for (size_t r = 0; r < bankCategory.size(); ++r)
        {
            if (bankCategory[r] != category || used[r])
                continue;
            double d = 0;
            for (size_t i = 0; i < size_t (zDim); ++i)
                d += juce::square (double (bankZ[r * zDim + i]) - mu[size_t (k) * zDim + i]);
            if (nearest < 0 || d < nearestDistance) // strict: equal distance keeps the lower row
                nearest = (int) r, nearestDistance = d;
        }
        if (nearest < 0)
            break;
        used[size_t (nearest)] = true;
        rows.push_back (nearest);
    }
    return rows;
}

PromptWorker::~PromptWorker()
{
    ++state->request;
    ++state->loads;
    pool.removeAllJobs (
        true,
        -1); // shortcut: closing the editor waits for a running load or Find (load ~0.5 s, Find <= ~0.6 s; ONNX runs cannot be
    // cut short); upgrade to a process-wide or detached runner that owns the model if that wait is noticed
}

void PromptWorker::load (juce::File file, JointLayout lab, std::function<void (juce::Result)> callback)
{
    auto shared = state;
    ++shared->request; // results from the previous bundle are obsolete
    const auto id = ++shared->loads;
    pool.addJob (
        [shared, id, file, lab = std::move (lab), completion = std::move (callback)]
        {
            if (id != shared->loads.load())
                return;
            const auto result = shared->model.load (file, lab);
            juce::MessageManager::callAsync (
                [shared, id, result, completion]
                {
                    if (id == shared->loads.load() && completion)
                        completion (result);
                });
        });
}

void PromptWorker::find (juce::String text, Found callback)
{
    auto shared = state;
    const auto id = ++shared->request;
    pool.addJob (
        [shared, id, text, completion = std::move (callback)]
        {
            if (id != shared->request.load())
                return;
            std::vector<PromptCandidate> found;
            int category = -1;
            const auto result = shared->model.find (text, found, category);
            juce::MessageManager::callAsync (
                [shared, id, result, category, found = std::move (found), completion]
                {
                    if (id == shared->request.load() && completion)
                        completion (result, category, found);
                });
        });
}
} // namespace vae::experimental
