#include "JointVae.h"
#include <algorithm>
#include <cmath>
#include <set>

namespace vae::experimental
{
namespace
{
bool number (const juce::var& v) { return v.isDouble() || v.isInt() || v.isInt64(); }
bool shape (const ml::TensorInfo& t, const char* name, int size)
{
    return t.name == name && t.isFloat && t.shape.size() == 2 && (t.shape[0] == 1 || t.shape[0] == -1)
           && t.shape[1] == size;
}
juce::Result validate (const ml::OnnxModel& m, bool encoder, int count)
{
    const auto& in = m.getInputs();
    const auto& out = m.getOutputs();
    if (in.size() != 2 || out.size() != 1 || ! shape (in[0], encoder ? "sound" : "latent", encoder ? 48 : 32)
        || ! shape (in[1], "category", 9) || ! shape (out[0], encoder ? "latent" : "codes", count))
        return juce::Result::fail ("Joint VAE tensor contract mismatch");
    return juce::Result::ok();
}
juce::Result run (ml::OnnxModel* model, std::span<const float> values, int category, int count,
                  std::vector<float>& output)
{
    if (model == nullptr)
        return juce::Result::fail ("Load a joint VAE bundle first");
    if (category < 0 || category >= 9
        || ! std::all_of (values.begin(), values.end(), [] (float v) { return std::isfinite (v); }))
        return juce::Result::fail ("Invalid category or non-finite input");
    std::array<float, 9> label {};
    label[(size_t) category] = 1.0f;
    const std::array<int64_t, 2> inputShape { 1, (int64_t) values.size() }, categoryShape { 1, 9 };
    const std::array<ml::FloatTensorView, 2> inputs { ml::FloatTensorView { values, inputShape },
                                                      { label, categoryShape } };
    std::vector<std::vector<float>> outputs;
    if (auto r = model->run (inputs, outputs); r.failed())
        return r;
    if (outputs.size() != 1 || outputs[0].size() != (size_t) count
        || ! std::all_of (outputs[0].begin(), outputs[0].end(), [] (float v) { return std::isfinite (v); }))
        return juce::Result::fail ("Joint VAE returned invalid output");
    output = std::move (outputs[0]);
    return juce::Result::ok();
}

// Noun forms per category from vae/pitcher/text2preset/vocabulary_v2.yaml (v2.1), in jointLabels order.
const char* const labelNouns[8][19] = {
    { "bass", "synth bass", "bass sound", "sub bass", "sub", "reese bass", "reese", "neuro bass", "neurofunk bass",
      "acid bass", "303-style bass", "tb-303-style bass", "fm bass", "dx-style bass", "wobble bass", "lfo bass", "808",
      "808 bass", "808-style bass" },
    { "lead", "synth lead", "lead synth", "saw lead", "sawtooth lead", "square lead", "square-wave lead", "solo lead",
      "mono lead", "sync lead", "hard-sync lead", "whistle lead", "g-funk-style lead", "screech lead",
      "screamer lead" },
    { "pad", "synth pad", "pad sound", "string pad", "string-like pad", "choir pad", "vocal pad", "synth choir" },
    { "keys", "keyboard sound", "synth keys", "electric piano", "e-piano", "rhodes-style piano",
      "wurlitzer-style piano", "organ", "synth organ", "drawbar organ", "house organ", "clavinet", "clav" },
    { "pluck", "synth pluck", "plucked synth", "bell", "bell tone", "synth bell", "mallet", "mallet sound",
      "marimba-like pluck", "vibraphone-like synth", "harp-like pluck", "plucked string synth", "music-box-like pluck",
      "kalimba-like pluck" },
    { "synth", "synth tone", "synthesizer sound", "poly synth", "polysynth", "synth stab", "stab", "hoover",
      "hoover synth", "brass synth", "synth brass" },
    { "atmosphere", "atmospheric sound", "ambience", "drone", "drone tone", "soundscape", "sound texture", "texture",
      "wind-like texture", "noise bed", "riser", "uplifter", "riser sweep", "sweep", "noise sweep", "whoosh" },
    { "strings", "string ensemble", "synth strings", "brass", "brass section", "horn", "orchestral hit",
      "orchestra stab", "cello-like synth", "violin-like synth", "flute-like synth", "woodwind-like synth" },
};

bool containsWord (const juce::String& text, const juce::String& word)
{
    for (int i = text.indexOf (word); i >= 0; i = text.indexOf (i + 1, word))
    {
        const int end = i + word.length();
        if ((i == 0 || ! juce::CharacterFunctions::isLetterOrDigit (text[i - 1]))
            && (end == text.length() || ! juce::CharacterFunctions::isLetterOrDigit (text[end])))
            return true;
    }
    return false;
}
} // namespace

JointVae::LabelMatch JointVae::labelFor (const juce::String& prompt)
{
    const auto text = prompt.toLowerCase();
    LabelMatch best { 8, {} };
    for (int c = 0; c < 8; ++c)
        for (const auto* noun : labelNouns[c])
            if (noun != nullptr && juce::String (noun).length() > best.noun.length() && containsWord (text, noun))
                best = { c, noun };
    return best;
}

juce::Result JointLayout::parse (const juce::var& value, JointLayout& output)
{
    if (value["format"].toString() != "pitcher.joint-vae" || ! value["version"].isInt() || (int) value["version"] != 1
        || (int) value["latent_dimensions"] != 32 || (int) value["sound_dimensions"] != 48)
        return juce::Result::fail ("Expected pitcher.joint-vae bundle version 1 (32 latent / 48 sound)");
    auto* labels = value["labels"].getArray();
    if (labels == nullptr || labels->size() != 9)
        return juce::Result::fail ("Bundle must contain nine ordered category labels");
    for (int i = 0; i < 9; ++i)
        if ((*labels)[i].toString() != jointLabels[i])
            return juce::Result::fail ("Bundle category order differs from the trained model");
    auto* features = value["features"].getArray();
    if (features == nullptr || features->isEmpty() || features->size() > 10000)
        return juce::Result::fail ("Missing or invalid feature layout");
    JointLayout candidate;
    candidate.id = value["id"].toString();
    if (candidate.id.isEmpty())
        return juce::Result::fail ("Bundle needs an id");
    std::set<juce::String> seen;
    for (const auto& feature : *features)
    {
        auto name = feature["name"].toString();
        if (name.isEmpty() || ! seen.insert (name).second || ! number (feature["min"]) || ! number (feature["max"])
            || ! std::isfinite ((double) feature["min"]) || ! std::isfinite ((double) feature["max"])
            || (double) feature["min"] > (double) feature["max"])
            return juce::Result::fail ("Invalid or duplicate feature in bundle");
        candidate.names.add (name);
    }
    output = std::move (candidate);
    return juce::Result::ok();
}

juce::Result JointVae::load (const juce::File& file)
{
    juce::var manifest;
    if (auto r = juce::JSON::parse (file.loadFileAsString(), manifest); r.failed())
        return r;
    JointLayout next;
    if (auto r = JointLayout::parse (manifest, next); r.failed())
        return r;
    // Fixed relative names keep a bundle self-contained.
    auto dec = std::make_unique<ml::OnnxModel>();
    auto enc = std::make_unique<ml::OnnxModel>();
    if (auto r = dec->load (file.getSiblingFile ("decoder.onnx")); r.failed())
        return r;
    if (auto r = validate (*dec, false, next.names.size()); r.failed())
        return r;
    if (auto r = enc->load (file.getSiblingFile ("sound_encoder.onnx")); r.failed())
        return r;
    StudentCodec nextCodec;
    if (auto r = StudentCodec::load (file.getParentDirectory(), nextCodec); r.failed())
        return r;
    if (auto r = validate (*enc, true, 32); r.failed())
        return r;
    // Exercise both models before replacing a working bundle.
    std::vector<float> probe;
    if (auto r = run (enc.get(), SoundCode {}, 0, 32, probe); r.failed())
        return r;
    if (auto r = run (dec.get(), probe, 0, next.names.size(), probe); r.failed())
        return r;
    metadata = std::move (next);
    decoder = std::move (dec);
    encoder = std::move (enc);
    codec = std::move (nextCodec);
    return juce::Result::ok();
}

juce::Result JointVae::decode (const JointLatent& latent, int category, std::vector<float>& codes)
{
    return run (decoder.get(), latent, category, metadata.names.size(), codes);
}

juce::Result JointVae::patch (const std::vector<float>& codes, juce::MemoryBlock& output)
{
    StudentPlan plan;
    if (auto r = codec.plan (metadata.names, codes, plan); r.failed())
        return r;
#if VAE_WITH_SURGE
    if (builder == nullptr)
    {
        // Template + wavetables: installed by scripts/install-models.sh, else the build tree's Surge checkout
        auto data = ml::appDataDir().getChildFile ("models/surge-data");
        if (! data.isDirectory())
            data = juce::File (VAE_SURGE_DATA_DIR);
        builder = std::make_unique<SurgePatchBuilder> (data);
    }
    return builder->build (plan, output);
#else
    juce::ignoreUnused (output);
    return juce::Result::fail ("Full Surge patches need a build with VAE_WITH_SURGE=ON");
#endif
}

juce::Result JointVae::encode (const SoundCode& sound, int category, JointLatent& latent)
{
    std::vector<float> result;
    if (auto r = run (encoder.get(), sound, category, 32, result); r.failed())
        return r;
    std::copy (result.begin(), result.end(), latent.begin());
    return juce::Result::ok();
}

juce::Result JointVae::slice (const JointLatent& anchor, int xDim, int yDim, float x, float y, float radius,
                              JointLatent& result)
{
    if (xDim < 0 || yDim < 0 || xDim >= 32 || yDim >= 32 || xDim == yDim || ! std::isfinite (x) || ! std::isfinite (y)
        || ! std::isfinite (radius) || radius < 0 || std::abs (x) > 1 || std::abs (y) > 1
        || ! std::all_of (anchor.begin(), anchor.end(), [] (float v) { return std::isfinite (v); }))
        return juce::Result::fail ("Choose distinct latent axes and finite coordinates");
    auto next = anchor;
    next[(size_t) xDim] += x * radius;
    next[(size_t) yDim] += y * radius;
    if (! std::all_of (next.begin(), next.end(), [] (float v) { return std::isfinite (v); }))
        return juce::Result::fail ("Latent coordinate overflow");
    result = next;
    return juce::Result::ok();
}

JointVaeWorker::~JointVaeWorker()
{
    ++state->request;
    pool.removeAllJobs (true, -1);
}

void JointVaeWorker::submit (Work work, Callback callback)
{
    auto shared = state;
    const auto id = ++shared->request;
    pool.addJob (
        [shared, id, task = std::move (work), completion = std::move (callback)]
        {
            if (id != shared->request.load())
                return;
            Output output;
            auto result = task (shared->model, output);
            output.layout = shared->model.layout();
            juce::MessageManager::callAsync (
                [shared, id, result, decoded = std::move (output), completion]
                {
                    if (id == shared->request.load() && completion)
                        completion (result, decoded);
                });
        });
}

void JointVaeWorker::load (juce::File file, Callback cb)
{
    submit ([file] (JointVae& model, Output&) { return model.load (file); }, std::move (cb));
}
void JointVaeWorker::decode (JointLatent z, int category, Callback cb)
{
    submit (
        [z, category] (JointVae& model, Output& out)
        {
            out.latent = z;
            if (auto r = model.decode (z, category, out.codes); r.failed())
                return r;
            out.patchResult = model.patch (out.codes, out.patch);
            return juce::Result::ok();
        },
        std::move (cb));
}
void JointVaeWorker::encode (SoundCode sound, int category, Callback cb)
{
    submit (
        [sound, category] (JointVae& model, Output& out)
        {
            if (auto r = model.encode (sound, category, out.latent); r.failed())
                return r;
            if (auto r = model.decode (out.latent, category, out.codes); r.failed())
                return r;
            out.patchResult = model.patch (out.codes, out.patch);
            return juce::Result::ok();
        },
        std::move (cb));
}
} // namespace vae::experimental
