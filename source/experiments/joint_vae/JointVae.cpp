#include "JointVae.h"
#include "LiteTranslation.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
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
juce::Result validate (const ml::OnnxModel& m, bool encoder, int count, const char* outName = nullptr)
{
    const auto& in = m.getInputs();
    const auto& out = m.getOutputs();
    if (in.size() != 2 || out.size() != 1 || ! shape (in[0], encoder ? "sound" : "latent", encoder ? 48 : 32)
        || ! shape (in[1], "category", 9)
        || ! shape (out[0],
                    outName != nullptr ? outName
                    : encoder          ? "latent"
                                       : "codes",
                    count))
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

juce::Result applyTail (const JointLayout& layout, const std::optional<std::vector<float>>& tail,
                        std::vector<float>& codes)
{
    if (! tail)
        return juce::Result::ok();
    const auto from = (size_t) layout.numNumeric;
    if (codes.size() != (size_t) layout.names.size() || tail->size() != codes.size() - from)
        return juce::Result::fail ("Locked structure does not match the bundle layout");
    std::copy (tail->begin(), tail->end(), codes.begin() + (std::ptrdiff_t) from);
    return juce::Result::ok();
}

// KnobGroups::names order; NaN when the plan writes no Value for it
std::vector<float> knobValues (const StudentPlan& plan, const KnobGroups& groups)
{
    std::map<juce::String, float> written;
    for (const auto& v : plan.values)
        written[v.feature] = v.code;
    std::vector<float> knobs;
    for (const auto& name : groups.names)
    {
        const auto it = written.find (name);
        knobs.push_back (it != written.end() ? it->second : std::numeric_limits<float>::quiet_NaN());
    }
    return knobs;
}

// Page drawReadouts "Decoded preset" rows. Osc display names by Surge osc type id.
const char* const oscNames[] = { "Classic", "Sine",   "Wavetable", "S&H noise", "Audio input", "FM3",
                                 "FM2",     "Window", "Modern",    "String",    "Twist",       "Alias" };
const char* const fxNames[][2] = { { "reverb", "reverb" },         { "reverb2", "reverb 2" },
                                   { "delay", "delay" },           { "chorus", "chorus" },
                                   { "flanger", "flanger" },       { "phaser", "phaser" },
                                   { "distortion", "distortion" }, { "eq", "EQ" },
                                   { "rotary", "rotary" },         { "resonator", "resonator" },
                                   { "freqshift", "freq shift" },  { "conditioner", "conditioner" },
                                   { "nimbus", "nimbus" } };
std::vector<std::pair<juce::String, juce::String>> presetRows (const StudentPlan& plan, const JointLayout& layout,
                                                               const std::vector<float>& codes,
                                                               const KnobGroups& groups,
                                                               const std::vector<float>& knobs)
{
    std::vector<std::pair<juce::String, juce::String>> rows;
    for (int o = 0; o < 3; ++o)
    {
        juce::String type = "off";
        for (const auto& t : plan.types)
            if (t.path == "scene/0/osc/" + juce::String (o) + "/type" && t.value >= 0 && t.value < 12)
                type = oscNames[t.value];
        for (const auto& v : plan.values)
            if (v.path == "scene/0/mute_o" + juce::String (o + 1) && v.code >= 0.5f)
                type << " (muted)";
        rows.emplace_back ("Osc " + juce::String (o + 1), type);
    }
    juce::StringArray fx;
    for (const auto& f : fxNames)
        if (const int i = layout.names.indexOf ("fx/" + juce::String (f[0]) + "/present");
            i >= 0 && (size_t) i < codes.size() && codes[(size_t) i] >= 0.5f)
            fx.add (f[1]);
    rows.emplace_back ("Effects", fx.isEmpty() ? juce::String ("none") : fx.joinIntoString (", "));
    const auto knob = [&] (const juce::String& name)
    {
        const int i = groups.names.indexOf (name);
        return groups.format (i, i >= 0 && (size_t) i < knobs.size() ? knobs[(size_t) i] : 0.0f);
    };
    rows.emplace_back ("Cutoff", knob ("scene/filterunit/0/cutoff"));
    rows.emplace_back ("Resonance", knob ("scene/filterunit/0/resonance"));
    rows.emplace_back ("Amp A / D / S / R",
                       knob ("scene/adsr/0/a") + juce::String (juce::CharPointer_UTF8 (" \xc2\xb7 "))
                           + knob ("scene/adsr/0/d") + juce::String (juce::CharPointer_UTF8 (" \xc2\xb7 "))
                           + knob ("scene/adsr/0/s") + juce::String (juce::CharPointer_UTF8 (" \xc2\xb7 "))
                           + knob ("scene/adsr/0/r"));
    return rows;
}

juce::Result computeMap (JointVae& model, const MapRequest& req, MapResult& out, const std::function<bool()>& stale)
{
    if (req.groupX >= 0)
    {
        const auto& kg = model.knobGroups();
        const auto groups = (int) kg.groups.size();
        if (req.groupX >= groups || req.groupY < 0 || req.groupY >= groups
            || req.knobs.size() != (size_t) kg.names.size())
            return juce::Result::fail ("Bad knob pad map request");
        if (req.kind == "none")
            return juce::Result::ok();
        auto vals = req.knobs;
        for (int j = 0; j < out.n; ++j)
            for (int i = 0; i < out.n; ++i)
            {
                if (stale())
                    return juce::Result::fail ("Map cancelled");
                kg.set (req.groupX, ((float) i + 0.5f) / (float) out.n, vals);
                kg.set (req.groupY, 1.0f - ((float) j + 0.5f) / (float) out.n, vals);
                const auto p = model.forest().predict (vals);
                const auto best = std::max_element (p.begin(), p.end());
                out.cells.push_back ((int) (best - p.begin()));
                out.conf.push_back (*best);
            }
        return juce::Result::ok();
    }
    std::vector<const LatentAxis*> axes;
    for (const auto& id : req.axisIds)
        if (const auto* a = model.xypad().find (id))
            axes.push_back (a);
        else
            return juce::Result::fail ("Unknown pad axis " + id);
    if (axes.size() < 2 || axes.size() > 3)
        return juce::Result::fail ("A map needs X and Y axes");
    float SoundMeasures::* measure = nullptr;
    if (req.kind == "brightness")
        measure = &SoundMeasures::brightness;
    else if (req.kind == "length")
        measure = &SoundMeasures::length;
    else if (req.kind == "attack")
        measure = &SoundMeasures::attack;
    else if (req.kind == "noise")
        measure = &SoundMeasures::noise;
    else if (req.kind == "none")
        return juce::Result::ok();
    else if (req.kind != "cat")
        return juce::Result::fail ("Unknown map kind " + req.kind);
    const int n = out.n;
    std::vector<float> targets (axes.size());
    if (axes.size() == 3)
        targets[2] = axisTarget (*axes[2], req.zPos);
    std::vector<float> codes;
    SoundCode sound;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
        {
            if (stale())
                return juce::Result::fail ("Map cancelled");
            targets[0] = axisTarget (*axes[0], ((float) i + 0.5f) / (float) n);
            targets[1] = axisTarget (*axes[1], 1.0f - ((float) j + 0.5f) / (float) n);
            const auto z = constrain (req.z, axes, targets);
            if (measure != nullptr)
            {
                if (auto r = model.decodeSound (z, req.category, sound); r.failed())
                    return r;
                out.val.push_back (measures (model.xypad(), melFromSound (model.soundPca(), sound)).*measure);
                continue;
            }
            StudentPlan plan;
            if (auto r = model.decode (z, req.category, codes); r.failed())
                return r;
            if (auto r = applyTail (model.layout(), req.lockedTail, codes); r.failed())
                return r;
            if (auto r = model.plan (codes, plan); r.failed())
                return r;
            const auto p = model.forest().predict (knobValues (plan, model.knobGroups()));
            const auto best = std::max_element (p.begin(), p.end());
            out.cells.push_back ((int) (best - p.begin()));
            out.conf.push_back (*best);
        }
    if (measure != nullptr)
    {
        auto sorted = out.val;
        std::sort (sorted.begin(), sorted.end());
        out.lo = sorted[(size_t) (sorted.size() * 2 / 100)];
        out.hi = sorted[(size_t) (sorted.size() * 98 / 100)];
    }
    return juce::Result::ok();
}

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
        if (feature["kind"].toString() == "bool" && candidate.numNumeric == 0)
            candidate.numNumeric = candidate.names.size();
        candidate.names.add (name);
    }
    if (candidate.numNumeric == 0)
        candidate.numNumeric = candidate.names.size();
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
    const auto soundFile = manifest["sound_decoder"].toString();
    if (soundFile.isEmpty() || soundFile != juce::File::createFileWithoutCheckingPath (soundFile).getFileName())
        return juce::Result::fail ("Bundle has no \"sound_decoder\" entry (re-export it with sound_decoder.onnx)");
    auto sdec = std::make_unique<ml::OnnxModel>();
    if (auto r = sdec->load (file.getSiblingFile (soundFile)); r.failed())
        return r;
    if (auto r = validate (*sdec, false, 48, "sound"); r.failed())
        return r;
    const auto xypadFile = file.getSiblingFile ("xypad.json");
    if (! xypadFile.existsAsFile())
        return juce::Result::fail ("Bundle has no xypad.json (re-export it with tools/joint_vae/export_xypad.py)");
    XyPadData nextXy;
    if (auto r = XyPadData::load (xypadFile, nextXy); r.failed())
        return r;
    const auto knobs = juce::JSON::parse (xypadFile)["knobs"];
    CategoryForest nextForest;
    if (auto r = CategoryForest::load (knobs["forest"], nextForest); r.failed())
        return r;
    KnobGroups nextGroups;
    if (auto r = KnobGroups::load (knobs, nextGroups); r.failed())
        return r;
    SoundPca nextPca;
    if (auto r = loadPca (file.getSiblingFile ("sound_pca.npz"), nextPca); r.failed())
        return r;
    // Exercise both models before replacing a working bundle.
    std::vector<float> probe;
    if (auto r = run (enc.get(), SoundCode {}, 0, 32, probe); r.failed())
        return r;
    std::vector<float> soundProbe;
    if (auto r = run (sdec.get(), probe, 0, 48, soundProbe); r.failed())
        return r;
    if (auto r = run (dec.get(), probe, 0, next.names.size(), probe); r.failed())
        return r;
    metadata = std::move (next);
    soundDecoder = std::move (sdec);
    decoder = std::move (dec);
    encoder = std::move (enc);
    codec = std::move (nextCodec);
    xy = std::move (nextXy);
    categoryForest = std::move (nextForest);
    groups = std::move (nextGroups);
    pca = std::move (nextPca);
    return juce::Result::ok();
}

juce::Result JointVae::decode (const JointLatent& latent, int category, std::vector<float>& codes)
{
    return run (decoder.get(), latent, category, metadata.names.size(), codes);
}

juce::Result JointVae::decodeSound (const JointLatent& latent, int category, SoundCode& sound)
{
    std::vector<float> result;
    if (auto r = run (soundDecoder.get(), latent, category, 48, result); r.failed())
        return r;
    std::copy (result.begin(), result.end(), sound.begin());
    return juce::Result::ok();
}

juce::Result JointVae::plan (const std::vector<float>& codes, StudentPlan& output) const
{
    return codec.plan (metadata.names, codes, output);
}

juce::Result JointVae::patch (const std::vector<float>& codes, juce::MemoryBlock& output, StudentPlan* planOut)
{
    StudentPlan local;
    auto& plan = planOut != nullptr ? *planOut : local;
    if (auto r = this->plan (codes, plan); r.failed())
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
    ++state->mapRequest;
    pool.removeAllJobs (true, -1);
}

void JointVaeWorker::submit (Work work, Callback callback)
{
    auto shared = state;
    const auto id = ++shared->request;
    ++shared->mapRequest; // any newer request cancels a map
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
    submit (
        [file] (JointVae& model, Output& out)
        {
            auto r = model.load (file);
            if (r.wasOk())
            {
                out.xypad = std::make_shared<const XyPadData> (model.xypad());
                out.knobGroups = std::make_shared<const KnobGroups> (model.knobGroups());
            }
            return r;
        },
        std::move (cb));
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
void JointVaeWorker::lab (LabRequest request, Callback cb)
{
    submit (
        [req = std::move (request)] (JointVae& model, Output& out)
        {
            const auto n = (size_t) model.layout().names.size();
            out.latent = req.z;
            if (req.codes)
            {
                if (req.codes->size() != n
                    || ! std::all_of (req.codes->begin(), req.codes->end(), [] (float v) { return std::isfinite (v); }))
                    return juce::Result::fail ("Codes do not match the bundle layout");
                out.codes = *req.codes;
            }
            else if (auto r = model.decode (req.z, req.category, out.codes); r.failed())
                return r;
            if (auto r = applyTail (model.layout(), req.lockedTail, out.codes); r.failed())
                return r;
            if (auto r = model.decodeSound (req.z, req.category, out.sound); r.failed())
                return r;
            out.mel = melFromSound (model.soundPca(), out.sound);

            StudentPlan plan;
            out.patchResult = model.patch (out.codes, out.patch, &plan);
            if (out.patchResult.wasOk())
                out.patchResult = liteControls (out.patch, out.lite);
            if (out.patchResult.failed())
                out.patch.reset();
            out.knobs = knobValues (plan, model.knobGroups());
            out.guess = model.forest().predict (out.knobs);
            out.preset = presetRows (plan, model.layout(), out.codes, model.knobGroups(), out.knobs);
            return juce::Result::ok();
        },
        std::move (cb));
}
void JointVaeWorker::map (MapRequest request, std::function<void (juce::Result, MapResult)> callback)
{
    auto shared = state;
    const auto id = ++shared->mapRequest;
    pool.addJob (
        [shared, id, req = std::move (request), completion = std::move (callback)]
        {
            const auto stale = [&] { return id != shared->mapRequest.load(); };
            if (stale())
                return;
            MapResult out;
            shared->mapRunning = true;
            auto result = computeMap (shared->model, req, out, stale);
            shared->mapRunning = false;
            if (stale())
                return;
            juce::MessageManager::callAsync (
                [shared, id, result, map = std::move (out), completion]
                {
                    if (id == shared->mapRequest.load() && completion)
                        completion (result, map);
                });
        });
}
} // namespace vae::experimental
