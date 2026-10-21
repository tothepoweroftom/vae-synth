#include "schema/PresetFile.h"
#include "PluginProcessor.h"

#include "BinaryData.h"
#include "PluginEditor.h"
#include "schema/SurgeCurves.h"
#if VAE_WITH_SURGE
#include "engine/surge/SurgeEngine.h"
#endif

namespace vae
{
namespace
{
const juce::Identifier kStateType { "VAESynth" };
const juce::Identifier kParamsTree { "VAESynthParams" };
constexpr const char* kAttrParams = "params";
constexpr const char* kAttrModel = "decoderModel";
constexpr const char* kAttrPrompt = "prompt";
constexpr const char* kAttrSchema = "schemaVersion";
} // namespace

PluginProcessor::PluginProcessor()
    : AudioProcessor (BusesProperties().withOutput ("Output", juce::AudioChannelSet::stereo(), true))
    , engineKinds (availableEngines())
{
    for (auto kind : engineKinds)
        engines.push_back (createEngine (kind));

    std::vector<params::ParamSpec> specs;
    specs.reserve (schema::kControlCount + 1);

    for (int i = 0; i < schema::kControlCount; ++i)
    {
        const auto& c = schema::controls()[(size_t) i];
        params::ParamSpec s;
        s.id = c.id;
        s.name = c.name;
        s.defaultValue = c.defaultValue;
        switch (c.kind)
        {
            case schema::Kind::Continuous:
                s.kind = params::ParamKind::Float;
                s.range = { 0.0f, 1.0f };
                s.toText = [i] (float v) { return juce::String (schema::surge::describe (i, v)); };
                break;
            case schema::Kind::Categorical:
                s.kind = params::ParamKind::Choice;
                for (int o = 0; o < c.numOptions; ++o)
                    s.choices.add (c.options[o]);
                break;
            case schema::Kind::Toggle: s.kind = params::ParamKind::Bool; break;
        }
        s.onAudioChange = [this, i] (float v)
        {
            audioControls[(size_t) i] = v;
            if (activeEngine != nullptr)
                activeEngine->setControl (i, v);
        };
        specs.push_back (std::move (s));
    }

    {
        params::ParamSpec s;
        s.id = kEngineParamId;
        s.name = "Engine";
        s.kind = params::ParamKind::Choice;
        for (const auto& engine : engines)
            s.choices.add (engine->displayName());
        // Default to the reference engine when it is built in.
        s.defaultValue = float (engineKinds.size() - 1);
        s.hostVisible = false; // a build/setup choice, not automation; saved with the state
        s.onAudioChange = [this] (float v) { selectEngine (juce::roundToInt (v)); };
        specs.push_back (std::move (s));
    }

    params.build (std::move (specs), *this, kParamsTree);

    for (int i = 0; i < schema::kControlCount; ++i)
        controlIndex[(size_t) i] = params.indexOf (schema::controls()[(size_t) i].id);

    // Power on with the embedded stub decoder so the prompt path works out of the box.
    loadDecoderModel ({}, nullptr);
}

PluginProcessor::~PluginProcessor() = default;

//==============================================================================
void PluginProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    const auto current = getControls();
    for (auto& e : engines)
    {
        e->setControls (current);
        e->prepare (sampleRate, samplesPerBlock);
    }
    audioControls = current.values;
    keyboardState.reset();
    selectEngine (juce::roundToInt (params.get (kEngineParamId)));
}

void PluginProcessor::releaseResources()
{
    for (auto& e : engines)
        e->reset();
}

bool PluginProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
    return layouts.getMainOutputChannelSet() == juce::AudioChannelSet::stereo();
}

void PluginProcessor::selectEngine (int choiceIndex)
{
    choiceIndex = juce::jlimit (0, int (engines.size()) - 1, choiceIndex);
    auto* next = engines[(size_t) choiceIndex].get();
    if (next == activeEngine)
        return;

    if (activeEngine != nullptr)
    {
        activeEngine->reset();
        // The new engine never saw the held notes: release them everywhere
        // (on-screen keyboard included) rather than leave keys shown as down.
        keyboardState.allNotesOff (0);
    }

    schema::Snapshot s;
    s.values = audioControls;
    next->setControls (s);
    activeEngine = next;
}

void PluginProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midi)
{
    juce::ScopedNoDenormals noDenormals;

    params.dispatch();
    keyboardState.processNextMidiBuffer (midi, 0, buffer.getNumSamples(), true);

    if (activeEngine == nullptr)
    {
        buffer.clear();
        return;
    }
    activeEngine->process (buffer, midi);
}

//==============================================================================
schema::Snapshot PluginProcessor::getControls() const
{
    schema::Snapshot s;
    for (int i = 0; i < schema::kControlCount; ++i)
        s[i] = params.get (controlIndex[(size_t) i]); // raw: 0..1, option index or 0/1
    return s;
}

juce::Result PluginProcessor::loadPresetFile (const juce::File& file)
{
    if (! file.existsAsFile() || file.getSize() > 1024 * 1024)
        return juce::Result::fail ("Missing or oversized preset file");
    const auto json = juce::JSON::parse (file);
    if (json["format"].toString() != "vae-preset-1" || int (json["schema_version"]) != schema::kSchemaVersion)
        return juce::Result::fail ("Incompatible preset format or schema");
    const auto name = json["engine"].toString();
    if (name != "lite" && name != "surge")
        return juce::Result::fail ("Unknown preset engine");
    const auto kind = name == "lite" ? EngineKind::Lite : EngineKind::Surge;
    const auto engine = std::find (engineKinds.begin(), engineKinds.end(), kind);
    if (engine == engineKinds.end())
        return juce::Result::fail ("Preset engine is unavailable in this build");
    schema::Snapshot controls;
    const auto result = schema::readPresetValues (json, controls);
    if (result.failed())
        return result;
    params.set (kEngineParamId, float (std::distance (engineKinds.begin(), engine)));
    applyControls (controls);
    return juce::Result::ok();
}

juce::Result PluginProcessor::loadSurgePatch (const juce::MemoryBlock& patch)
{
#if VAE_WITH_SURGE
    const auto kind = std::find (engineKinds.begin(), engineKinds.end(), EngineKind::Surge);
    if (kind != engineKinds.end())
    {
        const auto index = std::distance (engineKinds.begin(), kind);
        static_cast<SurgeEngine&> (*engines[(size_t) index]).loadPatchData (patch);
        params.set (kEngineParamId, float (index));
        return juce::Result::ok();
    }
#endif
    juce::ignoreUnused (patch);
    return juce::Result::fail ("Full Surge patches need a build with VAE_WITH_SURGE=ON");
}

void PluginProcessor::applyControls (const schema::Snapshot& snapshot)
{
#if VAE_WITH_SURGE
    for (auto& e : engines)
        if (e->kind() == EngineKind::Surge)
            static_cast<SurgeEngine&> (*e).useSchemaControls();
#endif
    for (int i = 0; i < schema::kControlCount; ++i)
    {
        const auto index = controlIndex[(size_t) i];
        params.beginGesture (index);
        params.set (index, schema::sanitiseValue (i, snapshot[i]));
        params.endGesture (index);
    }
}

void PluginProcessor::generateFromPrompt (const juce::String& prompt, std::function<void (const juce::String&)> done)
{
    setLastPrompt (prompt);
    generator.generate (prompt,
                        [this, done] (const ml::PresetGenerator::Result& r)
                        {
                            applyControls (r.controls);
                            if (done)
                                done (r.note);
                        });
}

void PluginProcessor::generateRandom (std::function<void (const juce::String&)> done)
{
    schema::Latent latent;
    for (auto& l : latent)
        l = random.nextFloat() * 2.0f - 1.0f;

    generator.generateFromLatent (latent,
                                  [this, done] (const ml::PresetGenerator::Result& r)
                                  {
                                      applyControls (r.controls);
                                      if (done)
                                          done (r.note);
                                  });
}

void PluginProcessor::loadDecoderModel (const juce::File& file, std::function<void (juce::Result)> done)
{
    requestModel (file, file, false, std::move (done));
}

void PluginProcessor::requestModel (const juce::File& toLoad, const juce::File& toRecord, bool missing,
                                    std::function<void (juce::Result)> done)
{
    const auto id = ++modelRequest;
    auto onLoaded = [this, id, toRecord, missing, done = std::move (done)] (juce::Result r)
    {
        // Only the latest request may change what is saved, and only on success.
        if (r.wasOk() && id == modelRequest.load())
        {
            const juce::ScopedLock sl (stateLock);
            modelPath = toRecord;
            modelMissing = missing;
        }
        if (done)
            done (r);
    };

    if (toLoad == juce::File())
        generator.loadDecoder (juce::MemoryBlock (BinaryData::stub_decoder_onnx, BinaryData::stub_decoder_onnxSize),
                               "stub_decoder.onnx (embedded)", std::move (onLoaded));
    else
        generator.loadDecoder (toLoad, std::move (onLoaded));
}

juce::File PluginProcessor::getDecoderModelFile() const
{
    const juce::ScopedLock sl (stateLock);
    return modelPath;
}

juce::String PluginProcessor::getLastPrompt() const
{
    const juce::ScopedLock sl (stateLock);
    return lastPrompt;
}

juce::String PluginProcessor::describeModel() const
{
    auto text = generator.describe();
    const juce::ScopedLock sl (stateLock);
    if (modelMissing)
        text += "  (saved model not found: " + modelPath.getFullPathName() + ")";
    return text;
}

//==============================================================================
void PluginProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    juce::MemoryBlock paramState;
    params.save (paramState);

    juce::XmlElement xml (kStateType);
    xml.setAttribute (kAttrSchema, schema::kSchemaVersion);
    xml.setAttribute (kAttrParams, paramState.toBase64Encoding());
    {
        const juce::ScopedLock sl (stateLock);
        xml.setAttribute (kAttrModel, modelPath.getFullPathName());
        xml.setAttribute (kAttrPrompt, lastPrompt);
    }
    copyXmlToBinary (xml, destData);
}

void PluginProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    const auto xml = getXmlFromBinary (data, sizeInBytes);
    if (xml == nullptr || ! xml->hasTagName (kStateType))
        return;

    // A different schema version means different control semantics: keep the
    // current (default) controls rather than misreading them.
    const bool sameSchema = xml->getIntAttribute (kAttrSchema, -1) == schema::kSchemaVersion;

    juce::MemoryBlock paramState;
    if (sameSchema && paramState.fromBase64Encoding (xml->getStringAttribute (kAttrParams)))
        params.load (paramState.getData(), (int) paramState.getSize());

    const juce::File model (xml->getStringAttribute (kAttrModel));
    const bool missing = model != juce::File() && ! model.existsAsFile();
    {
        // Keep the saved path even when the file is missing (unplugged drive),
        // so saving again does not lose the reference.
        const juce::ScopedLock sl (stateLock);
        lastPrompt = xml->getStringAttribute (kAttrPrompt);
        modelPath = model;
        modelMissing = missing;
    }
    requestModel (missing ? juce::File() : model, model, missing, nullptr);
}

juce::AudioProcessorEditor* PluginProcessor::createEditor() { return new PluginEditor (*this); }

} // namespace vae

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() { return new vae::PluginProcessor(); }

void vae::PluginProcessor::setLastPrompt (const juce::String& prompt)
{
    const juce::ScopedLock sl (stateLock);
    lastPrompt = prompt;
}
