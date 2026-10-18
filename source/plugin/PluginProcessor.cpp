#include "PluginProcessor.h"

#include "BinaryData.h"
#include "PluginEditor.h"
#include "schema/SurgeCurves.h"

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

juce::String engineName (EngineKind k)
{
    return k == EngineKind::Surge ? "Surge XT" : "Lite";
}
} // namespace

PluginProcessor::PluginProcessor()
    : AudioProcessor (BusesProperties().withOutput ("Output", juce::AudioChannelSet::stereo(), true)),
      engineKinds (availableEngines())
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
        s.kind = params::ParamKind::Float;
        s.range = { 0.0f, 1.0f };
        s.defaultValue = c.defaultValue;
        s.toText = [i] (float v) { return juce::String (schema::surge::describe (i, v)); };
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
        for (auto kind : engineKinds)
            s.choices.add (engineName (kind));
        // Default to the reference engine when it is built in.
        s.defaultValue = float (engineKinds.size() - 1);
        s.hostVisible = false;          // a build/setup choice, not automation
        s.restoreFromPreset = false;    // presets are engine-neutral vectors
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
        activeEngine->reset();

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
        s[i] = params.getNormalised (controlIndex[(size_t) i]);
    return s;
}

void PluginProcessor::applyControls (const schema::Snapshot& snapshot)
{
    for (int i = 0; i < schema::kControlCount; ++i)
    {
        const auto index = controlIndex[(size_t) i];
        params.beginGesture (index);
        params.setNormalised (index, snapshot[i]);
        params.endGesture (index);
    }
}

void PluginProcessor::generateFromPrompt (const juce::String& prompt, std::function<void (const juce::String&)> done)
{
    lastPrompt = prompt;
    generator.generate (prompt, [this, done] (const ml::PresetGenerator::Result& r)
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

    generator.generateFromLatent (latent, [this, done] (const ml::PresetGenerator::Result& r)
    {
        applyControls (r.controls);
        if (done)
            done (r.note);
    });
}

void PluginProcessor::loadDecoderModel (const juce::File& file, std::function<void (juce::Result)> done)
{
    decoderModelFile = file;
    if (file == juce::File())
        generator.loadDecoder (juce::MemoryBlock (BinaryData::stub_decoder_onnx, BinaryData::stub_decoder_onnxSize),
                               "stub_decoder.onnx (embedded)", std::move (done));
    else
        generator.loadDecoder (file, std::move (done));
}

//==============================================================================
void PluginProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    juce::MemoryBlock paramState;
    params.save (paramState);

    juce::XmlElement xml (kStateType);
    xml.setAttribute (kAttrSchema, schema::kSchemaVersion);
    xml.setAttribute (kAttrParams, paramState.toBase64Encoding());
    xml.setAttribute (kAttrModel, decoderModelFile.getFullPathName());
    xml.setAttribute (kAttrPrompt, lastPrompt);
    copyXmlToBinary (xml, destData);
}

void PluginProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    const auto xml = getXmlFromBinary (data, sizeInBytes);
    if (xml == nullptr || ! xml->hasTagName (kStateType))
        return;

    // A different schema version would mean different control semantics.
    jassert (xml->getIntAttribute (kAttrSchema, schema::kSchemaVersion) == schema::kSchemaVersion);

    juce::MemoryBlock paramState;
    if (paramState.fromBase64Encoding (xml->getStringAttribute (kAttrParams)))
        params.load (paramState.getData(), (int) paramState.getSize());

    lastPrompt = xml->getStringAttribute (kAttrPrompt);
    const juce::File model (xml->getStringAttribute (kAttrModel));
    if (model != decoderModelFile)
        loadDecoderModel (model.existsAsFile() ? model : juce::File(), nullptr);
}

juce::AudioProcessorEditor* PluginProcessor::createEditor()
{
    return new PluginEditor (*this);
}

} // namespace vae

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new vae::PluginProcessor();
}
