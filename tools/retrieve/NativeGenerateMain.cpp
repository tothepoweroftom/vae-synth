#include "ml/StableAudioGenerator.h"
#include "ml/SameSearch.h"
#include "schema/PresetFile.h"
#include <iostream>
extern "C" void* vae_tokenizer_load (const char*);
extern "C" void vae_tokenizer_free (void*);
int main (int argc, char** argv)
{
    auto file = [] (const char* p) { return juce::File::getCurrentWorkingDirectory().getChildFile (p); };
    if (argc == 4 && juce::String (argv[1]) == "--tokenize")
    {
        auto* tokenizer = vae_tokenizer_load (argv[2]);
        std::array<float, 256> ids {}, mask {};
        auto result = vae::ml::StableAudioGenerator::tokenize (tokenizer, juce::String::fromUTF8 (argv[3]), ids, mask);
        vae_tokenizer_free (tokenizer);
        if (result.failed())
            return 1;
        for (auto id : ids)
            std::cout << int (id) << ' ';
        std::cout << '\n';
        for (auto value : mask)
            std::cout << int (value) << ' ';
        std::cout << '\n';
        return 0;
    }
    if (argc != 5 && argc != 6)
    {
        std::cerr << "Usage: vae_generate generator.json native-bank.json prompt new-output-directory [noise.raw]\n";
        return 1;
    }
    const auto output = file (argv[4]);
    if (output.exists() || output.createDirectory().failed())
        return 1;
    std::vector<float> noise;
    if (argc == 6)
    {
        juce::MemoryBlock bytes;
        if (! file (argv[5]).loadFileAsData (bytes) || bytes.getSize() % sizeof (float) != 0)
            return 1;
        noise.resize (bytes.getSize() / sizeof (float));
        std::memcpy (noise.data(), bytes.getData(), bytes.getSize());
    }
    vae::ml::StableAudioGenerator generator;
    vae::ml::SameSearch search;
    auto r = search.load (file (argv[2]));
    if (r.wasOk())
        r = generator.load (file (argv[1]));
    const auto start = juce::Time::getMillisecondCounterHiRes();
    if (r.wasOk())
        r = generator.generate (
            juce::String::fromUTF8 (argv[3]), 42, output.getChildFile ("reference.wav"), {},
            [] (juce::String message) { std::cout << message << std::endl; }, noise);
    std::vector<vae::ml::SameSearch::Match> matches;
    if (r.wasOk())
        r = search.search (output.getChildFile ("reference.wav"), matches);
    if (r.failed())
    {
        std::cerr << r.getErrorMessage() << '\n';
        return 1;
    }
    juce::Array<juce::var> results;
    for (const auto& m : matches)
    {
        auto preset = vae::schema::presetJson (m.controls, m.name, "lite");
        preset.getDynamicObject()->setProperty ("score", m.distance);
        results.add (preset);
        if (! output.getChildFile ("match-" + juce::String (results.size()) + ".json")
                  .replaceWithText (juce::JSON::toString (preset)))
            return 1;
        std::cout << m.name << " " << m.distance << '\n';
    }
    std::cout << "Generation + search seconds: " << (juce::Time::getMillisecondCounterHiRes() - start) / 1000 << '\n';
    return output.getChildFile ("results.json").replaceWithText (juce::JSON::toString (results)) ? 0 : 1;
}
