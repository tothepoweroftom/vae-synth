#include "ml/SameSearch.h"
#include "schema/PresetFile.h"
#include <iostream>

int main (int argc, char** argv)
{
    if (argc != 4)
    {
        std::cerr << "Usage: vae_same_search native-bank.json reference.wav new-output-directory\n";
        return 1;
    }
    auto file = [] (const char* p) { return juce::File::getCurrentWorkingDirectory().getChildFile (p); };
    auto output = file (argv[3]);
    if (output.exists())
        return 1;
    vae::ml::SameSearch search;
    auto r = search.load (file (argv[1]));
    std::vector<vae::ml::SameSearch::Match> matches;
    if (r.wasOk())
        r = search.search (file (argv[2]), matches);
    if (r.failed())
    {
        std::cerr << r.getErrorMessage() << '\n';
        return 1;
    }
    if (output.createDirectory().failed())
        return 1;
    juce::Array<juce::var> results;
    for (const auto& m : matches)
    {
        auto preset = vae::schema::presetJson (m.controls, m.name, "lite");
        preset.getDynamicObject()->setProperty ("score", m.distance);
        preset.getDynamicObject()->setProperty ("note", m.note);
        results.add (preset);
        if (! output.getChildFile ("match-" + juce::String (results.size()) + ".json")
                  .replaceWithText (juce::JSON::toString (preset)))
            return 1;
        std::cout << m.name << " " << m.distance << '\n';
    }
    if (! output.getChildFile ("results.json").replaceWithText (juce::JSON::toString (results)))
        return 1;
    // Diagnostic tensors let tests compare exact preprocessing and embedding parity.
    std::vector<float> audio;
    r = vae::ml::SameSearch::preprocess (file (argv[2]), audio);
    vae::ml::SameSearch::Embedding embedding;
    if (r.wasOk())
        r = search.encode (audio, embedding);
    if (r.failed())
        return 1;
    return output.getChildFile ("input.raw").replaceWithData (audio.data(), audio.size() * sizeof (float))
                   && output.getChildFile ("embedding.raw")
                          .replaceWithData (embedding.data(), embedding.size() * sizeof (float))
               ? 0
               : 1;
}
