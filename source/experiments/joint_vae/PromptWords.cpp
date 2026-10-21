#include "PromptWords.h"
#include <algorithm>
#include <cmath>

namespace vae::experimental
{
namespace
{
struct Word
{
    const char* word;
    const char* axis;
    int sign;
};
const Word adjectives[] = {
    { "dark", "s:brightness", -1 },    { "darker", "s:brightness", -1 }, { "dull", "s:brightness", -1 },
    { "muted", "s:brightness", -1 },   { "warm", "s:brightness", -1 },   { "bright", "s:brightness", 1 },
    { "brighter", "s:brightness", 1 }, { "crisp", "s:brightness", 1 },   { "sharp", "s:brightness", 1 },
    { "airy", "s:brightness", 1 },     { "short", "s:length", -1 },      { "shorter", "s:length", -1 },
    { "tight", "s:length", -1 },       { "staccato", "s:length", -1 },   { "long", "s:length", 1 },
    { "longer", "s:length", 1 },       { "sustained", "s:length", 1 },   { "tail", "s:length", 1 },
    { "snappy", "s:attack", -1 },      { "punchy", "s:attack", -1 },     { "fast", "s:attack", -1 },
    { "plucky", "s:attack", -1 },      { "slow", "s:attack", 1 },        { "soft", "s:attack", 1 },
    { "swelling", "s:attack", 1 },     { "tonal", "s:noise", -1 },       { "clean", "s:noise", -1 },
    { "pure", "s:noise", -1 },         { "noisy", "s:noise", 1 },        { "breathy", "s:noise", 1 },
    { "gritty", "s:noise", 1 },        { "static", "s:movement", -1 },   { "still", "s:movement", -1 },
    { "steady", "s:movement", -1 },    { "moving", "s:movement", 1 },    { "evolving", "s:movement", 1 },
    { "wobbly", "s:movement", 1 },
};
const char* const stopWords[] = { "a",    "an",   "the", "it",   "make", "sound", "more",
                                  "less", "with", "and", "some", "bit",  "lot" };
const juce::String minus = juce::String (juce::CharPointer_UTF8 ("\xe2\x88\x92"));
const juce::String dot = juce::String (juce::CharPointer_UTF8 (" \xc2\xb7 "));

bool isIn (const juce::String& w, const char* const* list, size_t n)
{
    for (size_t i = 0; i < n; ++i)
        if (w == list[i])
            return true;
    return false;
}

// Lowercase runs of letters, digits and '-' (nouns like "303-style bass" keep their hyphen).
juce::StringArray tokenise (const juce::String& text)
{
    juce::StringArray out;
    juce::String cur;
    for (auto c : text.toLowerCase())
        if (juce::CharacterFunctions::isLetterOrDigit (c) || c == '-')
            cur += c;
        else if (cur.isNotEmpty())
            out.add (std::exchange (cur, {}));
    if (cur.isNotEmpty())
        out.add (cur);
    return out;
}
} // namespace

PromptReply interpretPrompt (const juce::String& text, bool hasStart, const XyPadData& data)
{
    PromptReply r;
    const auto tokens = tokenise (text);
    std::vector<bool> used ((size_t) tokens.size(), false);
    juce::StringArray parts;

    const auto noun = JointVae::labelFor (text);
    if (noun.noun.isNotEmpty())
    {
        const auto nounWords = tokenise (noun.noun);
        for (int i = 0; i + nounWords.size() <= tokens.size(); ++i)
        {
            bool match = true;
            for (int k = 0; k < nounWords.size() && match; ++k)
                match = tokens[i + k] == nounWords[k];
            if (match)
            {
                for (int k = 0; k < nounWords.size(); ++k)
                    used[(size_t) (i + k)] = true;
                break;
            }
        }
        const auto it = data.typical.find (jointLabels[noun.category]);
        if (it != data.typical.end())
            r.start = std::make_pair (it->second, noun.category);
        parts.add (jointLabels[noun.category]);
    }
    else if (! hasStart)
    {
        JointLatent mean {};
        int n = 0;
        for (size_t i = 0; i < data.presetZ.size() && i < data.presetLabel.size(); ++i)
            if (data.presetLabel[i] == 8)
            {
                for (size_t d = 0; d < mean.size(); ++d)
                    mean[d] += data.presetZ[i][d];
                ++n;
            }
        for (auto& v : mean)
            v /= (float) juce::jmax (1, n);
        r.start = std::make_pair (mean, 8);
        parts.add ("no sound type found");
    }

    float scale = 1.0f, flip = 1.0f;
    for (int i = 0; i < tokens.size(); ++i)
    {
        const auto& w = tokens[i];
        if (used[(size_t) i])
            continue;
        if (w == "slightly" || (w == "bit" && i > 0 && tokens[i - 1] == "a"))
            scale = 0.5f;
        else if (w == "much" || w == "very" || w == "way" || (w == "lot" && i > 0 && tokens[i - 1] == "a"))
            scale = 2.0f;
        else if (w == "less")
            flip = -1.0f;
        else if (w == "more")
            flip = 1.0f;
        else if (const auto* adj = std::find_if (std::begin (adjectives), std::end (adjectives),
                                                 [&] (const Word& a) { return w == a.word; });
                 adj != std::end (adjectives))
        {
            const float delta = 0.1f * scale * flip * (float) adj->sign;
            r.nudges.emplace_back (adj->axis, delta);
            const auto* axis = data.find (adj->axis);
            const int steps = juce::roundToInt (delta / 0.1f);
            auto s = (axis != nullptr ? axis->name : juce::String (adj->axis)) + " " + (steps < 0 ? minus : "+")
                     + juce::String (std::abs (steps)) + (std::abs (steps) == 1 ? " step" : " steps");
            if (juce::String (adj->axis) == "s:noise" || juce::String (adj->axis) == "s:movement")
                s += " (small effect)";
            parts.add (s);
            scale = flip = 1.0f;
        }
        else if (! isIn (w, stopWords, std::size (stopWords)))
            r.ignored.add (w);
    }

    if (! r.start && r.nudges.empty())
    {
        r.message = "Nothing to change. Try a sound type (bass, pad…) or brighter / longer / snappier.";
        return r;
    }
    if (r.ignored.size() > 0)
    {
        juce::StringArray q;
        for (auto& w : r.ignored)
            q.add ("'" + w + "'");
        parts.add ("didn't understand " + q.joinIntoString (", "));
    }
    r.message = parts.joinIntoString (dot);
    return r;
}
} // namespace vae::experimental
