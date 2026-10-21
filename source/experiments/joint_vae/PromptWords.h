#pragma once

#include "XyPad.h"
#include <optional>
#include <utility>
#include <vector>

namespace vae::experimental
{
struct PromptReply
{
    std::optional<std::pair<JointLatent, int>> start;   // set when this message picks a new start
    std::vector<std::pair<juce::String, float>> nudges; // axis id ("s:brightness"), pad-position delta
    juce::StringArray ignored;
    juce::String message; // one line shown in the chat log
};

/** Pure text -> reply. The one function the text -> z encoder replaces later. */
PromptReply interpretPrompt (const juce::String& text, bool hasStart, const XyPadData&);
} // namespace vae::experimental
