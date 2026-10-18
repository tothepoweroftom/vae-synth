#pragma once

#include <juce_core/juce_core.h>

namespace vae::params
{

/** Describes a link between two parameters. See docs/specs/parameter-manager.md section 7. */
struct LinkRule
{
    enum class Mode
    {
        Mirror,     ///< The other member takes the same value.
        Offset,     ///< The offset (first - second) captured at enable time is preserved.
        Opposite,   ///< The other member moves by the negated delta of the moved member.
        Ordered     ///< Keeps first <= second - minGap, pushing the other member when needed.
    };

    /** The value space in which values, offsets and gaps are measured. */
    enum class Space
    {
        Raw,
        Normalised
    };

    juce::String first;
    juce::String second;
    juce::String enabledBy;     ///< Optional id of a parameter that enables the rule (>= 0.5 = on). Empty = always on.
    Mode mode = Mode::Mirror;
    Space space = Space::Raw;
    float minGap = 0.0f;        ///< Minimum gap for Ordered, in the rule's space.

    /** Id of the hidden value used to persist the captured Offset: "<first>__<second>__offset". */
    juce::String offsetStateId() const  { return first + "__" + second + "__offset"; }
};

} // namespace vae::params
