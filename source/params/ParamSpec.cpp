#include "params/ParamSpec.h"

#include <cmath>

namespace vae::params
{

juce::NormalisableRange<float> ParamSpec::frequencyRange (float lo, float hi)
{
    jassert (lo > 0.0f && hi > lo);
    juce::NormalisableRange<float> r { lo, hi };
    r.setSkewForCentre (std::sqrt (lo * hi));
    return r;
}

bool ParamSpec::rangeIsUnset() const noexcept
{
    return juce::exactlyEqual (range.start, 0.0f)
        && juce::exactlyEqual (range.end, 1.0f)
        && juce::exactlyEqual (range.interval, 0.0f)
        && juce::exactlyEqual (range.skew, 1.0f)
        && ! range.symmetricSkew;
}

ParamSpec ParamSpec::completed() const
{
    ParamSpec s = *this;

    if (s.name.isEmpty())
        s.name = s.id;

    switch (s.kind)
    {
        case ParamKind::Bool:
            s.range = juce::NormalisableRange<float> { 0.0f, 1.0f, 1.0f };
            break;

        case ParamKind::Choice:
            if (s.rangeIsUnset())
                s.range = juce::NormalisableRange<float> { 0.0f, (float) juce::jmax (1, s.choices.size() - 1), 1.0f };
            break;

        case ParamKind::Int:
            if (s.range.interval < 1.0f)
                s.range.interval = 1.0f;
            break;

        case ParamKind::Float:
            break;
    }

    s.defaultValue = s.range.snapToLegalValue (s.defaultValue);
    return s;
}

static juce::String withUnit (juce::String text, const juce::String& unit)
{
    if (unit.isNotEmpty())
        text << " " << unit;

    return text;
}

juce::String ParamSpec::valueToText (float raw) const
{
    if (toText)
        return toText (raw);

    switch (kind)
    {
        case ParamKind::Float:  return withUnit (juce::String (raw, 2), unit);
        case ParamKind::Int:    return withUnit (juce::String (juce::roundToInt (raw)), unit);
        case ParamKind::Bool:   return raw >= 0.5f ? "On" : "Off";
        case ParamKind::Choice:
        {
            const auto index = juce::roundToInt (raw);
            return juce::isPositiveAndBelow (index, choices.size()) ? choices[index] : juce::String (index);
        }
    }

    return {};
}

float ParamSpec::textToValue (const juce::String& text) const
{
    if (fromText)
        return fromText (text);

    const auto t = text.trim();

    switch (kind)
    {
        case ParamKind::Float:  return t.getFloatValue();
        case ParamKind::Int:    return (float) t.getIntValue();
        case ParamKind::Bool:
        {
            const auto lower = t.toLowerCase();
            return (lower == "on" || lower == "yes" || lower == "y" || lower == "true" || lower == "1") ? 1.0f : 0.0f;
        }
        case ParamKind::Choice:
        {
            const auto index = choices.indexOf (t, true);
            return (float) (index >= 0 ? index : t.getIntValue());
        }
    }

    return 0.0f;
}

} // namespace vae::params
