#pragma once

#include "ControlSchema.h"
#include <juce_core/juce_core.h>

namespace vae::schema
{
// Shared by the offline retriever and message-thread preset loader. Never used in processBlock.
inline juce::Result readPresetValues (const juce::var& json, Snapshot& result)
{
    const auto* values = json["values"].getArray();
    if (values == nullptr || values->size() != kControlCount)
        return juce::Result::fail ("Expected " + juce::String (kControlCount) + " control values");
    Snapshot candidate;
    for (int i = 0; i < kControlCount; ++i)
    {
        const auto& value = (*values)[i];
        if (! (value.isDouble() || value.isInt() || value.isInt64()))
            return juce::Result::fail ("Control values must be numbers");
        candidate[i] = float (double (value));
    }
    if (! candidate.isValid())
        return juce::Result::fail ("Invalid control value");
    result = candidate;
    return juce::Result::ok();
}

inline juce::var presetJson (const Snapshot& controls, const juce::String& name, const juce::String& engine)
{
    auto* obj = new juce::DynamicObject();
    obj->setProperty ("format", "vae-preset-1");
    obj->setProperty ("schema_version", kSchemaVersion);
    obj->setProperty ("name", name);
    obj->setProperty ("engine", engine);
    juce::Array<juce::var> values;
    for (auto v : controls.values)
        values.add (double (v));
    obj->setProperty ("values", values);
    return juce::var (obj);
}
} // namespace vae::schema
