#pragma once

#include "schema/ControlSchema.h"
#include <juce_core/juce_core.h>

namespace vae::experimental
{
/** Surge patch data (an .fxp, or SurgePatchBuilder output; the XML from "<?xml") -> the 93 Lite controls.
    Line-by-line port of tools/surge_preset_to_vector.py (Patch + Converter + convert): same rules, same
    order, docs/SCHEMA.md. Fails without touching `out`. */
juce::Result liteControls (const juce::MemoryBlock& surgePatchData, schema::Snapshot& out);
} // namespace vae::experimental
