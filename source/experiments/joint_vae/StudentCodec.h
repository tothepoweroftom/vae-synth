#pragma once

#include <juce_core/juce_core.h>
#include <map>
#include <span>
#include <vector>

namespace vae::experimental
{
// Port of the student's codec (vae/pitcher/core/codec.py: decode, plus inverter.data.full_row):
// decoded codes -> a write plan in surgepy patch paths ("scene/0/osc/1/p/3"). Engine-free;
// SurgePatchWriter executes it. Checked against codec-parity.json (tools/joint_vae/codec_parity.py).
struct StudentPlan
{
    struct Item
    {
        juce::String path;
        int value;
    };
    struct Flag
    {
        juce::String path, method; // Deactivated, TempoSync, Extend, Absolute, Deform
        int value;
    };
    struct Value
    {
        juce::String path, kind; // cont, ord, bool, cat
        float code;
        int natural; // cat only
    };
    struct Wavetable
    {
        int osc;
        juce::String name;
    };
    struct Routing
    {
        int source; // Surge modsources id
        juce::String path;
        float depth; // -1..1
    };
    std::vector<Item> types, subtypes;
    std::vector<Flag> flags;
    std::vector<Value> values;
    std::vector<Wavetable> wavetables;
    std::vector<Routing> routings;
};

class StudentCodec
{
public:
    /** schema.json and pruning.json of the bundle (copied there by export_bundle.py). */
    static juce::Result load (const juce::File& bundleDir, StudentCodec&);

    /** names/codes: the decoder's output in bundle feature order. */
    juce::Result plan (const juce::StringArray& names, std::span<const float> codes, StudentPlan&) const;

private:
    struct Feature
    {
        juce::String name, kind;
        juce::Array<juce::var> options;
        int routeSource = -1;
        juce::String routeDest, routeOscType;
    };
    std::vector<Feature> features; // schema order
    std::map<juce::String, int> index;
    std::vector<float> frozen; // full_row's defaults, NaN = not written
};
} // namespace vae::experimental
