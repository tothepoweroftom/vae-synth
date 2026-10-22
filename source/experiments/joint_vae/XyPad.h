#pragma once

#include <array>
#include <juce_core/juce_core.h>
#include <limits>
#include <map>
#include <span>
#include <utility>
#include <vector>

namespace vae::experimental
{
// Independent of ControlSchema: these are the student's full Surge features.
using JointLatent = std::array<float, 32>;
using SoundCode = std::array<float, 48>;
inline constexpr const char* jointLabels[] = { "Bass",  "Lead",       "Pad",        "Keys",   "Pluck",
                                               "Synth", "Atmosphere", "Orchestral", "Unknown" };

// Port of the student's XY-pad page maths (vae/pitcher/xypad/page_v2.template.html). Engine-free.
struct LatentAxis
{
    juce::String id, kind /* "sound"|"dim"|"group" */, name, low, high, desc, key;
    juce::String tags;    // "group" rows: the tag line (knob count, |rho|, share of presets)
    bool unclear = false; // "group" rows: importance within 2 se, drawn grey
    JointLatent u {};
    float mean = 0, sd = 1, impact = 0;
    bool minor = false;
    std::vector<std::pair<juce::String, float>> sections;
};

struct XyPadData
{
    std::vector<LatentAxis> axes;                // sound axes first, then dims by impact (page order)
    std::map<juce::String, JointLatent> typical; // label name -> mean_z
    std::vector<JointLatent> presetZ;
    std::vector<int> presetLabel;
    juce::StringArray presetName;
    std::vector<float> bandHz;
    float dt = 0, f0 = 0;
    int holdFrames = 0;

    /** Fails without touching `out`. */
    static juce::Result load (const juce::File& xypadJson, XyPadData& out);
    const LatentAxis* find (const juce::String& id) const;
};

float axisTarget (const LatentAxis&, float pos);            // page tOf
float axisPosition (const LatentAxis&, const JointLatent&); // page posOf
/** Smallest change of z0 so that each axis reads its target. Dependent axes are dropped, not divided by. */
JointLatent constrain (const JointLatent& z0, std::span<const LatentAxis* const> axes, std::span<const float> targets);

struct SoundMeasures
{
    float brightness = 0, attack = 0, length = 0, noise = 0, movement = 0;
};

struct SoundPca
{
    std::vector<float> components /* 48 x F */, mean;
    float scale = 1;
    int frames = 0;
};
/** `sound_pca.npz`: a zip of float32/float64 C-order .npy files (components, mean, scale). */
juce::Result loadPca (const juce::File& soundPcaNpz, SoundPca&);
/** Page decodeA after the sound decoder: frames x 64 dB, peak-relative, floored at -80. */
std::vector<float> melFromSound (const SoundPca&, const SoundCode&);
/** Page measures(); `mel` is frames x bandHz.size(). */
SoundMeasures measures (const XyPadData&, std::span<const float> mel);

/** The page's random forest over the 123 knob codes; NaN = missing. */
struct CategoryForest
{
    struct Tree
    {
        std::vector<int> f, l, r;
        std::vector<float> t;
        std::vector<std::array<float, 8>> v;
    };
    std::vector<Tree> trees;
    /** `xypad.json` knobs.forest. */
    static juce::Result load (const juce::var&, CategoryForest&);
    std::array<float, 8> predict (std::span<const float> knobs123) const;
};

/** Knob <-> group maths (page unitOf/codeOf/groupPos/setGroup). */
struct KnobGroups
{
    struct Group
    {
        juce::String name, section, desc;
        std::vector<int> knob;
        std::vector<float> sign;
        float importance = 0, se = 0, active = 0;
        float meanAbsRho = std::numeric_limits<float>::quiet_NaN(); // NaN when the file has null
    };
    juce::StringArray names; // knobs.knobs[].name: student feature names
    std::vector<float> p5, p95;
    juce::StringArray ctrltype;      // Surge ctrltype per knob ("" = percent)
    std::vector<float> natLo, natHi; // natural_range per knob
    std::vector<Group> groups;
    /** `xypad.json` knobs (the object holding `knobs` and `groups`). */
    static juce::Result load (const juce::var&, KnobGroups&);
    int find (const juce::String& name) const;
    float position (int group, std::span<const float> knobs123) const; // NaN if none set
    void set (int group, float pos, std::span<float> knobs123) const;
    /** Page fmt: a knob code (0..1, NaN = off) in its unit, e.g. "354 ms", "440 Hz", "26%". */
    juce::String format (int knob, float code) const;
};
} // namespace vae::experimental
