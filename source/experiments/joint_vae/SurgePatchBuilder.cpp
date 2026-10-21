#include "SurgePatchBuilder.h"

// Surge's headers are not written for JUCE's recommended warning set.
JUCE_BEGIN_IGNORE_WARNINGS_GCC_LIKE ("-Wsign-conversion", "-Wfloat-equal", "-Wpedantic", "-Wshadow", "-Wconversion",
                                     "-Wfloat-conversion", "-Wold-style-cast", "-Wzero-as-null-pointer-constant",
                                     "-Wunused-parameter", "-Wswitch-enum", "-Wimplicit-int-float-conversion",
                                     "-Wshorten-64-to-32", "-Wextra-semi", "-Wcast-align", "-Wunused-value",
                                     "-Wunused-variable", "-Wunused-local-typedefs", "-Wstrict-aliasing",
                                     "-Wsign-compare")
JUCE_BEGIN_IGNORE_WARNINGS_MSVC (4244 4267 4100 4456 4458)
#include "SurgeSynthesizer.h"
JUCE_END_IGNORE_WARNINGS_MSVC
JUCE_END_IGNORE_WARNINGS_GCC_LIKE

#include <cmath>
#include <limits>
#include <set>

namespace vae::experimental
{
namespace
{
struct SilentPluginLayer final : SurgeSynthesizer::PluginLayer
{
    void surgeParameterUpdated (const SurgeSynthesizer::ID&, float) override {}
    void surgeMacroUpdated (long, float) override {}
};

float natural (const Parameter& p, const pdata& v)
{
    return p.valtype == vt_float ? v.f : p.valtype == vt_int ? float (v.i) : v.b ? 1.0f : 0.0f;
}

/** surgepy getPatch() paths (src/surge-python/surgepy.cpp asDict, with the student's send_level fix). */
std::map<juce::String, Parameter*> patchPaths (SurgePatch& patch)
{
    std::map<juce::String, Parameter*> m;
    auto& sc = patch.scene[0];
    const juce::String s = "scene/0/";
    for (int o = 0; o < n_oscs; ++o)
    {
        const auto o_ = s + "osc/" + juce::String (o) + "/";
        auto& osc = sc.osc[o];
        m[o_ + "type"] = &osc.type;
        m[o_ + "pitch"] = &osc.pitch;
        m[o_ + "octave"] = &osc.octave;
        m[o_ + "keytrack"] = &osc.keytrack;
        m[o_ + "retrigger"] = &osc.retrigger;
        for (int i = 0; i < n_osc_params; ++i)
            m[o_ + "p/" + juce::String (i)] = &osc.p[i];
    }
    for (int f = 0; f < 2; ++f)
    {
        const auto f_ = s + "filterunit/" + juce::String (f) + "/";
        auto& fu = sc.filterunit[f];
        m[f_ + "type"] = &fu.type;
        m[f_ + "subtype"] = &fu.subtype;
        m[f_ + "cutoff"] = &fu.cutoff;
        m[f_ + "resonance"] = &fu.resonance;
        m[f_ + "envmod"] = &fu.envmod;
        m[f_ + "keytrack"] = &fu.keytrack;
    }
    m[s + "wsunit/type"] = &sc.wsunit.type;
    m[s + "wsunit/drive"] = &sc.wsunit.drive;
    for (int e = 0; e < 2; ++e)
    {
        const auto e_ = s + "adsr/" + juce::String (e) + "/";
        auto& a = sc.adsr[e];
        for (auto [name, p] : { std::pair { "a", &a.a },
                                { "d", &a.d },
                                { "s", &a.s },
                                { "r", &a.r },
                                { "a_s", &a.a_s },
                                { "d_s", &a.d_s },
                                { "r_s", &a.r_s },
                                { "mode", &a.mode } })
            m[e_ + name] = p;
    }
    for (int l = 0; l < n_lfos; ++l)
    {
        const auto l_ = s + "lfo/" + juce::String (l) + "/";
        auto& lfo = sc.lfo[l];
        for (auto [name, p] : { std::pair { "rate", &lfo.rate },
                                { "shape", &lfo.shape },
                                { "start_phase", &lfo.start_phase },
                                { "magnitude", &lfo.magnitude },
                                { "deform", &lfo.deform },
                                { "trigmode", &lfo.trigmode },
                                { "unipolar", &lfo.unipolar },
                                { "delay", &lfo.delay },
                                { "hold", &lfo.hold },
                                { "attack", &lfo.attack },
                                { "decay", &lfo.decay },
                                { "sustain", &lfo.sustain },
                                { "release", &lfo.release } })
            m[l_ + name] = p;
    }
    for (auto [name, p] : { std::pair { "pitch", &sc.pitch },
                            { "octave", &sc.octave },
                            { "fm_depth", &sc.fm_depth },
                            { "fm_switch", &sc.fm_switch },
                            { "drift", &sc.drift },
                            { "noise_colour", &sc.noise_colour },
                            { "keytrack_root", &sc.keytrack_root },
                            { "level_o1", &sc.level_o1 },
                            { "mute_o1", &sc.mute_o1 },
                            { "solo_o1", &sc.solo_o1 },
                            { "route_o1", &sc.route_o1 },
                            { "level_o2", &sc.level_o2 },
                            { "mute_o2", &sc.mute_o2 },
                            { "solo_o2", &sc.solo_o2 },
                            { "route_o2", &sc.route_o2 },
                            { "level_o3", &sc.level_o3 },
                            { "mute_o3", &sc.mute_o3 },
                            { "solo_o3", &sc.solo_o3 },
                            { "route_o3", &sc.route_o3 },
                            { "level_ring_12", &sc.level_ring_12 },
                            { "mute_ring_12", &sc.mute_ring_12 },
                            { "solo_ring_12", &sc.solo_ring_12 },
                            { "route_ring_12", &sc.route_ring_12 },
                            { "level_ring_23", &sc.level_ring_23 },
                            { "mute_ring_23", &sc.mute_ring_23 },
                            { "solo_ring_23", &sc.solo_ring_23 },
                            { "route_ring_23", &sc.route_ring_23 },
                            { "level_noise", &sc.level_noise },
                            { "mute_noise", &sc.mute_noise },
                            { "solo_noise", &sc.solo_noise },
                            { "route_noise", &sc.route_noise },
                            { "level_pfg", &sc.level_pfg },
                            { "vca_level", &sc.vca_level },
                            { "pbrange_dn", &sc.pbrange_dn },
                            { "pbrange_up", &sc.pbrange_up },
                            { "vca_velsense", &sc.vca_velsense },
                            { "polymode", &sc.polymode },
                            { "portamento", &sc.portamento },
                            { "volume", &sc.volume },
                            { "pan", &sc.pan },
                            { "width", &sc.width },
                            { "f2_cutoff_is_offset", &sc.f2_cutoff_is_offset },
                            { "f2_link_resonance", &sc.f2_link_resonance },
                            { "feedback", &sc.feedback },
                            { "filterblock_configuration", &sc.filterblock_configuration },
                            { "filter_balance", &sc.filter_balance },
                            { "lowcut", &sc.lowcut } })
        m[s + name] = p;
    for (int i = 0; i < n_send_slots; ++i)
        m[s + "send_level/" + juce::String (i)] = &sc.send_level[i];
    for (int f = 0; f < n_fx_slots; ++f)
    {
        const auto f_ = "fx/" + juce::String (f) + "/";
        m[f_ + "type"] = &patch.fx[f].type;
        m[f_ + "return_level"] = &patch.fx[f].return_level;
        for (int i = 0; i < n_fx_params; ++i)
            m[f_ + "p/" + juce::String (i)] = &patch.fx[f].p[i];
    }
    for (auto [name, p] : { std::pair { "scene_active", &patch.scene_active },
                            { "scenemode", &patch.scenemode },
                            { "splitpoint", &patch.splitpoint },
                            { "volume", &patch.volume },
                            { "polylimit", &patch.polylimit },
                            { "fx_bypass", &patch.fx_bypass },
                            { "fx_disable", &patch.fx_disable },
                            { "character", &patch.character } })
        m[name] = p;
    return m;
}
} // namespace

struct SurgePatchBuilder::Impl
{
    SilentPluginLayer layer;
    std::unique_ptr<SurgeSynthesizer> synth;
    juce::File data;
    std::map<juce::String, Parameter*> paths;
    std::vector<FxStorage> pristineFx;             // FX slots exactly as a fresh synth has them
    std::map<juce::String, juce::File> wavetables; // display name -> file, factory folder first

    void settle()
    {
        // SurgeIO.settle: a few blocks apply queued osc / FX type changes and wavetable loads
        for (int i = 0; i < 8; ++i)
            synth->process();
    }

    void setVal (Parameter& p, float f)
    {
        synth->setParameter01 (synth->idForParameter (&p), p.value_to_normalized (f));
    }

    void setInt (Parameter& p, float target)
    {
        // SurgeIO.set_int: hit Surge's intScaledToFloat exactly (plain linear is off by one above ~100)
        const float lo = natural (p, p.val_min), range = natural (p, p.val_max) - lo;
        if (p.valtype != vt_int || range <= 0)
            setVal (p, target);
        else
            setVal (p, lo + range * (0.005f + 0.99f * (target - lo) / range));
    }

    Parameter* find (const juce::String& path) const
    {
        const auto it = paths.find (path);
        return it == paths.end() ? nullptr : it->second;
    }
};

SurgePatchBuilder::SurgePatchBuilder (const juce::File& surgeDataDir)
    : impl (std::make_unique<Impl>())
{
    impl->data = surgeDataDir;
    impl->synth = std::make_unique<SurgeSynthesizer> (&impl->layer, surgeDataDir.getFullPathName().toStdString());
    impl->synth->setSamplerate (48000.0f);
    impl->synth->time_data.tempo = 120.0;
    impl->paths = patchPaths (impl->synth->storage.getPatch());
    for (int f = 0; f < n_fx_slots; ++f)
        impl->pristineFx.push_back (impl->synth->storage.getPatch().fx[f]);
    for (auto folder : { "wavetables", "wavetables_3rdparty" })
        for (const auto& f : surgeDataDir.getChildFile (folder).findChildFiles (juce::File::findFiles, true))
            if (f.hasFileExtension ("wav;wt"))
                impl->wavetables.emplace (f.getFileNameWithoutExtension(), f); // emplace keeps the first
}

SurgePatchBuilder::~SurgePatchBuilder() = default;

juce::Result SurgePatchBuilder::build (const StudentPlan& plan, juce::MemoryBlock& output)
{
    auto& synth = *impl->synth;
    // pruning.json "template": the reference every frozen "reference" feature keeps
    const auto templ = impl->data.getChildFile ("patches_factory/Templates/Init Saw.fxp");
    if (! templ.existsAsFile())
        return juce::Result::fail ("Surge template missing: " + templ.getFullPathName());
    // Surge keeps per-slot FX parameter defaults/ranges from the last effect type (loadFx retypes the
    // params in place), and a template load copies those into fxsync. Unwritten params of a retyped
    // slot then inherit the previous build's values, so put the slots back first.
    for (int f = 0; f < n_fx_slots; ++f)
        synth.storage.getPatch().fx[f] = impl->pristineFx[(size_t) f];
    if (! synth.loadPatchByPath (templ.getFullPathName().toRawUTF8(), -1, "Init Saw"))
        return juce::Result::fail ("Surge could not load the template");
    impl->settle();
    // shortcut: write_plan's phase 0 (drop routings onto FX slots that change type) is skipped:
    // Init Saw has no routings. Port it if the template changes.

    auto required = [&] (const juce::String& path) -> Parameter&
    {
        if (auto* p = impl->find (path))
            return *p;
        throw std::runtime_error (("Unknown Surge path " + path).toStdString());
    };
    try
    {
        for (const auto& [path, value] : plan.types) // phase 1: types and switches
        {
            if (path == "fx_disable")
                synth.storage.getPatch().fx_disable.val.i = value; // as surgepy setFxDisable
            else
                impl->setInt (required (path), (float) value);
        }
        impl->settle();
        for (const auto& w : plan.wavetables)
        {
            const auto it = impl->wavetables.find (w.name);
            if (it == impl->wavetables.end())
                continue; // as SurgeIO.wavetable_path: an unknown name keeps the current table
            auto& osc = synth.storage.getPatch().scene[0].osc[w.osc];
            synth.storage.load_wt (it->second.getFullPathName().toStdString(), &osc.wt, &osc);
        }
        if (! plan.wavetables.empty())
            impl->settle();
        for (const auto& [path, value] : plan.subtypes) // phase 2: subtypes + flags
            impl->setInt (required (path), (float) value);
        // Flags belong to the parameter slot and survive type changes: clear, then apply the planned ones
        std::set<juce::String> touched;
        for (const auto& v : plan.values)
            touched.insert (v.path);
        for (const auto& f : plan.flags)
            touched.insert (f.path);
        for (const auto& path : touched)
        {
            auto& p = required (path);
            p.temposync = false;
            p.set_extend_range (false);
            p.absolute = false;
        }
        auto applyFlags = [&] (bool tempoSync)
        {
            for (const auto& f : plan.flags)
            {
                if ((f.method == "TempoSync") != tempoSync)
                    continue;
                auto& p = required (f.path);
                if (f.method == "Extend" && p.can_extend_range())
                    p.set_extend_range (f.value != 0);
                else if (f.method == "Absolute" && p.can_be_absolute())
                    p.absolute = f.value != 0;
                else if (f.method == "Deform" && p.has_deformoptions())
                    p.deform_type = f.value;
                else if (f.method == "Deactivated" && p.can_deactivate())
                    p.deactivated = f.value != 0;
                else if (f.method == "TempoSync" && p.can_temposync())
                    p.temposync = f.value != 0;
            }
        };
        // flags before values, except temposync: writing a value with temposync on snaps it to the grid
        applyFlags (false);
        impl->settle();
        for (const auto& v : plan.values) // phase 3: values, with live ranges
        {
            auto& p = required (v.path);
            const float lo = natural (p, p.val_min), hi = natural (p, p.val_max);
            if (v.kind == "cont")
                impl->setVal (p, lo + v.code * (hi - lo));
            else if (v.kind == "ord")
                impl->setInt (p, std::round (lo + v.code * (hi - lo)));
            else if (v.kind == "bool")
                impl->setVal (p, v.code >= 0.5f ? 1.0f : 0.0f);
            else
                impl->setInt (p, (float) v.natural);
        }
        applyFlags (true);
        for (const auto& r : plan.routings)
            synth.setModDepth01 (synth.idForParameter (&required (r.path)).getSynthSideId(), (modsources) r.source, 0,
                                 0, r.depth);
    }
    catch (const std::exception& e)
    {
        return juce::Result::fail (e.what());
    }
    impl->settle();
    void* data = nullptr;
    const auto size = synth.saveRaw (&data); // buffer owned by the patch
    if (data == nullptr || size == 0)
        return juce::Result::fail ("Surge returned an empty patch");
    output.replaceAll (data, size);
    return juce::Result::ok();
}

float SurgePatchBuilder::value (const juce::String& path) const
{
    const auto* p = impl->find (path);
    return p == nullptr ? std::numeric_limits<float>::quiet_NaN() : natural (*p, p->val);
}
} // namespace vae::experimental
