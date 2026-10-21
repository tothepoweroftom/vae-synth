"""Feature kinds and naming rules for Step 1.3 (PIPELINE_VAE_SURGE.md, Step 1.3).

Kinds:
  cont  float parameter          -> (v - min) / (max - min), in [0, 1]
  ord   int with a meaningful order (counts, octaves) -> same scaling, round on decode
  cat   int without order (types, modes, shapes)   -> one-hot over the observed options
  bool  on/off                   -> 0 / 1
"""

# Int ctrltypes whose order means something (PIPELINE_VAE_SURGE.md, Step 1.3 table)
ORDINAL_CTRLTYPES = {
    "ct_pitch_octave", "ct_osccount", "ct_osccount_or_playcount", "ct_fmratio_int", "ct_pbdepth",
    "ct_polylimit", "ct_midikey", "ct_phaser_stages", "ct_ensemble_stages", "ct_vocoder_bandcount",
}
# Int ctrltypes that are names without order
CATEGORICAL_CTRLTYPES = {
    "ct_osctype", "ct_filtertype", "ct_filtersubtype", "ct_fxtype", "ct_wstype", "ct_lfotype",
    "ct_lfotrigmode", "ct_envmode", "ct_envshape", "ct_envshape_attack", "ct_fbconfig", "ct_fmconfig",
    "ct_oscroute", "ct_polymode", "ct_character", "ct_sineoscmode", "ct_sinefmlegacy",
    "ct_ringmod_sineoscmode", "ct_twist_engine", "ct_alias_wave", "ct_stringosc_excitation_model",
    "ct_wt2window", "ct_reverbshape", "ct_distortion_waveshape", "ct_flangermode", "ct_fxlfowave",
    "ct_fxlfowave_extended", "ct_reson_mode", "ct_mscodec", "ct_nimbusmode", "ct_nimbusquality",
    "ct_bonsai_sat_filter", "ct_bonsai_sat_mode", "ct_bonsai_noise_mode", "ct_vocoder_modulator_mode",
}

# Raw paths that never become features
EXCLUDED_GLOBAL = {
    "volume": "global performance volume; not reset on load, so load-order dependent (Appendix A.4)",
    "fx_bypass": "global performance setting; not reset on load",
    "polylimit": "polyphony limit, no effect on a single note; not reset on load",
    "scene_active": "used to pick the active scene, not a feature",
    "scenemode": "always single in the kept set",
    "splitpoint": "only used in split modes",
    "fx_disable": "bitmask, used for the FX masks instead of as a feature",
}
EXCLUDED_SCENE = {
    "pbrange_dn": "pitch-bend range, no effect on a held note",
    "pbrange_up": "pitch-bend range, no effect on a held note",
    "send_level": "moved into the FX blocks (send_level of the FX that sits on that send)",
}

OSC_TYPES = ["classic", "sine", "wavetable", "shnoise", "audioinput", "fm3", "fm2", "window",
             "modern", "string", "twist", "alias"]  # osc_type enum, SurgeStorage.h:296
FX_TYPES = ["off", "delay", "reverb", "phaser", "rotary", "distortion", "eq", "freqshift", "conditioner",
            "chorus", "vocoder", "reverb2", "flanger", "ringmod", "airwindows", "neuron", "geq11",
            "resonator", "chow", "exciter", "ensemble", "combulator", "nimbus", "tape", "treemonster",
            "waveshaper", "mstool", "spring", "bonsai", "audioinput", "floatydelay", "convolution"]
EXCLUDED_FX = {
    "airwindows": "not available in the Rack build (Section 10), sub-algorithm list is its own world",
    "vocoder": "needs an audio input",
    "audioinput": "needs an audio input",
    "convolution": "needs an impulse-response file",
}

# FX slot layout (fxslot_positions, SurgeStorage.h:344-362)
INSERT_SLOTS = {0: [0, 1, 8, 9], 1: [2, 3, 10, 11]}  # per scene
SEND_SLOTS = {4: 0, 5: 1, 12: 2, 13: 3}              # slot -> send index
GLOBAL_SLOTS = [6, 7, 14, 15]
FX_PLACEMENTS = ["insert", "send", "global"]

WAVETABLE_TOP_N = 32           # wavetable name categorical: top N names + "other"

# Standard modulation routings (template option B, Round 12): the routings used by >= 5% of train
# presets whose source is audible in a plain note render (no mod wheel / macros / aftertouch) and
# whose destination has a fixed meaning (no oscillator knobs p0-p6, no FX knobs). Each becomes a
# feature "route/<source>/<destination>" = depth (normalised -1..1, 0 = not routed).
STANDARD_ROUTES = [
    ("Velocity", "filterunit/0/cutoff"),     # 22.2% of train presets
    ("Voice LFO 1", "pitch"),                # 17.2%
    ("Velocity", "filterunit/1/cutoff"),     # 12.0%
    ("Velocity", "adsr/0/a"),                # 11.4%
    ("Velocity", "filterunit/0/envmod"),     # 10.0%
    ("Voice LFO 2", "pitch"),                # 8.2%
    ("Voice LFO 1", "filterunit/0/cutoff"),  # 7.5%
    ("Voice LFO 1", "osc/0/pitch"),          # 6.1%
    ("Voice LFO 2", "filterunit/0/cutoff"),  # 5.7%
    ("Velocity", "fm_depth"),                # 5.7%
    ("Voice LFO 1", "level_o1"),             # 5.7%
    ("Velocity", "filterunit/0/resonance"),  # 5.3%
    ("Voice LFO 3", "osc/1/pitch"),          # 5.0%
]


def route_feature(source, dest):
    return "route/" + source.lower().replace(" ", "_") + "/" + dest
WAVETABLE_OSC_TYPES = {2, 7}   # wavetable, window

# Per-parameter flags (Appendix A.3) -> feature suffix and kind
FLAGS = {"deactivated": ("off", "bool"), "temposync": ("sync", "bool"), "extend": ("ext", "bool"),
         "absolute": ("abs", "bool"), "deform": ("deform", "cat")}


def kind_of(valtype, ctrltype):
    """Feature kind for one raw parameter, or None if it should not become a feature."""
    if ctrltype == "ct_none":
        return None
    if valtype == "float":
        return "cont"
    if valtype == "bool":
        return "bool"
    if ctrltype in ORDINAL_CTRLTYPES:
        return "ord"
    if ctrltype in CATEGORICAL_CTRLTYPES:
        return "cat"
    return "cat_unlisted"  # int ctrltype in neither list: treated as cat, reported for review
