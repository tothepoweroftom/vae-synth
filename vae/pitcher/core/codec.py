"""Decode a feature vector (one row of codes, Step 1.3) into a write plan for Surge (Step 7.1).

The plan is synth-specific (Surge parameter paths) but engine-free: SurgeIO.write_plan executes
it in the safe order of Step 7.2 (types -> settle -> subtypes + flags -> settle -> values).

Canonical -> Surge:
  - everything is written into scene A (scene 0), with scene_active = 0 and scenemode = single
  - osc knobs scene/osc/<o>/<type>/p<i> -> scene/0/osc/<o>/p/<i> (only for the decoded osc type)
  - filter model "t:s"                    -> filterunit type t (phase 1), subtype s (phase 2)
  - standard routings route/<source>/<dest> -> setModDepth01 (depth 0 = no routing), after values
  - FX blocks fx/<type>/...               -> a slot per present effect, at its stored position: insert -> A insert slots,
    send -> send slots (+ scene A send level, return level), global -> global slots; in FX type
    order. All other FX slots are set to off. (The original slot order is not in the features.)
"""

import math
import re
from dataclasses import dataclass, field

from pitcher.core.schema import FX_PLACEMENTS, FX_TYPES, OSC_TYPES, WAVETABLE_OSC_TYPES

FLAG_METHODS = {"off": "Deactivated", "sync": "TempoSync", "ext": "Extend", "abs": "Absolute",
                "deform": "Deform"}
SLOTS = {"insert": [0, 1, 8, 9], "send": [4, 5, 12, 13], "global": [6, 7, 14, 15]}
SEND_INDEX = {4: 0, 5: 1, 12: 2, 13: 3}
TYPE_FIELDS = {"type", "shape", "fm_switch", "filterblock_configuration"}  # written in phase 1
# Wavetable / window oscillator without a named table ("other" or not given): the oscillator would
# keep whatever table the engine loaded last (the previous render's), so a fixed one is loaded.
# The most common named table in the presets.
FALLBACK_WAVETABLE = "Sine Octaves"


@dataclass
class Plan:
    types: list = field(default_factory=list)     # (path, natural)                 phase 1
    subtypes: list = field(default_factory=list)  # (path, natural)                 phase 2
    flags: list = field(default_factory=list)     # (feature, path, method, value)  phase 2
    values: list = field(default_factory=list)    # (feature, path, kind, code, natural or None) phase 3
    wavetables: list = field(default_factory=list)  # (scene, osc, name)            after phase 1
    routings: list = field(default_factory=list)    # (feature, source_id, path, depth -1..1) last
    fx_slots: dict = field(default_factory=dict)  # fx type name -> slot
    warnings: list = field(default_factory=list)


def _isnan(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


def decode(codes, schema, slot_map=None, wavetable_fallback=True):
    """codes: {feature: code or NaN}; schema: parsed schema.json. Returns a Plan.
    slot_map: optional {fx type name: slot} to place effects in given slots (diagnostics).
    wavetable_fallback: load FALLBACK_WAVETABLE where no table is named (template rebuilds); False
    when writing onto the preset's own patch (round-trip A), whose table is then the right one."""
    feats = {f["name"]: f for f in schema["features"]}
    plan = Plan()

    def opt(name):
        c = codes.get(name)
        return None if _isnan(c) else feats[name]["options"][int(round(c))]

    # --- global scene setup -------------------------------------------------------------------
    plan.types += [("scenemode", 0), ("scene_active", 0), ("fx_disable", 0)]

    # --- FX slot assignment -----------------------------------------------------------------------
    used = {k: [] for k in SLOTS}
    for t, name in enumerate(FX_TYPES):
        if codes.get(f"fx/{name}/present", 0) != 1:
            continue
        place = opt(f"fx/{name}/placement") or "insert"
        if slot_map and name in slot_map:
            used[place].append(slot_map[name])
            plan.fx_slots[name] = slot_map[name]
            plan.types.append((f"fx/{slot_map[name]}/type", t))
            continue
        free = [s for s in SLOTS[place] if s not in used[place]]
        if not free:
            plan.warnings.append(f"no free {place} slot for {name}")
            continue
        pos = opt(f"fx/{name}/position")  # chain position within the group of 4 (send: which send)
        wanted = SLOTS[place][int(pos)] if pos is not None else None
        if wanted is not None and wanted not in free:
            plan.warnings.append(f"{name}: {place} position {pos} taken, using the next free slot")
        slot = wanted if wanted in free else free[0]
        used[place].append(slot)
        plan.fx_slots[name] = slot
        plan.types.append((f"fx/{slot}/type", t))
    taken = {s for v in plan.fx_slots.values() for s in [v]}
    plan.types += [(f"fx/{s}/type", 0) for s in range(16) if s not in taken]

    # --- oscillator types (needed to route the per-type knob blocks) ------------------------------
    osc_type = {}
    for o in range(3):
        t = opt(f"scene/osc/{o}/type")
        if t is not None:
            osc_type[o] = int(t)

    # --- every feature ------------------------------------------------------------------------------
    for name, c in codes.items():
        if _isnan(c) or name not in feats:
            continue
        f = feats[name]
        base, _, suffix = name.rpartition(".")
        if suffix in FLAG_METHODS and base:
            path = _path_for(base, plan, osc_type)
            if path:
                val = bool(c >= 0.5) if f["kind"] == "bool" else int(f["options"][int(round(c))])
                plan.flags.append((name, path, FLAG_METHODS[suffix], val))
            continue

        if name.startswith("route/"):                 # routing matrix (Rounds 12, 15)
            r = f.get("route") or {}
            if r.get("source_id") is None:
                continue
            if r.get("osc_type"):                      # oscillator-knob routing: only for that osc type
                o = int(r["dest"].split("/")[1])
                if osc_type.get(o) is None or OSC_TYPES[osc_type[o]] != r["osc_type"]:
                    continue
            plan.routings.append((name, int(r["source_id"]), "scene/0/" + r["dest"], 2.0 * float(c) - 1.0))
            continue

        m = re.fullmatch(r"scene/filterunit/(\d)/model", name)
        if m:
            ftype, sub = map(int, f["options"][int(round(c))].split(":"))
            plan.types.append((f"scene/0/filterunit/{m[1]}/type", ftype))
            plan.subtypes.append((f"scene/0/filterunit/{m[1]}/subtype", sub))
            continue
        m = re.fullmatch(r"scene/osc/(\d)/(\w+)/wt", name)
        if m:
            o = int(m[1])
            if osc_type.get(o) is not None and OSC_TYPES[osc_type[o]] == m[2]:  # only the decoded type's table
                wt = f["options"][int(round(c))]
                if wt != "other":
                    plan.wavetables.append((0, o, wt))
                elif wavetable_fallback:
                    plan.wavetables.append((0, o, FALLBACK_WAVETABLE))
            continue
        if name.startswith("fx/") and name.split("/")[2] in ("present", "placement", "position"):
            continue

        path = _path_for(name, plan, osc_type)
        if path is None:
            continue
        kind = f["kind"]
        nat = f["options"][int(round(c))] if kind == "cat" else None
        if kind == "cat" and path.rsplit("/", 1)[-1] in TYPE_FIELDS:
            plan.types.append((path, nat))
        else:
            plan.values.append((name, path, kind, float(c), nat))
    for o, t in osc_type.items():
        if wavetable_fallback and t in WAVETABLE_OSC_TYPES and not any(w[1] == o for w in plan.wavetables):
            plan.wavetables.append((0, o, FALLBACK_WAVETABLE))
    return plan


def _path_for(name, plan, osc_type):
    """Surge parameter path for a canonical feature name (scene A), or None if not writable."""
    if name == "character":
        return "character"
    m = re.fullmatch(r"scene/filterunit/(\d)/model", name)  # flags of the filter type (e.g. bypass)
    if m:
        return f"scene/0/filterunit/{m[1]}/type"
    m = re.fullmatch(r"fx/(\w+)/(p(\d+)|send_level|return_level)", name)
    if m:
        slot = plan.fx_slots.get(m[1])
        if slot is None:
            return None
        if m[3] is not None:
            return f"fx/{slot}/p/{m[3]}"
        if slot not in SEND_INDEX:
            return None
        return f"scene/0/send_level/{SEND_INDEX[slot]}" if m[2] == "send_level" else f"fx/{slot}/return_level"
    m = re.fullmatch(r"scene/osc/(\d)/(\w+)/p(\d)", name)
    if m:
        o = int(m[1])
        if osc_type.get(o) is None or OSC_TYPES[osc_type[o]] != m[2]:
            return None
        return f"scene/0/osc/{o}/p/{m[3]}"
    if name.startswith("scene/"):
        return "scene/0/" + name[len("scene/"):]
    return None


def placement_of(slot):
    return next(k for k, v in SLOTS.items() if slot in v) if slot in sum(SLOTS.values(), []) else None


assert set(SLOTS) == set(FX_PLACEMENTS)
