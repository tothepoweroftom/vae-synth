"""Step 1.3: raw long-format table -> canonical features, masks, schema, model-ready matrix.

Reads   data/presets.parquet, data/raw_values.parquet, data/routings.parquet (from pitcher/step1_dataset/extract.py)
Writes  data/schema.json        one entry per feature: kind, options, ranges, sources, counts
        data/codes.parquet      one row per preset, one column per feature: 0-1 value (cont/ord),
                                0/1 (bool) or option index (cat); NaN where the feature is inactive
        data/masks.parquet      same shape: 1 = audible/active for this preset, 0 = masked
        data/defaults.parquet   same shape: code of Surge's default value for the loaded type (Step 2.1)
        data/dataset.parquet    model-ready: cont/ord/bool columns + one-hot columns for cat
                                features; masked entries filled (cont: default, bool/one-hot: 0)

Canonical layout (PIPELINE_VAE_SURGE.md Step 1.3):
  - only the active scene, under "scene/..." (scene index removed)
  - oscillator knobs per oscillator type: scene/osc/<o>/<type>/p<i>
  - FX per effect type, not per slot: fx/<type>/{present,placement,send_level,return_level,p<i>}
  - filter type + subtype as one categorical "model"
  - per-parameter flags (Appendix A.3) as "<feature>.<off|sync|ext|abs|deform>"

Run:  .venv/bin/python -m pitcher.step1_dataset.encode
"""

import json
import re
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from pitcher.core.schema import (EXCLUDED_FX, EXCLUDED_GLOBAL, EXCLUDED_SCENE, FLAGS, FX_PLACEMENTS, FX_TYPES,
                    GLOBAL_SLOTS, INSERT_SLOTS, OSC_TYPES, SEND_SLOTS, WAVETABLE_OSC_TYPES,
                    WAVETABLE_TOP_N, STANDARD_ROUTES, kind_of, route_feature)
from pitcher.core.surge_io import PITCHER_ROOT

DATA = PITCHER_ROOT / "data"
_matrix_file = DATA / "routing_matrix.json"
ROUTING_MATRIX = (json.loads(_matrix_file.read_text()) if _matrix_file.exists()
                  else [dict(source=s, dest=d, osc_type=None, source_id=None) for s, d in STANDARD_ROUTES])
RAW_COLS = ["path", "value", "min", "max", "default", "valtype", "ctrltype"] + list(FLAGS)


def lfo_index(source):
    m = re.fullmatch(r"(Voice|Scene) LFO (\d)", source)
    if not m:
        return None
    return int(m[2]) - 1 + (6 if m[1] == "Scene" else 0)


class PresetView:
    """Raw values of one preset plus the 'is it audible' logic used for the masks."""

    def __init__(self, rows, active_scene, routings):
        self.r = rows  # path -> dict(RAW_COLS)
        self.sc = active_scene
        self.S = f"scene/{active_scene}"
        self.routings = routings
        self.used_lfos = {lfo_index(s) for s, scope, scene, src_scene, *_ in routings
                          if (scope in ("scene", "voice") and scene == active_scene)
                          or (scope == "global" and src_scene == active_scene)} - {None}
        # parameters that something modulates: a level of 0 can still be raised (macro, LFO, ...)
        self.modulated = {r[4] for r in routings if r[4]}

    def route_depth(self, source, rel):
        """(normalised depth -1..1, source id) of a scene/voice routing in the active scene; (0, None) if absent."""
        for s, scope, scene, _, dest, depth01, source_id in self.routings:
            if s == source and scope in ("scene", "voice") and scene == self.sc and dest == f"{self.S}/{rel}":
                return float(depth01), int(source_id)
        return 0.0, None

    def gt0(self, path):
        """Base value above 0, or raised by modulation."""
        return self.v(path) > 0 or path in self.modulated

    def v(self, path):
        return self.r[path]["value"]

    def sv(self, rel):
        return self.v(f"{self.S}/{rel}")

    # --- mixer: what reaches the filters --------------------------------------------------
    def audible(self, src):
        solos = [self.sv(f"solo_{s}") for s in ("o1", "o2", "o3", "noise", "ring_12", "ring_23")]
        on = self.sv(f"solo_{src}") if any(solos) else not self.sv(f"mute_{src}")
        return bool(on) and self.gt0(f"{self.S}/level_{src}")

    def osc_active(self, o):
        if self.audible(f"o{o + 1}"):
            return True
        if o in (0, 1) and self.audible("ring_12"):
            return True
        if o in (1, 2) and self.audible("ring_23"):
            return True
        fm = int(self.sv("fm_switch"))  # 0 off, 1 2->1, 2 3->2->1, 3 2&3->1 (SurgeStorage.h:543)
        return (fm == 1 and o == 1) or (fm in (2, 3) and o in (1, 2))

    def osc_type(self, o):
        return int(self.sv(f"osc/{o}/type"))

    # --- FX: which slots are heard -----------------------------------------------------------
    def heard_fx(self):
        """[(slot, fx type index, placement, send index or None)] in slot order."""
        out, disabled = [], int(self.v("fx_disable"))
        for slot in range(16):
            t = int(self.v(f"fx/{slot}/type"))
            if t == 0 or disabled & (1 << slot):
                continue
            if slot in SEND_SLOTS:
                k = SEND_SLOTS[slot]
                if not self.gt0(f"{self.S}/send_level/{k}") or not self.gt0(f"fx/{slot}/return_level"):
                    continue
                out.append((slot, t, "send", k))
            elif slot in GLOBAL_SLOTS:
                out.append((slot, t, "global", None))
            elif slot in INSERT_SLOTS[self.sc]:
                out.append((slot, t, "insert", None))
        return out


def entries_for(pv, fx_types_in_data):
    """feature name -> entry dict for one preset (natural value, range, kind info, active)."""
    E = {}

    def add(name, path, active, **override):
        row = pv.r[path]
        e = dict(src=path, nat=row["value"], mn=row["min"], mx=row["max"], dflt=row["default"],
                 vt=row["valtype"], ct=row["ctrltype"], active=bool(active))
        e.update(override)
        if e["ct"] == "ct_none" and "kind" not in override:
            return
        E[name] = e
        for flag, (suffix, fkind) in FLAGS.items():
            fv = row.get(flag)
            if fv is not None and not (isinstance(fv, float) and np.isnan(fv)):
                E[f"{name}.{suffix}"] = dict(src=f"{path} [{flag}]", nat=fv, kind=fkind,
                                             active=bool(active), flag_of=name)

    # global
    add("character", "character", True)

    # active scene
    S = pv.S
    osc_on = [pv.osc_active(o) for o in range(3)]
    for path in pv.r:
        if not path.startswith(S + "/"):
            continue
        rel = path[len(S) + 1:]
        top = rel.split("/")[0]
        if top in EXCLUDED_SCENE:
            continue
        m = re.fullmatch(r"osc/(\d)/p/(\d)", rel)
        if m:
            o, i = int(m[1]), int(m[2])
            t = pv.osc_type(o)
            add(f"scene/osc/{o}/{OSC_TYPES[t]}/p{i}", path, osc_on[o])
            continue
        m = re.fullmatch(r"osc/(\d)/(\w+)", rel)
        if m:
            add(f"scene/{rel}", path, osc_on[int(m[1])])
            continue
        m = re.fullmatch(r"filterunit/(\d)/(\w+)", rel)
        if m:
            f, field = int(m[1]), m[2]
            ftype = int(pv.sv(f"filterunit/{f}/type"))
            if field == "type":
                sub = int(pv.sv(f"filterunit/{f}/subtype"))
                d_sub = pv.r[f"{S}/filterunit/{f}/subtype"]["default"]
                add(f"scene/filterunit/{f}/model", path, True, nat=f"{ftype}:{sub}", kind="cat",
                    dflt=f"{int(pv.r[path]['default'])}:{int(d_sub)}")
            elif field != "subtype":
                add(f"scene/{rel}", path, ftype != 0)
            continue
        m = re.fullmatch(r"lfo/(\d+)/\w+", rel)
        if m:
            add(f"scene/{rel}", path, int(m[1]) in pv.used_lfos)
            continue
        if rel == "wsunit/drive":
            add("scene/wsunit/drive", path, int(pv.sv("wsunit/type")) != 0)
        elif rel == "noise_colour":
            add("scene/noise_colour", path, pv.audible("noise"))
        elif rel == "fm_depth":
            add("scene/fm_depth", path, int(pv.sv("fm_switch")) != 0)
        else:
            add(f"scene/{rel}", path, True)

    # wavetable name per oscillator (wavetable / window types)
    for o in range(3):
        t = pv.osc_type(o)
        if t in WAVETABLE_OSC_TYPES:
            E[f"scene/osc/{o}/{OSC_TYPES[t]}/wt"] = dict(
                src=f"{S}/osc/{o} [wavetable name]", nat=pv.wt[o] or "", kind="cat", active=osc_on[o])

    # routing matrix (Round 15, data/routing_matrix.json; falls back to the 13 STANDARD_ROUTES):
    # depth -1..1, 0 = not routed. Oscillator-knob routings are per oscillator type, active only
    # when that oscillator has that type (like the knob blocks).
    for m in ROUTING_MATRIX:
        source, rel, otype = m["source"], m["dest"], m["osc_type"]
        d, sid = pv.route_depth(source, rel)
        name, active = route_feature(source, rel), True
        if otype is not None:
            o, i = int(rel.split("/")[1]), int(rel.split("/")[3])
            name = route_feature(source, f"osc/{o}/{otype}/p{i}")
            active = osc_on[o] and OSC_TYPES[pv.osc_type(o)] == otype
        E[name] = dict(src=f"routing {source} -> {rel}" + (f" [{otype}]" if otype else ""), nat=d, mn=-1.0, mx=1.0,
                       dflt=0.0, vt="float", ct="route_depth", kind="cont", active=active,
                       route=dict(source=source, dest=rel, osc_type=otype), source_id=sid or m["source_id"])

    # FX blocks, by type (first heard instance in slot order)
    heard = {}
    for slot, t, place, k in pv.heard_fx():
        heard.setdefault(t, (slot, place, k))
    for t in fx_types_in_data:
        name = FX_TYPES[t]
        if t not in heard:
            E[f"fx/{name}/present"] = dict(src="(derived)", nat=0, kind="bool", active=True)
            continue
        slot, place, k = heard[t]
        E[f"fx/{name}/present"] = dict(src=f"fx/{slot}/type", nat=1, kind="bool", active=True)
        E[f"fx/{name}/placement"] = dict(src=f"fx/{slot} [slot {slot}]", nat=place, kind="cat", active=True)
        # position within its group of 4 slots = chain order (and, for sends, which send 1-4)
        group = {"insert": INSERT_SLOTS[pv.sc], "send": list(SEND_SLOTS), "global": GLOBAL_SLOTS}[place]
        E[f"fx/{name}/position"] = dict(src=f"fx/{slot} [slot {slot}]", nat=group.index(slot), kind="cat",
                                        active=True)
        if place == "send":
            add(f"fx/{name}/send_level", f"{S}/send_level/{k}", True)
            add(f"fx/{name}/return_level", f"fx/{slot}/return_level", True)
        for i in range(12):
            add(f"fx/{name}/p{i}", f"fx/{slot}/p/{i}", True)
    pv.dup_fx = [FX_TYPES[t] for t, c in Counter(t for _, t, _, _ in pv.heard_fx()).items() if c > 1]
    return E


def default_code(f, e):
    """Code of Surge's default value for this entry (NaN if not defined, e.g. flags, FX placement)."""
    d = e.get("dflt")
    if d is None or "flag_of" in e:
        return np.nan
    k = f["kind"]
    if k in ("cont", "ord"):
        return float(min(max((d - e["mn"]) / (e["mx"] - e["mn"]), 0.0), 1.0)) if e["mx"] > e["mn"] else 0.0
    if k == "bool":
        return float(bool(d))
    if isinstance(d, (float, np.floating)) and float(d).is_integer():
        d = int(d)
    return float(f["options"].index(d)) if d in f["options"] else np.nan


def main():
    t0 = time.time()
    presets = pd.read_parquet(DATA / "presets.parquet")
    raw = pd.read_parquet(DATA / "raw_values.parquet")
    routings = pd.read_parquet(DATA / "routings.parquet")

    rows_by_preset = {pid: dict(zip(g["path"], g[RAW_COLS].to_dict("records")))
                      for pid, g in raw.groupby("preset_id")}
    routs_by_preset = defaultdict(list)
    for r in routings.itertuples():
        routs_by_preset[r.preset_id].append((r.source, r.scope, r.scene, r.source_scene, r.dest, r.depth01,
                                             r.source_id))

    # --- curation (Round 7): which presets the canonical form can represent ----------------------
    audio = pd.read_parquet(DATA / "audio_index.parquet")
    silent = set(audio[(audio["mode"] == "fixed") & audio["silent"]]["preset_id"])
    views, excluded = {}, []
    for p in presets.itertuples():
        pv = PresetView(rows_by_preset[p.preset_id], p.active_scene, routs_by_preset[p.preset_id])
        pv.wt = [p.wt_osc1, p.wt_osc2, p.wt_osc3]
        heard_excluded = sorted({FX_TYPES[t] for _, t, _, _ in pv.heard_fx() if FX_TYPES[t] in EXCLUDED_FX})
        if p.preset_id in silent:
            reason = "silent render"
        elif heard_excluded:
            reason = "heard excluded FX: " + ",".join(heard_excluded)  # Airwindows etc. not representable
        elif p.active_scene != 0:
            reason = "plays scene B"  # the writer writes scene A; B's routings/wavetables are lost
        else:
            views[p.preset_id] = pv
            continue
        excluded.append(dict(preset_id=p.preset_id, name=p.name, label=p.label, reason=reason))
    pd.DataFrame(excluded).to_csv(DATA / "excluded_presets.csv", index=False)
    presets = presets[presets["preset_id"].isin(views)].reset_index(drop=True)

    # FX types that are heard anywhere (minus excluded ones)
    fx_heard = Counter()
    for pv in views.values():
        fx_heard.update({t for _, t, _, _ in pv.heard_fx()})
    fx_types_in_data = sorted(t for t in fx_heard if FX_TYPES[t] not in EXCLUDED_FX)

    # --- pass 1: entries per preset -----------------------------------------------------------
    entries = {}
    dup_fx = Counter()
    for pid, pv in views.items():
        entries[pid] = entries_for(pv, fx_types_in_data)
        dup_fx.update(pv.dup_fx)
    names = sorted({n for E in entries.values() for n in E})

    # --- pass 2: decide kind, options and ranges per feature ---------------------------------
    wt_counts = Counter(e["nat"] for E in entries.values() for n, e in E.items()
                        if n.endswith("/wt") and e["active"])
    # embedded tables/samples ("(Patch Wavetable)", "(Patch Sample)") have no file -> "other"
    wt_top = [w for w, _ in wt_counts.most_common() if w and not w.startswith("(")][:WAVETABLE_TOP_N]

    features, unlisted = [], Counter()
    for name in names:
        es = [E[name] for E in entries.values() if name in E]
        act = [e for e in es if e["active"]]
        if not act:
            continue
        if "kind" in act[0]:
            kind = act[0]["kind"]
        else:
            ct_mode = Counter(e["ct"] for e in act).most_common(1)[0][0]
            vt_mode = Counter(e["vt"] for e in act).most_common(1)[0][0]
            kind = kind_of(vt_mode, ct_mode)
            if kind == "cat_unlisted":
                unlisted[ct_mode] += 1
                kind = "cat"
        f = dict(name=name, kind=kind, n_active=len(act),
                 sources=sorted({re.sub(r"/\d+/", "/#/", e["src"]) for e in act})[:4])
        if "ct" in act[0]:
            f["ctrltypes"] = sorted({e["ct"] for e in act})
            f["valtype"] = Counter(e["vt"] for e in act).most_common(1)[0][0]
        if "route" in act[0]:
            sids = {e["source_id"] for e in act if e.get("source_id") is not None}
            f["route"] = dict(act[0]["route"], source_id=sids.pop() if len(sids) == 1 else None)
        if "flag_of" in act[0]:
            f["flag_of"] = act[0]["flag_of"]
        if kind == "cat":
            if name.endswith("/wt"):
                f["options"] = wt_top + ["other"]
            elif name.endswith("/placement"):
                f["options"] = FX_PLACEMENTS
            elif name.endswith("/position"):
                f["options"] = [0, 1, 2, 3]
            else:
                vals = {e["nat"] for e in act}
                f["options"] = sorted(vals, key=lambda x: (isinstance(x, str), x if not isinstance(x, str)
                                                         else tuple(map(int, x.split(":")))))
                f["options"] = [int(x) if isinstance(x, (float, np.floating)) and float(x).is_integer() else x
                                for x in f["options"]]
        if kind in ("cont", "ord"):
            rng = Counter((e["mn"], e["mx"]) for e in act).most_common()
            f["natural_range"] = list(rng[0][0])
            f["n_distinct_ranges"] = len(rng)
            d01 = [(e["dflt"] - e["mn"]) / (e["mx"] - e["mn"]) if e["mx"] > e["mn"] else 0.0 for e in act]
            f["fill01"] = float(np.clip(np.median(d01), 0, 1))
        features.append(f)

    # --- pass 3: encode ----------------------------------------------------------------------
    fidx = {f["name"]: f for f in features}
    pids = list(presets["preset_id"])
    codes = pd.DataFrame(np.nan, index=pids, columns=[f["name"] for f in features], dtype="float64")
    defaults = codes.copy()  # code of Surge's default value (for the loaded type), where known
    masks = pd.DataFrame(0, index=pids, columns=codes.columns, dtype="int8")
    clipped = Counter()
    roundtrip_bad = []

    for pid in pids:
        E = entries[pid]
        for name, e in E.items():
            f = fidx.get(name)
            if f is None or not e["active"]:
                continue
            k = f["kind"]
            if k in ("cont", "ord"):
                c = (e["nat"] - e["mn"]) / (e["mx"] - e["mn"]) if e["mx"] > e["mn"] else 0.0
                if c < -1e-6 or c > 1 + 1e-6:
                    clipped[name] += 1
                c = min(max(c, 0.0), 1.0)
                back = e["mn"] + c * (e["mx"] - e["mn"])
                if k == "ord":
                    back = round(back)
                if abs(back - e["nat"]) > 1e-4 * max(1.0, abs(e["nat"])) and not clipped[name]:
                    roundtrip_bad.append((pid, name, e["nat"], back))
            elif k == "bool":
                c = float(bool(e["nat"]))
            else:
                nat = e["nat"]
                if name.endswith("/wt"):
                    nat = nat if nat in f["options"] else "other"
                elif isinstance(nat, (float, np.floating)) and float(nat).is_integer():
                    nat = int(nat)
                c = float(f["options"].index(nat))
                if f["options"][int(c)] != nat:
                    roundtrip_bad.append((pid, name, nat, f["options"][int(c)]))
            codes.at[pid, name] = c
            defaults.at[pid, name] = default_code(f, e)
            masks.at[pid, name] = 1

    # model-ready matrix
    cols = {}
    for f in features:
        n, k = f["name"], f["kind"]
        m = masks[n].to_numpy().astype(bool)
        if k == "cat":
            idx = codes[n].to_numpy()
            f["columns"] = [f"{n}={o}" for o in f["options"]]
            for j, col in enumerate(f["columns"]):
                cols[col] = np.where(m & (idx == j), 1.0, 0.0).astype("float32")
        else:
            fill = f.get("fill01", 0.0)
            cols[n] = np.where(m, codes[n].to_numpy(), fill).astype("float32")
            f["columns"] = [n]
    dataset = pd.DataFrame(cols, index=pids)

    # --- write ---------------------------------------------------------------------------------
    codes.index.name = masks.index.name = dataset.index.name = "preset_id"
    codes.to_parquet(DATA / "codes.parquet")
    defaults.index.name = "preset_id"
    defaults.to_parquet(DATA / "defaults.parquet")
    masks.to_parquet(DATA / "masks.parquet")
    dataset.to_parquet(DATA / "dataset.parquet")
    kinds = Counter(f["kind"] for f in features)
    schema = dict(
        created=time.strftime("%Y-%m-%d %H:%M %Z"),
        n_presets=len(pids),
        n_features=len(features),
        n_columns=dataset.shape[1],
        kinds=dict(kinds),
        excluded=dict(**EXCLUDED_GLOBAL, **{f"scene/{k}": v for k, v in EXCLUDED_SCENE.items()},
                      **{f"fx/{k}": v for k, v in EXCLUDED_FX.items()}),
        notes=dict(
            fx_duplicates_dropped=dict(dup_fx.most_common()),
            excluded_presets=dict(Counter(e["reason"].split(":")[0] for e in excluded)),
            fx_position="fx/<type>/position = index of the slot within its group of 4 (insert: A1-A4, "
                        "send: send 1-4, global: G1-G4) = chain order",
            wavetable_top_n=WAVETABLE_TOP_N,
            masks="levels and sends count as > 0 if their base value is > 0 OR they are a modulation destination; osc: audible, ring-mod or FM modulator; lfo: has a routing from it in the active "
                  "scene; filter params: type != none; waveshaper drive: type != none; noise colour: "
                  "noise audible; fm_depth: FM on; fx: heard (not off/disabled; insert of the active "
                  "scene, global, or send with send and return level > 0)",
        ),
        features=features,
    )
    (DATA / "schema.json").write_text(json.dumps(schema, indent=1, default=str))

    # --- checks ----------------------------------------------------------------------------------
    print(f"excluded: {dict(Counter(e['reason'] for e in excluded))} -> data/excluded_presets.csv")
    print(f"encoded {len(pids)} presets in {time.time() - t0:.0f}s")
    print(f"features: {len(features)} ({dict(kinds)}), model-ready columns: {dataset.shape[1]}")
    print(f"FX types kept: {[FX_TYPES[t] for t in fx_types_in_data]}")
    print(f"FX duplicates (2nd instance not represented): {dict(dup_fx.most_common())}")
    if unlisted:
        print(f"WARNING int ctrltypes not in ORDINAL/CATEGORICAL lists (treated as cat): {dict(unlisted)}")
    x = dataset.to_numpy()
    print(f"check: values in [0,1]: {bool(((x >= 0) & (x <= 1)).all())}, NaN: {int(np.isnan(x).sum())}")
    bad_onehot = 0
    for f in features:
        if f["kind"] == "cat":
            s = dataset[f["columns"]].sum(axis=1).to_numpy()
            bad_onehot += int((s != masks[f["name"]].to_numpy()).sum())
    print(f"check: one-hot rows sum to the mask: {bad_onehot == 0} ({bad_onehot} bad)")
    print(f"check: encode -> decode round trip mismatches: {len(roundtrip_bad)}")
    for r in roundtrip_bad[:5]:
        print("   ", r)
    if clipped:
        print(f"note: values outside their range, clipped to [0,1]: {dict(clipped.most_common(8))}")
    groups = Counter(f["name"].split("/")[0] if not f["name"].startswith("scene/")
                     else "scene/" + f["name"].split("/")[1] for f in features)
    print("features per group:", dict(groups.most_common()))


if __name__ == "__main__":
    main()
