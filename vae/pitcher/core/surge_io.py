"""The only module that talks to surgepy (PIPELINE_VAE_SURGE.md, Step 0.4).

Everything else in the pipeline works with plain paths, numbers and tables, so this file
is the one to replace once the VAE drives a different synth (Section 10).
"""

import os
import re
import sys
from pathlib import Path

import numpy as np

PITCHER_ROOT = Path(__file__).resolve().parents[2]  # pitcher/core/surge_io.py -> project root
SURGE_ROOT = PITCHER_ROOT / "surge-rack-main" / "surge"
DATA_HOME = SURGE_ROOT / "resources" / "data"
SURGEPY_BUILD = SURGE_ROOT / "ignore" / "bpy" / "src" / "surge-python"

# No Surge XT app is installed, so point the engine at the repo's factory data
# (SurgeStorage.cpp:1991-2001). Must happen before surgepy is imported.
os.environ.setdefault("SURGE_DATA_HOME", str(DATA_HOME))
if SURGEPY_BUILD.is_dir():
    sys.path.insert(0, str(SURGEPY_BUILD))

import surgepy  # noqa: E402

SR = 48000  # CLAP embeddings (Step 8) expect 48 kHz


def _read_ctrltype_names():
    """Enum ctrltypes from Parameter.h; it has no explicit values, so position == value."""
    src = (SURGE_ROOT / "src" / "common" / "Parameter.h").read_text()
    body = src[src.index("enum ctrltypes"):]
    body = body[body.index("{") + 1: body.index("};")]
    return [m.group(1) for line in body.splitlines() if (m := re.match(r"\s*(ct_\w+)", line))]


CTRLTYPE_NAMES = _read_ctrltype_names()


def flatten(d, prefix=""):
    """getPatch() -> [(path, SurgeNamedParamId)], e.g. 'scene/0/osc/1/p/3'."""
    if isinstance(d, dict):
        return [x for k, v in d.items() for x in flatten(v, f"{prefix}/{k}" if prefix else k)]
    if isinstance(d, list):
        return [x for i, v in enumerate(d) for x in flatten(v, f"{prefix}/{i}")]
    return [(prefix, d)]


def patch_revision(fxp_path):
    """Patch file-format revision (<patch revision="N">), metadata only."""
    m = re.search(rb'<patch revision="(\d+)"', Path(fxp_path).read_bytes())
    return int(m.group(1)) if m else None


class SurgeIO:
    # (name, supported?, getter): per-parameter flags that change meaning (Appendix A.3)
    FLAGS = [
        ("extend", "canExtend", "getExtend"),
        ("temposync", "canTempoSync", "getTempoSync"),
        ("absolute", "canBeAbsolute", "getAbsolute"),
        ("deform", "canDeform", "getDeform"),
        ("deactivated", "canDeactivate", "getDeactivated"),
    ]

    def __init__(self, sample_rate=SR):
        self.s = surgepy.createSurge(sample_rate)
        self.bs = self.s.getBlockSize()
        self.params = flatten(self.s.getPatch())  # fixed order -> column order
        self.by_path = dict(self.params)
        self.path_by_synth_id = {p.getId().getSynthSideId(): path for path, p in self.params}
        # Unpatched surgepy lists 16 aliased return levels and misses send levels 3/4
        assert len(self.path_by_synth_id) == len(self.params), \
            "getPatch() has duplicate parameters: rebuild surgepy with the local getPatch fixes"
        self.has_extras = hasattr(self.s, "getParamCtrlType")  # Step 0.3 additions built in?

    def settle(self, n=8):
        """Run a few blocks so queued type changes and wavetable loads are applied."""
        for _ in range(n):
            self.s.process()

    def load(self, fxp_path):
        ok = self.s.loadPatch(str(fxp_path))
        self.settle()
        return ok

    def value(self, path):
        return self.s.getParamVal(self.by_path[path])

    def read_values(self):
        """One tuple per parameter, in natural units, with the ranges valid for this patch."""
        s = self.s
        rows = []
        for path, p in self.params:
            flags = {}
            for name, can, get in self.FLAGS:
                flags[name] = getattr(s, get)(p) if getattr(s, can)(p) else None
            rows.append(dict(
                path=path,
                name=s.getParameterName(p.getId()),  # current display name, e.g. 'A Osc 1 Morph'
                value=s.getParamVal(p),
                min=s.getParamMin(p),
                max=s.getParamMax(p),
                default=s.getParamDef(p),
                valtype=s.getParamValType(p),
                ctrltype=CTRLTYPE_NAMES[s.getParamCtrlType(p)] if self.has_extras else None,
                **flags,
            ))
        return rows

    def read_wavetable_names(self, scene):
        if not self.has_extras:
            return [None] * 3
        return [self.s.getWavetableName(scene, o) for o in range(3)]

    def read_routings(self):
        """All modulation routings: global, and per scene the scene and voice lists."""
        mr = self.s.getAllModRoutings()
        rows = []

        def add(r, scope, scene):
            rows.append(dict(
                scope=scope,
                scene=scene,
                source=r.getSource().getName(),
                source_id=r.getSource().getModSource(),
                source_scene=r.getSourceScene(),
                source_index=r.getSourceIndex(),
                dest=self.path_by_synth_id.get(r.getDest().getId().getSynthSideId()),
                dest_name=r.getDest().getName(),
                depth=r.getDepth(),
                depth01=r.getNormalizedDepth(),
            ))

        for r in mr["global"]:
            add(r, "global", None)
        for sc, d in enumerate(mr["scene"]):
            for scope in ("scene", "voice"):
                for r in d[scope]:
                    add(r, scope, sc)
        return rows

    # --- writing (Step 7.2) -------------------------------------------------------------------
    TYPE_CTRLTYPES = {"ct_osctype", "ct_fxtype", "ct_filtertype", "ct_filtersubtype", "ct_wstype",
                      "ct_lfotype", "ct_fmconfig", "ct_fbconfig"}
    NEVER_TOUCH = {"volume", "fx_bypass", "polylimit"}  # global performance settings
    CAN = {"Deactivated": "canDeactivate", "TempoSync": "canTempoSync", "Extend": "canExtend",
           "Absolute": "canBeAbsolute", "Deform": "canDeform"}

    def ctrltype_name(self, path):
        return CTRLTYPE_NAMES[self.s.getParamCtrlType(self.by_path[path])]

    def remove_routings_to(self, dest_filter):
        """Remove every modulation routing whose destination path matches dest_filter."""
        s, mr, n = self.s, self.s.getAllModRoutings(), 0
        routings = list(mr["global"]) + [r for d in mr["scene"] for k in ("scene", "voice") for r in d[k]]
        for r in routings:
            dest = self.path_by_synth_id.get(r.getDest().getId().getSynthSideId())
            if dest_filter(dest):
                s.setModDepth01(self.by_path[dest], r.getSource(), 0.0, r.getSourceScene(), r.getSourceIndex())
                n += 1
        return n

    def clear_flags(self):
        """Switch off temposync / extend / absolute on every parameter that currently allows it.

        Flags belong to the parameter slot, not to the oscillator or effect in it, and survive
        type changes. Parameter::bound_value snaps a value to the tempo grid whenever temposync
        is set, even for a control type that can't sync, and setTempoSync refuses to change it
        then. So this must run BEFORE types change (e.g. a Delay's sync flag snapped the Drive of
        a Tape effect written into the same slot to log2(1.5) = 0.585)."""
        s = self.s
        for path, p in self.params:
            for method in ("TempoSync", "Extend", "Absolute"):
                if getattr(s, self.CAN[method])(p):
                    getattr(s, "set" + method)(p, False)

    def neutralise(self, reset_types=True):
        """Set parameters to their defaults. reset_types=False keeps oscillator/filter/FX/LFO
        types as they are: changing those clears modulation routed onto them
        (SurgeSynthesizer.cpp:3415-3433, :3530)."""
        s = self.s
        self.clear_flags()
        typed = [(path, p) for path, p in self.params if self.ctrltype_name(path) in self.TYPE_CTRLTYPES]
        if reset_types:
            # remove routings onto FX slots while their effect still exists (see write_plan)
            self.remove_routings_to(lambda dest: dest is not None and dest.startswith("fx/"))
            for _, p in typed:
                s.setParamVal(p, s.getParamDef(p))
            self.settle()
        typed_paths = {path for path, _ in typed}
        for path, p in self.params:
            # ct_none params (fx_disable, unused FX knobs) have a +-2^31 range: setParamVal's float
            # normalisation corrupts them (fx_disable 0 -> -648, i.e. most FX slots disabled)
            if path in typed_paths or path in self.NEVER_TOUCH or self.ctrltype_name(path) == "ct_none":
                continue
            s.setParamVal(p, s.getParamDef(p))
        s.setFxDisable(0)
        self.settle()

    def write_plan(self, plan):
        """Execute a codec.Plan in the safe order; return read-back info per written item."""
        s, P = self.s, self.by_path
        # phase 0: FX slots that change type lose the modulation routed onto them. Surge keeps
        # those routings across an FX type change; a macro offset then landed on the Resonator's
        # integer mode knob and crashed the engine (array index out of range, ResonatorEffect.cpp:120)
        changing = {int(path.split("/")[1]) for path, nat in plan.types
                    if re.fullmatch(r"fx/\d+/type", path) and int(self.value(path)) != int(nat)}
        if changing:
            self.remove_routings_to(lambda dest: dest is not None and dest.startswith("fx/")
                                    and int(dest.split("/")[1]) in changing)
        for path, nat in plan.types:                       # phase 1: types and switches
            if path == "fx_disable":
                s.setFxDisable(int(nat))                   # see neutralise(): not via setParamVal
            else:
                self.set_int(path, nat)
        self.settle()
        for scene, osc, name in plan.wavetables:           # wavetable by name (needs the osc type)
            path = self.wavetable_path(name)
            if path is not None:
                s.loadWavetable(scene, osc, str(path))
        if plan.wavetables:
            self.settle()
        for path, nat in plan.subtypes:                    # phase 2: subtypes + flags
            self.set_int(path, nat)
        # Flags belong to the parameter slot and survive type changes; an FX type change even
        # restores them from the copy taken at patch load (fxsync). Clear them unconditionally
        # now that the types are in place, then apply the planned ones (see clear_flags()).
        for path in {path for _, path, *_ in plan.values} | {path for _, path, _, _ in plan.flags}:
            s.clearParamFlags(P[path])
        # flags before values, except temposync: setExtend rescales the current value, while
        # writing a value with temposync on snaps it to the tempo grid (Parameter::bound_value);
        # setTempoSync itself does not snap, so it is applied after the values.
        def apply_flags(methods):
            for _, path, method, val in plan.flags:
                if method in methods and getattr(s, self.CAN[method])(P[path]):
                    getattr(s, "set" + method)(P[path], val)
        apply_flags({"Extend", "Absolute", "Deform", "Deactivated"})
        self.settle()
        for _, path, kind, code, nat in plan.values:       # phase 3: values, with live ranges
            p = P[path]
            if kind == "cont":
                lo, hi = s.getParamMin(p), s.getParamMax(p)
                s.setParamVal(p, lo + code * (hi - lo))
            elif kind == "ord":
                lo, hi = s.getParamMin(p), s.getParamMax(p)
                self.set_int(path, round(lo + code * (hi - lo)))
            elif kind == "bool":
                s.setParamVal(p, 1.0 if code >= 0.5 else 0.0)
            else:
                self.set_int(path, nat)
        apply_flags({"TempoSync"})
        for _, source_id, path, depth in plan.routings:    # standard routings (template option B)
            s.setModDepth01(P[path], s.getModSource(source_id), depth, 0, 0)
        self.settle()

    def wavetable_path(self, name):
        """Factory wavetable file for a display name (factory folder first, then 3rd party)."""
        if not hasattr(self, "_wt_index"):
            self._wt_index = {}
            for folder in ("wavetables", "wavetables_3rdparty"):
                for f in sorted((DATA_HOME / folder).rglob("*")):
                    if f.suffix.lower() in (".wav", ".wt"):
                        self._wt_index.setdefault(f.stem, f)
        return self._wt_index.get(name)

    def set_int(self, path, target):
        """Set an int parameter exactly. setParamVal normalises linearly, but Surge turns 0-1 back
        into an int with intUnscaledFromFloat (0.005 + 0.99 x, Parameter.h:640-651); for ranges
        above ~100 (e.g. MIDI keys 0-127) that is off by one. So pass the value whose linear
        normalisation equals intScaledToFloat(target)."""
        s, p = self.s, self.by_path[path]
        if s.getParamValType(p) != "int":
            s.setParamVal(p, float(target))
            return
        lo, hi = s.getParamMin(p), s.getParamMax(p)
        rng = hi - lo
        if rng <= 0:
            s.setParamVal(p, float(target))
            return
        s.setParamVal(p, lo + rng * (0.005 + 0.99 * (float(target) - lo) / rng))

    def read_back(self, plan):
        """Values as they are now in the engine, for every item the plan wrote."""
        s, P = self.s, self.by_path
        out = dict(types=[(path, nat, s.getParamVal(P[path])) for path, nat in plan.types + plan.subtypes],
                   flags=[], values=[])
        for name, path, method, val in plan.flags:
            out["flags"].append((name, path, val, getattr(s, "get" + method)(P[path])))
        for name, path, kind, code, nat in plan.values:
            p = P[path]
            out["values"].append((name, path, kind, code, nat, s.getParamVal(p), s.getParamMin(p), s.getParamMax(p)))
        out["routings"] = [(name, depth, s.getModDepth01(P[path], s.getModSource(sid), 0, 0))
                           for name, sid, path, depth in plan.routings]
        out["wavetables"] = [((scene, osc), name, self.s.getWavetableName(scene, osc))
                             for scene, osc, name in plan.wavetables if self.wavetable_path(name) is not None]
        return out

    # --- rendering (Step 1.4) ----------------------------------------------------------------
    WIN_S = 0.05  # analysis/processing chunk: 50 ms

    def _chunk(self):
        """Process one 50 ms chunk; returns a (2, n) float32 copy."""
        if not hasattr(self, "_buf"):
            self._nblk = max(1, round(self.WIN_S * SR / self.bs))
            self._buf = self.s.createMultiBlock(self._nblk)
        self.s.processMultiBlock(self._buf)
        return self._buf.copy()

    def flush(self, seconds=0.5):
        """Stop all notes and run silence so no tail of an earlier note leaks in; returns residual RMS."""
        self.s.allNotesOff()
        last = None
        for _ in range(max(1, round(seconds / self.WIN_S))):
            last = self._chunk()
        return float(np.sqrt(np.mean(last ** 2)))

    @staticmethod
    def _db(x):
        return 20 * np.log10(max(x, 1e-12))

    def _release(self, chunks, rms, min_tail_s, max_tail_s, rel_db, below_windows):
        """After note-off: record until the 50 ms RMS stays rel_db below the loudest window."""
        n_below, n = 0, 0
        while True:
            c = self._chunk()
            chunks.append(c)
            r = float(np.sqrt(np.mean(c ** 2)))
            rms.append(r)
            n += 1
            t = n * self.WIN_S
            peak_rms = max(rms)
            silent_preset = peak_rms < 1e-6  # -120 dBFS: nothing to wait for
            n_below = n_below + 1 if (silent_preset or self._db(r) < self._db(peak_rms) + rel_db) else 0
            if t >= min_tail_s and n_below >= below_windows:
                return t, False
            if t >= max_tail_s:
                return t, True

    def render(self, note=48, velocity=100, hold_s=2.0, natural=False,
               min_hold_s=0.5, max_hold_s=6.0, steady_db=0.5, steady_win_s=0.25,
               min_tail_s=0.5, max_tail_s=12.0, rel_db=-60.0, below_windows=4, seed=None):
        """Play one note on the loaded patch and return (audio (2, n) float32, info dict).

        fixed   (natural=False): hold exactly hold_s, then release until the level stays
                 rel_db below the loudest 50 ms window (>= min_tail_s, <= max_tail_s).
        natural (natural=True):  hold until the sound is steady (two consecutive 250 ms
                 windows differ < steady_db) or has died out, within [min_hold_s, max_hold_s];
                 then the same release rule.
        seed: reseed Surge's random sources first (needs the local seedRandom patch), so two renders
        of the same patch and seed are identical; None keeps Surge's own (time-based) randomness.
        """
        residual = self.flush()
        if seed is not None:
            self.s.seedRandom(int(seed))
        chunks, rms = [], []
        self.s.playNote(0, note, velocity, 0)
        hold_reason = "fixed"
        per_steady = max(1, round(steady_win_s / self.WIN_S))
        steady_hits = 0
        while True:
            c = self._chunk()
            chunks.append(c)
            rms.append(float(np.sqrt(np.mean(c ** 2))))
            t = len(chunks) * self.WIN_S
            if not natural:
                if t >= hold_s - 1e-9:
                    break
                continue
            if t >= max_hold_s:
                hold_reason = "max"
                break
            if t < min_hold_s or len(rms) < 2 * per_steady:
                continue
            a = np.sqrt(np.mean(np.square(rms[-2 * per_steady:-per_steady])))
            b = np.sqrt(np.mean(np.square(rms[-per_steady:])))
            if max(rms) > 1e-6 and self._db(b) < self._db(max(rms)) + rel_db:
                hold_reason = "died"
                break
            steady_hits = steady_hits + 1 if abs(self._db(b) - self._db(a)) < steady_db else 0
            if steady_hits >= 2:
                hold_reason = "steady"
                break
        hold = len(chunks) * self.WIN_S
        self.s.releaseNote(0, note, 0)
        tail, capped = self._release(chunks, rms, min_tail_s, max_tail_s, rel_db, below_windows)
        audio = np.concatenate(chunks, axis=1)
        info = dict(hold_s=round(hold, 3), tail_s=round(tail, 3), total_s=round(audio.shape[1] / SR, 3),
                    hold_reason=hold_reason, tail_capped=capped,
                    peak_dbfs=round(self._db(float(np.abs(audio).max())), 2),
                    max_rms_dbfs=round(self._db(max(rms)), 2),
                    n_over_0dbfs=int((np.abs(audio) > 1.0).sum()),
                    silent=max(rms) < 1e-6, flush_residual_rms=residual)
        return audio, info

    def render_plan(self, plan, template, seed=0, **render_kw):
        """Deterministic render of a codec.Plan on the template: the same plan and seed give the same
        audio whatever was rendered before in this engine. Two things make it so (Round 29):
          - seed: Surge's random sources are reseeded (local seedRandom patch);
          - priming: some DSP state survives patch loads (the previous preset changed the next
            render by up to 7 dB with identical parameters); a short note on the template first
            puts that state in the same place every time.
        Returns (audio, info) like render()."""
        self.load(template)
        # same key as the real note: Surge remembers the last key across patch loads (portamento
        # would otherwise glide from the priming note)
        self.render(note=render_kw.get("note", 48), seed=seed, hold_s=0.1, min_tail_s=0.05, max_tail_s=0.5)
        self.load(template)
        self.write_plan(plan)
        return self.render(seed=seed, **render_kw)

    def save_patch(self, path):
        """Write the loaded patch (e.g. after load(template) + write_plan(plan)) as a .fxp file
        (surgepy savePatch; Surge XT opens it like any preset)."""
        self.s.savePatch(str(path))

    def param_catalog(self):
        """Static info per parameter path: XML storage name, OSC address (Step 7)."""
        return [dict(path=path,
                     synth_id=p.getId().getSynthSideId(),
                     storage_name=self.s.getParamStorageName(p) if self.has_extras else None,
                     osc_name=self.s.getParamOSCName(p) if self.has_extras else None)
                for path, p in self.params]
