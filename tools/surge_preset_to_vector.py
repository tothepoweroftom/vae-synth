#!/usr/bin/env python3
"""Convert Surge XT presets (.fxp) into schema control vectors.

Reference implementation of the import rules in docs/SCHEMA.md, for the
dataset sampler. Reads the patch XML embedded in each .fxp, uses the active
scene (scene A unless scene B is active in single-scene mode) and writes one
JSON line per preset:

    {"file": ..., "values": [...], "features": [...], "lossy": [...]}

`values` are raw schema values (continuous 0..1, categorical index, toggle
0/1); `features` are the model features; `lossy` lists everything that did
not map exactly (e.g. a ladder filter folded to lp24, a step-sequencer LFO).

Usage:
    surge_preset_to_vector.py PRESET_OR_DIR... [-o out.jsonl] [--stats]
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import vae_schema  # noqa: E402

# Surge enums at the pinned revision (src/common/SurgeStorage.h, ModulationSource.h).
OSC_TYPES = ["classic", "sine", "wavetable", "sh_noise", "audio_in", "fm3", "fm2",
             "window", "modern", "string", "twist", "alias"]
MS_FILTEREG, MS_LFO1, MS_LFO2 = 16, 17, 18
FX_DELAY, FX_REVERB1, FX_PHASER, FX_DIST, FX_EQ, FX_CHORUS, FX_REVERB2, FX_FLANGER = 1, 2, 3, 5, 6, 9, 11, 12
# Slot numbers in the patch XML (fx<N>_*), in signal order per scene.
# Surge XT appended A3/A4 (and B3/B4, sends 3/4, globals 3/4) after the
# original eight slots, so A3/A4 are fx9/fx10 in the patch XML (1-based).
FX_SLOTS = {"a": [1, 2, 9, 10], "b": [3, 4, 11, 12]}
SEND_SLOTS = [5, 6, 13, 14]
GLOBAL_SLOTS = [7, 8, 15, 16]

FILTER_FAMILY = {  # sst::filters::FilterType -> schema option
    0: "off", 1: "lp12", 11: "lp12",
    2: "lp24", 3: "lp24", 10: "lp24", 12: "lp24", 13: "lp24", 15: "lp24", 16: "lp24", 28: "lp24",
    4: "hp12", 20: "hp12", 5: "hp24", 14: "hp24", 17: "hp24", 29: "hp24",
    6: "bp", 19: "bp", 22: "bp", 23: "bp", 31: "bp",
    7: "notch", 18: "notch", 21: "notch", 24: "notch", 30: "notch",
    33: "lp24", 34: "lp12", 35: "lp24",  # multi-mode filters: their usual LP setting
}
LFO_SHAPES = {0: "sine", 1: "triangle", 2: "square", 3: "saw", 4: "smooth_random", 5: "random"}
PLAY_MODES = {0: "poly", 1: "mono", 2: "legato", 3: "mono", 4: "legato", 5: "poly"}
# Surge values the engine renders, in schema option order (source/engine/surge/SurgeEngine.cpp
# kPlayModes / kFilterTypes); any other Surge value is folded and reported as lossy.
EXACT_PLAY_MODES = [0, 1, 2]                 # poly, mono, legato
EXACT_FILTER_TYPES = [0, 1, 2, 4, 5, 6, 7]   # off, lp12, lp24, hp12, hp24, bp, notch
# Destination -> (storage name suffix, schema curve of the Surge param) for the
# provisional routes; a route's depth is relative to that curve's full range.
DESTINATIONS = {
    "pitch": ("pitch", "pitch7"),
    "filter1_cutoff": ("filter1_cutoff", "cutoff"),
    "filter2_cutoff": ("filter2_cutoff", "cutoff"),
    "filter1_resonance": ("filter1_resonance", "unit"),
    "volume": ("volume", "amplitude"),
    "pan": ("pan", "bipolar"),
}


class Patch:
    """Parameter values and modulation routes from a Surge patch XML."""

    def __init__(self, xml_text: str):
        root = ET.fromstring(xml_text)
        self.values: dict[str, float] = {}
        self.routes: dict[str, list[tuple[int, float]]] = {}
        params = root.find("parameters")
        if params is None:
            raise ValueError("no <parameters> in patch")
        # Surge inserted a modulation source after ms_ctrl7 (13) at patch
        # revision 9 and renumbers older patches on load (SurgePatch.cpp).
        revision = int(root.attrib.get("revision", "999"))
        for p in params:
            if "value" in p.attrib:
                self.values[p.tag] = float(p.attrib["value"])
            for r in p.findall("modrouting"):
                if r.attrib.get("muted", "0") == "1":
                    continue
                source = int(r.attrib["source"])
                if revision < 9 and source > 13:
                    source += 1
                self.routes.setdefault(p.tag, []).append((source, float(r.attrib["depth"])))

    def get(self, name: str, default: float = 0.0) -> float:
        return self.values.get(name, default)


def read_fxp(path: pathlib.Path) -> Patch:
    data = path.read_bytes()
    start = data.find(b"<?xml")
    end = data.find(b"</patch>")
    if start < 0 or end < 0:
        raise ValueError("no patch XML found")
    text = data[start:end + len(b"</patch>")].decode("utf-8", errors="replace")
    # Some patch names carry control-character references that are not valid XML.
    def keep_valid(m: re.Match) -> str:
        code = int(m.group(1)[1:], 16) if m.group(1)[0] in "xX" else int(m.group(1))
        return m.group(0) if code in (9, 10, 13) or code >= 32 else ""

    text = re.sub(r"&#([xX][0-9a-fA-F]+|\d+);", keep_valid, text)
    return Patch(text)


class Converter:
    """Maps one Surge patch onto the schema vector, one section at a time.

    `values` starts at the schema defaults; `lossy` collects every mapping that
    is not exact. Each section method reads the patch through `p()` (the active
    scene's storage names) and writes through the put* helpers."""

    def __init__(self, patch: Patch, schema: vae_schema.Schema):
        self.patch = patch
        self.schema = schema
        self.values = schema.defaults()
        self.lossy: list[str] = []

        scenemode = int(patch.get("scenemode"))
        active = int(patch.get("scene_active"))
        self.scene = "b" if scenemode == 0 and active == 1 else "a"
        if scenemode != 0:
            self.lossy.append(f"scene mode {scenemode}: only scene A used")

    def p(self, name: str) -> str:
        return f"{self.scene}_{name}"

    def get(self, name: str, default: float = 0.0) -> float:
        """A parameter of the active scene."""
        return self.patch.get(self.p(name), default)

    def put(self, control_id: str, value: float) -> None:
        c = self.schema[control_id]
        self.values[c.index] = self.schema.sanitise(c, value)

    def put_option(self, control_id: str, option: str) -> None:
        self.put(control_id, self.schema[control_id].options.index(option))

    def put_units(self, control_id: str, units: float) -> None:
        c = self.schema[control_id]
        v = self.schema.from_units(c, units)
        if abs(self.schema.to_units(c, v) - units) > 1e-3:
            self.lossy.append(f"{control_id}: {units:g} clipped to the schema range")
        self.put(control_id, v)

    def run(self) -> tuple[list[float], list[str]]:
        self.voice()
        self.oscillators()
        self.filters()
        self.envelopes()
        self.lfos()
        self.routes()
        self.fx()
        self.put_units("master_volume", self.patch.get("volume", 0.0))
        return self.values, self.lossy

    def voice(self) -> None:
        mode = int(self.get("polymode"))
        self.put_option("play_mode", PLAY_MODES.get(mode, "poly"))
        if mode not in EXACT_PLAY_MODES:
            self.lossy.append(f"play_mode: Surge mode {mode} folded")
        self.put_units("glide_time", self.get("portamento", -8.0))

    def oscillators(self) -> None:
        for n in (1, 2, 3):
            o = f"osc{n}"
            self.put(f"{o}_on", 0.0 if self.get(f"mute_o{n}") >= 0.5 else 1.0)
            self.put(f"{o}_level", self.get(f"level_o{n}", 1.0))
            self.put(f"{o}_octave", int(self.get(f"osc{n}_octave")) + 3)
            self.put_units(f"{o}_pitch", self.get(f"osc{n}_pitch"))
            self.put(f"{o}_filter", int(self.get(f"route_o{n}", 1)))
            self.put_option(f"{o}_wave", self.wave(n))

        if self.get("mute_noise", 1.0) < 0.5 and self.get("level_noise", 0.0) > 0.0:
            self.lossy.append("noise mixer channel ignored (not in schema)")
        for ring in ("ring12", "ring23"):
            if self.get(f"mute_{ring}", 1.0) < 0.5:
                self.lossy.append(f"{ring} ring modulation ignored")

    def wave(self, n: int) -> str:
        """Schema wave for Surge osc n (may also switch the osc off)."""
        o = f"osc{n}"
        t = int(self.get(f"osc{n}_type"))
        name = OSC_TYPES[t] if 0 <= t < len(OSC_TYPES) else "classic"
        shape = self.get(f"osc{n}_param0")
        if name == "classic":
            wave = "square" if shape <= -0.5 else "saw"
            if -0.5 < shape < -0.05 or shape > 0.05:
                self.lossy.append(f"{o}_wave: Classic shape {shape:.2f} -> {wave}")
            return wave
        if name == "sine":
            return "sine"
        if name == "modern":
            mix = {"saw": abs(shape), "square": abs(self.get(f"osc{n}_param1")),
                   "triangle": abs(self.get(f"osc{n}_param2"))}
            wave = max(mix, key=mix.get)
            self.lossy.append(f"{o}_wave: Modern mix -> {wave}")
            return wave
        if name in ("fm2", "fm3"):
            return "fm"
        if name == "sh_noise":
            return "noise"
        if name == "audio_in":
            self.put(f"{o}_on", 0.0)
            self.lossy.append(f"{o}: Audio Input osc -> off")
            return "sine"
        # wavetable, window, string, twist, alias
        wave = "wavetable" if name in ("wavetable", "window") else "saw"
        if name != "wavetable":
            self.lossy.append(f"{o}_wave: {name} -> {wave}")
        return wave

    def filters(self) -> None:
        config = int(self.get("fb_config", 7))
        self.put_option("filter_routing", "serial" if config in (0, 1, 2) else "parallel")
        if config not in (0, 3):
            self.lossy.append(f"filter_routing: Surge config {config} folded")
        f1_cutoff = self.get("filter1_cutoff", 3.0)
        for n in (1, 2):
            f = f"filter{n}"
            t = int(self.get(f"{f}_type"))
            option = FILTER_FAMILY.get(t, "off")
            if t not in EXACT_FILTER_TYPES:
                self.lossy.append(f"{f}_type: Surge type {t} -> {option}")
            self.put_option(f"{f}_type", option)
            cutoff = self.get(f"{f}_cutoff", 3.0)
            if n == 2 and self.get("f2_cf_is_offset") >= 0.5:
                cutoff += f1_cutoff
            self.put_units(f"{f}_cutoff", cutoff)
            self.put(f"{f}_resonance", self.get(f"{f}_resonance"))
            self.put_units(f"{f}_env", self.get(f"{f}_envmod"))
            self.put_units(f"{f}_keytrack", self.get(f"{f}_keytrack"))

    def envelopes(self) -> None:
        for prefix, env in (("amp", "env1"), ("feg", "env2")):
            for stage in ("attack", "decay", "release"):
                self.put_units(f"{prefix}_{stage}", self.get(f"{env}_{stage}", -8.0))
                self.put(f"{prefix}_{stage}_curve", int(self.get(f"{env}_{stage}_shape", 1)))
            self.put(f"{prefix}_sustain", self.get(f"{env}_sustain", 1.0))

    def lfos(self) -> None:
        for n, lfo in ((1, "lfo0"), (2, "lfo1")):
            l = f"lfo{n}"  # noqa: E741
            shape = int(self.get(f"{lfo}_shape"))
            if shape not in LFO_SHAPES:
                self.lossy.append(f"{l}_shape: Surge LFO type {shape} -> random")
            self.put_option(f"{l}_shape", LFO_SHAPES.get(shape, "random"))
            self.put_units(f"{l}_rate", self.get(f"{lfo}_rate"))
            self.put(f"{l}_amount", self.get(f"{lfo}_magnitude", 1.0))
            self.put(f"{l}_unipolar", self.get(f"{lfo}_unipolar"))
            self.put(f"{l}_retrigger", 1.0 if int(self.get(f"{lfo}_trigmode", 1)) == 1 else 0.0)
            self.put(f"{l}_phase", self.get(f"{lfo}_phase"))
            self.put_units(f"{l}_fade_in", self.get(f"{lfo}_attack", -8.0))

    def routes(self) -> None:
        """PROVISIONAL routing: the strongest supported route per source."""
        supported = {self.p(suffix) for suffix, _ in DESTINATIONS.values()}
        for source, prefix in ((MS_FILTEREG, "feg"), (MS_LFO1, "lfo1"), (MS_LFO2, "lfo2")):
            best = None
            others = 0
            for dest, (suffix, curve) in DESTINATIONS.items():
                span = self.schema.curves[curve]["max"] - self.schema.curves[curve]["min"]
                for src, depth in self.patch.routes.get(self.p(suffix), []):
                    if src != source:
                        continue
                    d = depth / span
                    if best is None or abs(d) > abs(best[1]):
                        if best is not None:
                            others += 1
                        best = (dest, d)
                    else:
                        others += 1
            unsupported = sum(1 for target, routes in self.patch.routes.items()
                              for src, _ in routes if src == source and target not in supported)
            if best is not None:
                self.put_option(f"{prefix}_destination", best[0])
                self.put(f"{prefix}_depth", (max(-1.0, min(1.0, best[1])) + 1.0) / 2.0)
            if others or unsupported:
                self.lossy.append(f"{prefix}: {others + unsupported} extra modulation route(s) dropped")

    def fx(self) -> None:
        """First audible effect of each type along the chain (scene inserts,
        sends, globals). fx_disable has one bit per slot (bit slot-1)."""
        disabled = int(self.patch.get("fx_disable"))
        seen: set[int] = set()
        for slot in FX_SLOTS[self.scene] + SEND_SLOTS + GLOBAL_SLOTS:
            t = int(self.patch.get(f"fx{slot}_type"))
            if t == 0 or (disabled >> (slot - 1)) & 1:
                continue
            wet = self.send_wetness(slot)
            if wet <= 0.0:
                continue  # nothing is sent to it: inaudible, so treat as off
            family = {FX_REVERB1: FX_REVERB2, FX_FLANGER: FX_PHASER}.get(t, t)
            if family in seen:
                self.lossy.append(f"fx slot {slot}: second {t} ignored")
                continue
            mixed = self.fx_unit(slot, t, wet)
            if mixed is None:
                self.lossy.append(f"fx slot {slot}: type {t} not in schema")
                continue
            if mixed and self.values[self.schema[f"{mixed}_mix"].index] <= 0.0:
                # Mix 0 is inaudible (Surge loads it that way too): label it off so
                # "on, silent" and "off" don't become two labels for one sound.
                section = self.schema[f"{mixed}_on"].section
                for c in self.schema.controls:
                    if c.section == section:
                        self.values[c.index] = c.default
                self.lossy.append(f"fx slot {slot}: {mixed} with mix 0 -> off")
                continue
            seen.add(family)

    def send_wetness(self, slot: int) -> float:
        """How much of the signal reaches an effect slot: 1 for inserts and
        globals; for a send, the scene send level times the send's return level,
        both applied by Surge as amplitude^3 (send N = slot 5, 6, 13, 14)."""
        n = {5: 1, 6: 2, 13: 3, 14: 4}.get(slot)
        if n is None:
            return 1.0
        return self.get(f"send_fx_{n}", 0.0) ** 3 * self.patch.get(f"volume_FX{n}", 1.0) ** 3

    def fx_unit(self, slot: int, t: int, wet: float) -> str | None:
        """Writes one effect's controls. Returns the schema prefix whose mix
        decides audibility ("" when the effect has no mix), or None when the
        type is not in the schema."""
        put, put_units = self.put, self.put_units
        fx = lambda k: self.patch.get(f"fx{slot}_p{k}")  # noqa: E731
        if t == FX_DIST:
            put("dist_on", 1)
            put_units("dist_drive", fx(4))
            put_units("dist_tone", fx(9))
            return ""
        if t == FX_EQ:
            put("eq_on", 1)
            put_units("eq_low_gain", fx(0))
            put_units("eq_mid_freq", fx(4))
            put_units("eq_mid_gain", fx(3))
            put_units("eq_high_gain", fx(6))
            return ""
        if t == FX_CHORUS:
            put("chorus_on", 1)
            put_units("chorus_rate", fx(1))
            put("chorus_depth", fx(2))
            put("chorus_mix", fx(6) * wet)
            return "chorus"
        if t == FX_PHASER:
            put("modfx_on", 1)
            self.put_option("modfx_type", "phaser")
            put_units("modfx_rate", fx(3))
            put("modfx_depth", fx(4))
            put("modfx_feedback", max(0.0, fx(1)))
            put("modfx_mix", fx(6) * wet)
            return "modfx"
        if t == FX_FLANGER:
            put("modfx_on", 1)
            self.put_option("modfx_type", "flanger")
            put_units("modfx_rate", fx(2))
            put("modfx_depth", fx(3))
            put("modfx_feedback", fx(7))
            put("modfx_mix", max(0.0, fx(10)) * wet)
            return "modfx"
        if t == FX_DELAY:
            put("delay_on", 1)
            put_units("delay_time", fx(0))
            put("delay_feedback", fx(2))
            put("delay_mix", fx(10) * wet)
            return "delay"
        if t == FX_REVERB2:
            put("reverb_on", 1)
            put("reverb_size", (fx(1) + 1.0) / 2.0)
            put_units("reverb_decay", fx(2))
            put("reverb_damping", fx(7))
            put("reverb_mix", fx(9) * wet)
            return "reverb"
        if t == FX_REVERB1:
            put("reverb_on", 1)
            put("reverb_size", fx(2))
            put_units("reverb_decay", fx(3))
            put("reverb_damping", fx(4))
            put("reverb_mix", fx(9) * wet)
            self.lossy.append("reverb: Reverb 1 mapped onto Reverb 2 controls")
            return "reverb"
        return None


def convert(patch: Patch, schema: vae_schema.Schema) -> tuple[list[float], list[str]]:
    return Converter(patch, schema).run()


def iter_presets(paths: list[str]):
    for arg in paths:
        path = pathlib.Path(arg)
        if path.is_dir():
            yield from sorted(path.rglob("*.fxp"))
        else:
            yield path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("-o", "--output", type=pathlib.Path)
    parser.add_argument("--stats", action="store_true", help="print per-control value distributions")
    args = parser.parse_args()

    schema = vae_schema.load()
    out = open(args.output, "w") if args.output else sys.stdout
    counts = collections.defaultdict(collections.Counter)
    lossy_counts = collections.Counter()
    n_ok = n_fail = 0
    for preset in iter_presets(args.inputs):
        try:
            values, lossy = convert(read_fxp(preset), schema)
        except Exception as e:  # one bad third-party file must not stop the batch
            print(f"skip {preset}: {e}", file=sys.stderr)
            n_fail += 1
            continue
        n_ok += 1
        out.write(json.dumps({"file": str(preset), "values": values,
                              "features": schema.encode_features(values), "lossy": lossy}) + "\n")
        for c, v in zip(schema.controls, values):
            if c.kind == "categorical":
                counts[c.id][c.options[int(v)]] += 1
            elif c.kind == "toggle":
                counts[c.id]["on" if v >= 0.5 else "off"] += 1
        for entry in lossy:
            lossy_counts[entry.split(":")[0]] += 1

    if args.output:
        out.close()
    print(f"converted {n_ok} presets, skipped {n_fail}", file=sys.stderr)
    if args.stats:
        for cid, counter in counts.items():
            print(f"{cid:20s} " + ", ".join(f"{k}={v}" for k, v in counter.most_common()), file=sys.stderr)
        print("lossy conversions:", file=sys.stderr)
        for key, n in lossy_counts.most_common(25):
            print(f"  {n:5d}  {key}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
