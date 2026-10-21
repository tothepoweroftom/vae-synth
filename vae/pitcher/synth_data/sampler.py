"""Stage 1 of the synthetic Surge dataset (TEXT2PRESET_PLAN.md §3.1): where the parameter rows come from.

Three sources, each giving full codes rows ({feature: code or NaN}, what codec.decode() takes):

  perturb   a cleaned TRAIN human preset (equal share per category), changed:
            - 0-3 switch groups (oscillator / filter / effect types, mutes, ...) take a donor preset's value;
              a newly switched-in oscillator or effect brings the donor's knobs along (as in stage 0)
            - knobs moved on their perceptual scale (stage 0b rulers) with a step size by screening tier
              (stage 0): tier 1 all moved, tier 2 a random share, tier 3 left alone. Gated knobs (tier 3
              but audible in some presets) count as tier 2: they are only moved where active, so the gate
              rule holds by construction
            - routings: existing depths moved, a few added (donor depth) or removed
            three strengths (small / medium / large): small steps teach the inverter that nearby settings
            sound alike, large ones reach further
  layered   structure first, knobs second: a human preset's switch settings (rare structures upweighted:
            weight frequency^-0.5 per preset, so structures are drawn ~ frequency^0.5), 0-3 switch
            mutations; then every active knob drawn from a scrambled Sobol point pushed through a Gaussian
            copula fitted on the human presets (keeps typical pairings, e.g. resonance vs cutoff) and the
            knob's human value distribution, widened: 5% of the values uniform on the perceptual scale.
            Filter knobs use the distribution of presets with the same filter type; gain-only knobs
            (stage 0) keep the structure preset's value (the set is level-normalised).
            Routings: the structure preset's routings, depths drawn the same way
  vae       per-category latent mixture of a VAE run (step6_sample.latent "gmm", tau 1) -> decoder;
            near-zero routing depths (|depth| < 0.04) snapped to "no routing" (the decoder never outputs
            exactly 0)

Every row then goes through finish(): features that became active (an oscillator unmuted, an LFO newly
routed, an effect switched on) are filled from a donor preset in which they are active, and features
that are inactive (activity rules as in step1_dataset.encode, written for canonical rows) are set to NaN,
so the stored row's NaN pattern is the inverter's target mask.
"""

import json
import math
import re
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import norm, qmc

from pitcher.core.schema import OSC_TYPES
from pitcher.step4_vae.data import LABELS
from pitcher.synth_data.common import OUT, Base, block_prefix

FLAG_SUFFIXES = {"off", "sync", "ext", "abs", "deform"}
NO_ROUTE = 0.5          # routing depth code of "not routed" (depth -1..1 -> code 0..1)
ROUTE_SNAP = 0.02       # VAE rows: |code - 0.5| below this (|depth| < 0.04) = not routed
WIDEN = 0.05            # layered: share of knob values drawn uniform on the perceptual scale (per knob:
                        # with ~100 active knobs a row still gets ~5 values outside the human habits)
TIER_WEIGHT = {1: 3.0, 2: 1.0, 3: 0.1}
MIN_COND = 15          # layered: presets needed for a filter-type-specific value distribution  # how often a switch group of that tier is picked for a change

STRENGTHS = {
    #          sd on the 0-1 perceptual scale   share of tier-2   P(0, 1, 2, 3 switch changes)   routings added / removed (mean count)
    "small":  dict(sd1=0.015, sd2=0.010, frac2=0.20, n_switch=(0.9, 0.1, 0.0, 0.0), route_add=0.1, route_del=0.05),
    "medium": dict(sd1=0.050, sd2=0.035, frac2=0.35, n_switch=(0.5, 0.45, 0.05, 0.0), route_add=0.3, route_del=0.15),
    "large":  dict(sd1=0.150, sd2=0.100, frac2=0.60, n_switch=(0.1, 0.4, 0.35, 0.15), route_add=0.6, route_del=0.3),
}
STRENGTH_P = {"small": 0.3, "medium": 0.4, "large": 0.3}
LAYERED_SWITCH_P = (0.3, 0.4, 0.2, 0.1)


def lfo_of(source):
    """'Voice LFO 1' -> 0 ... 'Scene LFO 1' -> 6 (step1_dataset.encode.lfo_index)."""
    m = re.fullmatch(r"(Voice|Scene) LFO (\d)", source or "")
    return None if not m else int(m[2]) - 1 + (6 if m[1] == "Scene" else 0)


def _rule(name, feat):
    """Activity rule of one kept feature (encode.entries_for, on canonical names)."""
    base, _, suf = name.rpartition(".")
    n = base if suf in FLAG_SUFFIXES and base else name
    if n.startswith("route/"):
        r = feat.get("route") or {}
        if r.get("osc_type"):
            return ("osc_type", int(r["dest"].split("/")[1]), r["osc_type"])
        return ("always",)
    m = re.fullmatch(r"scene/osc/(\d)/(\w+)/(p\d|wt)", n)
    if m:
        return ("osc_type", int(m[1]), m[2])
    m = re.fullmatch(r"scene/osc/(\d)/\w+", n)
    if m:
        return ("osc", int(m[1]))
    if re.fullmatch(r"scene/filterunit/\d/model", n):
        return ("always",)
    m = re.fullmatch(r"scene/filterunit/(\d)/\w+", n)
    if m:
        return ("filter", int(m[1]))
    m = re.fullmatch(r"scene/lfo/(\d+)/\w+", n)
    if m:
        return ("lfo", int(m[1]))
    if n == "scene/wsunit/drive":
        return ("ws",)
    if n == "scene/noise_colour":
        return ("noise",)
    if n == "scene/fm_depth":
        return ("fm",)
    m = re.fullmatch(r"fx/(\w+)/(present|placement|position|send_level|return_level|p\d+)", n)
    if m:
        if m[2] == "present":
            return ("always",)
        return ("fx_send", m[1]) if m[2] in ("send_level", "return_level") else ("fx", m[1])
    return ("always",)


def _block(name):
    """Features filled from the same donor when they become active together."""
    p = block_prefix(name)
    if p:
        return p
    m = re.match(r"(scene/(?:lfo|filterunit)/\d+/)", name)
    return m[1] if m else name


class Space(Base):
    """The cleaned train presets + everything stage 0 / 0b measured, and the three row sources."""

    def __init__(self):
        super().__init__()
        self.idx = {f: i for i, f in enumerate(self.kept)}
        self.KM = self.K.to_numpy(np.float64)                     # train presets x kept features
        self.train_ids = np.array(self.K.index)
        self.rules = {f: _rule(f, self.feats[f]) for f in self.kept}
        self.routes = [f for f in self.kept if f.startswith("route/")]
        self.route_info = {f: self.feats[f]["route"] for f in self.routes}
        self.knobs = [f for f in self.kept if self.feats[f]["kind"] in ("cont", "ord") and not f.startswith("route/")]
        self.grid = {}
        for f in self.kept:
            if self.feats[f]["kind"] == "ord":
                lo, hi = self.feats[f]["natural_range"]
                self.grid[f] = 1.0 / (hi - lo) if hi > lo else 1.0
        self.active_rows = {f: np.flatnonzero(~np.isnan(self.KM[:, i])) for f, i in self.idx.items()}

        # tiers (stage 0): per group; untested / unreliable groups count as tier 2, gated tier 3 as tier 2
        G = pd.read_csv(OUT / "screening_groups.csv").set_index("group")
        self.groups_all = [(g, [f for f in fs if f in self.idx]) for g, fs in self.groups()]
        self.groups_all = [(g, fs) for g, fs in self.groups_all if fs]
        self.tier, self.gkind = {}, {}
        for g, fs in self.groups_all:
            t = 2
            if g in G.index and bool(G.at[g, "reliable"]):
                t = int(G.at[g, "tier"])
                if t == 3 and bool(G.at[g, "gated"]):
                    t = 2
            self.tier[g], self.gkind[g] = t, self.kind(fs)
        self.switch_groups = [(g, fs) for g, fs in self.groups_all if self.gkind[g] == "switch"]
        self.knob_groups = [(g, [f for f in fs if f in set(self.knobs)]) for g, fs in self.groups_all if self.gkind[g] == "knob"]
        self.knob_groups = [(g, fs) for g, fs in self.knob_groups if fs]
        self.route_tier = {fs[0]: self.tier[g] for g, fs in self.groups_all if self.gkind[g] == "route"}
        self.group_cols = {g: np.array([self.idx[f] for f in fs]) for g, fs in self.switch_groups}

        # perceptual scales (stage 0b)
        R = json.loads((OUT / "rulers.json").read_text())
        self.rulers = {f: (np.array(v["codes"]), np.array(v["ruler"])) for f, v in R.items() if f in self.idx}

        # human value distributions (routings: depths of the presets that use the routing)
        self.human = {}
        for f in self.knobs + self.routes:
            v = self.KM[:, self.idx[f]]
            v = v[~np.isnan(v)]
            if f.startswith("route/"):
                v = v[np.abs(v - NO_ROUTE) > 1e-6]
            self.human[f] = np.sort(v)

        # gain-only knobs (stage 0: most of the raw change is loudness): the set is level-normalised, so the
        # layered source keeps the structure preset's value instead of drawing them (drawing only risks silence)
        gain = set(G.index[G.gain_mostly.astype(bool)])
        self.gain_only = {f for g, fs in self.groups_all if g in gain for f in fs}
        # filter knobs depend on the filter type (a high-pass wants other cutoffs than a low-pass): human values
        # per (filter unit, type) where at least MIN_COND presets have it
        self.human_filter = {}
        for k in range(2):
            mc = self.idx[f"scene/filterunit/{k}/model"]
            types = np.array([self.opt(f"scene/filterunit/{k}/model", c) or "0:0" for c in self.KM[:, mc]])
            types = np.array([t.split(":")[0] for t in types])
            for f in self.knobs:
                if not f.startswith(f"scene/filterunit/{k}/"):
                    continue
                v = self.KM[:, self.idx[f]]
                for t in np.unique(types):
                    x = v[(types == t) & ~np.isnan(v)]
                    if len(x) >= MIN_COND:
                        self.human_filter[(f, t)] = np.sort(x)
        # structures (layered source): key = the switch settings that matter most
        sw = [f for g, fs in self.switch_groups if self.tier[g] == 1 for f in fs]
        keys = [tuple(np.nan_to_num(self.KM[i, [self.idx[f] for f in sw]], nan=-1).round(3)) for i in range(len(self.KM))]
        cnt = Counter(keys)
        self.struct_w = np.array([cnt[k] ** -0.5 for k in keys])
        self.n_structures = len(cnt)
        self.labels_tr = np.array([self.label[p] for p in self.train_ids])
        self._copula = None
        # fast row(): kept + mode-frozen codes of every preset as arrays
        self._pos = {int(p): i for i, p in enumerate(self.codes.index)}
        self._kept_np = self.codes[self.kept].to_numpy(np.float64)
        self._mode_np = self.pruned.loc[self.codes.index, self.mode_frozen].to_numpy(np.float64)
        self._empty = {n: math.nan for n in self.names}

    def row(self, pid):
        """Base.row (template rebuild row of a preset), from arrays."""
        i = self._pos[int(pid)]
        r = dict(self._empty)
        r.update(zip(self.mode_frozen, self._mode_np[i].tolist()))
        r.update(zip(self.kept, self._kept_np[i].tolist()))
        return r

    # --- small helpers ----------------------------------------------------------------------------
    def opt(self, name, code):
        if code is None or math.isnan(code):
            return None
        return self.feats[name]["options"][int(round(code))]

    def to_u(self, f, c):
        if f in self.rulers:
            codes, ruler = self.rulers[f]
            return float(np.interp(c, codes, ruler))
        return float(c)

    def from_u(self, f, u):
        if f in self.rulers:
            codes, ruler = self.rulers[f]
            c = float(np.interp(u, ruler, codes))
        else:
            c = float(u)
        return self.snap(f, c)

    def snap(self, f, c):
        c = min(max(c, 0.0), 1.0)
        if f in self.grid:
            c = round(c / self.grid[f]) * self.grid[f]
        return c

    # --- activity (encode.entries_for for canonical rows) --------------------------------------------
    def activity(self, row):
        """Set of kept features that are heard / used in this row."""
        v = lambda n: row.get(n, math.nan)
        nz = lambda x: not math.isnan(x) and abs(x - NO_ROUTE) > 1e-6
        otype = {}
        for o in range(3):
            t = self.opt(f"scene/osc/{o}/type", v(f"scene/osc/{o}/type"))
            otype[o] = OSC_TYPES[int(t)] if t is not None else None
        routed, lfos = set(), set()
        for f, r in self.route_info.items():
            if not nz(v(f)):
                continue
            if r.get("osc_type") and otype.get(int(r["dest"].split("/")[1])) != r["osc_type"]:
                continue
            routed.add(r["dest"])
            lfos.add(lfo_of(r["source"]))

        def audible(src):
            mute = v(f"scene/mute_{src}")
            if not math.isnan(mute) and mute >= 0.5:
                return False
            lv = v(f"scene/level_{src}")
            return (not math.isnan(lv) and lv > 0) or f"level_{src}" in routed

        fm = self.opt("scene/fm_switch", v("scene/fm_switch")) or 0
        osc_on = [audible(f"o{o + 1}") or (o in (0, 1) and audible("ring_12")) or (o in (1, 2) and audible("ring_23"))
                  or (fm == 1 and o == 1) or (fm in (2, 3) and o in (1, 2)) for o in range(3)]
        ftype = {}
        for k in range(2):
            m = self.opt(f"scene/filterunit/{k}/model", v(f"scene/filterunit/{k}/model"))
            ftype[k] = int(str(m).split(":")[0]) if m is not None else 0
        ws = self.opt("scene/wsunit/type", v("scene/wsunit/type")) or 0
        noise = audible("noise")
        act = set()
        for f, r in self.rules.items():
            k = r[0]
            if k == "always":
                ok = True
            elif k == "osc_type":
                ok = osc_on[r[1]] and otype[r[1]] == r[2]
            elif k == "osc":
                ok = osc_on[r[1]]
            elif k == "filter":
                ok = ftype[r[1]] != 0
            elif k == "lfo":
                ok = r[1] in lfos
            elif k == "ws":
                ok = ws != 0
            elif k == "noise":
                ok = noise
            elif k == "fm":
                ok = fm != 0
            elif k == "fx":
                ok = v(f"fx/{r[1]}/present") == 1
            else:  # fx_send
                ok = v(f"fx/{r[1]}/present") == 1 and self.opt(f"fx/{r[1]}/placement", v(f"fx/{r[1]}/placement")) == "send"
            if ok:
                act.add(f)
        return act

    # --- shared moves -------------------------------------------------------------------------------
    def complete(self, row, rng):
        """Active features without a value get one from a donor preset where they are active (per block,
        one donor, so an oscillator's or effect's knobs stay a consistent set)."""
        act = self.activity(row)
        missing = [f for f in self.kept if f in act and math.isnan(row[f])]
        blocks = {}
        for f in missing:
            blocks.setdefault(_block(f), []).append(f)
        for fs in blocks.values():
            cand = self.active_rows[fs[0]]
            donor = int(rng.choice(cand)) if len(cand) else None
            for f in fs:
                x = self.KM[donor, self.idx[f]] if donor is not None else math.nan
                if math.isnan(x):
                    c = self.active_rows[f]
                    x = self.KM[int(rng.choice(c)), self.idx[f]] if len(c) else float(self.feats[f].get("fill01") or 0.0)
                row[f] = float(x)
        return len(missing)

    def finish(self, row, rng):
        for _ in range(3):                      # filling can switch more things on (e.g. an LFO's routings)
            if not self.complete(row, rng):
                break
        act = self.activity(row)
        for f in self.kept:
            if f not in act:
                row[f] = math.nan
        return row

    def switch(self, row, k, rng):
        """k switch groups take a donor's value (donor: a train preset where the group is active and differs)."""
        if k <= 0:
            return []
        live = [(g, fs) for g, fs in self.switch_groups if any(not math.isnan(row[f]) for f in fs)]
        w = np.array([TIER_WEIGHT[self.tier[g]] for g, _ in live])
        picks = rng.choice(len(live), size=min(k, len(live)), replace=False, p=w / w.sum())
        done = []
        for i in picks:
            g, fs = live[i]
            cols = self.group_cols[g]
            b = np.array([row[f] for f in fs])
            V = self.KM[:, cols]
            ok = ~np.isnan(V).all(1) & ~np.isclose(V, b, equal_nan=True).all(1)
            cand = np.flatnonzero(ok)
            if not len(cand):
                continue
            d = int(rng.choice(cand))
            for f, c in zip(fs, cols):
                row[f] = float(self.KM[d, c])
            for p in {block_prefix(f) for f in fs} - {None}:   # knobs of a newly switched-in type
                for f in self.kept:
                    if f.startswith(p) and math.isnan(row[f]):
                        row[f] = float(self.KM[d, self.idx[f]])
            done.append(g)
        return done

    # --- source 1: perturbations of human presets -----------------------------------------------------
    def perturb(self, pid, strength, rng):
        S = STRENGTHS[strength]
        row = self.row(pid)
        sw = self.switch(row, rng.choice(4, p=S["n_switch"]), rng)
        self.complete(row, rng)
        moved = 0
        for g, fs in self.knob_groups:
            t = self.tier[g]
            if t == 3 or (t == 2 and rng.random() > S["frac2"]):
                continue
            sd = S["sd1"] if t == 1 else S["sd2"]
            for f in fs:
                c = row[f]
                if math.isnan(c):
                    continue
                u = self.to_u(f, c) + rng.normal(0, sd)
                u = abs(u) if u < 0 else (2 - u if u > 1 else u)        # reflect at the ends
                row[f] = self.from_u(f, min(max(u, 0.0), 1.0))
                moved += 1
        on = [f for f in self.routes if not math.isnan(row[f]) and abs(row[f] - NO_ROUTE) > 1e-6]
        off = [f for f in self.routes if not math.isnan(row[f]) and abs(row[f] - NO_ROUTE) <= 1e-6 and len(self.human[f])]
        for f in on:
            t = self.route_tier.get(f, 2)
            if t != 3 and (t == 1 or rng.random() < S["frac2"]):
                sd = (S["sd1"] if t == 1 else S["sd2"]) / 2
                row[f] = min(max(row[f] + rng.normal(0, sd), 0.0), 1.0)
        for f in rng.permutation(on)[:rng.poisson(S["route_del"])]:
            row[f] = NO_ROUTE
        for f in rng.permutation(off)[:rng.poisson(S["route_add"])]:
            row[f] = float(rng.choice(self.human[f]))
        return self.finish(row, rng), dict(switches=len(sw), moved=moved)

    # --- source 2: layered exploration -----------------------------------------------------------------
    def copula(self):
        """Cholesky factor of the knobs' + routings' correlation on normal scores (pairwise over the train
        presets where both are active; routings: only presets that use them), shrunk towards independence."""
        if self._copula is not None:
            return self._copula
        cols = self.knobs + self.routes
        S = np.full((len(self.KM), len(cols)), np.nan)
        for j, f in enumerate(cols):
            v = self.KM[:, self.idx[f]].copy()
            if f.startswith("route/"):
                v[np.abs(v - NO_ROUTE) < 1e-6] = np.nan
            m = ~np.isnan(v)
            if m.sum() >= 2:
                r = pd.Series(v[m]).rank().to_numpy()
                S[m, j] = norm.ppf((r - 0.5) / m.sum())
        M = (~np.isnan(S)).astype(float)
        X = np.nan_to_num(S)
        n = M.T @ M
        num = X.T @ X
        den = np.sqrt((X ** 2).T @ M * (M.T @ X ** 2))
        with np.errstate(invalid="ignore", divide="ignore"):
            R = np.where((n >= 30) & (den > 0), num / den, 0.0)
        R = np.nan_to_num((R + R.T) / 2)
        np.fill_diagonal(R, 1.0)
        R = 0.7 * R + 0.3 * np.eye(len(cols))
        w, V = np.linalg.eigh(R)
        R = (V * np.maximum(w, 2e-2)) @ V.T
        R = (R + R.T) / 2
        d = np.sqrt(np.diag(R))
        R = R / d[:, None] / d[None, :]
        for jitter in (0.0, 1e-4, 1e-3, 1e-2):
            try:
                L = np.linalg.cholesky(R + jitter * np.eye(len(cols)))
                break
            except np.linalg.LinAlgError:
                continue
        self._copula = (cols, L)
        return self._copula

    def copula_draw(self, n, seed):
        """(n, d) uniforms: scrambled Sobol -> normal -> correlated -> uniform."""
        cols, L = self.copula()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            P = qmc.Sobol(len(cols), scramble=True, seed=seed).random(n)
        Z = norm.ppf(np.clip(P, 1e-6, 1 - 1e-6)) @ L.T
        return cols, norm.cdf(Z)

    def human_q(self, f, u, h=None):
        h = self.human[f] if h is None else h
        if not len(h):
            return self.from_u(f, u)
        return self.snap(f, float(h[min(int(u * len(h)), len(h) - 1)]))

    def layered(self, label, u_row, rng):
        """u_row: {feature: uniform} from copula_draw (one row)."""
        ids = np.flatnonzero(self.labels_tr == label)
        w = self.struct_w[ids]
        pid = int(self.train_ids[rng.choice(ids, p=w / w.sum())])
        row = self.row(pid)
        sw = self.switch(row, rng.choice(4, p=LAYERED_SWITCH_P), rng)
        self.complete(row, rng)
        ftype = [str(self.opt(f"scene/filterunit/{k}/model", row[f"scene/filterunit/{k}/model"]) or "0:0").split(":")[0]
                 for k in range(2)]
        for f in self.knobs:
            if math.isnan(row[f]) or f in self.gain_only:
                continue
            u = u_row[f]
            pool = None
            if f.startswith("scene/filterunit/"):
                pool = self.human_filter.get((f, ftype[int(f.split("/")[2])]))
            row[f] = self.from_u(f, u) if rng.random() < WIDEN else self.human_q(f, u, pool)
        for f in self.routes:
            c = row[f]
            if not math.isnan(c) and abs(c - NO_ROUTE) > 1e-6:
                row[f] = self.human_q(f, u_row[f])
        return self.finish(row, rng), dict(structure_preset=pid, switches=len(sw))

    # --- source 3: VAE samples ---------------------------------------------------------------------------
    def vae_rows(self, run, label, n, seed):
        from pitcher.step6_sample.latent import decode_rows, get_gmms, load_model, sample_z
        from pitcher.step4_vae.data import Spec
        if not hasattr(self, "_vae"):
            model, cfg = load_model(run)
            self._vae = (model, Spec(), get_gmms(run, model=model), cfg["z"])
        model, spec, gmms, zd = self._vae
        rng = np.random.default_rng(seed)
        z = sample_z("gmm", label, n, 1.0, zd, gmms=gmms, rng=rng)
        out = []
        for r in decode_rows(model, spec, z, label):
            row = {k: r.get(k, math.nan) for k in self.names}
            for f in self.routes:
                if not math.isnan(row[f]) and abs(row[f] - NO_ROUTE) < ROUTE_SNAP:
                    row[f] = NO_ROUTE
            for f in self.grid:
                if not math.isnan(row[f]):
                    row[f] = self.snap(f, row[f])
            out.append(self.finish(row, rng))
        return out


def check_activity(space, n=None):
    """Agreement of Space.activity with the masks of step1_dataset.encode on the train presets."""
    M = pd.read_parquet(space.codes.attrs.get("masks_path", OUT.parent / "masks.parquet"))
    ids = space.train_ids[:n] if n else space.train_ids
    agree, rows = [], []
    for pid in ids:
        act = space.activity(space.row(int(pid)))
        mine = np.array([f in act for f in space.kept])
        truth = M.loc[int(pid), space.kept].to_numpy() > 0
        agree.append(mine == truth)
    A = np.array(agree)
    per = pd.Series(1 - A.mean(0), index=space.kept)
    return float(A.mean()), per[per > 0].sort_values(ascending=False)


if __name__ == "__main__":
    sp = Space()
    acc, worst = check_activity(sp)
    print(f"activity rules vs encode masks: {acc:.4f} of (preset, feature) cells agree; "
          f"{len(worst)} features ever differ; worst:")
    print(worst.head(25).round(3).to_string())
