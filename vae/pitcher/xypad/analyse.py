"""XY-pad concept: which continuous knobs decide the category, and which can move together.

Knobs: continuous Surge parameters (no switches, no choices, no stepped knobs, no routing depths) that are
active in >= 25% of the cleaned presets (1,719: train + val + test). Inactive = NaN (an effect knob
without the effect, a Classic knob on a wavetable oscillator).

1. Do knobs move together in real presets? Spearman correlation over the presets where both are active
   (>= 150). Average-linkage clusters at |rho| >= 0.5 give only 8 small groups, several of them chance
   pairs (an LFO's hold with the delay's return level): the presets don't tie knobs together. So the
   groups are FUNCTIONAL: the same job on several units (cutoff of both filters, the detune of both
   Classic oscillators, the reverb's size + decay). The mean |rho| inside each group is reported.
2. Label importance: random forest (300 trees, balanced) on all knobs, 5-fold stratified CV, permutation
   importance in the held-out fold (5 shuffles; a group's knobs shuffled together = one slider's worth);
   mean +- SE over folds; drop in balanced accuracy.
3. For the page: every preset's position on every group (mean of its knobs' positions between the knob's
   own 5th and 95th percentile, sign-aligned), category medians, and a small random forest (60 trees,
   depth 10) on the knobs, exported as arrays, for the category map behind the pad.

Writes  data/xypad/knobs.json + data/xypad/importance.csv
Run:    .venv/bin/python -m pitcher.xypad.analyse
"""

import json

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from pitcher.core.surge_io import PITCHER_ROOT
from pitcher.step4_vae.data import LABELS, Spec

DATA = PITCHER_ROOT / "data"
OUT = DATA / "xypad"
MIN_ACTIVE = 0.25
MIN_PAIR = 150
RHO = 0.5

S, F1, F2, O1, O2 = "scene/", "scene/filterunit/0/", "scene/filterunit/1/", "scene/osc/0/", "scene/osc/1/"
# (group, section, [knobs], plain description); sections order the page's list headings only
GROUPS = [
    ("Amp attack", "Amp envelope", [S + "adsr/0/a"], "how fast the sound starts"),
    ("Amp decay", "Amp envelope", [S + "adsr/0/d"], "how fast it falls to the sustain level"),
    ("Amp sustain", "Amp envelope", [S + "adsr/0/s"], "level while the key is held"),
    ("Amp release", "Amp envelope", [S + "adsr/0/r"], "how long it rings after the key is released"),
    ("Velocity → volume", "Amp envelope", [S + "vca_velsense"], "how much harder playing gets louder"),
    ("Filter env attack", "Filter envelope", [S + "adsr/1/a"], "how fast the filter opens"),
    ("Filter env decay", "Filter envelope", [S + "adsr/1/d"], "how fast the filter closes again"),
    ("Filter env sustain", "Filter envelope", [S + "adsr/1/s"], "filter opening while held"),
    ("Filter env release", "Filter envelope", [S + "adsr/1/r"], "filter movement after release"),
    ("Cutoff", "Filters", [F1 + "cutoff", F2 + "cutoff"], "brightness: both filters' cutoff together"),
    ("Resonance", "Filters", [F1 + "resonance", F2 + "resonance"], "peak at the cutoff (both filters)"),
    ("Filter env amount", "Filters", [F1 + "envmod", F2 + "envmod"], "how far the filter envelope moves the cutoff"),
    ("Filter keytrack", "Filters", [F1 + "keytrack", F2 + "keytrack"], "cutoff follows the note"),
    ("Filter balance", "Filters", [S + "filter_balance"], "filter 1 ↔ filter 2"),
    ("Drive", "Filters", [S + "wsunit/drive", S + "level_pfg"], "waveshaper drive + gain into the filters"),
    ("Glide", "Voice", [S + "portamento"], "pitch slides between notes"),
    ("Drift", "Voice", [S + "drift"], "slow analog pitch wander"),
    ("Osc 2/3 detune", "Oscillators", [O2 + "pitch", "scene/osc/2/pitch"], "pitch offset of osc 2 and 3"),
    ("Osc 1 pitch", "Oscillators", [O1 + "pitch"], "fine pitch of osc 1"),
    ("Osc 1 level", "Oscillators", [S + "level_o1"], "mixer level of osc 1"),
    ("Osc 2/3 level", "Oscillators", [S + "level_o2", S + "level_o3"], "mixer level of osc 2 and 3"),
    ("Noise level", "Oscillators", [S + "level_noise"], "mixer level of the noise generator"),
    ("Ring mod level", "Oscillators", [S + "level_ring_12", S + "level_ring_23"], "ring modulation 1×2 and 2×3"),
    ("Feedback", "Oscillators", [S + "feedback"], "filter output back into the input"),
    ("FM depth", "Oscillators", [S + "fm_depth"], "frequency modulation between oscillators"),
    ("Classic shape", "Oscillators", [O1 + "classic/p0", O2 + "classic/p0"], "triangle ↔ saw ↔ square"),
    ("Pulse width", "Oscillators", [O1 + "classic/p1", O1 + "classic/p2", O2 + "classic/p1", O2 + "classic/p2"], "width 1 and 2"),
    ("Sub mix", "Oscillators", [O1 + "classic/p3", O2 + "classic/p3"], "sub-oscillator one octave lower"),
    ("Sync", "Oscillators", [O1 + "classic/p4", O2 + "classic/p4"], "hard-sync pitch"),
    ("Unison detune", "Oscillators", [O1 + "classic/p5", O2 + "classic/p5", O1 + "wavetable/p5"], "spread of the stacked voices"),
    ("Wavetable morph", "Oscillators", [O1 + "wavetable/p0"], "position in the wavetable"),
    ("Wavetable colour", "Oscillators", [O1 + "wavetable/p1", O1 + "wavetable/p2", O1 + "wavetable/p3", O1 + "wavetable/p4"], "skew, saturate, formant"),
    ("LFO 1 rate", "Modulation", [S + "lfo/0/rate"], "speed of LFO 1"),
    ("LFO 1 amount", "Modulation", [S + "lfo/0/magnitude"], "depth of LFO 1"),
    ("LFO 1 shape tweak", "Modulation", [S + "lfo/0/deform", S + "lfo/0/start_phase"], "deform + start phase"),
    ("LFO 1 envelope", "Modulation", [S + f"lfo/0/{p}" for p in ("delay", "attack", "hold", "decay", "sustain", "release")], "fade-in / out of LFO 1"),
    ("LFO 2 rate", "Modulation", [S + "lfo/1/rate"], "speed of LFO 2"),
    ("LFO 2 amount", "Modulation", [S + "lfo/1/magnitude"], "depth of LFO 2"),
    ("LFO 2 shape tweak", "Modulation", [S + "lfo/1/deform", S + "lfo/1/start_phase"], "deform + start phase"),
    ("LFO 2 envelope", "Modulation", [S + f"lfo/1/{p}" for p in ("delay", "attack", "hold", "decay", "sustain", "release")], "fade-in / out of LFO 2"),
    ("LFO 3 rate / amount", "Modulation", [S + "lfo/2/rate", S + "lfo/2/magnitude"], "LFO 3"),
    ("LFO 3 envelope", "Modulation", [S + f"lfo/2/{p}" for p in ("delay", "attack", "hold", "decay", "sustain", "release", "deform", "start_phase")], "LFO 3 envelope + shape"),
    ("Reverb size / decay", "Effects", ["fx/reverb/p2", "fx/reverb/p3"], "room size + decay time"),
    ("Reverb mix", "Effects", ["fx/reverb/p9", "fx/reverb/send_level", "fx/reverb/return_level"], "how much reverb"),
    ("Reverb pre-delay", "Effects", ["fx/reverb/p0"], "gap before the reverb"),
    ("Reverb tone", "Effects", ["fx/reverb/p4", "fx/reverb/p5", "fx/reverb/p6", "fx/reverb/p7", "fx/reverb/p8"], "damping, low / high cut, EQ"),
    ("Reverb width", "Effects", ["fx/reverb/p10"], "stereo width of the reverb"),
    ("Delay time", "Effects", ["fx/delay/p0", "fx/delay/p1"], "left + right delay time"),
    ("Delay feedback", "Effects", ["fx/delay/p2", "fx/delay/p3"], "feedback + crossfeed: number of echoes"),
    ("Delay tone", "Effects", ["fx/delay/p4", "fx/delay/p5"], "low / high cut of the echoes"),
    ("Delay modulation", "Effects", ["fx/delay/p6", "fx/delay/p7"], "wobble rate + depth"),
    ("Delay mix", "Effects", ["fx/delay/p10", "fx/delay/send_level", "fx/delay/return_level"], "how much delay"),
    ("Delay width / input", "Effects", ["fx/delay/p11", "fx/delay/p8"], "stereo width, input channel"),
    ("EQ gains", "Effects", ["fx/eq/p0", "fx/eq/p3", "fx/eq/p6"], "boost / cut of the 3 bands"),
    ("EQ frequencies", "Effects", ["fx/eq/p1", "fx/eq/p4", "fx/eq/p7"], "where the 3 bands sit"),
    ("EQ bandwidth", "Effects", ["fx/eq/p2", "fx/eq/p5", "fx/eq/p8"], "how wide the bands are"),
    ("EQ output", "Effects", ["fx/eq/p9", "fx/eq/p10"], "EQ gain + mix"),
    ("Volume", "Output", [S + "volume", S + "vca_level"], "scene volume + amp level"),
    ("Stereo width", "Output", [S + "width", S + "pan"], "width + pan"),
    ("Low cut", "Output", [S + "lowcut"], "removes the lowest frequencies"),
    ("Pitch", "Output", [S + "pitch"], "fine tune of the whole sound"),
]


def presets():
    C = pd.read_parquet(DATA / "codes.parquet")
    sp = pd.read_parquet(DATA / "splits.parquet").set_index("preset_id")
    excl = set(pd.read_csv(DATA / "vae_excluded.csv").preset_id)
    ids = [p for p in sp.index if p not in excl and p in C.index]
    return C.loc[ids], sp.loc[ids, "label"]


def importance(X, y, sets, folds=5, reps=5, seed=0):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold
    rng = np.random.default_rng(seed)
    drops, accs = {k: [] for k in sets}, []
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        rf = RandomForestClassifier(300, class_weight="balanced", min_samples_leaf=2, n_jobs=4, random_state=seed).fit(X[tr], y[tr])
        base = balanced_accuracy_score(y[te], rf.predict(X[te]))
        accs.append(base)
        for k, idx in sets.items():
            d = []
            for _ in range(reps):
                Xp = X[te].copy()
                Xp[:, idx] = Xp[rng.permutation(len(te))][:, idx]
                d.append(base - balanced_accuracy_score(y[te], rf.predict(Xp)))
            drops[k].append(np.mean(d))
    return {k: (float(np.mean(v)), float(np.std(v) / np.sqrt(len(v)))) for k, v in drops.items()}, float(np.mean(accs))


def export_forest(rf):
    trees = []
    for est in rf.estimators_:
        t = est.tree_
        leaf = t.children_left == -1
        val = t.value[:, 0, :] / t.value[:, 0, :].sum(1, keepdims=True)
        trees.append(dict(f=t.feature.tolist(), t=np.round(t.threshold, 5).tolist(), l=t.children_left.tolist(),
                          r=t.children_right.tolist(), v=[np.round(val[i], 3).tolist() if leaf[i] else 0 for i in range(t.node_count)]))
    return trees


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s = Spec()
    F = {f["name"]: f for f in s.feats}
    C, lab = presets()
    cand = [n for n in s.cont if F[n]["kind"] == "cont" and not n.startswith("route/")]
    cols = [n for n in cand if C[n].notna().mean() >= MIN_ACTIVE]
    K = C[cols].astype(float)
    print(f"{len(K)} presets, {len(cols)} continuous knobs active in >= {MIN_ACTIVE:.0%}")
    groups = [(g, sec, [k for k in ks if k in cols], d) for g, sec, ks, d in GROUPS]
    covered = {k for _, _, ks, _ in groups for k in ks}
    assert covered == set(cols), f"knobs without a group: {sorted(set(cols) - covered)}"
    groups = [g for g in groups if g[2]]

    # 1. correlation structure
    R = K.corr(method="spearman", min_periods=MIN_PAIR)
    D = 1 - R.abs().fillna(0).to_numpy()
    np.fill_diagonal(D, 0)
    cl = fcluster(linkage(squareform(D, checks=False), "average"), 1 - RHO, "distance")
    multi = [[c for c, k in zip(cols, cl) if k == q] for q in set(cl)]
    multi = [m for m in multi if len(m) > 1]
    print(f"correlation clusters (|rho| >= {RHO}) with > 1 knob: {len(multi)}")
    for m in multi:
        print("   ", ", ".join(m))
    off = R.where(~np.eye(len(cols), dtype=bool)).abs()
    print(f"share of knob pairs with |rho| >= 0.3: {(off >= 0.3).sum().sum() / off.notna().sum().sum():.1%}")

    # 2. importance (knobs and groups)
    y = lab.map({l: i for i, l in enumerate(LABELS)}).to_numpy()
    X = K.fillna(-1.0).to_numpy()
    sets = {c: [i] for i, c in enumerate(cols)} | {f"G:{g}": [cols.index(k) for k in ks] for g, _, ks, _ in groups}
    cache = OUT / "_importance_cache.json"
    if cache.exists() and set(json.loads(cache.read_text())["imp"]) == set(sets):
        c = json.loads(cache.read_text()); imp, acc = {k: tuple(v) for k, v in c["imp"].items()}, c["acc"]
    else:
        imp, acc = importance(X, y, sets)
        cache.write_text(json.dumps(dict(imp=imp, acc=acc)))
    print(f"random forest on the knobs: balanced accuracy {acc:.3f} (8 categories, chance 0.125)")

    # 3. page data
    lo = K.quantile(0.05)
    hi = K.quantile(0.95)
    span = (hi - lo).replace(0, 1e-6)
    out_groups, pos = [], {}
    for g, sec, ks, desc in groups:
        lead = max(ks, key=lambda c: imp[c][0])
        sign = {c: (1 if (len(ks) == 1 or c == lead or (R.at[lead, c] if R.at[lead, c] == R.at[lead, c] else 0) >= 0) else -1) for c in ks}
        unit = lambda c: ((K[c] - lo[c]) / span[c]).clip(0, 1)
        u = pd.concat([unit(c) if sign[c] > 0 else 1 - unit(c) for c in ks], axis=1).mean(1)
        pos[g] = u
        rho = float(np.nanmean([abs(R.at[a, b]) for a in ks for b in ks if a < b])) if len(ks) > 1 else None
        by_label = {l: (float(u[lab == l].median()) if u[lab == l].notna().mean() > 0.3 else None) for l in LABELS}
        out_groups.append(dict(name=g, section=sec, desc=desc, knobs=ks, signs=sign, importance=imp[f"G:{g}"][0],
                               se=imp[f"G:{g}"][1], mean_abs_rho=rho, active=float(u.notna().mean()), by_label=by_label))
    out_groups.sort(key=lambda d: -d["importance"])

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    small = RandomForestClassifier(60, max_depth=10, min_samples_leaf=3, class_weight="balanced", n_jobs=4, random_state=0)
    small_acc = float(cross_val_score(small, X, y, cv=5, scoring="balanced_accuracy").mean())
    small.fit(X, y)
    print(f"page forest (60 trees, depth 10): balanced accuracy {small_acc:.3f}")

    knobs = [dict(name=c, ctrltype=F[c]["ctrltypes"][0], natural_range=F[c]["natural_range"], p5=float(lo[c]), p95=float(hi[c]),
                  default=float(F[c]["fill01"]), active=float(K[c].notna().mean()), importance=imp[c][0], se=imp[c][1]) for c in cols]
    P = np.round(K.to_numpy(), 3)
    res = dict(n_presets=len(K), labels=LABELS, rf_balanced_accuracy=acc, page_forest_balanced_accuracy=small_acc,
               correlation_clusters=multi, knobs=knobs, groups=out_groups,
               presets=dict(label=[LABELS.index(l) for l in lab], values=[[None if v != v else float(v) for v in row] for row in P]),
               forest=dict(features=cols, missing=-1.0, trees=export_forest(small)))
    (OUT / "knobs.json").write_text(json.dumps(res, separators=(",", ":")))
    T = pd.DataFrame([dict(group=g["name"], section=g["section"], n_knobs=len(g["knobs"]), importance=g["importance"], se=g["se"],
                           mean_abs_rho=g["mean_abs_rho"], active=g["active"], knobs=";".join(g["knobs"])) for g in out_groups])
    T.to_csv(OUT / "importance.csv", index=False)
    pd.set_option("display.width", 220)
    print(T.drop(columns="knobs").round(4).to_string(index=False))
    print(f"knobs.json: {(OUT / 'knobs.json').stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
