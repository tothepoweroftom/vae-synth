"""Step 3.1-3.2: rank parameter groups for the core set (PIPELINE_VAE_SURGE.md, Step 3).

Group = a kept feature with its flags (e.g. LFO rate + .sync); an effect's present / placement /
position form one "fx/<type>" group. Only the 536 features kept in Step 2.1 are candidates.

Three criteria, all on the TRAIN split only (val is used later to choose K, test stays untouched):
  1. category importance: random forest; grouped permutation importance (all columns of a group
     shuffled together), drop in balanced accuracy; 5-fold StratifiedGroupKFold inside train
     (near-twin groups from Step 1.5 never on both sides), 5 permutations per fold
  2. within-category variation ("character"): per category, the spread of the group's main feature
     among presets where it's active (cont/ord: std of the 0-1 code; cat/bool: Gini impurity),
     times its active share; averaged over the categories
  3. structural must-haves (sound would break or go silent without them): oscillator types, levels
     and mutes, filter 1 model + cutoff, amp envelope A/D/S/R, scene volume
Combined order: must-haves first, then alternately the next group from criterion 1 and from 2.
Dependencies: an effect knob pulls in its fx/<type> group; per-type oscillator knobs need the
oscillator type (a must-have).

Writes  data/core_ranking.csv   one row per group: features, columns, importance, variation, ranks, order
Run:    .venv/bin/python -m pitcher.step3_core.rank
"""

import json
import re
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

from pitcher.core.surge_io import PITCHER_ROOT

DATA = PITCHER_ROOT / "data"
MUST_HAVE = (["scene/osc/%d/type" % o for o in range(3)] + ["scene/level_o%d" % o for o in (1, 2, 3)]
             + ["scene/mute_o%d" % o for o in (1, 2, 3)]
             + ["scene/filterunit/0/model", "scene/filterunit/0/cutoff", "scene/volume"]
             + ["scene/adsr/0/%s" % x for x in "adsr"])
N_PERM, SEED = 5, 0


def build_groups(kept, feats):
    groups = {}
    for name in kept:
        f = feats[name]
        if "flag_of" in f:
            continue
        m = re.fullmatch(r"fx/(\w+)/(present|placement|position)", name)
        key = f"fx/{m[1]}" if m else name
        groups.setdefault(key, []).append(name)
    for name in kept:  # flags join their parent's group
        parent = feats[name].get("flag_of")
        if parent:
            key = next((k for k, v in groups.items() if parent in v), None)
            if key:
                groups[key].append(name)
    return groups


def main():
    t0 = time.time()
    schema = json.loads((DATA / "schema.json").read_text())
    feats = {f["name"]: f for f in schema["features"]}
    kept = json.loads((DATA / "pruning.json").read_text())["kept"]
    groups = build_groups(kept, feats)
    cols_of = {g: [c for n in names for c in feats[n]["columns"]] for g, names in groups.items()}

    X = pd.read_parquet(DATA / "dataset.parquet")
    codes, masks = pd.read_parquet(DATA / "codes.parquet"), pd.read_parquet(DATA / "masks.parquet")
    sp = pd.read_parquet(DATA / "splits.parquet").set_index("preset_id")
    tr = sp.index[sp.split == "train"]
    cols = [c for g in groups for c in cols_of[g]]
    Xtr, ytr, gtr = X.loc[tr, cols], sp.loc[tr, "label"].to_numpy(), sp.loc[tr, "group_id"].to_numpy()

    # --- 1. grouped permutation importance, 5-fold inside train ------------------------------
    rng = np.random.default_rng(SEED)
    col_idx = {c: i for i, c in enumerate(cols)}
    drops = {g: [] for g in groups}
    base_scores = []
    for fold, (a, b) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(Xtr, ytr, gtr)):
        rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1)
        rf.fit(Xtr.iloc[a], ytr[a])
        Xb = Xtr.iloc[b].to_numpy()
        base = balanced_accuracy_score(ytr[b], rf.predict(Xb))
        base_scores.append(base)
        for g in groups:
            ix = [col_idx[c] for c in cols_of[g]]
            for _ in range(N_PERM):
                Xp = Xb.copy()
                Xp[:, ix] = Xb[rng.permutation(len(Xb))][:, ix]
                drops[g].append(base - balanced_accuracy_score(ytr[b], rf.predict(Xp)))
        print(f"  fold {fold + 1}/5: balanced accuracy {base:.3f}, {time.time() - t0:.0f}s", flush=True)
    imp = pd.Series({g: float(np.mean(v)) for g, v in drops.items()})
    imp_sd = pd.Series({g: float(np.std(v) / np.sqrt(len(v))) for g, v in drops.items()})

    # --- 2. within-category variation of the group's main feature ------------------------------
    labels = sp.loc[tr, "label"]
    var = {}
    for g, names in groups.items():
        main = names[0]
        kind = feats[main]["kind"]
        per_class = []
        for lab in labels.unique():
            ids = labels.index[labels == lab]
            act = masks.loc[ids, main] == 1
            v = codes.loc[ids, main][act]
            if len(v) < 2:
                per_class.append(0.0)
                continue
            spread = float(v.std()) if kind in ("cont", "ord") else float(1 - (v.value_counts(normalize=True) ** 2).sum())
            per_class.append(spread * act.mean())
        var[g] = float(np.mean(per_class))
    var = pd.Series(var)

    # --- 3. combined order -----------------------------------------------------------------------------
    must = [g for g in groups if g in MUST_HAVE]
    by_imp, by_var = list(imp.sort_values(ascending=False).index), list(var.sort_values(ascending=False).index)
    order, seen = [], set()

    def add(g):
        if g in seen:
            return
        m = re.fullmatch(r"fx/(\w+)/\w+", g)  # an effect knob needs its effect
        if m and f"fx/{m[1]}" in groups:
            add(f"fx/{m[1]}")
        seen.add(g)
        order.append(g)

    for g in must:
        add(g)
    i = j = 0
    while len(order) < len(groups):
        while i < len(by_imp) and by_imp[i] in seen:
            i += 1
        if i < len(by_imp):
            add(by_imp[i])
        while j < len(by_var) and by_var[j] in seen:
            j += 1
        if j < len(by_var):
            add(by_var[j])

    out = pd.DataFrame({
        "group": order,
        "order": range(1, len(order) + 1),
        "must_have": [g in MUST_HAVE for g in order],
        "importance": [imp[g] for g in order],
        "importance_se": [imp_sd[g] for g in order],
        "importance_rank": [by_imp.index(g) + 1 for g in order],
        "variation": [var[g] for g in order],
        "variation_rank": [by_var.index(g) + 1 for g in order],
        "n_columns": [len(cols_of[g]) for g in order],
        "features": [";".join(groups[g]) for g in order],
    })
    out["cum_columns"] = out.n_columns.cumsum()
    out.to_csv(DATA / "core_ranking.csv", index=False)

    print(f"\n{len(groups)} groups from {len(kept)} kept features; RF balanced accuracy (5-fold inside train): "
          f"{np.mean(base_scores):.3f} ± {np.std(base_scores):.3f}; {time.time() - t0:.0f}s")
    print(f"must-haves present: {len(must)} of {len(MUST_HAVE)}")
    print(f"importance > 0 (beyond 2 standard errors): {(imp > 2 * imp_sd).sum()} groups")
    print("\ntop 25 by category importance (drop in balanced accuracy):")
    for g in by_imp[:25]:
        print(f"  {imp[g]:+.4f} ± {imp_sd[g]:.4f}  {g}")
    print("\ntop 15 by within-category variation:")
    for g in by_var[:15]:
        print(f"  {var[g]:.3f}  {g}")
    print("\ncombined order, first 40:", ", ".join(order[:40]))
    print("columns at K = 30 / 45 / 60 / 90 / 150 groups:", [int(out.cum_columns.iloc[k - 1]) for k in (30, 45, 60, 90, 150)])


if __name__ == "__main__":
    main()
