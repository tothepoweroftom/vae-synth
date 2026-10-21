"""Step 1.5: train / val / test split, 80 / 10 / 10 (PIPELINE_VAE_SURGE.md, Step 1.5).

- Uses the curated set from Step 1.3 (data/codes.parquet; exclusions in data/excluded_presets.csv).
- Groups presets that must stay on the same side (union of):
    * same author + same name stem (numbered variants: "Cute 1", "Cute 2"; series like "Winter 1-10")
    * same author + same name in two category folders (same patch filed twice)
    * identical encoded parameter vector (duplicates under another name)
- Stratified by label: each label's share is (nearly) the same in train, val and test.
  StratifiedGroupKFold with 10 folds; over many seeds and all (test, val) fold choices, the split
  with the smallest maximum deviation of label shares (and of split sizes) is kept.

Writes  data/splits.json      {preset_id: "train" | "val" | "test"}  - written once, then fixed
        data/splits.parquet   preset_id, split, group_id, label, name, author
Run:    .venv/bin/python -m pitcher.step1_dataset.split
"""

import json
import re
from itertools import permutations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from pitcher.core.surge_io import PITCHER_ROOT

DATA = PITCHER_ROOT / "data"
N_FOLDS, SEEDS = 10, range(300)
# trailing variant marker: number (may be attached), (..), [..]; or a separate v2 / roman numeral / single a-d
SUFFIX = re.compile(r"(?:[\s_\-.]*(?:\(.*?\)|\[.*?\]|#?\d+)|[\s_\-.]+(?:v\d+|[ivx]{1,4}|[a-d]))$", re.I)


def name_stem(name):
    s = name.lower().strip()
    for _ in range(3):
        s2 = SUFFIX.sub("", s).strip()
        if s2 == s or not s2:
            break
        s = s2
    return s


class UnionFind:
    def __init__(self, items):
        self.p = {i: i for i in items}

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def score(labels, assign):
    """Max deviation (percentage points) of label shares per split from the overall shares,
    plus how far split sizes are from 80/10/10."""
    overall = labels.value_counts(normalize=True)
    dev = 0.0
    for s in ("train", "val", "test"):
        share = labels[assign == s].value_counts(normalize=True).reindex(overall.index, fill_value=0)
        dev = max(dev, float((share - overall).abs().max()) * 100)
    sizes = pd.Series(assign).value_counts(normalize=True)
    size_dev = max(abs(sizes.get("train", 0) - .8), abs(sizes.get("val", 0) - .1), abs(sizes.get("test", 0) - .1)) * 100
    return dev + size_dev, dev, size_dev


def main():
    # the curated set = the presets encoded in Step 1.3 (exclusions in data/excluded_presets.csv)
    presets = pd.read_parquet(DATA / "presets.parquet")
    codes = pd.read_parquet(DATA / "codes.parquet")
    excluded = pd.read_csv(DATA / "excluded_presets.csv")
    presets = presets[presets["preset_id"].isin(codes.index)].reset_index(drop=True)
    codes = codes.loc[presets["preset_id"]]

    # --- groups ---------------------------------------------------------------------------------
    uf = UnionFind(presets["preset_id"])
    presets["stem"] = presets["name"].map(name_stem)
    n_union = {"name stem": 0, "identical parameters": 0}
    for _, g in presets.groupby(["author", "stem"]):  # also covers same name in two folders
        ids = list(g["preset_id"])
        for i in ids[1:]:
            uf.union(ids[0], i)
            n_union["name stem"] += 1
    h = codes.round(4).fillna(-1).apply(lambda r: hash(tuple(r)), axis=1)
    for _, ids in h.groupby(h).groups.items():
        ids = list(ids)
        for i in ids[1:]:
            if uf.find(ids[0]) != uf.find(i):
                n_union["identical parameters"] += 1
            uf.union(ids[0], i)
    roots = presets["preset_id"].map(uf.find)
    presets["group_id"] = pd.factorize(roots)[0]

    # --- best stratified group split ----------------------------------------------------------------
    y, groups = presets["label"], presets["group_id"]
    best = None
    for seed in SEEDS:
        folds = np.zeros(len(presets), int)
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for k, (_, te) in enumerate(sgkf.split(presets, y, groups)):
            folds[te] = k
        for t, v in permutations(range(N_FOLDS), 2):
            assign = np.where(folds == t, "test", np.where(folds == v, "val", "train"))
            s = score(y, assign)
            if best is None or s[0] < best[0][0]:
                best = (s, seed, t, v, assign)
    (total, dev, size_dev), seed, t, v, assign = best
    presets["split"] = assign

    # --- write ----------------------------------------------------------------------------------------
    out = presets[["preset_id", "split", "group_id", "label", "name", "author"]]
    out.to_parquet(DATA / "splits.parquet", index=False)
    (DATA / "splits.json").write_text(json.dumps({int(p): s for p, s in zip(out.preset_id, out.split)}, indent=0))

    # --- report + checks --------------------------------------------------------------------------------
    sizes = out.groupby("group_id").size()
    print(f"curated set: {len(presets)} presets; excluded in Step 1.3: {excluded.reason.str.split(":").str[0].value_counts().to_dict()}")
    print(f"{len(out)} presets in {out.group_id.nunique()} groups "
          f"({(sizes > 1).sum()} groups with >1 preset, covering {sizes[sizes > 1].sum()} presets; largest {sizes.max()})")
    print(f"unions: {n_union}")
    print(f"chosen: seed {seed}, test fold {t}, val fold {v}; max label-share deviation {dev:.2f} pp, "
          f"split-size deviation {size_dev:.2f} pp")
    ct = pd.crosstab(out.label, out.split)[["train", "val", "test"]]
    pct = (ct / ct.sum() * 100).round(1)
    table = ct.astype(str) + " (" + pct.astype(str) + "%)"
    table["all"] = out.label.value_counts().astype(str) + " (" + (out.label.value_counts(normalize=True) * 100).round(1).astype(str) + "%)"
    tot = ct.sum()
    table.loc["TOTAL"] = [f"{tot[s]} ({tot[s] / len(out) * 100:.1f}%)" for s in ("train", "val", "test")] + [str(len(out))]
    print(table.to_string())
    leaks = out.groupby("group_id").split.nunique()
    print(f"check: groups spanning more than one split: {(leaks > 1).sum()}")
    print(f"check: every preset assigned once: {out.preset_id.is_unique and len(out) == len(presets)}")


if __name__ == "__main__":
    main()
