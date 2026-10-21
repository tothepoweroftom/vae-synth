"""Step 2.2: PCA / UMAP, and a check of the 1%-pruning (PIPELINE_VAE_SURGE.md, Steps 2.1-2.2).

Question: are the features that 1% pruning would drop important for the structure of the data?
  1. PCA (fit on train, model-ready matrix) -> per-feature importance = variance it carries in the
     principal components that explain 90% of the variance (sum over its columns of
     explained_variance_k x loading_kc^2). Where do the dropped features rank?
     Caveat: a feature that changes in < 1% of presets can hardly carry variance, so PCA ranks such
     features low almost by construction; hence also check 2 and 3.
  2. UMAP (train) on all features vs on the 1%-kept features: do presets keep the same neighbours?
     (UMAP has no loadings, so it cannot rank features; it can show whether pruning changes the map.)
  3. Category check: random forest on train, balanced accuracy on val, all vs kept features.
(The 3% comparison of Round 9 is recorded in IMPLEMENTATION_LOG.md; the pruning is now final at 1%.)
Plots (small multiples, one panel per category): runs/step2/{pca,umap_all,umap_kept1pct}.png

Run:  .venv/bin/python -m pitcher.step2_explore.explore
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import NearestNeighbors

from pitcher.core.surge_io import PITCHER_ROOT

DATA, OUT = PITCHER_ROOT / "data", PITCHER_ROOT / "runs" / "step2"
SURFACE, INK, MUTED, OTHERS, HIGHLIGHT = "#fcfcfb", "#1f1f1d", "#6b6a63", "#d9d7d0", "#2a78d6"
LABELS = ["Bass", "Lead", "Pad", "Keys", "Pluck", "Synth", "Atmosphere", "Orchestral"]


def knn_sets(X, k=15):
    idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X, return_distance=False)[:, 1:]
    return [set(r) for r in idx]


def overlap(a, b):
    return float(np.mean([len(x & y) / len(x | y) for x, y in zip(a, b)]))


def small_multiples(emb, labels, title, path, xlabel, ylabel):
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.2), sharex=True, sharey=True, facecolor=SURFACE)
    for ax, lab in zip(axes.flat, LABELS):
        sel = labels == lab
        ax.set_facecolor(SURFACE)
        ax.scatter(emb[~sel, 0], emb[~sel, 1], s=4, c=OTHERS, linewidths=0, rasterized=True)
        ax.scatter(emb[sel, 0], emb[sel, 1], s=9, c=HIGHLIGHT, linewidths=0, rasterized=True)
        ax.set_title(f"{lab} (n={sel.sum()})", fontsize=11, color=INK, loc="left")
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.grid(color="#ecebe6", linewidth=0.6)
        ax.set_axisbelow(True)
    for ax in axes[1]:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=INK)
    fig.text(0.01, 0.935, "Each panel highlights one category (blue) against all other train presets (grey).",
             fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    schema = json.loads((DATA / "schema.json").read_text())
    cols_of = {f["name"]: f["columns"] for f in schema["features"]}
    X = pd.read_parquet(DATA / "dataset.parquet")
    splits = pd.read_parquet(DATA / "splits.parquet").set_index("preset_id")
    tr, va = splits.index[splits.split == "train"], splits.index[splits.split == "val"]
    y_tr, y_va = splits.loc[tr, "label"].to_numpy(), splits.loc[va, "label"].to_numpy()
    kept1 = json.loads((DATA / "pruning.json").read_text())["kept"]
    dropped1 = [f for f in cols_of if f not in set(kept1)]
    cols1 = [c for f in kept1 for c in cols_of[f]]
    Xtr = X.loc[tr].to_numpy()

    # --- 1. PCA ----------------------------------------------------------------------------------
    pca = PCA(n_components=min(150, Xtr.shape[1]), random_state=0).fit(Xtr)
    cum = np.cumsum(pca.explained_variance_ratio_)
    K = int(np.searchsorted(cum, 0.90) + 1)
    w = pca.explained_variance_[:K, None] * pca.components_[:K] ** 2  # (K, columns)
    col_imp = pd.Series(w.sum(axis=0), index=X.columns)
    imp = pd.Series({f: col_imp[c].sum() for f, c in cols_of.items()})
    imp = (imp / imp.sum()).sort_values(ascending=False)
    rank = pd.Series(np.arange(1, len(imp) + 1), index=imp.index)
    pd.DataFrame({"pca_importance_share": imp, "rank": rank,
                  "kept": [f in set(kept1) for f in imp.index]}).rename_axis("feature") \
        .to_csv(DATA / "pca_feature_importance.csv")
    r_drop = rank[dropped1]
    print(f"PCA (train, {Xtr.shape[1]} columns): PC1 {pca.explained_variance_ratio_[0]:.1%}, PC2 "
          f"{pca.explained_variance_ratio_[1]:.1%}; {K} components explain 90% (80%: "
          f"{int(np.searchsorted(cum, 0.80) + 1)})")
    print(f"\n1%-dropped features: {len(dropped1)} of {len(imp)}")
    print(f"  in PCA top 25 / 50 / 100 / 200: {(r_drop <= 25).sum()} / {(r_drop <= 50).sum()} / "
          f"{(r_drop <= 100).sum()} / {(r_drop <= 200).sum()}")
    print(f"  their share of PCA importance: {imp[dropped1].sum():.2%} (kept features: {imp[kept1].sum():.2%})")
    print(f"  best-ranked dropped features:")
    for f in r_drop.sort_values().index[:10]:
        print(f"    rank {rank[f]:4d}  {imp[f]:.3%}  {f}")
    print("  top 15 features overall:", ", ".join(f"{f} ({imp[f]:.1%})" for f in imp.index[:15]))

    # --- 2. UMAP, all vs 1%-kept ----------------------------------------------------------------------
    Xtr1 = X.loc[tr, cols1].to_numpy()
    reducer = dict(n_neighbors=15, min_dist=0.1, random_state=0)
    emb_all = umap.UMAP(**reducer).fit_transform(Xtr)
    emb_1 = umap.UMAP(**reducer).fit_transform(Xtr1)
    nn_feat = overlap(knn_sets(Xtr), knn_sets(Xtr1))
    nn_umap = overlap(knn_sets(emb_all), knn_sets(emb_1))
    rnd = np.random.default_rng(0)
    baseline = overlap(knn_sets(Xtr), [set(rnd.choice(len(Xtr), 15, replace=False)) for _ in range(len(Xtr))])
    print(f"\nUMAP / neighbours (k=15, train): overlap of each preset's neighbours, all vs 1%-kept features:")
    print(f"  in feature space: {nn_feat:.2f}   in the 2-D UMAP maps: {nn_umap:.2f}   (random neighbours: {baseline:.3f})")

    # --- 3. category check -------------------------------------------------------------------------------
    print("\ncategory check (random forest on train, balanced accuracy on val):")
    acc = {}
    for name, cols in [("all features", list(X.columns)), ("1%-kept", cols1)]:
        rf = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=0, n_jobs=-1)
        rf.fit(X.loc[tr, cols], y_tr)
        acc[name] = balanced_accuracy_score(y_va, rf.predict(X.loc[va, cols]))
        print(f"  {name:13s} {len(cols):5d} columns: {acc[name]:.3f}")

    # --- plots ----------------------------------------------------------------------------------------------
    pcs = pca.transform(Xtr)[:, :2]
    ev = pca.explained_variance_ratio_
    small_multiples(pcs, y_tr, "PCA of train presets, all features", OUT / "pca.png",
                    f"PC1 ({ev[0]:.1%})", f"PC2 ({ev[1]:.1%})")
    small_multiples(emb_all, y_tr, "UMAP of train presets, all features", OUT / "umap_all.png", "UMAP 1", "UMAP 2")
    small_multiples(emb_1, y_tr, "UMAP of train presets, 1%-kept features", OUT / "umap_kept1pct.png",
                    "UMAP 1", "UMAP 2")
    (OUT / "summary.json").write_text(json.dumps(dict(
        pca_components_90pct=K, pc1=float(ev[0]), pc2=float(ev[1]),
        dropped1_in_top_25_50_100=[int((r_drop <= k).sum()) for k in (25, 50, 100)],
        dropped1_pca_share=float(imp[dropped1].sum()), neighbour_overlap_feature_space=nn_feat,
        neighbour_overlap_umap=nn_umap, neighbour_overlap_random=baseline, rf_balanced_acc_val=acc), indent=1))
    print(f"\nplots: {OUT}/pca.png, umap_all.png, umap_kept1pct.png")


if __name__ == "__main__":
    main()
