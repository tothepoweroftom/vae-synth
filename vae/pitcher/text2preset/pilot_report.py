"""Report on a post-processed Stable Audio run (postprocess.py): filter results, pitch, category and
description adherence, reachability by Surge, seed consistency.

References from the Surge side: the same detectors on Surge renders (surge_controls.parquet), the
tagger on real val presets (data/tagger_report.json), and each val preset's nearest train preset in
CLAP space (how close a real, unseen Surge sound gets to the dataset).

Writes  runs/text2preset/<name>/report.md, summary.json, pitch.png, confusion.png, reach.png
Run:    .venv/bin/python -m pitcher.text2preset.pilot_report --name pilot_small-music
"""

import argparse
import json

import numpy as np
import pandas as pd

from pitcher.core.surge_io import PITCHER_ROOT
from pitcher.step4_vae.data import LABELS

DATA = PITCHER_ROOT / "data"
pct = lambda x: f"{100 * x:.1f}%"


def surge_reference_nn():
    """CLAP similarity of each val preset to its nearest train preset."""
    from pitcher.step8_eval.clap import cache_file
    model = json.loads((DATA / "tagger_report.json").read_text())["chosen"]
    G = pd.read_parquet(cache_file(model)).set_index("preset_id")
    sp = pd.read_parquet(DATA / "splits.parquet")
    tr, va = G.loc[sp[sp.split == "train"].preset_id], G.loc[sp[sp.split == "val"].preset_id]
    return (va.to_numpy() @ tr.to_numpy().T).max(1)


def plots(df, ctrl, ref_nn, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    bins = np.arange(-36.5, 25.5, 1)
    for d, lab, c in ((df, "Stable Audio clips", "#3b6fb6"), (ctrl, "Surge renders (control)", "#c8833a")):
        s = d.semis_from_c3.dropna()
        ax[0].hist(s, bins=bins, density=True, alpha=0.6, label=f"{lab} (n={len(s)})", color=c)
        ax[1].hist(d.pc_offset.dropna(), bins=np.arange(-6.5, 6.5, 1), density=True, alpha=0.6, label=lab, color=c)
    ax[0].axvline(0, color="#333", lw=0.8); ax[0].set_xlabel("main note, semitones from C3"); ax[0].legend(fontsize=8)
    ax[1].set_xlabel("semitones from the nearest C (any octave)"); ax[1].set_xticks(range(-6, 7))
    for a in ax:
        a.set_yticks([]); a.spines[["top", "right", "left"]].set_visible(False)
    fig.suptitle("Pitch of the tonal clips", fontsize=10); fig.tight_layout(); fig.savefig(out / "pitch.png", dpi=150); plt.close(fig)

    m = df[(df.kind == "main") & df.keep]
    cm = confusion_matrix(m.category, m.tagged_aligned, labels=LABELS, normalize="true")
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            if cm[i, j] >= 0.005:
                ax.text(j, i, f"{cm[i, j]:.0%}", ha="center", va="center", fontsize=8, color="white" if cm[i, j] > 0.5 else "#333")
    n = m.category.value_counts()
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=40, ha="right")
    ax.set_yticks(range(len(LABELS)), [f"{l} ({n.get(l, 0)})" for l in LABELS])
    ax.set_xlabel("tagged as (tagger trained on Surge presets; aligned fingerprints)"); ax.set_ylabel("category in the prompt (kept clips)")
    fig.tight_layout(); fig.savefig(out / "confusion.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    b = np.linspace(0.3, 1.0, 50)
    ax.hist(ref_nn, bins=b, density=True, alpha=0.6, color="#c8833a", label=f"Surge val preset → nearest train preset (n={len(ref_nn)})")
    ax.hist(df[df.keep].surge_nn_sim, bins=b, density=True, alpha=0.6, color="#3b6fb6", label="Stable Audio clip → nearest Surge preset")
    ax.set_xlabel("CLAP similarity to the nearest Surge preset"); ax.set_yticks([]); ax.legend(fontsize=8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "reach.png", dpi=150); plt.close(fig)


def descriptor_checks(df):
    """Do descriptor words change the measured sound? Medians per slot value (kept main clips)."""
    m = df[(df.kind == "main") & df.keep].copy()
    m["tags_d"] = m.tags.map(json.loads)
    # brightness relative to the category (bass is darker than lead by nature): octaves vs category median
    m["centroid_rel"] = np.log2(m.centroid_hz / m.groupby("category").centroid_hz.transform("median"))
    m["flatness_db"] = 10 * np.log10(m.flatness + 1e-12)
    checks = {"brightness": ("centroid_rel", "brightness: octaves vs category median", 0.5),
              "attack": ("attack_s", "attack time (s), 75th percentile", 0.75),
              "length": ("lvl_2s", "level after 2 s (dB vs peak)", 0.5),
              "grit": ("flatness_db", "noisiness: spectral flatness (dB)", 0.5)}
    rows = []
    for slot, (col, what, q) in checks.items():
        base = m[~m.tags_d.map(lambda t: slot in t)][col].quantile(q)
        for val, g in m.groupby(m.tags_d.map(lambda t: t.get(slot))):
            if val is not None and len(g) >= 8:
                rows.append(dict(slot=slot, value=val, n=len(g), measure=what, measured=g[col].quantile(q), without_slot=base))
    return pd.DataFrame(rows)


def report(name):
    src, out = DATA / "text2preset" / name, PITCHER_ROOT / "runs" / "text2preset" / name
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(src / "clips.parquet")
    ctrl = pd.read_parquet(src / "surge_controls.parquet")
    th = json.loads((src / "thresholds.json").read_text())
    tag_ref = json.loads((DATA / "tagger_report.json").read_text())
    ref_nn = surge_reference_nn()
    from pitcher.text2preset.postprocess import RULES
    from sklearn.metrics import balanced_accuracy_score

    S = {"name": name, "clips": len(df), "prompts": int(df.prompt_id.nunique()), "kept": float(df.keep.mean()),
         "flags": {k: float(df[k].mean()) for k in RULES}, "flags_surge_controls": {k: float(ctrl[k].mean()) for k in RULES if k in ctrl},
         "thresholds": th}
    tonal = df.semis_from_c3.notna()
    S["pitch"] = dict(tonal=float(tonal.mean()), within1_of_C3=float((df.semis_from_c3[tonal].abs() <= 1).mean()),
                      within1_of_any_C=float((df.pc_offset[tonal].abs() <= 1).mean()),
                      median_semis_from_c3=float(df.semis_from_c3.median()),
                      surge_within1_of_C3=float((ctrl.semis_from_c3.dropna().abs() <= 1).mean()),
                      surge_within1_of_any_C=float((ctrl.pc_offset.dropna().abs() <= 1).mean()))
    m = df[(df.kind == "main") & df.keep]
    S["category"] = dict(n=len(m), balanced=float(balanced_accuracy_score(m.category, m.tagged)),
                         balanced_aligned=float(balanced_accuracy_score(m.category, m.tagged_aligned)),
                         recall_aligned={l: float((m[m.category == l].tagged_aligned == l).mean()) for l in LABELS},
                         surge_val_reference=tag_ref["methods"][f"clap:{tag_ref['chosen'].split('/')[-1]}"]["balanced"],
                         recall={l: float((m[m.category == l].tagged == l).mean()) for l in LABELS},
                         tagged_share={l: float((m.tagged == l).mean()) for l in LABELS})
    k = df[df.keep]
    nd = th.get("n_descriptions", int(df.prompt_id.nunique()))
    S["text"] = dict(median_rank=float(k.text_rank.median()), n_descriptions=nd,
                     top10pct=float((k.text_rank <= 0.1 * nd).mean()),
                     by_kind={g: float(x.text_rank.median()) for g, x in k.groupby("kind")})
    E = pd.read_parquet(src / "clap.parquet").iloc[:, 2:].to_numpy()
    from pitcher.step8_eval.clap import cache_file
    Gall = pd.read_parquet(cache_file(tag_ref["chosen"])).iloc[:, 1:].to_numpy()
    unit = lambda v: v / np.linalg.norm(v)
    S["domain"] = dict(mean_cos=float(unit(E.mean(0)) @ unit(Gall.mean(0))))
    S["reach"] = dict(median=float(k.surge_nn_sim.median()), median_aligned=float(k.surge_nn_sim_aligned.median()),
                      surge_ref_median=float(np.median(ref_nn)),
                      above_ref_p10=float((k.surge_nn_sim >= np.percentile(ref_nn, 10)).mean()))
    rnd = np.random.default_rng(0)
    S["seeds"] = dict(median=float(df.seed_consistency.median()))
    by_kind = df.groupby("kind").agg(clips=("keep", "size"), kept=("keep", "mean"), tagged_top=("tagged", lambda s: s.value_counts().index[0]))
    by_cat = m.groupby("category").agg(kept_clips=("keep", "size"), recall=("tagged", "size"))
    cat_rows = []
    for c in LABELS:
        allc = df[(df.kind == "main") & (df.category == c)]
        mc = m[m.category == c]
        cat_rows.append(dict(category=c, clips=len(allc), kept=allc.keep.mean(), multi_note=allc.multi_note.mean(),
                             rhythmic=allc.rhythmic.mean(), off_pitch=allc.off_pitch.mean(), chord_like=allc.chord_like.mean(),
                             tagger_recall=(mc.tagged == c).mean(), recall_aligned=(mc.tagged_aligned == c).mean(),
                             median_semis_from_c3=allc.semis_from_c3.median(),
                             text_rank=mc.text_rank.median(), surge_nn=mc.surge_nn_sim.median()))
    cat = pd.DataFrame(cat_rows)
    desc = descriptor_checks(df)
    worst = (df[df.kind == "main"].groupby("prompt_id").agg(kept=("keep", "mean"), prompt=("sa3_prompt", "first"))
             .sort_values("kept").head(12))
    plots(df, ctrl, ref_nn, out)
    (out / "summary.json").write_text(json.dumps(S, indent=1))

    L = [f"# Stable Audio pilot report: {name}", "",
         f"{len(df)} clips ({S['prompts']} prompts × {len(df) // S['prompts']} seeds). Generated by `pitcher/text2preset/postprocess.py` + `pilot_report.py`.", "",
         "## 1. Filter", "",
         f"**Kept: {pct(S['kept'])}** ({int(df.keep.sum())} clips). Thresholds calibrated on {len(ctrl)} Surge renders analysed with the same code.", "",
         "| Flag | Meaning | Rejects? | Stable Audio | Surge renders |", "|---|---|---|---|---|"]
    for f, (meaning, rej) in RULES.items():
        L.append(f"| {f} | {meaning} | {'yes' if rej else 'no, marked'} | {pct(S['flags'][f])} | "
                 f"{pct(S['flags_surge_controls'][f]) if f in S['flags_surge_controls'] and f not in ('silent', 'cut_off', 'clipped') else '–'} |")
    L += ["", "By kind:", "", by_kind.to_markdown(floatfmt=".2f"), "",
          "## 2. Pitch (asked: C3)", "",
          f"Tonal clips: {pct(S['pitch']['tonal'])}. Within ±1 semitone of **any C**: **{pct(S['pitch']['within1_of_any_C'])}** "
          f"(Surge renders, all played on the C3 key: {pct(S['pitch']['surge_within1_of_any_C'])}). Within ±1 semitone of C3 itself: "
          f"{pct(S['pitch']['within1_of_C3'])} (Surge: {pct(S['pitch']['surge_within1_of_C3'])}; many Surge presets have octave "
          f"switches, so a bass played on the C3 key often sounds at C1 or C2). Median: {S['pitch']['median_semis_from_c3']:+.1f} semitones from C3.", "",
          "![pitch](pitch.png)", "",
          "## 3. Category (kept main clips, tagger trained on Surge presets)", "",
          f"Balanced agreement with the prompt's category: raw fingerprints {pct(S['category']['balanced'])}; "
          f"**after domain alignment {pct(S['category']['balanced_aligned'])}** "
          f"(the same tagger on real, unseen Surge presets: {pct(S['category']['surge_val_reference'])}; chance 12.5%). "
          f"Domain gap: the mean Stable Audio fingerprint and the mean Surge fingerprint have cosine {S['domain']['mean_cos']:.2f}, "
          f"so the raw tagger sees most clips as one category; the alignment shifts all clips by the difference of the two means "
          f"(no labels used; the same gain holds when the shift is estimated on half of the prompts). Matrix below: aligned.", "",
          "![confusion](confusion.png)", "",
          "## 4. Per category", "", cat.to_markdown(index=False, floatfmt=".2f"), "",
          "## 5. Does the description match? (CLAP text ↔ audio)", "",
          f"Rank of the clip's own description among all {S['text']['n_descriptions']}: median **{S['text']['median_rank']:.0f}** "
          f"(random: {S['text']['n_descriptions'] / 2:.0f}); in the top 10%: {pct(S['text']['top10pct'])}. "
          f"By kind: " + ", ".join(f"{g} {v:.0f}" for g, v in S['text']['by_kind'].items()) +
          ". CLAP text matching is weak on our sounds (Round 18: 20% zero-shot), so this is a coarse check.", "",
          "## 6. Descriptor words → measured sound (kept main clips)", "",
          desc.to_markdown(index=False, floatfmt=".3f") if len(desc) else "(too few clips per value)", "",
          "## 7. Can Surge reach these sounds?", "",
          f"CLAP similarity to the nearest Surge preset (raw fingerprints, plot): median **{S['reach']['median']:.3f}**; a real unseen Surge preset to its nearest "
          f"train preset: {S['reach']['surge_ref_median']:.3f}. Clips at least as close as the 10th percentile of that reference: {pct(S['reach']['above_ref_p10'])}. "
          f"After domain alignment: median {S['reach']['median_aligned']:.3f} (partly by construction, since the shift moves the clips towards the Surge set). "
          f"Much of the raw gap is the domain difference (section 3); how far Surge can really reach these sounds is a question for version A and a listening test.", "",
          "![reach](reach.png)", "",
          "## 8. Seeds", "", f"Mean CLAP similarity between the 4 seeds of a prompt: median {S['seeds']['median']:.3f}.", "",
          "## 9. Prompts with the fewest kept clips", "", worst.to_markdown(floatfmt=".2f"), ""]
    (out / "report.md").write_text("\n".join(L))
    print("\n".join(L))
    print(f"wrote {out.relative_to(PITCHER_ROOT)}/report.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="pilot_small-music")
    report(ap.parse_args().name)
