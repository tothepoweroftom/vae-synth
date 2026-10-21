"""Stage 0 of the synthetic Surge dataset (TEXT2PRESET_PLAN.md §3.2): sensitivity screening.

Which parameter groups change the sound, by how much, and does it depend on the rest of the patch?
A radial one-group-at-a-time design (Morris screening, Campolongo et al. 2011) from human presets:

  - 32 base presets: 4 per category, cleaned train only, rebuilt on the template (all kept features
    + routing matrix = what the VAE learns from)
  - per base and per parameter group that is active in it (452 core-ranking groups + 95 routing
    features): the group takes the value of a DONOR - another train preset where the group is active
    and differs - so changes have realistic sizes; when a switch brings in a new oscillator / effect
    type, the donor's knobs for that type come along
  - every version rendered at C3 (MIDI 48, the dataset note) and C4 (MIDI 60, the candidate second
    note) with the base's seed: renders are deterministic (Round 29), so the mel distance to the base
    is the effect of the change alone
  - reference: the base with another seed (Surge's own per-note randomness)

Two distances per version: the raw mel distance, and a LEVEL-MATCHED one (loudest 50 ms window of the
variant matched to the base; the synthetic set is level-normalised, so a pure gain change is not a
different sound). Per group, on the level-matched distance: mean (mu*), median, spread across bases
(sigma: interaction / nonlinearity), share of bases where the change alters the character (> 6.7 dB,
Round 14), C3 vs C4; tiers come from these. Raw mean and mean |level change| are kept alongside.

Writes  data/synth/screening.parquet (one row per base x group x note), data/synth/screening_groups.csv,
        runs/synth/screening.png, runs/synth/screening_summary.json
Run:    .venv/bin/python -m pitcher.synth_data.screen [--per-label 4] [--workers 4] [--analyse-only]
"""

import argparse
import json
import math
import time

import numpy as np
import pandas as pd

from pitcher.synth_data.common import (CHARACTER_DB, OUT, RUNS, Base, block_prefix, compare, render,
                                       run_pool)

NOTES = (48, 60)
MIN_BASES = 5  # a group's tier counts as measured from this many base presets on
CHUNK = 30  # variants per task (the base is re-rendered per task: cheap, keeps tasks independent)


def variants_for(base, pid, groups, rng):
    row0 = base.row(pid)
    K = base.K.drop(index=pid)
    out = []
    for g, feats in groups:
        b = np.array([row0[f] for f in feats], dtype=float)
        if np.isnan(b).all():
            continue                                            # group not active in this preset
        V = K[feats].to_numpy(float)
        active = ~np.isnan(V).all(1)
        differs = ~np.isclose(V, b, equal_nan=True).all(1)
        cand = K.index[active & differs]
        if len(cand) == 0:
            continue
        donor = int(rng.choice(cand))
        row = dict(row0)
        for f in feats:
            row[f] = float(base.codes.at[donor, f])
        prefixes = {block_prefix(f) for f in feats} - {None}
        for f in base.kept:                                     # knobs of a newly switched-in type
            if any(f.startswith(p) for p in prefixes) and math.isnan(row[f]):
                row[f] = float(base.codes.at[donor, f])
        delta = (float(base.codes.at[donor, feats[0]]) - b[0]) if len(feats) and not np.isnan(b[0]) else math.nan
        out.append((g, donor, delta, row))
    return row0, out


def _task(task):
    pid, seed, row0, chunk, with_ref = task
    rows = []
    for note in NOTES:
        a0, info0 = render(row0, note, seed)
        if with_ref:
            a1, _ = render(row0, note, seed + 100000)
            d, lv, dn = compare(a0, a1)
            rows.append(dict(base=pid, group="(other seed)", donor=-1, delta=math.nan, note=note, distance=d,
                             distance_norm=dn, level_diff=lv, silent=False, base_len=info0["total_s"], base_rms=info0["max_rms_dbfs"]))
        for g, donor, delta, row in chunk:
            a, info = render(row, note, seed)
            d, lv, dn = compare(a0, a)
            rows.append(dict(base=pid, group=g, donor=donor, delta=delta, note=note, distance=d, distance_norm=dn,
                             level_diff=lv,
                             silent=bool(info["silent"]), base_len=info0["total_s"], base_rms=info0["max_rms_dbfs"]))
    return rows


def run(base, args):
    groups = base.groups()
    bases = base.stratified(args.per_label, seed=0)
    rng = np.random.default_rng(1)
    tasks, n_var = [], 0
    for i, pid in enumerate(bases):
        row0, var = variants_for(base, pid, groups, rng)
        n_var += len(var)
        for j in range(0, len(var), CHUNK):
            tasks.append((pid, 1000 + i, row0, var[j:j + CHUNK], j == 0))
    print(f"{len(bases)} bases, {n_var} variants ({n_var / len(bases):.0f} per base), "
          f"{(n_var + 2 * len(bases)) * len(NOTES)} renders, {len(tasks)} tasks")
    t0, rows = time.time(), []
    for res in run_pool(_task, tasks, args.workers, "screening"):
        rows += res
    df = pd.DataFrame(rows)
    df["label"] = df.base.map(base.label)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "screening.parquet", index=False)
    print(f"{len(df)} rows in {time.time() - t0:.0f}s")
    return df


def analyse(base, df):
    groups = dict(base.groups())
    ref = df[df.group == "(other seed)"]
    v = df[df.group != "(other seed)"]
    piv = v.pivot_table(index=["base", "group"], columns="note", values=["distance", "distance_norm", "level_diff"]).dropna()
    piv.columns = [{"distance": "d", "distance_norm": "n", "level_diff": "lv"}[a] + str(c) for a, c in piv.columns]
    stats = []
    for g, s in piv.groupby(level="group"):
        d48, d60, n48, n60 = (s[c].to_numpy() for c in ("d48", "d60", "n48", "n60"))
        silent = v[(v.group == g)].silent.mean()
        stats.append(dict(group=g, kind=base.kind(groups[g]), n_features=len(groups[g]), n_bases=len(s),
                          mu_star=d48.mean(), median=np.median(d48), sigma=d48.std(ddof=1) if len(s) > 1 else 0.0,
                          share_character=(d48 > CHARACTER_DB).mean(), mu_star_c4=d60.mean(),
                          mu_norm=n48.mean(), median_norm=np.median(n48), sigma_norm=n48.std(ddof=1) if len(s) > 1 else 0.0,
                          share_character_norm=(n48 > CHARACTER_DB).mean(), mu_norm_c4=n60.mean(),
                          max_norm=max(n48.max(), n60.max()),
                          level_abs=np.abs(s.lv48.to_numpy()).mean(),
                          c4_vs_c3=(n60.mean() + .1) / (n48.mean() + .1), silent=silent))
    G = pd.DataFrame(stats)
    # tiers on the LEVEL-MATCHED distance (the synthetic set is level-normalised; a pure gain knob is
    # a level change, not a different sound): 1 = changes the character in at least a quarter of the
    # bases; 3 = below 1 dB on average at both notes (not audible in a single fixed-velocity note:
    # take human values, don't explore)
    G["tier"] = np.where(G.share_character_norm >= .25, 1, np.where((G.mu_norm < 1) & (G.mu_norm_c4 < 1), 3, 2))
    # tier 3 but clearly audible in a few presets: the group acts only when something else is on (a muted
    # oscillator's level, the noise level, an envelope-shape switch): explored when its source is on
    G["gated"] = (G.tier == 3) & (G.max_norm > 3)
    G["gain_mostly"] = (G.mu_star >= 3) & (G.mu_norm < .5 * G.mu_star)  # most of the raw change is loudness
    # groups of rare types (an oscillator or effect type few presets use) were tested on few bases:
    # their tier is a guess; the sampler treats them as tier 2 until measured on more presets
    G["reliable"] = G.n_bases >= MIN_BASES
    G = G.sort_values("mu_norm", ascending=False)
    G.to_csv(OUT / "screening_groups.csv", index=False)

    seed_ref = {int(n): float(ref[ref.note == n].distance_norm.median()) for n in NOTES}
    summ = dict(
        bases=int(v.base.nunique()), groups_tested=int(len(G)), renders=int(len(df)),
        other_seed_median_db_norm=seed_ref,
        tiers_reliable={int(t): int(((G.tier == t) & G.reliable).sum()) for t in (1, 2, 3)},
        groups_on_few_bases=int((~G.reliable).sum()),
        tier3_gated_reliable=int((G.gated & G.reliable).sum()),
        tiers_by_kind_reliable=G[G.reliable].groupby(["kind", "tier"]).size().unstack(fill_value=0).to_dict(orient="index"),
        variants_changing_character=float((piv.d48 > CHARACTER_DB).mean()),
        variants_changing_character_norm=float((piv.n48 > CHARACTER_DB).mean()),
        variants_below_1db_norm=float((piv.n48 < 1).mean()),
        groups_gain_mostly=G[G.gain_mostly & G.reliable].group.tolist(),
        variants_silent=float(v.silent.mean()),
        c4_vs_c3_corr=float(np.corrcoef(piv.n48, piv.n60)[0, 1]),
        groups_louder_at_c4=G[(G.c4_vs_c3 > 2) & (G.mu_star_c4 > 2)].group.tolist(),
        groups_quieter_at_c4=G[(G.c4_vs_c3 < .5) & (G.mu_star > 2)].group.tolist(),
    )
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "screening_summary.json").write_text(json.dumps(summ, indent=1, default=str))
    pd.set_option("display.width", 200)
    print(json.dumps(summ, indent=1, default=str))
    cols = ["group", "kind", "n_bases", "mu_norm", "median_norm", "sigma_norm", "share_character_norm", "mu_norm_c4",
            "mu_star", "level_abs", "tier"]
    R = G[G.reliable]
    print(f"\nTop 40 groups by mean level-matched distance at C3 (tested on >= {MIN_BASES} bases):")
    print(R[cols].head(40).round(2).to_string(index=False))
    print("\nTier 3 (no audible effect in one note), by kind:", R[R.tier == 3].kind.value_counts().to_dict())
    print(R[R.tier == 3].sort_values("n_bases", ascending=False).group.head(40).tolist())
    plot(G, seed_ref)
    return G


def plot(G, seed_ref):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1f1f1d", "#6b6a63", "#ecebe6"
    C3, C4 = "#2a78d6", "#eb6834"
    top = G[G.reliable].head(40).iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(13, 9.5), facecolor=SURFACE, gridspec_kw=dict(width_ratios=[3, 2]))
    ax = axes[0]
    ax.set_facecolor(SURFACE)
    y = np.arange(len(top))
    ax.barh(y + .2, top.mu_norm, height=.38, color=C3, label="C3 (MIDI 48)")
    ax.barh(y - .2, top.mu_norm_c4, height=.38, color=C4, label="C4 (MIDI 60)")
    ax.axvline(CHARACTER_DB, color=MUTED, linewidth=1, linestyle="--")
    ax.text(CHARACTER_DB, len(top) + .3, " 6.7 dB: other character", color=MUTED, fontsize=8)
    ax.axvline(seed_ref[48], color=MUTED, linewidth=1, linestyle=":")
    ax.text(seed_ref[48], -1.6, f" other seed {seed_ref[48]:.1f} dB", color=MUTED, fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([g if len(g) < 38 else g[:36] + "…" for g in top.group], fontsize=7.5, color=INK)
    ax.set_title("40 most sensitive parameter groups: mean level-matched mel distance when\nthe group takes another preset's value (groups tested on >= 5 of 32 base presets)",
                 loc="left", fontsize=10.5, color=INK)
    ax.set_xlabel("mean distance to the base (dB)", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK)
    ax2 = axes[1]
    ax2.set_facecolor(SURFACE)
    for kind, mk in (("switch", "s"), ("knob", "o"), ("route", "^")):
        s = G[G.kind == kind]
        ax2.scatter(s.mu_norm, s.sigma_norm, s=16, marker=mk, color=C3, alpha=.55, edgecolor="none", label=kind)
    ax2.set_xscale("symlog", linthresh=1)
    ax2.set_yscale("symlog", linthresh=1)
    ax2.set_xlabel("mean distance mu* (dB)", color=MUTED, fontsize=9)
    ax2.set_ylabel("spread across bases sigma (dB)", color=MUTED, fontsize=9)
    ax2.set_title("All groups: effect size vs dependence on the patch", loc="left", fontsize=10.5, color=INK)
    ax2.legend(frameon=False, fontsize=9, labelcolor=INK)
    for a in axes:
        a.tick_params(colors=MUTED, labelsize=8, length=0)
        for s in a.spines.values():
            s.set_visible(False)
        a.grid(color=GRID, linewidth=.6)
    fig.tight_layout()
    fig.savefig(RUNS / "screening.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-label", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--analyse-only", action="store_true")
    args = ap.parse_args()
    base = Base()
    df = pd.read_parquet(OUT / "screening.parquet") if args.analyse_only else run(base, args)
    analyse(base, df)


if __name__ == "__main__":
    main()
