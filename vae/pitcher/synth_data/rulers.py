"""Stage 0b of the synthetic Surge dataset (TEXT2PRESET_PLAN.md §3.2): perceptual scales per knob.

Equal steps on a knob's 0-1 code are not equal steps in sound. For the most sensitive knobs (by the
stage-0 screening, data/synth/screening_groups.csv) this sweeps the knob in 11 equal steps (code
0, 0.1, ..., 1) from 30 human base presets each (cleaned train, the knob active), renders at C3 with
the base's seed (deterministic, Round 29) and measures the level-matched mel distance (common.compare)
between neighbouring steps.

Per base, the cumulative distance along the sweep, divided by its total, is that base's perceptual
scale; the knob's scale ("ruler") is the mean over bases with a total above 1 dB (the mean, not the
median: where a knob only acts in some patches, e.g. filter balance, the median jumps between groups).
Sampling then draws evenly on the ruler (equal sound steps) instead of on the raw code.

Writes  data/synth/rulers_raw.parquet (knob x base x step), data/synth/rulers.json (per knob: codes,
        ruler, total change, where the change happens, human value quantiles), runs/synth/rulers.png
Run:    .venv/bin/python -m pitcher.synth_data.rulers [--knobs 40] [--bases 30] [--workers 4] [--analyse-only]
"""

import argparse
import json
import time

import numpy as np
import pandas as pd

from pitcher.synth_data.common import OUT, RUNS, Base, compare, render, run_pool

STEPS = np.linspace(0, 1, 11)
NOTE = 48


def pick_knobs(base, n):
    G = pd.read_csv(OUT / "screening_groups.csv")
    groups = dict(base.groups())
    out = []
    G = G[G.kind.isin(["knob", "route"]) & (G.n_bases >= 10)]  # common enough to sweep from many presets
    for g in G.sort_values("mu_norm", ascending=False).group:      # by level-matched effect: pure gain
        # knobs need no perceptual scale (the set is level-normalised; their scale is dB itself)
        f = next(x for x in groups[g] if base.feats[x]["kind"] in ("cont", "ord"))
        out.append((g, f))
        if len(out) == n:
            break
    return out


def _task(task):
    feature, pid, seed, row0 = task
    audio = []
    for c in STEPS:
        row = dict(row0)
        row[feature] = float(c)
        audio.append(render(row, NOTE, seed)[0])
    rows = []
    for i in range(1, len(STEPS)):
        d, lv, dn = compare(audio[i - 1], audio[i])
        rows.append(dict(feature=feature, base=pid, step=i, code=STEPS[i], d_prev=d, level_prev=lv, n_prev=dn))
    return rows


def run(base, args):
    knobs = pick_knobs(base, args.knobs)
    tasks = []
    for k, (g, f) in enumerate(knobs):
        act = base.K.index[base.K[f].notna()]
        rng = np.random.default_rng(100 + k)
        for j, pid in enumerate(rng.choice(act, min(args.bases, len(act)), replace=False)):
            tasks.append((f, int(pid), 2000 + j, base.row(int(pid))))
    print(f"{len(knobs)} knobs, {len(tasks)} sweeps, {len(tasks) * len(STEPS)} renders")
    t0, rows = time.time(), []
    for res in run_pool(_task, tasks, args.workers, "rulers"):
        rows += res
    df = pd.DataFrame(rows)
    df["label"] = df.base.map(base.label)
    df.to_parquet(OUT / "rulers_raw.parquet", index=False)
    print(f"{len(df)} rows in {time.time() - t0:.0f}s")
    return df


def analyse(base, df):
    out, per_base = {}, {}
    for f, s in df.groupby("feature", sort=False):
        # level-matched step distances (the set is level-normalised); raw ones kept as total_raw_db_median
        P = s.pivot_table(index="base", columns="step", values="n_prev").reindex(columns=range(1, len(STEPS)))
        raw = s.pivot_table(index="base", columns="step", values="d_prev").sum(axis=1)
        tot = P.sum(axis=1)
        live = P[tot > 1.0]
        cum = np.hstack([np.zeros((len(live), 1)), np.cumsum(live.to_numpy(), 1)]) / tot[live.index].to_numpy()[:, None]
        per_base[f] = cum
        ruler = np.maximum.accumulate(cum.mean(0)) if len(cum) else STEPS.copy()
        ruler = (ruler - ruler[0]) / (ruler[-1] - ruler[0]) if ruler[-1] > ruler[0] else STEPS.copy()
        human = base.K[f].dropna()
        out[f] = dict(
            n_bases=int(len(P)), n_live=int(len(live)),
            total_db_median=float(tot.median()), total_db_p90=float(tot.quantile(.9)),
            total_raw_db_median=float(raw.median()),
            codes=STEPS.round(3).tolist(), ruler=ruler.round(4).tolist(),
            half_point=float(np.interp(.5, ruler, STEPS)),            # code where half the change is reached
            unevenness=float(np.abs(ruler - STEPS).max()),            # 0 = the code is already perceptually even
            change_bottom_20=float(ruler[2]), change_top_20=float(1 - ruler[8]),
            spread=float(np.median(np.abs(cum - np.median(cum, 0)).max(1))) if len(cum) else None,
            human_q=dict(zip(["p1", "p5", "p50", "p95", "p99"], human.quantile([.01, .05, .5, .95, .99]).round(4).tolist())),
        )
    (OUT / "rulers.json").write_text(json.dumps(out, indent=1))
    T = pd.DataFrame(out).T[["n_live", "total_db_median", "half_point", "unevenness", "change_bottom_20",
                             "change_top_20", "spread"]]
    pd.set_option("display.width", 200)
    print(T.astype(float).round(2).to_string())
    summ = dict(knobs=len(out),
                median_unevenness=float(T.unevenness.astype(float).median()),
                knobs_uneven_over_0_2=int((T.unevenness.astype(float) > .2).sum()),
                knobs_change_mostly_in_bottom_half=int((T.half_point.astype(float) < .35).sum()),
                knobs_change_mostly_in_top_half=int((T.half_point.astype(float) > .65).sum()))
    (RUNS / "rulers_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))
    plot(out, per_base)


def plot(out, per_base):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURFACE, INK, MUTED, GRID, LINE, THIN, BAND = "#fcfcfb", "#1f1f1d", "#6b6a63", "#ecebe6", "#2a78d6", "#c9c8c1", "#eef3fb"
    feats = list(out)[:16]
    fig, axes = plt.subplots(4, 4, figsize=(13, 11.5), facecolor=SURFACE)
    for ax, f in zip(axes.flat, feats):
        o = out[f]
        ax.set_facecolor(SURFACE)
        ax.axvspan(o["human_q"]["p5"], o["human_q"]["p95"], color=BAND, zorder=0)
        for c in per_base[f]:
            ax.plot(STEPS, c, color=THIN, linewidth=.7, zorder=1)
        ax.plot(STEPS, STEPS, color=MUTED, linewidth=.8, linestyle="--", zorder=2)
        ax.plot(STEPS, o["ruler"], color=LINE, linewidth=2, zorder=3)
        name = f.replace("scene/", "").replace("route/", "route ")
        ax.set_title(f"{name if len(name) < 34 else name[:32] + '…'}\nsum of steps {o['total_db_median']:.0f} dB, {o['n_live']} presets",
                     loc="left", fontsize=8, color=INK)
        ax.tick_params(colors=MUTED, labelsize=7, length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.grid(color=GRID, linewidth=.5)
    for ax in axes.flat[len(feats):]:
        ax.axis("off")
    fig.suptitle("Perceptual scales: share of the total sound change reached along the knob's 0-1 code\n"
                 "(blue: mean over presets; grey: single presets; dashed: even; shaded: 5-95% of human values)",
                 x=.01, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(RUNS / "rulers.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--knobs", type=int, default=40)
    ap.add_argument("--bases", type=int, default=30)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--analyse-only", action="store_true")
    args = ap.parse_args()
    base = Base()
    df = pd.read_parquet(OUT / "rulers_raw.parquet") if args.analyse_only else run(base, args)
    analyse(base, df)


if __name__ == "__main__":
    main()
