#!/usr/bin/env python3
"""Turn vae_compare's CSV into a Markdown report (see docs/COMPARE.md).

Three renders per preset: the original Surge patch ("orig"), the schema vector
on the Surge backend ("svec") and on the Lite backend ("lite"). Pairs:
    os  orig -> svec   what the schema itself loses ("schema loss")
    sl  svec -> lite   what the Lite engine changes ("engine gap")
    ol  orig -> lite   both together

Usage:
    compare_report.py compare.csv vectors.jsonl -o report.md [--wav-list worst.txt --worst N]

Requires numpy.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import vae_schema  # noqa: E402

RENDERS = ("orig", "svec", "lite")
PAIRS = {"os": "orig→svec (schema loss)", "sl": "svec→lite (engine gap)", "ol": "orig→lite (total)"}
DISTANCES = {
    "frame_mel_db": "spectrum over time, dB (lower = closer)",
    "mel_db": "average spectrum shape, dB (lower = closer)",
    "env_corr": "loudness envelope correlation (higher = closer)",
    "d_level_db": "loudness difference, dB",
    "d_centroid_oct": "brightness difference, octaves",
    "d_flatness": "noisiness difference (0..1)",
}
SCALARS = ("level_db", "centroid_hz", "flatness", "attack_s", "release_s")
MEL = [f"mel{i}" for i in range(40)]
MIN_CATEGORY = 20


def load_rows(csv_path: pathlib.Path) -> list[dict]:
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def num(r: dict, key: str) -> float:
    v = r.get(key, "")
    return float(v) if v not in ("", None) else float("nan")


def usable(r: dict, pair: str) -> bool:
    """Both renders of the pair exist and make sound, and the original is not silent."""
    a, b = {"os": ("orig", "svec"), "sl": ("svec", "lite"), "ol": ("orig", "lite")}[pair]
    if r["orig_ok"] == "1" and r["orig_silent"] == "1":
        return False
    if "orig" in (a, b) and r["orig_ok"] != "1":
        return False
    return r[f"{a}_silent"] == "0" and r[f"{b}_silent"] == "0"


def ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share their mean rank)."""
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x), dtype=float)
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra, rb = ranks(a[ok]), ranks(b[ok])
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")  # a constant feature has no ordering
    return float(np.corrcoef(ra, rb)[0, 1])


def features(r: dict, render: str) -> np.ndarray:
    mel = [num(r, f"{render}_{m}") for m in MEL]
    return np.array(mel + [np.log2(max(num(r, f"{render}_centroid_hz"), 20.0)),
                           num(r, f"{render}_flatness"),
                           num(r, f"{render}_attack_s"),
                           num(r, f"{render}_release_s")])


def knn_predict(train_x, train_y, test_x, k=5):
    mu, sd = train_x.mean(axis=0), train_x.std(axis=0) + 1e-9
    tr = (train_x - mu) / sd
    te = (test_x - mu) / sd
    out = []
    for x in te:
        d = np.sqrt(((tr - x) ** 2).sum(axis=1))
        nearest = np.argsort(d)[:k]
        votes = Counter(train_y[i] for i in nearest)
        best = max(votes.values())
        # tie: the closest of the tied labels wins
        for i in nearest:
            if votes[train_y[i]] == best:
                out.append(train_y[i])
                break
    return out


def category_test(rows: list[dict]) -> list[str]:
    usable_rows = [r for r in rows if r["category"] != "unknown" and r["svec_silent"] == "0"
                   and r["lite_silent"] == "0"]
    counts = Counter(r["category"] for r in usable_rows)
    keep = {c for c, n in counts.items() if n >= MIN_CATEGORY}
    data = [r for r in usable_rows if r["category"] in keep]
    if len(keep) < 2:
        return ["Not enough categories with ≥ %d presets." % MIN_CATEGORY]

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(data))
    folds = np.array_split(idx, 5)
    y = [r["category"] for r in data]
    x = {render: np.array([features(r, render) for r in data]) for render in RENDERS}
    orig_ok = np.array([r["orig_ok"] == "1" and r["orig_silent"] == "0" for r in data])

    correct = Counter()
    total = Counter()
    for f in range(5):
        test = folds[f]
        train = np.concatenate([folds[g] for g in range(5) if g != f])
        ty = [y[i] for i in train]
        for render in RENDERS:
            t = test if render != "orig" else test[orig_ok[test]]
            if len(t) == 0:
                continue
            pred = knn_predict(x["svec"][train], ty, x[render][t])
            correct[render] += sum(p == y[i] for p, i in zip(pred, t))
            total[render] += len(t)

    majority = counts.most_common(1)[0]
    lines = [
        f"Categories with ≥ {MIN_CATEGORY} presets: {len(keep)} ({', '.join(f'{c} {counts[c]}' for c in sorted(keep))}); "
        f"{len(data)} presets. kNN (k=5) on the average spectrum shape, brightness, noisiness, attack and release, "
        "trained on svec renders, 5-fold cross-validation.",
        "",
        "| test renders | accuracy |",
        "|---|---|",
    ]
    names = {"svec": "svec (same engine as training)", "lite": "lite (vector on Lite)",
             "orig": "orig (original Surge preset)"}
    for render in ("svec", "lite", "orig"):
        if total[render]:
            lines.append(f"| {names[render]} | {correct[render] / total[render]:.1%} (n={total[render]}) |")
    lines.append(f"| chance: always '{majority[0]}' | {max(counts[c] for c in keep) / len(data):.1%} |")
    lines.append(f"| chance: uniform over {len(keep)} | {1 / len(keep):.1%} |")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv", type=pathlib.Path)
    ap.add_argument("jsonl", type=pathlib.Path)
    ap.add_argument("-o", "--output", type=pathlib.Path, required=True)
    ap.add_argument("--wav-list", type=pathlib.Path, help="write the worst presets' paths here (for vae_compare --wav-list)")
    ap.add_argument("--worst", type=int, default=20)
    args = ap.parse_args()

    schema = vae_schema.load()
    rows = load_rows(args.csv)
    vectors = {}
    with open(args.jsonl) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                vectors[d["file"]] = d["values"]

    out = ["# Lite vs Surge comparison", ""]
    n = len(rows)
    orig_ok = sum(r["orig_ok"] == "1" for r in rows)
    silent = {render: sum(r[f"{render}_silent"] == "1" for r in rows if render != "orig" or r["orig_ok"] == "1")
              for render in RENDERS}
    mismatch = sum((r["svec_silent"] == "1") != (r["lite_silent"] == "1") for r in rows)
    out += [f"{n} presets; original loaded for {orig_ok}. Silent renders: "
            f"orig {silent['orig']}, svec {silent['svec']}, lite {silent['lite']}; "
            f"svec/lite disagree on silence for {mismatch}. Distances below use presets where both renders make sound.",
            ""]

    # Distance table.
    out += ["## Schema loss vs engine gap", "",
            "Median / 75th / 90th percentile over presets. If the engine gap (svec→lite) is well below the schema "
            "loss (orig→svec), Lite adds little on top of what the schema already changes.", ""]
    out += ["| measure | " + " | ".join(PAIRS.values()) + " |", "|---|---|---|---|"]
    for key, label in DISTANCES.items():
        cells = []
        for p in PAIRS:
            v = np.array([num(r, f"{p}_{key}") for r in rows if usable(r, p)])
            v = v[np.isfinite(v)]
            cells.append(f"{np.median(v):.2f} / {np.percentile(v, 75):.2f} / {np.percentile(v, 90):.2f}"
                         if len(v) else "–")
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    out.append("")

    # Rank agreement.
    out += ["## Do the engines agree on which preset is brighter, louder, slower?", "",
            "Spearman rank correlation across presets (1 = same ordering).", "",
            "| feature | svec vs lite | orig vs svec |", "|---|---|---|"]
    both = [r for r in rows if usable(r, "sl")]
    tri = [r for r in rows if usable(r, "os") and usable(r, "sl")]
    for feat in SCALARS:
        a = np.array([num(r, f"svec_{feat}") for r in both])
        b = np.array([num(r, f"lite_{feat}") for r in both])
        c = np.array([num(r, f"orig_{feat}") for r in tri])
        d = np.array([num(r, f"svec_{feat}") for r in tri])
        out.append(f"| {feat} | {spearman(a, b):.2f} | {spearman(c, d):.2f} |")
    out.append("")

    # Category test.
    out += ["## Does the preset category survive?", ""] + category_test(rows) + [""]

    # Breakdown by control.
    out += ["## Where Lite diverges most", "",
            "Median engine gap (svec→lite, spectrum over time in dB / loudness difference in dB) grouped by a "
            "control's value; n = presets.", ""]
    groups = ["filter1_type", "filter2_type", "filter_routing", "osc1_wave", "osc2_on", "play_mode",
              "dist_on", "eq_on", "chorus_on", "modfx_on", "delay_on", "reverb_on"]
    by_file = {r["file"]: r for r in both}
    for g in groups:
        c = schema[g]
        buckets = defaultdict(list)
        for file, values in vectors.items():
            r = by_file.get(file)
            if r is None:
                continue
            v = values[c.index]
            label = c.options[int(v)] if c.kind == "categorical" else ("on" if v >= 0.5 else "off")
            buckets[label].append((num(r, "sl_frame_mel_db"), num(r, "sl_d_level_db")))
        cells = []
        for label, vals in sorted(buckets.items(), key=lambda kv: -np.median([v[0] for v in kv[1]])):
            arr = np.array(vals)
            cells.append(f"{label}: {np.median(arr[:, 0]):.1f} / {np.median(arr[:, 1]):.1f} (n={len(vals)})")
        out.append(f"- **{g}**: " + "; ".join(cells))
    out.append("")

    # Worst presets.
    scored = sorted(both, key=lambda r: -num(r, "sl_frame_mel_db"))
    spreads = {f: np.median([abs(num(r, f"svec_{f}") - num(r, f"lite_{f}")) for r in both]) + 1e-9
               for f in SCALARS}

    def main_difference(r: dict) -> str:
        diffs = {f: abs(num(r, f"svec_{f}") - num(r, f"lite_{f}")) / spreads[f] for f in SCALARS}
        f = max(diffs, key=diffs.get)
        return f"{f}: svec {num(r, f'svec_{f}'):.3g} vs lite {num(r, f'lite_{f}'):.3g}"

    out += [f"## Worst {args.worst} presets by engine gap", "",
            "| preset | category | gap dB | schema loss dB | lossy notes | main difference |", "|---|---|---|---|---|---|"]
    for r in scored[:args.worst]:
        name = "/".join(pathlib.Path(r["file"]).parts[-3:])
        os_ = num(r, "os_frame_mel_db")
        out.append(f"| {name} | {r['category']} | {num(r, 'sl_frame_mel_db'):.1f} | "
                   f"{'' if np.isnan(os_) else f'{os_:.1f}'} | {r['n_lossy']} | {main_difference(r)} |")
    out.append("")

    args.output.write_text("\n".join(out) + "\n")
    if args.wav_list:
        args.wav_list.write_text("\n".join(r["file"] for r in scored[:args.worst]) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
