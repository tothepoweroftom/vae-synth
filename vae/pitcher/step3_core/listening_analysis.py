"""Step 3.4: analyse the listening-test ratings -> distance threshold for "same character".

Reads   data/listening_ratings/ratings/*.json (exported from the artifact's db), data/listening_key.csv,
        data/core_curve.csv
Writes  data/listening_results.csv (one row per pair: rating, distance, version),
        data/listening_threshold.json, runs/step3/listening_fit.png,
        runs/step3/core_curve_character.png (K curve with the listening-based threshold)
Run:    .venv/bin/python -m pitcher.step3_core.listening_analysis
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from pitcher.core.surge_io import PITCHER_ROOT

DATA, OUT = PITCHER_ROOT / "data", PITCHER_ROOT / "runs" / "step3"
SCORE = {"same_sound": 3, "same_character": 2, "different": 1, "broken": 0}
SURFACE, INK, MUTED, BLUE, ORANGE = "#fcfcfb", "#1f1f1d", "#6b6a63", "#2a78d6", "#eb6834"


def fit(x, y):
    m = LogisticRegression(C=1e6).fit(x.reshape(-1, 1), y)
    a, b = m.intercept_[0], m.coef_[0, 0]
    return m, (lambda p: (np.log(p / (1 - p)) - a) / b)  # distance where P = p


def main():
    rows = [json.loads(f.read_text()) for f in sorted((DATA / "listening_ratings" / "ratings").glob("*.json"))]
    r = pd.DataFrame(rows)
    r = r.rename(columns={"pair": "page_id"}) if "pair" in r else r
    key = pd.read_csv(DATA / "listening_key.csv")
    df = key.merge(r[["page_id", "rating"]], on="page_id", how="inner")
    df["score"] = df.rating.map(SCORE)
    df["kind"] = df.version.str[0].map({"A": "A: original as base", "B": "B: template", "R": "control: re-render"})
    df.to_csv(DATA / "listening_results.csv", index=False)

    x = df.distance_db.to_numpy()
    char = (df.score >= 2).astype(int).to_numpy()   # same sound or same character
    sound = (df.score >= 3).astype(int).to_numpy()  # same sound
    m_char, inv_char = fit(x, char)
    m_sound, inv_sound = fit(x, sound)
    thr = {"same_character_p50": float(inv_char(0.5)), "same_character_p80": float(inv_char(0.8)),
           "same_sound_p50": float(inv_sound(0.5)), "n_pairs": int(len(df))}

    print(f"{len(df)} rated pairs")
    print(pd.crosstab(df.kind, df.rating).reindex(columns=list(SCORE), fill_value=0).to_string())
    print("\nratings by distance band (re-measured):")
    bins = pd.cut(df.distance_db, [-0.1, 2, 4, 6, 8, 10, 12, 15, 20, 30])
    print(pd.crosstab(bins, df.rating).reindex(columns=list(SCORE), fill_value=0).to_string())
    print(f"\nlogistic fit: P(same character or better) = 50% at {thr['same_character_p50']:.1f} dB, "
          f"80% at {thr['same_character_p80']:.1f} dB; P(same sound) = 50% at {thr['same_sound_p50']:.1f} dB")
    print("controls (original vs re-render):", df[df.version == "R"][["distance_db", "rating"]].round(1).values.tolist())
    print("per label, share >= same character:", df.groupby("label").score.apply(lambda s: (s >= 2).mean()).round(2).to_dict())

    (DATA / "listening_threshold.json").write_text(json.dumps(thr, indent=1))

    # --- K curve with the listening-based threshold ------------------------------------------------
    curve = pd.read_csv(DATA / "core_curve.csv")
    t = thr["same_character_p80"]
    s = curve.groupby(["variant", "k"]).agg(
        char80=("distance", lambda d: (d <= t).mean()),
        char50=("distance", lambda d: (d <= thr["same_character_p50"]).mean())).reset_index()
    print(f"\nK curve, share of val presets at 'same character' (P >= 80%, distance <= {t:.1f} dB) and (P >= 50%):")
    print(s.pivot(index="k", columns="variant", values=["char80", "char50"]).round(3).to_string())
    (DATA / "listening_threshold.json").write_text(json.dumps(dict(thr, k_curve=s.round(4).to_dict("records")), indent=1))

    # --- plots ---------------------------------------------------------------------------------------
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    jitter = np.random.default_rng(0).uniform(-0.08, 0.08, len(df))
    ax.scatter(x, df.score + jitter, s=28, c=MUTED, alpha=0.7, linewidths=0)
    for d, pct, style, side in ((t, "80%", "--", "right"), (thr["same_character_p50"], "50%", ":", "left")):
        ax.axvline(d, color=BLUE, linewidth=1.2, linestyle=style)
        ax.text(d - 0.25 if side == "right" else d + 0.25, 0.35, f"{pct} same character\nor better: {d:.1f} dB",
                color=INK, fontsize=8, ha=side)
    ax.set_yticks(list(SCORE.values()))
    ax.set_yticklabels(["Same sound", "Same character", "Different", "Broken"])
    ax.set_xlabel("mel distance to the original (dB)", color=MUTED, fontsize=9)
    ax.set_title("Listening ratings against measured distance", loc="left", fontsize=11, color=INK)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(color="#ecebe6", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(OUT / "listening_fit.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    names = {"static": "A: template without routings", "routes": "B: + 13 standard routings"}
    for v, color in (("routes", ORANGE), ("static", BLUE)):
        g = s[s.variant == v]
        ax.plot(g.k, 100 * g.char80, color=color, linewidth=2, marker="o", markersize=5, label=names[v])
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks(sorted(s.k.unique()))
    ax.set_xticklabels([str(k) for k in sorted(s.k.unique())])
    ax.set_ylabel("%", color=MUTED, fontsize=9)
    ax.set_xlabel("core parameter groups K (in ranking order)", color=MUTED, fontsize=9)
    ax.set_title(f"Val presets rebuilt with the same character (distance <= {t:.1f} dB)", loc="left", fontsize=11, color=INK)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(color="#ecebe6", linewidth=0.6)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "core_curve_character.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
