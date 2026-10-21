"""Step 4 prerequisite: minimal cleaning of the VAE training data.

A preset is excluded if its rebuild on the template from all kept features (+ routing matrix) is
clearly broken - the writer cannot express what makes it sound, and the VAE would learn settings
that only work through state it can't generate (e.g. level 0 raised by a macro):
  - silent: level more than 60 dB below the original, or
  - level jump: more than 12 dB louder or quieter than the original
(the listening test's "broken" ratings came with level jumps of +10..15 / -12..-23 dB).
Same rule for train, val and test; it uses no labels or model.

Reads   data/core_curve_all_kall.csv  (pitcher/step3_core/curve.py --ks 452 --variants routes --split all)
Writes  data/vae_excluded.csv
Run:    .venv/bin/python -m pitcher.step4_vae.clean
"""

import pandas as pd

from pitcher.core.surge_io import PITCHER_ROOT

DATA = PITCHER_ROOT / "data"
SILENT_DB, JUMP_DB = -60.0, 12.0


def main():
    c = pd.read_csv(DATA / "core_curve_all_kall.csv")
    sp = pd.read_parquet(DATA / "splits.parquet").set_index("preset_id")
    pr = pd.read_parquet(DATA / "presets.parquet").set_index("preset_id")
    c["reason"] = ""
    c.loc[c.level_diff_db.abs() > JUMP_DB, "reason"] = "level jump > 12 dB"
    c.loc[c.level_diff_db < SILENT_DB, "reason"] = "silent"
    ex = c[c.reason != ""].copy()
    ex["split"] = ex.preset_id.map(sp.split)
    ex["name"] = ex.preset_id.map(pr.name)
    ex[["preset_id", "name", "label", "split", "reason", "level_diff_db", "distance"]].to_csv(DATA / "vae_excluded.csv", index=False)
    print(f"excluded {len(ex)} of {len(c)} presets ({len(ex) / len(c):.1%})")
    print(pd.crosstab(ex.label, ex.split).reindex(columns=["train", "val", "test"], fill_value=0).to_string())
    print("by reason:", ex.reason.value_counts().to_dict())
    kept = sp[~sp.index.isin(ex.preset_id)]
    print("\nremaining per split:", kept.split.value_counts().to_dict())
    shares = pd.crosstab(kept.label, kept.split, normalize="columns")[["train", "val", "test"]] * 100
    print("label shares after cleaning (%):")
    print(shares.round(1).to_string())


if __name__ == "__main__":
    main()
