"""Step 2.1: drop features that (almost) never change (PIPELINE_VAE_SURGE.md, Step 2.1).

Statistics on the TRAIN split only. "Changed" means: the feature is active in the preset AND
  - value features: differs from its REFERENCE (data/reference_codes.parquet, pitcher/step2_explore/reference.py:
    template patch + this preset's types = what the writer produces when it does not write it)
  - flags and FX-block features (present / placement / position / wavetable name): differs from
    the most common value among active train presets (no reference defined for these)
  cont/ord: "differs" = by more than 0.005 of the range.
A feature is KEPT if its changed share >= THRESHOLD (1%, decided in Round 10). A dropped feature becomes:
  reference  value features: the reference value (by construction what >= 97% of presets have)
  mode       flags: their most common train value
  absent     effect types whose fx/<type>/present is dropped (with their whole block)
Flags are dropped with their parent feature.

History (Round 8): freezing every dropped feature at its most common value made 11 presets silent
(Modern oscillator waveform levels are "off" in most presets, but each preset needs one of them);
freezing at getParamDef was worse (17% transparent: getParamDef is not the value a type gets when
selected). The reference definition is consistent with what the writer does.

Writes  data/feature_activity.csv   one row per feature: shares, kept, reason, freeze
        data/pruning.json           threshold, kept features, freeze rule per dropped feature
        data/codes_pruned.parquet   codes with dropped features replaced as the writer would
Run:    .venv/bin/python -m pitcher.step2_explore.reference && .venv/bin/python -m pitcher.step2_explore.prune
"""

import argparse
import json

import numpy as np
import pandas as pd

from pitcher.core.surge_io import PITCHER_ROOT

DATA = PITCHER_ROOT / "data"
CONT_TOL = 0.005
MODE_RULE = ("/present", "/placement", "/position", "/wt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.01)  # decided in Round 10 (1%)
    args = ap.parse_args()

    schema = json.loads((DATA / "schema.json").read_text())
    feats = {f["name"]: f for f in schema["features"]}
    codes = pd.read_parquet(DATA / "codes.parquet")
    masks = pd.read_parquet(DATA / "masks.parquet")
    ref = pd.read_parquet(DATA / "reference_codes.parquet")
    splits = pd.read_parquet(DATA / "splits.parquet")
    train = splits.loc[splits.split == "train", "preset_id"]
    C, M, R = codes.loc[train], masks.loc[train], ref.loc[train]
    n = len(train)

    rows = []
    for f in schema["features"]:
        name, kind = f["name"], f["kind"]
        active = M[name] == 1
        vals = C.loc[active, name]
        use_mode = "flag_of" in f or name.endswith(MODE_RULE)
        mode = np.nan
        if len(vals) == 0:
            changed = 0
        elif use_mode:
            mode = float(vals.round(3).mode().iloc[0])
            diff = (vals.sub(mode).abs() > CONT_TOL) if kind in ("cont", "ord") else (vals != mode)
            changed = int(diff.sum())
        else:
            r = R.loc[active, name]
            diff = (vals.sub(r).abs() > CONT_TOL) if kind in ("cont", "ord") else (vals != r)
            changed = int((diff | r.isna()).sum())  # no reference = counts as changed (kept if common)
        rows.append(dict(feature=name, kind=kind, active_share=active.mean(), changed_share=changed / n,
                         n_changed=changed, rule="mode" if use_mode else "reference", mode_code=mode,
                         flag_of=f.get("flag_of")))
    act = pd.DataFrame(rows).set_index("feature")
    act["kept"] = act.changed_share >= args.threshold
    act["reason"] = np.where(act.kept, "", "changed in < %.0f%% of train presets" % (args.threshold * 100))
    act["freeze"] = np.where(act.kept, "", act.rule)

    # placement and position are structural: they stay whenever the effect type stays (freezing them
    # made effects claim the same slot -> wrong chain order; 107 presets in Round 8)
    for name in act.index:
        if name.endswith(("/placement", "/position")):
            present = "/".join(name.split("/")[:2]) + "/present"
            act.loc[name, ["kept", "reason", "freeze"]] = [bool(act.at[present, "kept"]), "", ""]

    for name in act.index:  # dependencies
        parent = act.at[name, "flag_of"]
        if isinstance(parent, str) and not act.at[parent, "kept"] and act.at[name, "kept"]:
            act.loc[name, ["kept", "reason", "freeze"]] = [False, "parent feature dropped", "mode"]
        if name.startswith("fx/"):
            present = "/".join(name.split("/")[:2]) + "/present"
            if not act.at[present, "kept"]:
                act.loc[name, ["kept", "freeze"]] = [False, "absent"]
                if name != present:
                    act.at[name, "reason"] = "effect type dropped"
    act.reset_index().to_csv(DATA / "feature_activity.csv", index=False)

    pruned = codes.copy()  # what the writer gets for a dropped feature
    for name in act.index[~act.kept]:
        how, on = act.at[name, "freeze"], masks[name] == 1
        if how == "absent":
            pruned[name] = 0.0 if name.endswith("/present") else np.nan
        elif how == "reference":
            pruned.loc[on, name] = ref.loc[on, name]
        elif not np.isnan(act.at[name, "mode_code"]):
            pruned.loc[on, name] = act.at[name, "mode_code"]
    pruned.to_parquet(DATA / "codes_pruned.parquet")

    kept = act.index[act.kept].tolist()
    n_cols = sum(len(feats[k]["columns"]) for k in kept)
    (DATA / "pruning.json").write_text(json.dumps(dict(
        threshold=args.threshold, n_train=n, statistic="changed share on train (reference / mode rule)",
        template="see pitcher/step2_explore/reference.py (provisional: Templates/Init Saw.fxp until Step 3)",
        n_features=len(act), n_kept=len(kept), n_model_columns_kept=n_cols, kept=kept,
        frozen={k: dict(how=act.at[k, "freeze"],
                        code=None if act.at[k, "freeze"] != "mode" or np.isnan(act.at[k, "mode_code"])
                        else float(act.at[k, "mode_code"])) for k in act.index[~act.kept]},
    ), indent=1))

    # --- report -----------------------------------------------------------------------------------
    print(f"train presets: {n}; keep if changed in >= {args.threshold:.0%} ({int(np.ceil(args.threshold * n))} presets)")
    print(f"features: {len(act)} -> kept {len(kept)} ({act.kept.mean():.0%}); model-ready columns: "
          f"{sum(len(f['columns']) for f in schema['features'])} -> {n_cols}")
    print("kept by kind:", act[act.kept].kind.value_counts().to_dict())
    print("dropped, by reason:", act[~act.kept].reason.value_counts().to_dict())
    print("dropped, by freeze:", act[~act.kept].freeze.value_counts().to_dict())
    for th in (0.01, 0.03, 0.05, 0.10):
        print(f"  sensitivity: threshold {th:.0%} would keep {(act.changed_share >= th).sum()} features (before dependencies)")
    grp = lambda s: "/".join(s.split("/")[:2]) if s.startswith("fx/") else "/".join(s.split("/")[:3]).split(".")[0]
    g = act.assign(group=[grp(i) for i in act.index]).groupby("group").kept.agg(["sum", "size"])
    print("\nkept / total per group (largest 20):")
    print(g.sort_values("size", ascending=False).head(20).to_string())
    print("\nFX types kept:", sorted(k.split("/")[1] for k in kept if k.endswith("/present")))


if __name__ == "__main__":
    main()
