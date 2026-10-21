"""Step 2.1 helper: the REFERENCE value of every feature, per preset.

Reference = what the writer produces for a feature it does NOT write: the template patch, with only
this preset's types written (oscillator/filter/FX types, FX slots), so oscillators and effects carry
the values Surge gives them when that type is selected (init_default_values), not the nominal
getParamDef (e.g. Modern oscillator: getParamDef saw level = 0 = silent; on selection = full level).

Step 2.1 drops a feature when it equals its reference in >= 97% of train presets; a dropped feature
then simply takes the reference value. This is the plan's "fix all other parameters to an init
patch (template)". Until Step 3 defines the final template, the template is Surge's Init Saw.

Writes  data/reference_codes.parquet   same shape as codes.parquet (NaN where not written/inactive)
Run:    .venv/bin/python -m pitcher.step2_explore.reference [--template PATH]
"""

import argparse
import json
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from pitcher.core.codec import Plan, decode
from pitcher.core.surge_io import DATA_HOME, PITCHER_ROOT, SurgeIO

DATA = PITCHER_ROOT / "data"
DEFAULT_TEMPLATE = DATA_HOME / "patches_factory" / "Templates" / "Init Saw.fxp"
_io = _schema = _codes = _template = None


def _init(template):
    global _io, _schema, _codes, _template
    _io, _template = SurgeIO(), template
    _schema = json.loads((DATA / "schema.json").read_text())
    _codes = pd.read_parquet(DATA / "codes.parquet")


def _one(pid):
    s = _io.s
    plan = decode({k: float(v) for k, v in _codes.loc[pid].items()}, _schema)
    _io.load(_template)
    _io.write_plan(Plan(types=plan.types, subtypes=plan.subtypes, fx_slots=plan.fx_slots))
    out = {}
    feats = {f["name"]: f for f in _schema["features"]}
    for name, path, kind, code, nat in plan.values:
        p = _io.by_path[path]
        v, lo, hi = s.getParamVal(p), s.getParamMin(p), s.getParamMax(p)
        if kind in ("cont", "ord"):
            out[name] = min(max((v - lo) / (hi - lo), 0.0), 1.0) if hi > lo else 0.0
        elif kind == "bool":
            out[name] = float(v >= 0.5)
        else:
            opts = feats[name]["options"]
            out[name] = float(opts.index(int(round(v)))) if int(round(v)) in opts else np.nan
    for name, c in _codes.loc[pid].items():  # template has no routings -> reference depth 0 (code 0.5)
        if name.startswith("route/") and not np.isnan(c):
            out[name] = 0.5
    return pid, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    codes = pd.read_parquet(DATA / "codes.parquet")
    t0 = time.time()
    with Pool(args.workers, initializer=_init, initargs=(args.template,)) as pool:
        res = dict(pool.imap_unordered(_one, list(codes.index), chunksize=8))
    ref = pd.DataFrame.from_dict(res, orient="index").reindex(index=codes.index, columns=codes.columns)
    ref.index.name = "preset_id"
    ref.to_parquet(DATA / "reference_codes.parquet")
    known = ref.notna().sum().sum()
    print(f"reference for {len(ref)} presets in {time.time() - t0:.0f}s (template: {args.template}); "
          f"{known} reference values")
    for k in ["scene/osc/0/modern/p0", "scene/adsr/0/a", "scene/filterunit/0/cutoff"]:
        if k in ref:
            print(f"  {k}: reference codes {ref[k].dropna().round(3).value_counts().head(3).to_dict()}")


if __name__ == "__main__":
    main()
