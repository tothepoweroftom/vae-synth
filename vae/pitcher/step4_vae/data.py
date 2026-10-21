"""Data for the CVAE (PIPELINE_VAE_SURGE.md, Step 4): kept features (Step 2.1), cleaned presets.

Per preset:
  x_in     encoder input: model-ready columns of the kept features (data/dataset.parquet) + one
           mask bit per kept feature (so the encoder can tell "inactive" from "value 0")
  x_cont   0-1 codes of cont/ord features, m_cont their masks
  x_bool   0/1 codes of bool features,   m_bool
  x_cat    option index per categorical feature (-1 where masked), m_cat
  y        label index (LABELS order)
Excluded: presets in data/vae_excluded.csv (pitcher/step4_vae/clean.py).

Also: to_codes() turns decoder outputs back into a full codes row for codec.decode(), filling the
features dropped in Step 2.1 with their freeze rule (mode value / not written / effect absent).
"""

import json
import math

import numpy as np
import pandas as pd

from pitcher.core.surge_io import PITCHER_ROOT

DATA = PITCHER_ROOT / "data"
LABELS = ["Bass", "Lead", "Pad", "Keys", "Pluck", "Synth", "Atmosphere", "Orchestral"]


class Spec:
    """Column layout of the kept features."""

    def __init__(self):
        schema = json.loads((DATA / "schema.json").read_text())
        self.pruning = json.loads((DATA / "pruning.json").read_text())
        kept = set(self.pruning["kept"])
        self.all_names = [f["name"] for f in schema["features"]]
        self.feats = [f for f in schema["features"] if f["name"] in kept]
        self.cont = [f["name"] for f in self.feats if f["kind"] in ("cont", "ord")]
        self.bool = [f["name"] for f in self.feats if f["kind"] == "bool"]
        self.cat = [f["name"] for f in self.feats if f["kind"] == "cat"]
        self.cat_sizes = [len(f["options"]) for f in self.feats if f["kind"] == "cat"]
        self.in_columns = [c for f in self.feats for c in f["columns"]]
        self.names = [f["name"] for f in self.feats]

    @property
    def n_in(self):
        return len(self.in_columns) + len(self.names)

    def to_codes(self, cont, bool_logit, cat_logits):
        """Decoder outputs for ONE preset -> {feature: code} for all schema features."""
        row = {n: math.nan for n in self.all_names}
        for n, v in zip(self.cont, cont):
            row[n] = float(v)
        for n, v in zip(self.bool, bool_logit):
            row[n] = 1.0 if v > 0 else 0.0
        for n, logits in zip(self.cat, cat_logits):
            row[n] = float(int(np.argmax(logits)))
        for n, fr in self.pruning["frozen"].items():  # features dropped in Step 2.1
            if fr["how"] == "mode" and fr["code"] is not None:
                row[n] = fr["code"]
            elif fr["how"] == "absent" and n.endswith("/present"):
                row[n] = 0.0
        return row


def load(spec=None):
    """{split: dict of numpy arrays} for train / val / test (cleaned)."""
    spec = spec or Spec()
    X = pd.read_parquet(DATA / "dataset.parquet")
    M = pd.read_parquet(DATA / "masks.parquet")
    C = pd.read_parquet(DATA / "codes.parquet")
    sp = pd.read_parquet(DATA / "splits.parquet").set_index("preset_id")
    excl = set(pd.read_csv(DATA / "vae_excluded.csv").preset_id)
    out = {}
    for split in ("train", "val", "test"):
        ids = [p for p in sp.index[sp.split == split] if p not in excl]
        m = M.loc[ids, spec.names].to_numpy(np.float32)
        x_in = np.hstack([X.loc[ids, spec.in_columns].to_numpy(np.float32), m])
        mc = M.loc[ids, spec.cont].to_numpy(np.float32)
        mb = M.loc[ids, spec.bool].to_numpy(np.float32)
        mk = M.loc[ids, spec.cat].to_numpy(np.float32)
        xc = np.nan_to_num(C.loc[ids, spec.cont].to_numpy(np.float32))
        xb = np.nan_to_num(C.loc[ids, spec.bool].to_numpy(np.float32))
        xk = np.where(mk > 0, np.nan_to_num(C.loc[ids, spec.cat].to_numpy(np.float32), nan=0), -1).astype(np.int64)
        y = np.array([LABELS.index(l) for l in sp.loc[ids, "label"]], dtype=np.int64)
        out[split] = dict(ids=np.array(ids), x_in=x_in, x_cont=xc, m_cont=mc, x_bool=xb, m_bool=mb,
                          x_cat=xk, m_cat=mk, y=y)
    return spec, out
