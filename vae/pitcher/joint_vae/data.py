"""Joint VAE data: parameter layout, training rows (stage-1 set) and the human val / test presets.

Layout (JSpec), from the kept features (step4_vae.data.Spec):
  num   continuous knobs incl. stepped knobs with > 32 values (keytrack root): 0-1 code, sigmoid + MSE
  bool  on / off: logit + BCE
  cat   the 91 categoricals + the stepped knobs with <= 32 values as categories (NumCat++, Le Vaillant
        2021: octave, unison voices, ...): logits + cross-entropy; level = round(code * (n - 1))
Masked (inactive) features don't count in the loss. Labels: the 8 categories + "unknown" (index 8; for
category-free prompts of the matched clips later). Every stage-1 row has a label (human: its own;
perturb: the base preset's; layered: the structure's; vae: the sampled one).
Sound: the 48-number code (sound_code.py), aligned with MelSet's row order.
"""

import math

import numpy as np
import pandas as pd

from pitcher.core.surge_io import PITCHER_ROOT
from pitcher.inverter.data import full_row
from pitcher.step4_vae.data import LABELS, Spec

DATA = PITCHER_ROOT / "data"
JDATA = DATA / "joint_vae"
MAX_LEVELS = 32
N_CLS = len(LABELS) + 1          # + unknown
UNKNOWN = len(LABELS)


class JSpec:
    def __init__(self, spec=None):
        self.spec = spec or Spec()
        s = self.spec
        kinds = {f["name"]: f for f in s.feats}
        self.ord_cat, self.ord_levels = [], []
        for n in s.cont:
            f = kinds[n]
            if f["kind"] == "ord":
                lo, hi = f["natural_range"]
                levels = int(round(hi - lo)) + 1
                if levels <= MAX_LEVELS:
                    self.ord_cat.append(n)
                    self.ord_levels.append(levels)
        self.num = [n for n in s.cont if n not in set(self.ord_cat)]
        self.bool = list(s.bool)
        self.cat = list(s.cat) + self.ord_cat
        self.cat_sizes = list(s.cat_sizes) + self.ord_levels
        self.n_ord = len(self.ord_cat)

    def targets(self, C):
        """C: DataFrame with the kept features (NaN = inactive) -> dict of numpy arrays."""
        xn = C[self.num].to_numpy(np.float32)
        xb = C[self.bool].to_numpy(np.float32)
        xk = C[self.spec.cat].to_numpy(np.float32)
        xo = C[self.ord_cat].to_numpy(np.float32)
        lv = np.array(self.ord_levels, np.float32)
        xo = np.where(np.isnan(xo), np.nan, np.clip(np.round(xo * (lv - 1)), 0, lv - 1))
        xk = np.hstack([xk, xo])
        mn, mb, mk = ~np.isnan(xn), ~np.isnan(xb), ~np.isnan(xk)
        return dict(x_cont=np.nan_to_num(xn), m_cont=mn.astype(np.float32),
                    x_bool=np.nan_to_num(xb), m_bool=mb.astype(np.float32),
                    x_cat=np.where(mk, np.nan_to_num(xk), -1).astype(np.int64), m_cat=mk.astype(np.float32))

    def to_row(self, num, bool_on, cat_idx):
        """One preset: num values (0-1), bool on/off (0/1), option index per cat -> full codes row."""
        v = dict(zip(self.num, map(float, num)))
        v.update(zip(self.bool, map(float, bool_on)))
        nk = len(self.spec.cat)
        v.update((n, float(i)) for n, i in zip(self.spec.cat, cat_idx[:nk]))
        v.update((n, float(i) / (lv - 1)) for n, i, lv in zip(self.ord_cat, cat_idx[nk:], self.ord_levels))
        return full_row(self.spec, v)


def label_index(labels):
    lut = {l: i for i, l in enumerate(LABELS)}
    return np.array([lut.get(l, UNKNOWN) if isinstance(l, str) else UNKNOWN for l in labels], np.int64)


class JointSet:
    """Training rows: the stage-1 set's valid rows (MelSet order) without the mels (only their codes)."""

    def __init__(self, name="pilot", jspec=None, val_share=0.02, seed=0):
        from pitcher.synth_data.stage1 import load
        self.js = jspec or JSpec()
        meta, _ = load(name)
        root = DATA / "synth" / "stage1" / name
        keep, C = [], []
        for c in sorted(meta.chunk.unique()):            # same order and filter as inverter.data.MelSet
            m = meta[meta.chunk == c]
            codes = pd.read_parquet(root / f"chunk_{c:03d}" / "codes.parquet")
            ok = ~(m.silent.to_numpy() | m.nan.to_numpy())
            C.append(codes.iloc[m.pos.to_numpy()[ok]])
            keep.append(m[ok])
        self.meta = pd.concat(keep, ignore_index=True)
        self.codes = pd.concat(C, ignore_index=True)
        self.t = self.js.targets(self.codes)
        self.y = label_index(self.meta.label)
        self.a = np.load(JDATA / f"codes_{name}.npy")
        assert len(self.a) == len(self.meta), "sound codes out of date: run pitcher.joint_vae.sound_code"
        rng = np.random.default_rng(seed)
        syn = (self.meta.source != "human").to_numpy()
        self.is_val = syn & (rng.random(len(self.meta)) < val_share)   # identical to MelSet's split


def load_eval(jspec):
    """Human val / test presets (data/inverter/eval): meta, mel, codes, targets, labels, sound codes."""
    from pitcher.inverter.data import load_eval as inv_eval
    meta, mel, codes, _ = inv_eval(jspec.spec)
    return meta, mel, codes, jspec.targets(codes), label_index(meta.label), np.load(JDATA / "codes_eval.npy")
