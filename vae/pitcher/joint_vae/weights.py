"""Sound-weighted parameter loss: how audible each parameter is, from the stage-0 screening.

Per parameter group, the screening's mean level-matched mel distance when the group is set to another
train preset's value (mu_norm, data/synth/screening_groups.csv; 514 groups, 32 base presets). Every
feature of the joint layout belongs to one group (synth_data.common.Base.groups). Groups tested on only
1-4 bases (rare oscillator / effect types) are shrunk towards the median of their kind (num / bool / cat)
with the weight of SHRINK bases; untested groups get the median. Weight = (dB / median dB of the kind)
^ power, clipped to [lo, hi] and scaled to mean 1 within each kind, so the balance between knobs, on /
off switches and choices stays as in the unweighted loss and only shifts within each kind: a wrong
cutoff costs more than a wrong chorus rate. (The stage-0b rulers cover only 40 knobs; the screening
covers all.)

Writes  data/joint_vae/loss_weights.csv (feature, kind, group, db, n_bases, weight) on fit()
"""

import numpy as np
import pandas as pd
import torch

from pitcher.joint_vae.data import JDATA

SCREEN = JDATA.parent / "synth" / "screening_groups.csv"
SHRINK = 5


def table(js, power=1.0, lo=0.1, hi=10.0):
    from pitcher.synth_data.common import Base
    f2g = {f: g for g, fs in Base().groups() for f in fs}
    G = pd.read_csv(SCREEN).set_index("group")
    rows = []
    for kind, feats in (("num", js.num), ("bool", js.bool), ("cat", js.cat)):
        g = [f2g.get(f) for f in feats]
        db = np.array([G.at[x, "mu_norm"] if x in G.index else np.nan for x in g])
        n = np.array([G.at[x, "n_bases"] if x in G.index else 0 for x in g], float)
        med = np.nanmedian(db[n >= SHRINK])
        db = np.where(n > 0, (n * np.nan_to_num(db) + SHRINK * med) / (n + SHRINK), med)
        w = np.clip((db / med) ** power, lo, hi)
        rows.append(pd.DataFrame(dict(feature=feats, kind=kind, group=g, db=db, n_bases=n, weight=w / w.mean())))
    return pd.concat(rows, ignore_index=True)


def fit(js, power=1.0, lo=0.1, hi=10.0):
    T = table(js, power, lo, hi)
    T.to_csv(JDATA / "loss_weights.csv", index=False)
    return T


def tensors(T):
    """-> dict(num, bool, cat) of float tensors in the joint layout's order."""
    return {k: torch.tensor(T[T.kind == k].weight.to_numpy(), dtype=torch.float32) for k in ("num", "bool", "cat")}
