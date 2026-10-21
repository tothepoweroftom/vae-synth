"""Inverter data (TEXT2PRESET_PLAN.md §4): (mel spectrogram, parameters) pairs.

Train  the stage-1 set (data/synth/stage1/<name>/): chunk 0 = the cleaned human train presets, chunks 1..
       = synthetic rows; valid rows only (not silent, no NaN). 2% of the synthetic rows are held out
       (by id) to watch over-fitting on the training distribution.
Eval   the human val / test presets (never used by the sampler), rendered exactly like the training rows
       (stage1.render_mel, seed = preset id + 7): build_eval() -> data/inverter/eval/{meta,codes}.parquet + mel.npy

Targets use the VAE decoder's layout (step4_vae.data.Spec): cont / ord codes + masks, bool + masks,
categorical option index + masks; NaN in a stored row = inactive = masked. full_row() turns kept-feature
values back into a codes row for codec.decode(): the mode-frozen features at their code everywhere (same
sound as each preset's own pattern: measured 0.0 dB on 40 val presets).
"""

import json
import math

import numpy as np
import pandas as pd

from pitcher.core.surge_io import PITCHER_ROOT
from pitcher.step4_vae.data import LABELS, Spec

DATA = PITCHER_ROOT / "data"
EVAL = DATA / "inverter" / "eval"


def targets(spec, C):
    """C: DataFrame with the kept features -> dict of numpy arrays (Spec layout)."""
    xc = C[spec.cont].to_numpy(np.float32)
    xb = C[spec.bool].to_numpy(np.float32)
    xk = C[spec.cat].to_numpy(np.float32)
    mc, mb, mk = ~np.isnan(xc), ~np.isnan(xb), ~np.isnan(xk)
    return dict(x_cont=np.nan_to_num(xc), m_cont=mc.astype(np.float32),
                x_bool=np.nan_to_num(xb), m_bool=mb.astype(np.float32),
                x_cat=np.where(mk, np.nan_to_num(xk), -1).astype(np.int64), m_cat=mk.astype(np.float32))


class MelSet:
    """All valid rows of a stage-1 run, in memory: mel (n, 559, 64) float16 + targets + meta."""

    def __init__(self, name, spec=None, val_share=0.02, seed=0):
        from pitcher.synth_data.stage1 import load
        self.spec = spec or Spec()
        meta, mels = load(name)
        root = DATA / "synth" / "stage1" / name
        keep, M, C = [], [], []
        for c, mm in zip(sorted(meta.chunk.unique()), mels):
            m = meta[meta.chunk == c]
            codes = pd.read_parquet(root / f"chunk_{c:03d}" / "codes.parquet")
            ok = ~(m.silent.to_numpy() | m.nan.to_numpy())
            M.append(np.asarray(mm)[m.pos.to_numpy()[ok]])
            C.append(codes.iloc[m.pos.to_numpy()[ok]])
            keep.append(m[ok])
        self.meta = pd.concat(keep, ignore_index=True)
        self.mel = np.concatenate(M)
        codes = pd.concat(C, ignore_index=True)
        self.codes = codes
        self.t = targets(self.spec, codes)
        rng = np.random.default_rng(seed)
        syn = (self.meta.source != "human").to_numpy()
        self.is_val = syn & (rng.random(len(self.meta)) < val_share)
        self.weight = np.ones(len(self.meta), np.float32)

    def __len__(self):
        return len(self.meta)


def full_row(spec, values):
    """{kept feature: code} (NaN allowed) -> full codes row for codec.decode()."""
    row = {n: math.nan for n in spec.all_names}
    for n, fr in spec.pruning["frozen"].items():
        if fr["how"] == "mode" and fr["code"] is not None:
            row[n] = fr["code"]
        elif fr["how"] == "absent" and n.endswith("/present"):
            row[n] = 0.0
    row.update(values)
    return row


def build_eval(workers=4):
    """Render the human val / test presets like the training rows."""
    from pitcher.synth_data.common import run_pool
    from pitcher.synth_data.stage1 import _task, init_worker
    sp = pd.read_parquet(DATA / "splits.parquet").set_index("preset_id")
    excl = set(pd.read_csv(DATA / "vae_excluded.csv").preset_id)
    ids = [int(p) for p in sp.index[sp.split.isin(["val", "test"])] if p not in excl]
    specs = [dict(source="human", base=p, label=sp.label[p], id=p, order=j, seed=p + 7, keep_audio=False, audio_path="")
             for j, p in enumerate(ids)]
    res = []
    for r in run_pool(_task, [specs[i:i + 10] for i in range(0, len(specs), 10)], workers, "eval renders", init=init_worker):
        res += r
    res.sort(key=lambda r: r[0]["order"])
    from pitcher.synth_data.sampler import Space
    kept = Space().kept
    meta = pd.DataFrame([r[0] for r in res])
    meta["split"] = meta.id.map(sp.split)
    EVAL.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(np.vstack([r[1] for r in res]), columns=kept).assign(id=meta.id.values).to_parquet(EVAL / "codes.parquet", index=False)
    np.save(EVAL / "mel.npy", np.stack([r[2] for r in res]))
    meta.to_parquet(EVAL / "meta.parquet", index=False)
    print(f"eval set: {len(meta)} presets ({meta.split.value_counts().to_dict()}) -> {EVAL}")


def load_eval(spec=None):
    spec = spec or Spec()
    meta = pd.read_parquet(EVAL / "meta.parquet")
    codes = pd.read_parquet(EVAL / "codes.parquet")
    return meta, np.load(EVAL / "mel.npy"), codes, targets(spec, codes)


if __name__ == "__main__":
    build_eval()
