"""Sound code: a stored mel (559 frames x 64 bands) -> K = 48 numbers.

Steps: average 4 frames in time (140 x 64), make it relative to its own peak and floor at -80 dB (as the
inverter sees it), PCA over the training rows. One global scale makes the average code dimension unit
variance, so distances in code space stay proportional to mel distances (no per-dimension whitening, which
would give the small, noisy components the same weight as the main ones).
K = 48 (user, 17 Oct): reconstruction distance on 600 held-out rows (stage1.exact, the threshold's dB scale)
median 2.6 dB, 90th percentile 4.1 dB; Surge's own render-to-render noise is 2.3 dB, the "same character"
threshold 7.3 dB; 64 numbers 2.4 / 3.9 dB, 32 numbers 2.9 / 4.7 dB (Round 36).

Writes  data/joint_vae/sound_pca.npz (components, mean, scale, explained variance)
        data/joint_vae/codes_<set>.npy (MelSet row order) and codes_eval.npy (human val / test presets)
Run:    .venv/bin/python -m pitcher.joint_vae.sound_code [--data pilot]
"""

import argparse
import json

import numpy as np

from pitcher.core.surge_io import PITCHER_ROOT

K = 48
POOL_T = 4
DYN_DB = 80.0
OUT = PITCHER_ROOT / "data" / "joint_vae"


def prepare(M):
    """(n, T, F) stored mels -> (n, T/4 * F) float32, peak-relative, floored."""
    T = (M.shape[1] // POOL_T) * POOL_T
    X = np.asarray(M[:, :T], dtype=np.float32).reshape(len(M), T // POOL_T, POOL_T, -1).mean(2)
    X = np.maximum(X - X.max(axis=(1, 2), keepdims=True), -DYN_DB)
    return X.reshape(len(M), -1)


class SoundCode:
    def __init__(self, path=OUT / "sound_pca.npz"):
        z = np.load(path)
        self.components, self.mean, self.scale = z["components"], z["mean"], float(z["scale"])

    def __call__(self, M, bs=2000):
        return np.vstack([(prepare(M[i:i + bs]) - self.mean) @ self.components.T / self.scale
                          for i in range(0, len(M), bs)]).astype(np.float32)

    def decode(self, C, T=559):
        """codes -> approximate stored mels (peak-relative), for checks."""
        X = (C * self.scale) @ self.components + self.mean
        X = np.repeat(X.reshape(len(C), -1, self.components.shape[1] // ((T // POOL_T))), POOL_T, axis=1)
        return np.concatenate([X, np.repeat(X[:, -1:], T - X.shape[1], axis=1)], axis=1)


def fit(mel, idx, k=K, n_fit=12000, seed=0):
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(idx, min(n_fit, len(idx)), replace=False))
    pca = PCA(k, svd_solver="randomized", random_state=seed).fit(prepare(mel[pick]))
    scale = float(np.sqrt(pca.explained_variance_.mean()))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "sound_pca.npz", components=pca.components_.astype(np.float32), mean=pca.mean_.astype(np.float32),
             scale=scale, explained_variance=pca.explained_variance_, explained_ratio=pca.explained_variance_ratio_)
    return pca


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="pilot")
    args = ap.parse_args()
    from pitcher.inverter.data import MelSet, load_eval
    from pitcher.step4_vae.data import Spec
    from pitcher.synth_data.stage1 import exact
    spec = Spec()
    ds = MelSet(args.data, spec)
    tr = np.flatnonzero(~ds.is_val)
    pca = fit(ds.mel, tr)
    sc = SoundCode()
    codes = sc(ds.mel)
    np.save(OUT / f"codes_{args.data}.npy", codes)
    _, ev_mel, _, _ = load_eval(spec)
    np.save(OUT / "codes_eval.npy", sc(ev_mel))
    # check on the held-out synthetic rows: reconstruction distance on the threshold's scale
    va = np.flatnonzero(ds.is_val)[:600]
    R = sc.decode(codes[va])
    orig = ds.mel[va].astype(np.float32)
    d = np.array([float(exact(orig[i] - orig[i].max(), R[i][None].astype(np.float32))[0]) for i in range(len(va))])
    rep = dict(k=K, rows=len(codes), explained=float(pca.explained_variance_ratio_.sum()),
               recon_median_db=float(np.median(d)), recon_p90_db=float(np.percentile(d, 90)),
               code_std_first_last=[float(codes[:, 0].std()), float(codes[:, -1].std())])
    (OUT / f"sound_code_{args.data}.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
