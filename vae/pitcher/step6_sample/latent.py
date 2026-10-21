"""Step 6 samplers: where in the latent space to draw z for a requested label.

  prior  z ~ tau x N(0, I), the CVAE's own prior (shared by all labels; the label only enters the decoder)
  gmm    z ~ a Gaussian mixture fitted per label to where that label's TRAIN presets actually sit:
         the aggregate posterior (5 draws z ~ q(z | x, label) per preset), number of components
         1..5 chosen by BIC, full covariances; tau scales each component's spread around its mean.
         Answers "the shared prior puts mass between the presets of a label" (Round 18).
The GMMs are fitted once per run and cached in runs/vae/<run>/latent_gmm.pkl.

Run:    .venv/bin/python -m pitcher.step6_sample.latent --run h256_d01   (fits + prints components)
"""

import argparse
import json
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture

from pitcher.core.surge_io import PITCHER_ROOT
from pitcher.step4_vae.data import LABELS, Spec, load
from pitcher.step4_vae.eval import rows_from
from pitcher.step4_vae.model import CVAE
from pitcher.step4_vae.train import tensors

RUNS = PITCHER_ROOT / "runs" / "vae"
SAMPLERS = ["prior", "gmm"]
DRAWS, MAX_K = 5, 5


def load_model(run):
    cfg = json.loads((RUNS / run / "config.json").read_text())
    model = CVAE(cfg["n_in"], cfg["n_cont"], cfg["n_bool"], cfg["cat_sizes"], len(LABELS), cfg["z"], cfg["hidden"], cfg["dropout"])
    model.load_state_dict(torch.load(RUNS / run / "model.pt"))
    return model.eval(), cfg


@torch.no_grad()
def fit_gmms(model, data, seed=0):
    """{label: GaussianMixture} on the aggregate posterior of the train presets of each label."""
    torch.manual_seed(seed)
    t = tensors(data["train"])
    mu, logvar = model.encode(t["x_in"], t["y1h"])
    y = data["train"]["y"]
    gmms = {}
    for li, lab in enumerate(LABELS):
        idx = torch.tensor(np.flatnonzero(y == li).tolist())
        m, s = mu[idx], (0.5 * logvar[idx]).exp()
        z = torch.cat([m + s * torch.randn_like(m) for _ in range(DRAWS)])
        Z = np.array(z.tolist())
        fits = [GaussianMixture(k, covariance_type="full", reg_covar=1e-4, n_init=3, random_state=seed).fit(Z)
                for k in range(1, MAX_K + 1)]
        bic = [g.bic(Z) for g in fits]
        gmms[lab] = fits[int(np.argmin(bic))]
        gmms[lab].n_presets_, gmms[lab].bic_ = len(idx), bic
    return gmms


def get_gmms(run, model=None, data=None, refit=False):
    path = RUNS / run / "latent_gmm.pkl"
    if path.exists() and not refit:
        return pickle.loads(path.read_bytes())
    if model is None:
        model, _ = load_model(run)
    if data is None:
        _, data = load(Spec())
    gmms = fit_gmms(model, data)
    path.write_bytes(pickle.dumps(gmms))
    return gmms


def sample_z(sampler, label, n, tau, z_dim, gmms=None, rng=None):
    """(n, z_dim) list of latent vectors."""
    rng = rng or np.random.default_rng(0)
    if sampler == "prior":
        return (tau * rng.standard_normal((n, z_dim))).tolist()
    if sampler == "gmm":
        g = gmms[label]
        comp = rng.choice(len(g.weights_), size=n, p=g.weights_)
        L = np.linalg.cholesky(g.covariances_)
        eps = rng.standard_normal((n, z_dim))
        return (g.means_[comp] + tau * np.einsum("nij,nj->ni", L[comp], eps)).tolist()
    raise ValueError(sampler)


@torch.no_grad()
def decode_rows(model, spec, z, label):
    """latent vectors (list) + label -> feature rows for the writer."""
    z = torch.tensor(z, dtype=torch.float32)
    y1h = F.one_hot(torch.full((len(z),), LABELS.index(label)), len(LABELS)).float()
    return rows_from(spec, *model.decode(z, y1h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="h256_d01")
    a = ap.parse_args()
    gmms = get_gmms(a.run, refit=True)
    print(f"per-label latent GMMs (run {a.run}; aggregate posterior, {DRAWS} draws per train preset, k by BIC)")
    for lab, g in gmms.items():
        spread = np.sqrt(np.trace(g.covariances_, axis1=1, axis2=2) / g.means_.shape[1])
        print(f"  {lab:<11} {g.n_presets_:>4} presets  k={len(g.weights_)}  weights {np.round(g.weights_, 2).tolist()}  "
              f"|mean| {np.round(np.linalg.norm(g.means_, axis=1), 2).tolist()}  rms spread {np.round(spread, 2).tolist()}")


if __name__ == "__main__":
    main()
