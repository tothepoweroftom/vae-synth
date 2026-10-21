"""Conditional VAE over Surge preset features (PIPELINE_VAE_SURGE.md, Step 4).

Encoder  (x_in, label) -> mu, logvar of a z-dimensional Gaussian
Decoder  (z, label)    -> sigmoid per cont/ord feature, logit per bool feature, logits per categorical
Loss     sum over features of masked MSE (cont/ord) + BCE (bool) + CE (cat), plus beta x KL;
         KL per dimension with an optional free-bits floor (against posterior collapse).
Inputs are plain torch tensors; numpy is only bridged via lists (torch 2.2.2 on Intel macOS cannot
use numpy 2's C API).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(n_in, n_out, hidden, dropout):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.SiLU(), nn.Dropout(dropout),
                         nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(dropout),
                         nn.Linear(hidden, n_out))


class CVAE(nn.Module):
    def __init__(self, n_in, n_cont, n_bool, cat_sizes, n_cls=8, z_dim=12, hidden=256, dropout=0.1):
        super().__init__()
        self.n_cont, self.n_bool, self.cat_sizes, self.n_cls, self.z_dim = n_cont, n_bool, list(cat_sizes), n_cls, z_dim
        self.enc = mlp(n_in + n_cls, 2 * z_dim, hidden, dropout)
        self.dec = mlp(z_dim + n_cls, n_cont + n_bool + sum(cat_sizes), hidden, dropout)

    def encode(self, x_in, y_onehot):
        mu, logvar = self.enc(torch.cat([x_in, y_onehot], -1)).chunk(2, -1)
        return mu, logvar.clamp(-8, 8)

    def decode(self, z, y_onehot):
        out = self.dec(torch.cat([z, y_onehot], -1))
        cont, bool_logit, rest = out.split([self.n_cont, self.n_bool, sum(self.cat_sizes)], -1)
        return torch.sigmoid(cont), bool_logit, list(rest.split(self.cat_sizes, -1))

    def forward(self, x_in, y_onehot):
        mu, logvar = self.encode(x_in, y_onehot)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        return self.decode(z, y_onehot), mu, logvar


def recon_terms(out, b):
    """Per-preset masked reconstruction terms: (cont MSE, bool BCE, cat CE), each summed over features."""
    cont, bool_logit, cats = out
    mse = (((cont - b["x_cont"]) ** 2) * b["m_cont"]).sum(-1)
    bce = (F.binary_cross_entropy_with_logits(bool_logit, b["x_bool"], reduction="none") * b["m_bool"]).sum(-1)
    ce = torch.zeros_like(mse)
    for j, logits in enumerate(cats):
        target = b["x_cat"][:, j].clamp(min=0)
        ce = ce + F.cross_entropy(logits, target, reduction="none") * b["m_cat"][:, j]
    return mse, bce, ce


def loss_fn(out, mu, logvar, b, beta, free_bits=0.0, w_cont=1.0, w_bool=1.0, w_cat=1.0):
    mse, bce, ce = recon_terms(out, b)
    rec = w_cont * mse + w_bool * bce + w_cat * ce
    kl_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)            # (batch, z)
    kl_used = torch.clamp(kl_dim.mean(0), min=free_bits).sum() if free_bits > 0 else kl_dim.sum(-1).mean()
    loss = rec.mean() + beta * kl_used
    return loss, dict(rec=rec.mean().item(), mse=mse.mean().item(), bce=bce.mean().item(), ce=ce.mean().item(),
                      kl=kl_dim.sum(-1).mean().item()), kl_dim.mean(0).detach()
