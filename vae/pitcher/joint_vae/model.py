"""Joint VAE: parameters and sound share one latent z (multimodal VAE with a product of experts,
Wu & Goodman 2018; modality dropout as in their sub-sampled training).

Encoders  params expert: (parameter values + masks + one-hot choices, label) -> mu_p, logvar_p
          sound expert:  (48-number sound code, label)                       -> mu_a, logvar_a
Posterior product of the experts that are present and a N(0, 1) prior expert: precisions add, so with
          only the sound the model works as an inverter (sound -> z -> preset), with only the parameters
          as an encoder of presets, with both as the training posterior
Decoders  params: (z, label) -> sigmoid per num knob, logit per on/off, logits per categorical (incl. the
          stepped knobs, NumCat++); sound: (z, label) -> the 48-number code
Loss      masked parameter terms (step4_vae.model.recon_terms) + w_sound * 0.5 * squared error of the sound
          code + beta * KL (free bits per dimension). The sound term is what makes z sound-aware.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pitcher.step4_vae.model import recon_terms


def mlp(n_in, n_out, hidden, dropout, layers=2):
    mods, c = [], n_in
    for _ in range(layers):
        mods += [nn.Linear(c, hidden), nn.SiLU(), nn.Dropout(dropout)]
        c = hidden
    return nn.Sequential(*mods, nn.Linear(c, n_out))


class JointVAE(nn.Module):
    def __init__(self, n_num, n_bool, cat_sizes, k_sound=48, n_cls=9, z_dim=32, hidden=512, dropout=0.1):
        super().__init__()
        self.n_num, self.n_bool, self.cat_sizes, self.n_cls, self.z_dim, self.k = n_num, n_bool, list(cat_sizes), n_cls, z_dim, k_sound
        self.register_buffer("offsets", torch.tensor([0] + list(torch.tensor(cat_sizes).cumsum(0)[:-1].tolist())))
        n_in = 2 * n_num + 2 * n_bool + sum(cat_sizes)
        self.enc_p = mlp(n_in + n_cls, 2 * z_dim, hidden, dropout)
        self.enc_a = mlp(k_sound + n_cls, 2 * z_dim, hidden, dropout)
        self.dec_p = mlp(z_dim + n_cls, n_num + n_bool + sum(cat_sizes), hidden, dropout)
        self.dec_a = mlp(z_dim + n_cls, k_sound, hidden // 2, dropout)

    def param_input(self, b):
        onehot = torch.zeros(len(b["x_cat"]), sum(self.cat_sizes))
        onehot.scatter_(1, self.offsets + b["x_cat"].clamp(min=0), b["m_cat"])
        return torch.cat([b["x_cont"] * b["m_cont"], b["m_cont"], b["x_bool"] * b["m_bool"], b["m_bool"], onehot], -1)

    def posterior(self, b, a, y1, use_p, use_a):
        """use_p / use_a: (B,) 0/1 floats, which experts are present per row."""
        T, M = [torch.ones(len(y1), self.z_dim)], [torch.zeros(len(y1), self.z_dim)]     # prior expert
        if b is not None and use_p.any():
            mu, lv = self.enc_p(torch.cat([self.param_input(b), y1], -1)).chunk(2, -1)
            t = torch.exp(-lv.clamp(-8, 8)) * use_p[:, None]
            T.append(t); M.append(t * mu)
        if a is not None and use_a.any():
            mu, lv = self.enc_a(torch.cat([a, y1], -1)).chunk(2, -1)
            t = torch.exp(-lv.clamp(-8, 8)) * use_a[:, None]
            T.append(t); M.append(t * mu)
        prec = sum(T)
        return sum(M) / prec, -torch.log(prec)

    def decode(self, z, y1):
        out = self.dec_p(torch.cat([z, y1], -1))
        num, bl, rest = out.split([self.n_num, self.n_bool, sum(self.cat_sizes)], -1)
        return (torch.sigmoid(num), bl, list(rest.split(self.cat_sizes, -1))), self.dec_a(torch.cat([z, y1], -1))

    def forward(self, b, a, y, use_p, use_a, sample=True):
        y1 = F.one_hot(y, self.n_cls).float()
        mu, lv = self.posterior(b, a, y1, use_p, use_a)
        z = mu + torch.randn_like(mu) * (0.5 * lv).exp() if sample else mu
        params, sound = self.decode(z, y1)
        return params, sound, mu, lv


def recon(params, b, w=None):
    """Masked parameter terms per preset; w = per-feature weights (weights.tensors) or None (unweighted)."""
    if w is None:
        return recon_terms(params, b)
    num, bl, cats = params
    mse = (((num - b["x_cont"]) ** 2) * b["m_cont"] * w["num"]).sum(-1)
    bce = (F.binary_cross_entropy_with_logits(bl, b["x_bool"], reduction="none") * b["m_bool"] * w["bool"]).sum(-1)
    ce = torch.zeros_like(mse)
    for j, logits in enumerate(cats):
        ce = ce + F.cross_entropy(logits, b["x_cat"][:, j].clamp(min=0), reduction="none") * b["m_cat"][:, j] * w["cat"][j]
    return mse, bce, ce


def loss_fn(params, sound, mu, lv, b, a, beta, w_sound=1.0, free_bits=0.05, w=None):
    mse, bce, ce = recon(params, b, w)
    snd = 0.5 * ((sound - a) ** 2).sum(-1)
    kl_dim = 0.5 * (mu.pow(2) + lv.exp() - 1 - lv)
    kl = torch.clamp(kl_dim.mean(0), min=free_bits).sum()
    rec_p = mse + bce + ce
    loss = rec_p.mean() + w_sound * snd.mean() + beta * kl
    return loss, dict(params=rec_p.mean().item(), sound=snd.mean().item(), kl=kl_dim.sum(-1).mean().item())
