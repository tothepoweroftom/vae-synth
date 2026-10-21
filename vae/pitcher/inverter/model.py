"""Inverter v0 (TEXT2PRESET_PLAN.md §4): mel spectrogram of a level-normalised 6 s note at C3 -> preset.

Input   the stored 64-band mel (559 frames x 64 bands, dB, floored 80 dB below its peak), averaged over 4
        frames in time (140 x 43 ms), scaled to [-1, 1] relative to its own peak
Encoder 4 conv blocks (2 x conv3x3 + batch norm + SiLU, then 2 x 2 max-pool), channels 16-32-64-128,
        pooled to 4 x 8, -> 512 -> audio code c (256)
Heads   the VAE decoder's (step4_vae.model.CVAE): sigmoid per cont / ord feature, logit per bool,
        logits per categorical; loss = the same masked sum (step4_vae.model.recon_terms)
Several candidates per sound (many presets give one sound): candidate 0 = the most likely values;
the others sample the categorical and on/off heads (temperature 1), so different switch settings
(filter model, oscillator type, ...) are tried; continuous values stay at the prediction. A generative
decoder (conditional VAE / flow matching) replaces this once the chain runs.
Augmentation (training only, in the mel domain): EQ tilt, raised floor, small onset delay, a little noise:
the Stable Audio clips are close to Surge renders but not identical.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

POOL_T = 4
DYN_DB = 80.0


def block(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.SiLU(),
                         nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.SiLU(),
                         nn.MaxPool2d(2))


def prepare(mel):
    """(B, T, F) dB -> (B, 1, F, T') in [-1, 1] relative to each mel's peak, T' = T / POOL_T."""
    x = mel - mel.amax(dim=(1, 2), keepdim=True)
    x = torch.clamp(x, min=-DYN_DB) / (DYN_DB / 2) + 1
    x = x.transpose(1, 2).unsqueeze(1)
    return F.avg_pool2d(x, (1, POOL_T))


def augment(mel, gen=None):
    """mel (B, T, F) in dB, before prepare()."""
    B, T, Fq = mel.shape
    dev = mel.device
    r = lambda *s: torch.rand(*s, generator=gen, device=dev)
    peak = mel.amax(dim=(1, 2), keepdim=True)
    # EQ tilt: up to +-6 dB across the bands
    tilt = (r(B, 1, 1) * 12 - 6) * (r(B, 1, 1) < 0.5)
    mel = mel + tilt * (torch.linspace(-0.5, 0.5, Fq, device=dev).view(1, 1, Fq))
    # raised floor (noise floor / less dynamic range): 50-80 dB below the peak
    floor = peak - (50 + 30 * r(B, 1, 1))
    mel = torch.where(r(B, 1, 1) < 0.3, torch.maximum(mel, floor), mel)
    # onset delay: 0-3 pooled frames (0-130 ms)
    shift = (r(B) * 4).long() * POOL_T * (r(B) < 0.3).long()
    if shift.any():
        idx = (torch.arange(T, device=dev).view(1, T) - shift.view(B, 1)).clamp(min=0)
        silent = torch.arange(T, device=dev).view(1, T) < shift.view(B, 1)
        mel = torch.gather(mel, 1, idx.unsqueeze(-1).expand(B, T, Fq))
        mel = torch.where(silent.unsqueeze(-1), peak - DYN_DB, mel)
    # a little noise
    mel = mel + torch.randn(mel.shape, generator=gen, device=dev) * 0.5 * (r(B, 1, 1) < 0.3)
    return mel


class Inverter(nn.Module):
    def __init__(self, n_cont, n_bool, cat_sizes, channels=(16, 32, 64, 128), code=256, dropout=0.2):
        super().__init__()
        self.n_cont, self.n_bool, self.cat_sizes = n_cont, n_bool, list(cat_sizes)
        layers, c = [], 1
        for ch in channels:
            layers.append(block(c, ch))
            c = ch
        self.conv = nn.Sequential(*layers, nn.AdaptiveAvgPool2d((4, 8)))
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(c * 32, 512), nn.SiLU(), nn.Dropout(dropout),
                                nn.Linear(512, code), nn.SiLU())
        self.head = nn.Linear(code, n_cont + n_bool + sum(cat_sizes))

    def encode(self, mel):
        return self.fc(self.conv(prepare(mel)))

    def forward(self, mel):
        c = self.encode(mel)
        cont, bl, rest = self.head(c).split([self.n_cont, self.n_bool, sum(self.cat_sizes)], -1)
        return (torch.sigmoid(cont), bl, list(rest.split(self.cat_sizes, -1))), c
