"""Latent pad data: the joint VAE's latent axes, described, plus the decoder weights for the page.

Per latent dimension (active ones only, KL > ACTIVE_NATS):
  sweep    400 human presets' z (posterior from parameters + sound); the dimension moved from its 10th to its
           90th percentile; both ends decoded (parameter decoder + sound decoder)
  sections share of the parameter change per synth section (num knobs |difference| x the screening loss
           weight, on/off |probability difference|, choices total-variation distance x weight)
  sound    decoded sound code -> approximate mel -> simple measures (brightness, attack, length, noise,
           movement); impact = mean |dB difference| of the two decoded mels
Sound axes: per measure, ridge regression of the measure (computed on each training row's own sound code)
  on z + category one-hot; the z part, normalised, is the direction ("dark <-> bright"); R^2 on held-out rows.
Activity: per page knob, logistic regression z + category -> active (the decoder's switches decide activity
  in Surge; the page has no codec, so it uses this small stand-in).

Writes  data/xypad/latent.json (axes, directions, presets' z, weights as float16 base64)
Run:    .venv/bin/python -m pitcher.xypad.latent [--run j48w_b05]
"""

import argparse
import base64
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from pitcher.core.audio_metrics import FMAX, FMIN, HOP, N_MELS, SR
from pitcher.core.surge_io import PITCHER_ROOT
from pitcher.inverter.train import to_t
from pitcher.joint_vae.data import JDATA, JointSet, JSpec, N_CLS
from pitcher.joint_vae.sound_code import POOL_T, SoundCode
from pitcher.joint_vae.train import load_model
from pitcher.step4_vae.data import LABELS

OUT = PITCHER_ROOT / "data" / "xypad"
ACTIVE_NATS = 0.1
DT = POOL_T * HOP / SR                       # seconds per pooled frame
F0 = 440 * 2 ** ((48 - 69) / 12)             # C3, the render note
HOLD_FRAMES = int(round(2.0 / DT))           # note held 2 s, then released
MEASURES = {                                 # name: (low word, high word, unit)
    "brightness": ("Dark", "Bright", "oct"),
    "attack": ("Snappy", "Slow attack", "s"),
    "length": ("Short", "Long", "s"),
    "noise": ("Tonal", "Noisy", ""),
    "movement": ("Static", "Moving", "dB"),
}
SECTIONS = ["Oscillators", "Mixer", "Filter / drive", "Envelopes", "LFOs", "Modulation", "Effects", "Voice"]
DISPLAY_BOOLS = [f"fx/{f}/present" for f in ("reverb", "reverb2", "delay", "chorus", "flanger", "phaser", "distortion",
                                             "eq", "rotary", "resonator", "freqshift", "conditioner", "nimbus")] + \
                [f"scene/mute_{m}" for m in ("o1", "o2", "o3", "noise")]
DISPLAY_CATS = [f"scene/osc/{o}/type" for o in range(3)]
OSC_TYPES = {0: "Classic", 1: "Sine", 2: "Wavetable", 3: "S&H noise", 5: "FM3", 6: "FM2", 7: "Window", 8: "Modern",
             9: "String", 10: "Twist", 11: "Alias"}


def section(name):
    if name.startswith("route/"):
        return "Modulation"
    if name.startswith("fx/"):
        return "Effects"
    if name.startswith(("scene/level_", "scene/mute_")):
        return "Mixer"
    if name.startswith(("scene/osc/", "scene/fm_", "character")):
        return "Oscillators"
    if name.startswith(("scene/filter", "scene/f2_", "scene/wsunit", "scene/feedback")):
        return "Filter / drive"
    if name.startswith("scene/adsr/"):
        return "Envelopes"
    if name.startswith("scene/lfo/"):
        return "LFOs"
    return "Voice"


def band_hz():
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    return hz(np.linspace(mel(FMIN), mel(FMAX), N_MELS + 2))[1:-1]


FC = band_hz()


def to_np(t):
    """torch -> numpy without the numpy C API (torch 2.2.2 + numpy 2)."""
    return np.array(t.detach().tolist(), dtype=np.float32)


def pooled_mels(sc, C):
    """sound codes (n, 48) -> peak-relative pooled mels (n, 140, 64), floored at -80 dB."""
    X = ((C * sc.scale) @ sc.components + sc.mean).reshape(len(C), -1, N_MELS)
    X = X - X.max(axis=(1, 2), keepdims=True)
    return np.maximum(X, -80.0)


def measures(X):
    """pooled mels (n, T, 64) -> dict of (n,) measures. The page computes the same in JavaScript."""
    P = 10 ** (X / 10)
    e = 10 * np.log10(P.sum(-1) + 1e-12)
    emax = e.max(1, keepdims=True)
    act = e >= emax - 30
    Pb = (P * act[..., None]).sum(1) / np.maximum(act.sum(1, keepdims=True), 1)
    bright = np.log2((Pb * FC).sum(-1) / Pb.sum(-1) / F0)
    attack = np.argmax(e >= emax - 3, axis=1) * DT
    length = (X.shape[1] - np.argmax((e >= emax - 40)[:, ::-1], axis=1)) * DT
    noise = np.exp(np.log(Pb + 1e-12).mean(-1)) / Pb.mean(-1)
    d = np.abs(np.diff(X[:, :HOLD_FRAMES], axis=1))
    loud = np.maximum(X[:, 1:HOLD_FRAMES], X[:, :HOLD_FRAMES - 1]) > -50
    move = (d * loud).sum((1, 2)) / np.maximum(loud.sum((1, 2)), 1)
    return dict(brightness=bright, attack=attack, length=length, noise=noise, movement=move)


def mel_db_diff(A, B):
    aud = (A > -80) | (B > -80)
    return (np.abs(A - B) * aud).sum((1, 2)) / np.maximum(aud.sum((1, 2)), 1)


@torch.no_grad()
def encode(model, ds, idx, bs=2048):
    mus = []
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        b = {k: to_t(v[j]) for k, v in ds.t.items()}
        y1 = F.one_hot(to_t(ds.y[j]), N_CLS).float()
        one = torch.ones(len(j))
        mu, lv = model.posterior(b, to_t(ds.a[j]), y1, one, one)
        mus.append(to_np(mu))
    return np.vstack(mus)


@torch.no_grad()
def decode(model, z, y):
    y1 = F.one_hot(to_t(np.asarray(y, np.int64)), N_CLS).float()
    (num, bl, cats), snd = model.decode(to_t(np.asarray(z, np.float32)), y1)
    return to_np(num), to_np(torch.sigmoid(bl)), [to_np(torch.softmax(c, -1)) for c in cats], to_np(snd)


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.float16).tobytes()).decode()


def layer(lin, rows=None):
    W, b = to_np(lin.weight), to_np(lin.bias)
    if rows is not None:
        W, b = W[rows], b[rows]
    return dict(shape=list(W.shape), w=b64(W), b=b64(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="j48w_b05")
    ap.add_argument("--sweep", type=int, default=400)
    args = ap.parse_args()
    from sklearn.linear_model import LogisticRegression, Ridge

    js = JSpec()
    model, cfg = load_model(args.run)
    ds = JointSet("pilot", js)
    sc = SoundCode()
    n = len(ds.y)
    print(f"{n} rows; encoding with {args.run}", flush=True)
    Z = encode(model, ds, np.arange(n))
    with torch.no_grad():                    # KL per dimension over the data
        lv_all = []
        for i in range(0, n, 4096):
            j = np.arange(i, min(i + 4096, n))
            b = {k: to_t(v[j]) for k, v in ds.t.items()}
            one = torch.ones(len(j))
            _, lv = model.posterior(b, to_t(ds.a[j]), F.one_hot(to_t(ds.y[j]), N_CLS).float(), one, one)
            lv_all.append(to_np(lv))
        LV = np.vstack(lv_all)
    kl = (0.5 * (Z ** 2 + np.exp(LV) - 1 - LV)).mean(0)
    active = [int(d) for d in np.argsort(-kl) if kl[d] > ACTIVE_NATS]
    print(f"active dims ({len(active)}): " + ", ".join(f"z{d} {kl[d]:.2f}" for d in active), flush=True)

    # measures of every training row's own sound
    M = {k: [] for k in MEASURES}
    for i in range(0, n, 4000):
        for k, v in measures(pooled_mels(sc, ds.a[i:i + 4000])).items():
            M[k].append(v)
    M = {k: np.concatenate(v) for k, v in M.items()}

    # sound axes: measure ~ z + category
    rng = np.random.default_rng(0)
    test = rng.random(n) < 0.1
    Y1 = np.eye(N_CLS)[ds.y]
    directions = []
    human = np.flatnonzero((ds.meta.source == "human").to_numpy())
    zs_h, ys_h = Z[human], ds.y[human]
    for k, (lo, hi, unit) in MEASURES.items():
        target = np.log10(M[k] + 0.005) if k in ("attack", "noise") else M[k]
        X = np.hstack([Z, Y1])
        r = Ridge(alpha=1.0).fit(X[~test], target[~test])
        r2 = 1 - np.mean((r.predict(X[test]) - target[test]) ** 2) / np.var(target[test])
        u = r.coef_[:Z.shape[1]]
        u = u / np.linalg.norm(u)
        proj = Z @ u
        sd = float(proj.std())
        # effect through the sound decoder: human presets moved -1 / +1 sd along u
        _, _, _, s_lo = decode(model, zs_h - sd * u, ys_h)
        _, _, _, s_hi = decode(model, zs_h + sd * u, ys_h)
        m_lo, m_hi = measures(pooled_mels(sc, s_lo))[k], measures(pooled_mels(sc, s_hi))[k]
        directions.append(dict(key=k, name=f"{lo} ↔ {hi}", low=lo, high=hi, unit=unit, u=[round(float(x), 5) for x in u],
                               mean=float(proj.mean()), sd=sd, r2=round(float(r2), 3),
                               effect=round(float(np.median(m_hi - m_lo)), 3)))
        print(f"  {k}: R2 {r2:.2f}, decoded change over 2 sd {np.median(m_hi - m_lo):+.3f} {unit}", flush=True)

    # latent dimensions: sweep around human presets
    w = pd.read_csv(JDATA / "loss_weights.csv").set_index("feature").weight.to_dict()
    sec_num = np.array([SECTIONS.index(section(f)) for f in js.num])
    sec_bool = np.array([SECTIONS.index(section(f)) for f in js.bool])
    sec_cat = np.array([SECTIONS.index(section(f)) for f in js.cat])
    w_num = np.array([w.get(f, 1.0) for f in js.num])
    w_bool = np.array([w.get(f, 1.0) for f in js.bool])
    w_cat = np.array([w.get(f, 1.0) for f in js.cat])
    pick = rng.choice(human, min(args.sweep, len(human)), replace=False)
    zp, yp = Z[pick], ds.y[pick]
    mask_num = ds.t["m_cont"][pick]
    feats = js.num + js.bool + js.cat
    sec_all = np.concatenate([sec_num, sec_bool, sec_cat])
    dims, change = [], []
    for d in active:
        q10, q90 = np.quantile(Z[:, d], [0.1, 0.9])
        z_lo, z_hi = zp.copy(), zp.copy()
        z_lo[:, d], z_hi[:, d] = q10, q90
        n_lo, b_lo, c_lo, s_lo = decode(model, z_lo, yp)
        n_hi, b_hi, c_hi, s_hi = decode(model, z_hi, yp)
        dn = (np.abs(n_hi - n_lo) * np.maximum(mask_num, 0.25)).mean(0) * w_num
        db = np.abs(b_hi - b_lo).mean(0) * w_bool
        dc = np.array([0.5 * np.abs(h - l).sum(-1).mean() for h, l in zip(c_hi, c_lo)]) * w_cat
        change.append(np.concatenate([dn, db, dc]))
        X_lo, X_hi = pooled_mels(sc, s_lo), pooled_mels(sc, s_hi)
        m_lo, m_hi = measures(X_lo), measures(X_hi)
        eff = {k: float(np.median(m_hi[k] - m_lo[k])) for k in MEASURES}
        dims.append(dict(d=d, kl=round(float(kl[d]), 3), q10=float(q10), q90=float(q90),
                         mean=float(Z[:, d].mean()), sd=float(Z[:, d].std()),
                         impact=round(float(np.median(mel_db_diff(X_lo, X_hi))), 2),
                         effect={k: round(v, 3) for k, v in eff.items()}))
    # what is SPECIFIC to a dimension: each parameter's change relative to its mean change over all dimensions
    # (raw sums only mirror how many parameters a section has); sections = mean relative change of their parameters
    C = np.array(change)
    base = C.mean(0)
    rel = C / (base + 0.5 * np.median(base))
    big = base >= np.median(base)
    for r, row in zip(dims, rel):
        score = np.array([row[sec_all == s].mean() if (sec_all == s).any() else 0 for s in range(len(SECTIONS))])
        r["sections"] = {s: round(float(v), 3) for s, v in zip(SECTIONS, score / score.sum())}
        r["top"] = [feats[i] for i in np.argsort(-(row * big))[:4]]
    dims.sort(key=lambda r: -r["impact"])
    for r in dims:
        top2 = sorted(r["sections"].items(), key=lambda kv: -kv[1])[:2]
        print(f"  z{r['d']}: impact {r['impact']:.1f} dB, " + ", ".join(f"{s} {v:.0%}" for s, v in top2), flush=True)

    # page knobs: decoder rows + activity stand-in
    knobs = json.loads((OUT / "knobs.json").read_text())["knobs"]
    kn = [k["name"] for k in knobs]
    num_rows = [js.num.index(k) for k in kn]
    nb, nn_ = len(js.bool), len(js.num)
    bool_rows = [nn_ + js.bool.index(b) for b in DISPLAY_BOOLS if b in js.bool]
    offs = np.cumsum([0] + js.cat_sizes[:-1])
    cat_rows, cat_meta = [], []
    for c in DISPLAY_CATS:
        j = js.cat.index(c)
        opts = next(f for f in js.spec.feats if f["name"] == c)["options"]
        cat_rows += list(range(nn_ + nb + offs[j], nn_ + nb + offs[j] + js.cat_sizes[j]))
        cat_meta.append(dict(name=c, options=[OSC_TYPES.get(int(o), str(o)) for o in opts]))
    m_page = ds.t["m_cont"][:, num_rows]
    Xa = np.hstack([Z, Y1])
    act_w, act_b, acc = [], [], []
    for i in range(len(kn)):
        y = m_page[:, i]
        if y.min() == y.max():
            act_w.append(np.zeros(Xa.shape[1])); act_b.append(8.0 if y[0] else -8.0); acc.append(1.0); continue
        lr = LogisticRegression(C=1.0, max_iter=300).fit(Xa[~test], y[~test])
        act_w.append(lr.coef_[0]); act_b.append(lr.intercept_[0]); acc.append(lr.score(Xa[test], y[test]))
    print(f"activity stand-in: median accuracy {np.median(acc):.3f}, min {np.min(acc):.3f}", flush=True)

    names = pd.read_parquet(PITCHER_ROOT / "data" / "splits.parquet").set_index("preset_id").name.to_dict()
    base = ds.meta.base.to_numpy()[human]
    dp, dpa = model.dec_p, model.dec_a
    page = dict(
        run=args.run, z_dim=int(cfg["z"]), n_cls=N_CLS, labels=list(LABELS) + ["Unknown"], sections=SECTIONS,
        n_rows=int(n), active=active, kl=[round(float(x), 3) for x in kl],
        dims=dims, directions=directions,
        measures={k: dict(low=v[0], high=v[1], unit=v[2]) for k, v in MEASURES.items()},
        knob_names=kn, bools=[b for b in DISPLAY_BOOLS if b in js.bool], cats=cat_meta,
        activity=dict(w=b64(np.array(act_w)), b=b64(np.array(act_b)), acc=round(float(np.median(acc)), 3)),
        dec_p=[layer(dp[0]), layer(dp[3]), layer(dp[6], num_rows + bool_rows + cat_rows)],
        dec_a=[layer(dpa[0]), layer(dpa[3]), layer(dpa[6])],
        pca=dict(components=b64(sc.components), mean=b64(sc.mean), scale=sc.scale, frames=int(sc.components.shape[1] // N_MELS)),
        band_hz=[round(float(f), 1) for f in FC], dt=DT, f0=F0, hold_frames=HOLD_FRAMES,
        presets=dict(z=b64(zs_h), label=[int(v) for v in ys_h], name=[names.get(int(b), "") for b in base]),
        mean_z={LABELS[c]: [round(float(x), 4) for x in zs_h[ys_h == c].mean(0)] for c in range(len(LABELS)) if (ys_h == c).any()},
    )
    (OUT / "latent.json").write_text(json.dumps(page, separators=(",", ":")))
    print(f"written {OUT / 'latent.json'} ({(OUT / 'latent.json').stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
