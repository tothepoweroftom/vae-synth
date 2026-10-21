"""Audio comparison metrics (used by the round-trip test, Step 3.4 and Step 8)."""

import numpy as np

SR = 48000
N_FFT, HOP, N_MELS, FMIN, FMAX = 2048, 512, 64, 30.0, 20000.0
FLOOR_DB = 80.0  # ignore everything more than 80 dB below the louder signal's peak


def _mono(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:
        return x.mean(axis=0) if x.shape[0] <= 2 else x.mean(axis=1)
    return x


def _mel_fb(n_fft=N_FFT, sr=SR, n_mels=N_MELS, fmin=FMIN, fmax=FMAX):
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    edges = hz(np.linspace(mel(fmin), mel(fmax), n_mels + 2))
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    fb = np.zeros((n_mels, len(freqs)))
    for i in range(n_mels):
        lo, mid, hi = edges[i], edges[i + 1], edges[i + 2]
        fb[i] = np.clip(np.minimum((freqs - lo) / (mid - lo), (hi - freqs) / (hi - mid)), 0, None)
    return fb


_FB = _mel_fb()


def mel_db(x):
    """(frames, 64) mel power spectrogram in dB."""
    x = _mono(x)
    if len(x) < N_FFT:
        x = np.pad(x, (0, N_FFT - len(x)))
    frames = np.lib.stride_tricks.sliding_window_view(x, N_FFT)[::HOP] * np.hanning(N_FFT)
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    return 10 * np.log10(power @ _FB.T + 1e-20)


def mel_distance_db(a, b):
    """Mean absolute difference (dB) between mel spectrograms, over the cells where at least one
    signal is within 80 dB of the louder peak. Shorter signal padded with silence, so a missing or
    extra tail counts. 0 = identical; ~1 dB = barely different; >3 dB = clearly different."""
    A, B = mel_db(a), mel_db(b)
    n = max(len(A), len(B))
    pad = lambda M: np.vstack([M, np.full((n - len(M), M.shape[1]), -200.0)])
    A, B = pad(A), pad(B)
    floor = max(A.max(), B.max()) - FLOOR_DB
    A, B = np.maximum(A, floor), np.maximum(B, floor)
    audible = (A > floor) | (B > floor)
    return float(np.abs(A - B)[audible].mean()) if audible.any() else 0.0


def rms_db(x):
    return float(20 * np.log10(max(np.sqrt(np.mean(_mono(x) ** 2)), 1e-12)))
