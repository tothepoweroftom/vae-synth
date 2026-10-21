"""Shared pieces of the synthetic Surge dataset (TEXT2PRESET_PLAN.md §3): human base presets as
template-rebuild rows, parameter groups, donors, and a deterministic render worker.

A "row" is a full codes row ({feature: code or NaN}) as codec.decode() takes it. The base row of a
preset is its template rebuild with all kept features + the routing matrix (as in Step 3.4 with
K = all), i.e. exactly what the VAE learns from.
"""

import json
import math
import time
from multiprocessing import Pool, TimeoutError

import numpy as np
import pandas as pd

from pitcher.core.audio_metrics import mel_distance_db, rms_db
from pitcher.core.codec import decode
from pitcher.core.surge_io import PITCHER_ROOT, SurgeIO
from pitcher.step2_explore.reference import DEFAULT_TEMPLATE
from pitcher.step4_vae.data import LABELS

DATA = PITCHER_ROOT / "data"
OUT = DATA / "synth"
RUNS = PITCHER_ROOT / "runs" / "synth"
CHARACTER_DB = 6.7  # "same character" threshold from the listening test (Round 14)


class Base:
    """Codes, groups and the cleaned train presets (never val / test: they stay clean for evaluation)."""

    def __init__(self):
        self.schema = json.loads((DATA / "schema.json").read_text())
        self.feats = {f["name"]: f for f in self.schema["features"]}
        self.names = list(self.feats)
        pruning = json.loads((DATA / "pruning.json").read_text())
        self.kept = pruning["kept"]
        self.mode_frozen = [k for k, v in pruning["frozen"].items() if v["how"] == "mode"]
        self.codes = pd.read_parquet(DATA / "codes.parquet")
        self.pruned = pd.read_parquet(DATA / "codes_pruned.parquet")
        sp = pd.read_parquet(DATA / "splits.parquet").set_index("preset_id")
        excl = set(pd.read_csv(DATA / "vae_excluded.csv").preset_id)
        self.label = sp.label
        self.train = [p for p in sp.index[sp.split == "train"] if p not in excl]
        self.K = self.codes.loc[self.train, self.kept]  # kept features of the train presets

    def row(self, pid):
        r = {n: math.nan for n in self.names}
        for n in self.mode_frozen:
            r[n] = float(self.pruned.at[pid, n])
        for n in self.kept:
            r[n] = float(self.codes.at[pid, n])
        return r

    def groups(self):
        """[(group name, [features])]: the 452 groups of the core ranking (a feature + its flags; an
        effect's present / placement / position) + every routing feature of the matrix on its own."""
        rk = pd.read_csv(DATA / "core_ranking.csv")
        out = [(g, f.split(";")) for g, f in zip(rk.group, rk.features)]
        out += [(n, [n]) for n in self.kept if n.startswith("route/")]
        return out

    def kind(self, feats):
        """knob (one cont / ord feature), route (a routing depth), or switch (cat / bool / FX block)."""
        if feats[0].startswith("route/"):
            return "route"
        main = [f for f in feats if self.feats[f]["kind"] in ("cont", "ord")]
        return "knob" if main and len(feats) - len(main) <= 2 and not any(
            self.feats[f]["kind"] == "cat" for f in feats) else "switch"

    def stratified(self, n_per_label, seed=0):
        rng = np.random.default_rng(seed)
        out = []
        for lab in LABELS:
            ids = [p for p in self.train if self.label[p] == lab]
            out += list(rng.choice(ids, n_per_label, replace=False))
        return [int(p) for p in out]


def block_prefix(feature):
    """Features that only exist with a type (an oscillator's knobs, an effect's knobs) share a prefix;
    when a group switch brings in a new type, the donor's knobs for it come along."""
    parts = feature.split("/")
    if parts[0] == "scene" and parts[1] == "osc":
        return "/".join(parts[:3]) + "/"
    if parts[0] == "fx":
        return "/".join(parts[:2]) + "/"
    return None


# --- render worker ---------------------------------------------------------------------------------
_io = _schema = None


def init_worker():
    global _io, _schema
    _io = SurgeIO()
    _schema = json.loads((DATA / "schema.json").read_text())


def render(row, note, seed, **kw):
    audio, info = _io.render_plan(decode(row, _schema), DEFAULT_TEMPLATE, seed=seed, note=note, **kw)
    return audio, info


def _loudest_window_db(x, win=2400):
    m = np.asarray(x, dtype=np.float64)
    m = m.mean(axis=0) if m.ndim == 2 else m
    n = len(m) // win
    if n == 0:
        return rms_db(m)
    return float(20 * np.log10(max(np.sqrt((m[:n * win].reshape(n, win) ** 2).mean(1)).max(), 1e-12)))


def compare(a, b):
    """(mel distance dB, level difference dB of b vs a, level-matched mel distance dB).
    Level-matched: b scaled so its loudest 50 ms window equals a's (the render rule of the
    post-processing), at most +-40 dB, so a change that only makes the sound louder or quieter
    scores ~0. The synthetic set's fingerprints and descriptors are computed on level-matched audio."""
    gain = float(np.clip(_loudest_window_db(a) - _loudest_window_db(b), -40, 40))
    return mel_distance_db(a, b), rms_db(b) - rms_db(a), mel_distance_db(a, np.asarray(b) * 10 ** (gain / 20))


def run_pool(fn, tasks, workers, label, timeout=300, init=init_worker):
    """Yield fn(task) results (unordered). A crashed worker loses its task: after `timeout` s without
    any result the run stops with what it has, and says so."""
    t0, n = time.time(), 0
    with Pool(workers, initializer=init) as pool:
        it = pool.imap_unordered(fn, tasks, chunksize=1)
        while True:
            try:
                res = it.next(timeout=timeout)
            except StopIteration:
                break
            except TimeoutError:
                print(f"  no result for {timeout} s - a worker probably crashed; {len(tasks) - n} tasks lost", flush=True)
                break
            n += 1
            if n % max(1, len(tasks) // 20) == 0 or n == len(tasks):
                el = time.time() - t0
                print(f"  {label}: {n}/{len(tasks)} tasks, {el:.0f}s, ETA {el / n * (len(tasks) - n):.0f}s", flush=True)
            yield res
        pool.terminate()
