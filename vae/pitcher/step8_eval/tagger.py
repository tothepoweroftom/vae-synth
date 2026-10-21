"""Step 8.0 the tagger's "judge": which category does a sound belong to?

Fits a logistic regression on CLAP fingerprints of the train renders (folder labels), chooses the
regularisation C by 5-fold cross-validation on train, and reports on val (the test split stays
untouched until the final Step 8 run). The smallest CLAP model is kept unless a larger one beats it
by > 1 pp in cross-validation. Compared with:
  majority    always the most common train label
  mel stats   the same classifier on plain spectrum statistics (mean + std of the 64-band log-mel,
              and of its frame-to-frame change: 256 numbers), to see whether CLAP's "ear" helps
  zero-shot   CLAP text matching without training: the label whose prompt fingerprint is closest
Accuracy, balanced accuracy (mean recall per label, so small labels count equally), top-2.

Writes  data/tagger.pkl (best CLAP model + classifier), data/tagger_report.json,
        data/tagger_val_predictions.csv, runs/step8/tagger_confusion.png
Run:    .venv/bin/python -m pitcher.step8_eval.tagger
"""

import json
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pitcher.core.audio_metrics import mel_db
from pitcher.core.surge_io import PITCHER_ROOT
from pitcher.step4_vae.data import LABELS
from pitcher.step8_eval.clap import MODELS, Clap, cache_file, read_render

DATA, OUT = PITCHER_ROOT / "data", PITCHER_ROOT / "runs" / "step8"
PROMPTS = {
    "Bass": "a deep synth bass sound",
    "Lead": "a synth lead melody sound",
    "Pad": "a soft sustained synth pad",
    "Keys": "an electric piano or keyboard sound",
    "Pluck": "a short plucked synth sound",
    "Synth": "a synthesizer sound",
    "Atmosphere": "an ambient atmospheric soundscape",
    "Orchestral": "an orchestral instrument sound, strings or brass",
}


def mel_stats(audio):
    m = mel_db(audio)
    m = m - m.max()  # level-independent shape
    d = np.diff(m, axis=0)
    return np.concatenate([m.mean(0), m.std(0), np.abs(d).mean(0), d.std(0)])


def classifier():
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced"))
    grid = {"logisticregression__C": [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]}
    return GridSearchCV(clf, grid, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                        scoring="balanced_accuracy", n_jobs=4)


def scores(y, proba, classes):
    pred = classes[proba.argmax(1)]
    top2 = classes[np.argsort(-proba, 1)[:, :2]]
    return dict(accuracy=float((pred == y).mean()), balanced=float(balanced_accuracy_score(y, pred)),
                top2=float((top2 == y[:, None]).any(1).mean()))


def per_label_recall(y, pred):
    return {l: float((pred[y == l] == l).mean()) for l in LABELS}


def plot_confusion(y, pred, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = confusion_matrix(y, pred, labels=LABELS, normalize="true")
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, f"{cm[i, j]:.0%}" if cm[i, j] >= 0.005 else "", ha="center", va="center",
                    fontsize=8, color="white" if cm[i, j] > 0.5 else "#333")
    n = pd.Series(y).value_counts()
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=40, ha="right")
    ax.set_yticks(range(len(LABELS)), [f"{l} ({n.get(l, 0)})" for l in LABELS])
    ax.set_xlabel("tagged as"); ax.set_ylabel("folder label (val presets)"); ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sp = pd.read_parquet(DATA / "splits.parquet")
    tr, va = sp[sp.split == "train"], sp[sp.split == "val"]
    ytr, yva = tr.label.to_numpy(), va.label.to_numpy()
    report = {"n_train": len(tr), "n_val": len(va), "methods": {}}

    major = pd.Series(ytr).mode()[0]
    report["methods"]["majority"] = dict(accuracy=float((yva == major).mean()), balanced=1 / len(LABELS), label=major)

    print("mel stats ...", flush=True)
    feats = {p: mel_stats(read_render(p)) for p in sp.preset_id}
    g = classifier().fit(np.stack([feats[p] for p in tr.preset_id]), ytr)
    proba = g.predict_proba(np.stack([feats[p] for p in va.preset_id]))
    report["methods"]["mel_stats"] = dict(scores(yva, proba, g.classes_), C=g.best_params_["logisticregression__C"],
                                          cv_balanced=float(g.best_score_))

    best = None
    for model in MODELS:
        if not cache_file(model).exists():
            print("missing", cache_file(model).name); continue
        E = pd.read_parquet(cache_file(model)).set_index("preset_id")
        Xtr, Xva = E.loc[tr.preset_id].to_numpy(), E.loc[va.preset_id].to_numpy()
        g = classifier().fit(Xtr, ytr)
        proba = g.predict_proba(Xva)
        r = dict(scores(yva, proba, g.classes_), C=g.best_params_["logisticregression__C"], cv_balanced=float(g.best_score_))
        T = Clap(model).text([PROMPTS[l] for l in LABELS])
        zs = Xva @ T.T
        r_zs = scores(yva, zs, np.array(LABELS))
        name = model.split("/")[-1]
        report["methods"][f"clap:{name}"] = r
        report["methods"][f"zero_shot:{name}"] = r_zs
        print(f"{name}: trained {r['balanced']:.1%} balanced (cv {r['cv_balanced']:.1%}), zero-shot {r_zs['balanced']:.1%}", flush=True)
        # selection on the train cross-validation score, so val stays an honest estimate; MODELS is
        # ordered small -> large, and a larger model must win by more than 1 pp (CV noise; Round 19)
        if best is None or r["cv_balanced"] > best[1]["cv_balanced"] + 0.01:
            best = (model, r, g.best_estimator_, proba, g.classes_)

    model, r, clf, proba, classes = best
    pred = classes[proba.argmax(1)]
    report["chosen"] = model
    report["val_recall_per_label"] = per_label_recall(yva, pred)
    pickle.dump({"model": model, "classifier": clf, "labels": list(classes)}, open(DATA / "tagger.pkl", "wb"))
    (DATA / "tagger_report.json").write_text(json.dumps(report, indent=1))
    out = va[["preset_id", "name", "label"]].assign(tagged=pred, p_own=proba[np.arange(len(va)), [list(classes).index(l) for l in yva]])
    out = out.join(pd.DataFrame(proba, columns=[f"p_{c}" for c in classes], index=out.index))
    out.to_csv(DATA / "tagger_val_predictions.csv", index=False)
    plot_confusion(yva, pred, f"Tagger on val ({model.split('/')[-1]}): {r['balanced']:.0%} balanced accuracy",
                   OUT / "tagger_confusion.png")
    print(json.dumps(report, indent=1))


class Tagger:
    """Load the fitted tagger and tag audio: tagger.proba(clips) -> DataFrame of label probabilities."""

    def __init__(self):
        d = pickle.load(open(DATA / "tagger.pkl", "rb"))
        self.clap_name, self.clf, self.labels = d["model"], d["classifier"], d["labels"]
        self.clap = Clap(d["model"])

    def proba(self, clips):
        return self.proba_from(self.clap.audio(clips))

    def proba_from(self, emb):
        """label probabilities from CLAP fingerprints (e.g. cached ones)."""
        return pd.DataFrame(self.clf.predict_proba(emb), columns=self.labels)


if __name__ == "__main__":
    main()
