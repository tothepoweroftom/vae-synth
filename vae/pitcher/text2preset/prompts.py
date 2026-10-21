"""Text -> preset dataset, step 2: prompts for Stable Audio 3 (TEXT2PRESET_PLAN.md, section 2).

Draws tag combinations from pitcher/text2preset/vocabulary_v2.yaml (or --vocab) and writes, per prompt:
  tags         {slot: value} + category / noun id (the structured truth behind the text)
  sa3_prompt   description + fixed technical suffix (single note, C3, dry, ...), sent to Stable Audio
  user_texts   6 paraphrases without the suffix, 2 per style: tags ("dark reese bass, gritty"),
               sentence ("I want a dark reese bass with a slow attack"), slang ("dark reese w/ slow attack")
  user_texts_unseen  (test split only) texts that use the held-out words
Kinds: main (one of the 8 categories, weighted), generic (character only, no category),
out_of_range (test only). Unique per tag combination; split by a hash of the combination
(80/10/10), so paraphrases and seeds of one prompt never cross splits.

Writes  data/text2preset/prompts_<name>.jsonl (full) + .csv (one row per prompt, for review)
Run:    .venv/bin/python -m pitcher.text2preset.prompts --name pilot --n 500 [--vocab pitcher/text2preset/vocabulary_v2.yaml]
"""

import argparse
import hashlib
import re
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pitcher.core.surge_io import PITCHER_ROOT

VOCAB = Path(__file__).with_name("vocabulary_v2.yaml")  # v1: vocabulary.yaml
OUT = PITCHER_ROOT / "data" / "text2preset"
SHARES = {"main": 0.83, "generic": 0.12, "out_of_range": 0.05}
N_DESC = ([0, 1, 2, 3], [0.10, 0.30, 0.35, 0.25])  # number of descriptor slots per prompt
SPARSE = {"style": 0.15, "mood": 0.15}  # fuzzy slots: added on top of the descriptors with this probability
PHRASE_SLOTS = {"attack", "length", "movement"}  # these are often said after the noun ("with a slow attack")
ADJ_ORDER = ["mood", "style", "grit", "brightness", "body", "stereo", "colour", "movement", "attack", "length"]


def order(slot):
    return ADJ_ORDER.index(slot) if slot in ADJ_ORDER else len(ADJ_ORDER)  # new slots go last


def article(word):
    w = word.lower()
    if w.startswith(("uni", "use", "one", "eu")):
        return "a"
    return "an" if w[:1] in "aeiou" or word.startswith(("FM", "80s", "8-bit")) else "a"


class Grammar:
    def __init__(self, vocab_path=VOCAB, seed=0):
        self.v = yaml.safe_load(Path(vocab_path).read_text())
        self.rng = np.random.default_rng(seed)
        self.held = set(self.v["held_out"])

    def pick(self, items, p=None):
        return items[self.rng.choice(len(items), p=p)]

    def forms(self, slot, value, kind, unseen=False):
        fs = self.v["slots"][slot][value].get(kind) or []
        has_held = lambda f: any(w in self.held for w in f.split())
        return [f for f in fs if has_held(f) == unseen]

    def form(self, slot, value, kind, canonical_p=0.0):
        fs = self.forms(slot, value, kind)
        if not fs:
            return None
        return fs[0] if self.rng.random() < canonical_p else self.pick(fs)

    # --- tags ---------------------------------------------------------------------------------------
    def quotas(self, n):
        """prompts per category, proportional to the weights (largest remainder)."""
        cats = self.v["categories"]
        w = np.array([cats[c]["weight"] for c in cats], float)
        raw = n * w / w.sum()
        q = np.floor(raw).astype(int)
        q[np.argsort(raw - q)[::-1][:n - q.sum()]] += 1
        return dict(zip(cats, q.tolist()))

    def draw_tags(self, kind, cat=None):
        v = self.v
        if kind == "main":
            noun = self.pick(list(v["categories"][cat]["nouns"]))
            prefer, avoid = set(v["categories"][cat]["prefer"]), set(v["categories"][cat]["avoid"])
            entry = v["categories"][cat]["nouns"][noun]
            if isinstance(entry, dict):  # noun with its own avoid list: {forms: [...], avoid: [slot:value or slot]}
                for r in entry.get("avoid") or []:
                    avoid |= {f"{r}:{x}" for x in v["slots"][r]} if ":" not in r else {r}
        else:
            cat, prefer, avoid = None, set(), set()
            noun = self.pick(v["generic_nouns"] if kind == "generic" else v["out_of_range_nouns"])
        k = int(self.rng.choice(N_DESC[0], p=N_DESC[1]))
        if kind == "generic":
            k = max(k, 2)
        if kind == "out_of_range":
            k = min(k, 2)
        slots = [s for s in v["slots"] if s not in SPARSE]
        chosen = [str(s) for s in self.rng.choice(slots, size=k, replace=False)] if k else []
        chosen += [s for s, prob in SPARSE.items() if s in v["slots"] and self.rng.random() < prob]
        tags = {}
        for s in chosen:
            vals = list(v["slots"][s])
            w = np.array([0.0 if f"{s}:{x}" in avoid else 3.0 if f"{s}:{x}" in prefer else 1.0 for x in vals])
            if w.sum() == 0:
                continue
            tags[s] = self.pick(vals, w / w.sum())
        return dict(kind=kind, category=cat, noun=noun, tags=tags)

    @staticmethod
    def key(p):
        return p["kind"] + "|" + p["noun"] + "|" + ",".join(f"{s}:{x}" for s, x in sorted(p["tags"].items()))

    # --- surface texts ------------------------------------------------------------------------------
    def noun_form(self, p, canonical_p=0.0):
        if p["kind"] != "main":
            return p["noun"]
        fs = self.v["categories"][p["category"]]["nouns"][p["noun"]]
        fs = fs["forms"] if isinstance(fs, dict) else fs
        return fs[0] if self.rng.random() < canonical_p else self.pick(fs)

    def parts(self, p, canonical_p=0.0, p_phrase=0.5, unseen=False):
        """(adjectives in order, 'with' phrases) for one realisation of the tags."""
        adjs, withs = [], []
        for s in sorted(p["tags"], key=order):
            val = p["tags"][s]
            if unseen and self.forms(s, val, "adj", unseen=True):
                adjs.append(self.pick(self.forms(s, val, "adj", unseen=True)))
                continue
            w = self.form(s, val, "with", canonical_p)
            a = self.form(s, val, "adj", canonical_p)
            if w and (s in PHRASE_SLOTS and self.rng.random() < p_phrase or a is None):
                withs.append(w)
            elif a:
                adjs.append(a)
        return adjs, withs

    @staticmethod
    def join_phrases(withs):
        """'with a slow attack' + 'with a long release' -> 'with a slow attack and a long release';
        'that ...' / 'like ...' phrases follow after a comma ('..., which ...')."""
        w = [x for x in withs if x.startswith(("with ", "w/ ", "without"))]
        other = [x for x in withs if x not in w]
        out = ""
        if w:
            out = w[0] + "".join(" and " + x.removeprefix("with ").removeprefix("w/ ") for x in w[1:])
        for x in other:
            out = f"{out}, {'which' + x[4:] if x.startswith('that ') else x}" if out else x
        return out

    def describe(self, adjs, noun, withs):
        desc = " ".join(adjs + [noun])
        return f"{desc} {self.join_phrases(withs)}" if withs else desc

    def sa3_prompt(self, p):
        adjs, withs = self.parts(p, canonical_p=0.7)
        d = self.describe(adjs, self.noun_form(p, canonical_p=0.7), withs)
        return f"{d[0].upper() + d[1:]}, {self.v['suffix']}"

    def text_tags(self, p, unseen=False):
        adjs, withs = self.parts(p, p_phrase=0.0, unseen=unseen)
        noun = self.noun_form(p)
        if adjs and self.rng.random() < 0.5:
            return ", ".join([noun] + adjs)
        return ", ".join([" ".join(adjs[:2] + [noun])] + adjs[2:] + [w.replace("with ", "") for w in withs])

    def text_sentence(self, p, unseen=False):
        adjs, withs = self.parts(p, unseen=unseen)
        noun = self.noun_form(p)
        tmpls = [t for t in self.v["sentence_templates"] if ("{with}" in t) == bool(withs)] or ["a {desc}"]
        t = self.pick(tmpls)
        desc = " ".join(adjs + [noun])
        s = t.replace("{desc}", desc).replace("{with}", self.join_phrases(withs))
        return re.sub(r"(^|\s)a (?=" + re.escape(desc) + ")", lambda m: f"{m[1]}{article(desc)} ", s, count=1)

    def text_slang(self, p, unseen=False):
        adjs, withs = self.parts(p, p_phrase=0.3, unseen=unseen)
        noun = self.noun_form(p).replace(" sound", "")
        return self.describe(adjs, noun, [w.replace("with a ", "w/ ").replace("with ", "w/ ") for w in withs])

    def user_texts(self, p, n_per_style=2, unseen=False):
        out = []
        for f in (self.text_tags, self.text_sentence, self.text_slang):
            tries = 0
            got = set()
            while len(got) < n_per_style and tries < 10:
                got.add(f(p, unseen=unseen)); tries += 1
            out += sorted(got)
        return out


def split_of(key, kind):
    if kind == "out_of_range":
        return "test"
    h = int(hashlib.md5(key.encode()).hexdigest(), 16) % 10
    return "test" if h == 0 else "val" if h == 1 else "train"


def generate(n, seed=0, vocab=VOCAB):
    g = Grammar(vocab, seed)
    prompts, seen = [], set()
    n_main = round(n * SHARES["main"])
    jobs = [("main", c, q) for c, q in g.quotas(n_main).items()] + \
           [(k, None, round(n * s)) for k, s in SHARES.items() if k != "main"]
    for kind, cat, target in jobs:
        made, tries = 0, 0
        while made < target and tries < 1000 * target:
            tries += 1
            p = g.draw_tags(kind, cat)
            k = g.key(p)
            if k in seen:
                continue
            seen.add(k)
            p["key"], p["split"] = k, split_of(k, kind)
            p["sa3_prompt"] = g.sa3_prompt(p)
            p["user_texts"] = g.user_texts(p)
            p["user_texts_unseen"] = [t for t in g.user_texts(p, n_per_style=1, unseen=True)
                                      if any(h in t for h in g.held)] if p["split"] == "test" else []
            prompts.append(p)
            made += 1
    for i, p in enumerate(prompts):
        p["prompt_id"] = i
    return prompts, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="pilot")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vocab", default=str(VOCAB), help="vocabulary file (default: vocabulary_v2.yaml)")
    a = ap.parse_args()
    prompts, g = generate(a.n, a.seed, a.vocab)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"prompts_{a.name}.jsonl", "w") as f:
        for p in prompts:
            f.write(json.dumps({"vocab": Path(a.vocab).name} | {k: p[k] for k in ["prompt_id", "kind", "category", "noun", "tags", "split", "key",
                                                   "sa3_prompt", "user_texts", "user_texts_unseen"]}) + "\n")
    df = pd.DataFrame([dict(prompt_id=p["prompt_id"], kind=p["kind"], category=p["category"] or "", split=p["split"],
                            tags=", ".join(f"{s}:{x}" for s, x in p["tags"].items()), sa3_prompt=p["sa3_prompt"],
                            user_texts=" | ".join(p["user_texts"]), user_texts_unseen=" | ".join(p["user_texts_unseen"]))
                       for p in prompts])
    df.to_csv(OUT / f"prompts_{a.name}.csv", index=False)

    print(f"{len(prompts)} prompts -> {(OUT / f'prompts_{a.name}.jsonl').relative_to(PITCHER_ROOT)} (+ .csv)")
    print("kind:", dict(Counter(p["kind"] for p in prompts)), " split:", dict(Counter(p["split"] for p in prompts)))
    print("category:", dict(Counter(p["category"] for p in prompts if p["category"])))
    print("descriptors per prompt:", dict(sorted(Counter(len(p["tags"]) for p in prompts).items())))
    print("slot usage:", {str(k): v for k, v in Counter(s for p in prompts for s in p["tags"]).most_common()})
    words = Counter(w for p in prompts for t in p["user_texts"] for w in t.replace(",", " ").split())
    print(f"user texts: {sum(len(p['user_texts']) for p in prompts)}, distinct words {len(words)}; "
          f"held-out words in user_texts: {sum(words[h] for h in g.held)} (must be 0); "
          f"unseen-word texts (test): {sum(len(p['user_texts_unseen']) for p in prompts)}")


if __name__ == "__main__":
    main()
