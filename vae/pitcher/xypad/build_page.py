"""Build the XY-pad concept page: page.template.html + data/xypad/knobs.json -> runs/xypad/xypad.html.

Only what the page needs goes in: knob ranges and signs, the ranked groups, every preset's knob values
(3 decimals) and the small category forest. Run pitcher.xypad.analyse first.
v2 (--v2): page_v2.template.html + also data/xypad/latent.json (latent pad; run pitcher.xypad.latent first)
-> runs/xypad/xypad_v2.html.
Run:    .venv/bin/python -m pitcher.xypad.build_page [--v2]
"""

import argparse
import json
from pathlib import Path

from pitcher.core.surge_io import PITCHER_ROOT

SRC = PITCHER_ROOT / "data" / "xypad" / "knobs.json"
LATENT = PITCHER_ROOT / "data" / "xypad" / "latent.json"
HERE = Path(__file__).parent
OUT = PITCHER_ROOT / "runs" / "xypad"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", action="store_true", help="latent pad + knob pad")
    args = ap.parse_args()
    d = json.loads(SRC.read_text())
    page = dict(labels=d["labels"], n_presets=d["n_presets"], rf_acc=d["rf_balanced_accuracy"],
                page_acc=d["page_forest_balanced_accuracy"], n_clusters=len(d["correlation_clusters"]),
                knobs=[{k: v for k, v in kn.items() if k in ("name", "ctrltype", "natural_range", "p5", "p95", "default", "active")} for kn in d["knobs"]],
                groups=[{k: v for k, v in g.items() if k in ("name", "section", "desc", "knobs", "signs", "importance", "se", "mean_abs_rho", "active")}
                        for g in d["groups"]],
                presets=d["presets"], forest=d["forest"])
    template = HERE / ("page_v2.template.html" if args.v2 else "page.template.html")
    html = template.read_text().replace("/*DATA*/null", json.dumps(page, separators=(",", ":")))
    if args.v2:
        html = html.replace("/*LATENT*/null", LATENT.read_text())
    out = OUT / ("xypad_v2.html" if args.v2 else "xypad.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"{out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
