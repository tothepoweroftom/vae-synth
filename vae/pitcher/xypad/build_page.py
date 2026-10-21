"""Build the XY-pad concept page: page.template.html + data/xypad/knobs.json -> runs/xypad/xypad.html.

Only what the page needs goes in: knob ranges and signs, the ranked groups, every preset's knob values
(3 decimals) and the small category forest. Run pitcher.xypad.analyse first.
Run:    .venv/bin/python -m pitcher.xypad.build_page
"""

import json
from pathlib import Path

from pitcher.core.surge_io import PITCHER_ROOT

SRC = PITCHER_ROOT / "data" / "xypad" / "knobs.json"
TEMPLATE = Path(__file__).with_name("page.template.html")
OUT = PITCHER_ROOT / "runs" / "xypad" / "xypad.html"


def main():
    d = json.loads(SRC.read_text())
    page = dict(labels=d["labels"], n_presets=d["n_presets"], rf_acc=d["rf_balanced_accuracy"],
                page_acc=d["page_forest_balanced_accuracy"], n_clusters=len(d["correlation_clusters"]),
                knobs=[{k: v for k, v in kn.items() if k in ("name", "ctrltype", "natural_range", "p5", "p95", "default", "active")} for kn in d["knobs"]],
                groups=[{k: v for k, v in g.items() if k in ("name", "section", "desc", "knobs", "signs", "importance", "se", "mean_abs_rho", "active")}
                        for g in d["groups"]],
                presets=d["presets"], forest=d["forest"])
    html = TEMPLATE.read_text().replace("/*DATA*/null", json.dumps(page, separators=(",", ":")))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)
    print(f"{OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
