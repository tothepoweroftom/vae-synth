#!/usr/bin/env python3
"""Smoke test for vae_compare: three Surge init templates through the importer
and the comparison tool.

usage: test_compare.py VAE_COMPARE_EXE SURGE_TEMPLATES_DIR
"""

import csv
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]


def main() -> int:
    exe, templates = sys.argv[1], pathlib.Path(sys.argv[2])
    presets = [templates / f"Init {n}.fxp" for n in ("Saw", "Square", "Sine")]
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        jsonl = pathlib.Path(tmp) / "vectors.jsonl"
        out = pathlib.Path(tmp) / "compare.csv"
        subprocess.run([sys.executable, str(ROOT / "tools" / "surge_preset_to_vector.py"), *map(str, presets),
                        "-o", str(jsonl)], check=True)
        subprocess.run([exe, str(jsonl), "-o", str(out), "--jobs", "2"], check=True)
        rows = list(csv.DictReader(open(out, newline="")))

    if len(rows) != 3:
        failures.append(f"expected 3 rows, got {len(rows)}")
    for r in rows:
        if r["orig_ok"] != "1" or "1" in (r["orig_silent"], r["svec_silent"], r["lite_silent"]):
            failures.append(f"{r['file']}: a render failed or is silent")
    saw = next((r for r in rows if r["file"].endswith("Init Saw.fxp")), None)
    # Init Saw is the schema default: the vector on Surge must sound like the original
    # (measured: 0.4 dB spectrum-over-time difference, envelope correlation 0.999).
    if saw is None or float(saw["os_frame_mel_db"]) > 2.0 or float(saw["os_env_corr"]) < 0.98:
        failures.append(f"Init Saw: vector does not match the original ({saw and saw['os_frame_mel_db']})")

    for f in failures:
        print("FAIL:", f)
    print(f"compare smoke test: {'ok' if not failures else f'{len(failures)} failure(s)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
