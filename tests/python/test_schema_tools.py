#!/usr/bin/env python3
"""Checks the Python schema tools against the generated C++ schema.

usage: test_schema_tools.py GENERATED_HEADER [SURGE_TEMPLATES_DIR]
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import surge_preset_to_vector as importer  # noqa: E402
import vae_schema  # noqa: E402


def main() -> int:
    header = pathlib.Path(sys.argv[1]).read_text()
    schema = vae_schema.load()
    failures = []

    def check(ok: bool, what: str) -> None:
        if not ok:
            failures.append(what)

    for name, value in (("kControlCount", len(schema.controls)), ("kFeatureCount", schema.feature_count)):
        cpp = int(re.search(rf"{name} = (\d+);", header).group(1))
        check(cpp == value, f"{name}: C++ {cpp} != Python {value}")

    defaults = schema.defaults()
    check(schema.decode_features(schema.encode_features(defaults)) == defaults, "feature round trip")

    if len(sys.argv) > 2:
        init_saw = pathlib.Path(sys.argv[2]) / "Init Saw.fxp"
        values, lossy = importer.convert(importer.read_fxp(init_saw), schema)
        diff = [c.id for c, a, b in zip(schema.controls, values, defaults) if abs(a - b) > 1e-4]
        check(not diff, f"Init Saw differs from the schema defaults: {diff}")

    for f in failures:
        print("FAIL:", f)
    print(f"schema tools: {'ok' if not failures else f'{len(failures)} failure(s)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
