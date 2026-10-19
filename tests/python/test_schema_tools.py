#!/usr/bin/env python3
"""Checks the Python schema tools against the generated C++ schema.

usage: test_schema_tools.py GENERATED_HEADER [SURGE_TEMPLATES_DIR]
"""

import pathlib
import re
import sys
import tempfile

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
        templates = pathlib.Path(sys.argv[2])
        init_saw = templates / "Init Saw.fxp"
        values, lossy = importer.convert(importer.read_fxp(init_saw), schema)
        diff = [c.id for c, a, b in zip(schema.controls, values, defaults) if abs(a - b) > 1e-4]
        check(not diff, f"Init Saw differs from the schema defaults: {diff}")

        def imported(name: str) -> dict:
            values, _ = importer.convert(importer.read_fxp(templates.parent / "Basses" / name), schema)
            return {c.id: v for c, v in zip(schema.controls, values)}

        # Reverb on send 1 at level 0.707: mix is the cubed send level.
        behemoth = imported("Behemoth.fxp")
        check(behemoth["reverb_on"] == 1 and abs(behemoth["reverb_mix"] - 0.707143 ** 3) < 1e-3,
              f"Behemoth reverb: on={behemoth['reverb_on']} mix={behemoth['reverb_mix']:.3f}")
        # Its only delay sits in a slot switched off with fx_disable.
        check(imported("Bass 5.fxp")["delay_on"] == 0, "Bass 5: disabled delay imported as on")

        # A corrupt file is skipped; the batch still converts the rest.
        with tempfile.TemporaryDirectory() as tmp:
            (pathlib.Path(tmp) / "bad.fxp").write_bytes(b'<?xml version="1.0"?><patch><parameters>'
                                                         b'<a_polymode value="inf"/></parameters></patch>')
            (pathlib.Path(tmp) / "dir.fxp").mkdir()
            sys.argv_backup = sys.argv
            sys.argv = ["importer", tmp, str(init_saw), "-o", str(pathlib.Path(tmp) / "out.jsonl")]
            try:
                check(importer.main() == 0, "importer batch failed")
            finally:
                sys.argv = sys.argv_backup
            lines = (pathlib.Path(tmp) / "out.jsonl").read_text().splitlines()
            check(len(lines) == 1, f"expected 1 converted preset, got {len(lines)}")

    for f in failures:
        print("FAIL:", f)
    print(f"schema tools: {'ok' if not failures else f'{len(failures)} failure(s)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
