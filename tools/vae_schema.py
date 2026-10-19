"""Python view of schema/controls.json, mirroring source/schema/ControlSchema.h.

Raw values per kind:
    continuous   0..1 along the control's curve (Surge-normalised)
    categorical  option index
    toggle       0 or 1
Model features: one per continuous/toggle control, one per categorical option
(one-hot). encode_features / decode_features match the C++ exactly.
"""

from __future__ import annotations

import json
import math
import pathlib
from dataclasses import dataclass, field

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT / "schema" / "controls.json"


@dataclass(frozen=True)
class Control:
    index: int
    id: str
    name: str
    section: str
    kind: str
    default: float
    options: tuple[str, ...] = ()
    curve: str | None = None
    provisional: bool = False
    first_feature: int = 0

    @property
    def width(self) -> int:
        return len(self.options) if self.kind == "categorical" else 1


@dataclass
class Schema:
    version: int
    latent_dimensions: int
    curves: dict
    controls: list[Control] = field(default_factory=list)

    @property
    def feature_count(self) -> int:
        last = self.controls[-1]
        return last.first_feature + last.width

    def __getitem__(self, control_id: str) -> Control:
        for c in self.controls:
            if c.id == control_id:
                return c
        raise KeyError(control_id)

    def defaults(self) -> list[float]:
        return [c.default for c in self.controls]

    def sanitise(self, control: Control, v: float) -> float:
        if not math.isfinite(v):
            return control.default
        if control.kind == "continuous":
            return min(max(v, 0.0), 1.0)
        if control.kind == "toggle":
            return 1.0 if v >= 0.5 else 0.0
        # floor(v + 0.5) matches C++ std::round for the non-negative values used here
        # (Python's round() rounds halves to even).
        return float(min(max(math.floor(v + 0.5), 0), len(control.options) - 1))

    def to_units(self, control: Control, v: float) -> float:
        curve = self.curves[control.curve]
        return curve["min"] + min(max(v, 0.0), 1.0) * (curve["max"] - curve["min"])

    def from_units(self, control: Control, x: float) -> float:
        curve = self.curves[control.curve]
        return min(max((x - curve["min"]) / (curve["max"] - curve["min"]), 0.0), 1.0)

    def encode_features(self, values: list[float]) -> list[float]:
        f = [0.0] * self.feature_count
        for c, v in zip(self.controls, values):
            v = self.sanitise(c, v)
            if c.kind == "categorical":
                f[c.first_feature + int(v)] = 1.0
            else:
                f[c.first_feature] = v
        return f

    def decode_features(self, f: list[float]) -> list[float]:
        out = []
        for c in self.controls:
            if c.kind == "categorical":
                scores = f[c.first_feature:c.first_feature + c.width]
                out.append(float(max(range(c.width), key=lambda o: (scores[o], -o))))
            else:
                out.append(self.sanitise(c, f[c.first_feature]))
        return out


def load(path: pathlib.Path = SCHEMA_PATH) -> Schema:
    doc = json.loads(pathlib.Path(path).read_text())
    schema = Schema(doc["schema_version"], doc["latent_dimensions"], doc["curves"])
    feature = 0
    for i, c in enumerate(doc["controls"]):
        options = tuple(c.get("options", ()))
        if c["kind"] == "categorical":
            default = float(options.index(c["default"]))
        else:
            default = float(c["default"])
        control = Control(i, c["id"], c["name"], c["section"], c["kind"], default, options,
                          c.get("curve"), bool(c.get("provisional", False)), feature)
        schema.controls.append(control)
        feature += control.width
    return schema
