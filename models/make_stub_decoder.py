#!/usr/bin/env python3
"""Write models/stub_decoder.onnx: a placeholder latent -> controls decoder.

    controls = sigmoid(latent @ W + b)

input  "latent"   float32 [batch, latent_dimensions]
output "controls" float32 [batch, control_count]   (0..1, schema order)

b is logit(default) so the zero latent decodes to the schema defaults; W is a
small seeded random matrix. It has the exact I/O contract a trained decoder
must have, so the plug-in, tests and export tooling can be built against it.

Requires: pip install onnx numpy
"""

import argparse
import json
import pathlib

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--schema", default=ROOT / "schema" / "controls.json", type=pathlib.Path)
    parser.add_argument("--out", default=ROOT / "models" / "stub_decoder.onnx", type=pathlib.Path)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--scale", default=1.5, type=float, help="spread of W (how far the latent moves controls)")
    args = parser.parse_args()

    schema = json.loads(args.schema.read_text())
    latent_dims = schema["latent_dimensions"]
    defaults = np.array([c["default"] for c in schema["controls"]], dtype=np.float32)
    controls = len(defaults)

    rng = np.random.default_rng(args.seed)
    w = (rng.standard_normal((latent_dims, controls)) * args.scale / np.sqrt(latent_dims)).astype(np.float32)
    eps = 1e-3
    clipped = np.clip(defaults, eps, 1 - eps)
    b = np.log(clipped / (1 - clipped)).astype(np.float32)

    graph = helper.make_graph(
        nodes=[
            helper.make_node("MatMul", ["latent", "W"], ["xw"]),
            helper.make_node("Add", ["xw", "b"], ["logits"]),
            helper.make_node("Sigmoid", ["logits"], ["controls"]),
        ],
        name="vae_stub_decoder",
        inputs=[helper.make_tensor_value_info("latent", TensorProto.FLOAT, ["batch", latent_dims])],
        outputs=[helper.make_tensor_value_info("controls", TensorProto.FLOAT, ["batch", controls])],
        initializer=[numpy_helper.from_array(w, "W"), numpy_helper.from_array(b, "b")],
    )
    model = helper.make_model(
        graph,
        producer_name="vae-synth/make_stub_decoder.py",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 8  # loadable by ONNX Runtime >= 1.10
    helper.set_model_props(model, {
        "vae.schema_version": str(schema["schema_version"]),
        "vae.role": "latent_decoder",
        "vae.stub": "true",
    })
    onnx.checker.check_model(model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.out)
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes): [{latent_dims}] -> [{controls}]")


if __name__ == "__main__":
    main()
