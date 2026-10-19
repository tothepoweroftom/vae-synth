#!/usr/bin/env python3
"""Write models/stub_decoder.onnx: a placeholder latent -> features decoder.

    features = sigmoid(latent @ W + b)

input  "latent"    float32 [batch, latent_dimensions]
output "features"  float32 [batch, feature_count]
    continuous/toggle controls: one value in 0..1
    categorical controls: one score per option (the plug-in takes the argmax)

b is logit(default features) so the zero latent decodes to the schema
defaults; W is a small seeded random matrix. It has the exact I/O contract a
trained decoder must have (see docs/SCHEMA.md), so the plug-in, tests and
export tooling can be built against it.

Requires: pip install onnx numpy
"""

import argparse
import pathlib
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import vae_schema  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--schema", default=vae_schema.SCHEMA_PATH, type=pathlib.Path)
    parser.add_argument("--out", default=ROOT / "models" / "stub_decoder.onnx", type=pathlib.Path)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--scale", default=1.5, type=float, help="spread of W (how far the latent moves features)")
    args = parser.parse_args()

    schema = vae_schema.load(args.schema)
    latent_dims = schema.latent_dimensions
    defaults = np.array(schema.encode_features(schema.defaults()), dtype=np.float32)
    features = len(defaults)

    rng = np.random.default_rng(args.seed)
    w = (rng.standard_normal((latent_dims, features)) * args.scale / np.sqrt(latent_dims)).astype(np.float32)
    eps = 1e-3
    clipped = np.clip(defaults, eps, 1 - eps)
    b = np.log(clipped / (1 - clipped)).astype(np.float32)

    graph = helper.make_graph(
        nodes=[
            helper.make_node("MatMul", ["latent", "W"], ["xw"]),
            helper.make_node("Add", ["xw", "b"], ["logits"]),
            helper.make_node("Sigmoid", ["logits"], ["features"]),
        ],
        name="vae_stub_decoder",
        inputs=[helper.make_tensor_value_info("latent", TensorProto.FLOAT, ["batch", latent_dims])],
        outputs=[helper.make_tensor_value_info("features", TensorProto.FLOAT, ["batch", features])],
        initializer=[numpy_helper.from_array(w, "W"), numpy_helper.from_array(b, "b")],
    )
    model = helper.make_model(
        graph,
        producer_name="vae-synth/make_stub_decoder.py",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 8  # loadable by ONNX Runtime >= 1.10
    helper.set_model_props(model, {
        "vae.schema_version": str(schema.version),
        "vae.role": "latent_decoder",
        "vae.stub": "true",
    })
    onnx.checker.check_model(model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.out)
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes): [{latent_dims}] -> [{features}] features "
          f"({len(schema.controls)} controls)")


if __name__ == "__main__":
    main()
