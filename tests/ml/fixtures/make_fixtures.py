#!/usr/bin/env python3
"""Write the malformed decoder models used by tests/ml (requires onnx, numpy).

batch2.onnx  input [2, latent]: a fixed batch the plug-in cannot feed
rank1.onnx   input [latent]: wrong rank
nan.onnx     correct shapes, outputs NaN
"""

import pathlib
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "tools"))
import vae_schema  # noqa: E402


def write(name: str, in_shape: list, out_shape: list, bias: np.ndarray) -> None:
    latent = in_shape[-1]
    w = np.zeros((latent, bias.shape[0]), dtype=np.float32)
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["latent", "W"], ["xw"]), helper.make_node("Add", ["xw", "b"], ["features"])],
        name,
        [helper.make_tensor_value_info("latent", TensorProto.FLOAT, in_shape)],
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, out_shape)],
        [numpy_helper.from_array(w, "W"), numpy_helper.from_array(bias, "b")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, HERE / f"{name}.onnx")


schema = vae_schema.load()
n_lat, n_feat = schema.latent_dimensions, schema.feature_count
half = np.full(n_feat, 0.5, dtype=np.float32)
write("batch2", [2, n_lat], [2, n_feat], half)
write("rank1", [n_lat], [n_feat], half)
write("nan", ["batch", n_lat], ["batch", n_feat], np.full(n_feat, np.nan, dtype=np.float32))
print("fixtures written to", HERE)
