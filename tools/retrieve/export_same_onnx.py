#!/usr/bin/env python3
"""Export a fixed-four-second FP32 SAME encoder and check ONNX Runtime parity."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from same_search import Encoder, SAMPLES, pool, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    p.add_argument('references', nargs='+', type=Path)
    p.add_argument('--verify-existing', action='store_true')
    a = p.parse_args()
    if a.output.exists() and not a.verify_existing:
        p.error('Choose a new ONNX output path')
    encoder = Encoder('cpu')
    ae = encoder.model.autoencoder
    pad = (-SAMPLES) % ae.min_length

    class FixedEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ae = ae

        def forward(self, audio):
            return self.ae.encode_audio(torch.nn.functional.pad(audio, (0, pad)))

    wrapper = FixedEncoder().eval()
    x = encoder.audio(a.references[0]).unsqueeze(0)
    if not a.verify_existing:
      with torch.inference_mode():
        torch.onnx.export(wrapper, x, str(a.output), input_names=['audio'], output_names=['latents'],
                          opset_version=17, dynamo=False)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(a.output), options, providers=['CPUExecutionProvider'])
    checks = []
    for path in a.references:
        x = encoder.audio(path).unsqueeze(0)
        with torch.inference_mode():
            expected = encoder.model.encode(x, 44100).cpu().numpy()
        t = time.monotonic()
        actual = session.run(None, {'audio': x.numpy()})[0]
        elapsed = time.monotonic()-t
        relative_rms = float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
        if relative_rms > 1e-4:
            raise ValueError(f'Latent relative RMS error too large: {relative_rms}')
        cosine = float(np.sum(pool(actual)*pool(expected)))
        if cosine < .99999:
            raise ValueError('Pooled-feature parity failed')
        checks.append(dict(reference=str(path), max_abs_error=float(np.max(np.abs(actual-expected))),
                           pooled_cosine=cosine, relative_rms_error=relative_rms, cpu_seconds=elapsed))
    report = dict(encoder.identity, onnx_sha256=digest(a.output), input_shape=[1,2,SAMPLES], sample_rate=44100,
                  output_shape=list(actual.shape), runtime=ort.__version__, opset=17, checks=checks)
    a.output.with_suffix('.parity.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
