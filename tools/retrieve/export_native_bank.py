#!/usr/bin/env python3
"""Package a verified SAME ONNX model and existing Python index for native JUCE."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from same_search import digest, read_manifest, FORMAT, POOLING


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','index','model','output'):
        p.add_argument(name,type=Path)
    a=p.parse_args()
    if a.output.exists():
        p.error('Choose a new bundle directory')
    entries=read_manifest(a.manifest)
    with np.load(a.index,allow_pickle=False) as cache:
        meta=json.loads(str(cache['metadata']));embeddings=cache['embeddings']
    parity=json.loads(a.model.with_suffix('.parity.json').read_text())
    if (meta['format']!=FORMAT or meta['pooling']!=POOLING or meta['manifest_sha256']!=digest(a.manifest)
        or any(parity.get(k)!=meta.get(k) for k in ('weights_sha256','config_sha256','pooling','preprocessing'))):
        p.error('Model parity record and index identities must match')
    if parity.get('onnx_sha256') != digest(a.model):
        p.error('Parity record must fingerprint this ONNX file; rerun export verification')
    if embeddings.shape!=(len(entries),512) or not np.isfinite(embeddings).all():
        p.error('Invalid embedding matrix')
    if not np.allclose(np.linalg.norm(embeddings,axis=1),1,atol=1e-4):
        p.error('Embeddings must be unit vectors')
    from vae_schema import load
    bank=dict(format='vae-native-same-1',schema_version=load().version,engine='lite',
              pooling=POOLING,preprocessing=meta['preprocessing'],model_file='same-encoder.onnx',
              model_sha256=digest(a.model),source_index_sha256=digest(a.index),entries=[])
    for e,v in zip(entries,embeddings):
        bank['entries'].append(dict(name=e['name'],values=e['values'],note=e['note'],embedding=v.tolist()))
    a.output.mkdir(parents=True)
    shutil.copyfile(a.model,a.output/'same-encoder.onnx')
    (a.output/'native-bank.json').write_text(json.dumps(bank,separators=(',',':')))
    print(a.output/'native-bank.json')


if __name__=='__main__':
    main()
