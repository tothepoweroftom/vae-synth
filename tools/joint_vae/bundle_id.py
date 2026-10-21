"""The one authority for a joint VAE bundle id and its manifest file keys (used by export_bundle and export_xypad)."""
import hashlib
import json
from pathlib import Path

FILES = ('config.json', 'schema.json', 'pruning.json', 'sound_pca.npz', 'sound_decoder.onnx', 'xypad.json')


def bundle_id(bundle, run, model_sha256):
    """run + hash of the model.pt digest and the bundle's own files; xypad.json counts when present."""
    digest = hashlib.sha256(model_sha256.encode())
    for name in FILES:
        if (bundle / name).exists():
            digest.update((bundle / name).read_bytes())
    return run + '-' + digest.hexdigest()[:16]


def finalise(bundle):
    """Rewrite bundle.json / example-sound.json with the id and the optional xypad key for the files present."""
    manifest = json.loads((bundle / 'bundle.json').read_text())
    run = json.loads((bundle / 'config.json').read_text())['run']
    manifest['id'] = bundle_id(bundle, run, manifest['model_sha256'])
    if (bundle / 'xypad.json').exists():
        manifest['xypad'] = 'xypad.json'
    (bundle / 'bundle.json').write_text(json.dumps(manifest, indent=2) + '\n')
    example = json.loads((bundle / 'example-sound.json').read_text())
    example['bundle_id'] = manifest['id']
    (bundle / 'example-sound.json').write_text(json.dumps(example, indent=2) + '\n')
    return manifest['id']
