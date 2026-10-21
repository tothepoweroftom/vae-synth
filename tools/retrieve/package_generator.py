#!/usr/bin/env python3
"""Publish a native generator manifest only after all three graphs pass parity."""
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1])
def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
reports={stage:json.loads((root/(stage+'.parity.json')).read_text()) for stage in ['text','dit','decoder']}
hashes={name:digest(root/name) for name in ['text.onnx','dit.onnx','decoder.onnx','tokenizer.json']}
for stage,r in reports.items():
    if r['sha256']!=hashes[stage+'.onnx'] or r['relative_rms']>0.002:raise ValueError('Unverified graph: '+stage)
manifest=dict(format='vae-stable-audio-small-sfx-1',model='stabilityai/stable-audio-3-small-sfx',
    duration=4,sample_rate=44100,steps=8,latent_length=108,precision='fp32',sha256=hashes,
    schedule=reports['dit']['schedule'],decoder_noise_count=reports['decoder']['shapes'][1][0])
(root/'generator.json').write_text(json.dumps(manifest,indent=2))
print(root/'generator.json')
