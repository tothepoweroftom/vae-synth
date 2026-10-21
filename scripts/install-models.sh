#!/usr/bin/env bash
# Copy the models to where the plugin loads them on first use. Each part is installed when its source exists:
#   SAME bank + Stable Audio package (Prompt tab)  from [build dir]/native-same, native-generator
#   joint VAE bundle (VAE Lab tab)                 from models/joint-vae-j48 (tools/joint_vae/export_bundle.py)
#   Surge template + the bundle's wavetables       from build-surge's Surge checkout (VAE Lab playback)
# Usage: scripts/install-models.sh [build dir holding native-same/ and native-generator/]
set -euo pipefail
src="${1:-build-lite-au}"
bundle="models/joint-vae-j48"
surge="build-surge/_deps/surge-src/resources/data"
dest="$HOME/Library/Application Support/VAE Synth/models"

if [ -d "$src/native-same" ]; then
    mkdir -p "$dest/same" "$dest/generator"
    cp -p "$src/native-same/"{native-bank.json,same-encoder.onnx} "$dest/same/"
    cp -p "$src/native-generator/"{generator.json,text.onnx,dit.onnx,decoder.onnx,tokenizer.json} "$dest/generator/"
    echo "Installed SAME + Stable Audio"
fi

if [ -f "$bundle/bundle.json" ]; then
    mkdir -p "$dest/joint-vae"
    cp -p "$bundle/"* "$dest/joint-vae/"
    echo "Installed joint VAE bundle"
fi

if [ -d "$surge" ] && [ -f "$bundle/schema.json" ]; then
    rm -rf "$dest/surge-data"
    mkdir -p "$dest/surge-data/patches_factory/Templates"
    cp -p "$surge/patches_factory/Templates/Init Saw.fxp" "$dest/surge-data/patches_factory/Templates/"
    # Only the wavetables the model can pick, each the file the student's SurgeIO.wavetable_path resolves
    python3 - "$surge" "$bundle/schema.json" "$dest/surge-data" <<'EOF'
import json, pathlib, shutil, sys
surge, schema, out = pathlib.Path(sys.argv[1]), json.loads(pathlib.Path(sys.argv[2]).read_text()), pathlib.Path(sys.argv[3])
names = {o for f in schema['features'] if f['name'].endswith('/wt') for o in f['options']} - {'other'} | {'Sine Octaves'}
index = {}
for folder in ('wavetables', 'wavetables_3rdparty'):
    for f in sorted((surge / folder).rglob('*')):
        if f.suffix.lower() in ('.wav', '.wt'):
            index.setdefault(f.stem, f)
for name in sorted(names):
    target = out / index[name].relative_to(surge)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(index[name], target)
print(f'Installed Surge template + {len(names)} wavetables')
EOF
fi
echo "Models in $dest"
