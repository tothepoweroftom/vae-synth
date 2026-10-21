#!/usr/bin/env bash
# Copy the SAME bank and Stable Audio package to where the plugin loads them on first use.
# Usage: scripts/install-models.sh [build dir holding native-same/ and native-generator/]
set -euo pipefail
src="${1:-build-lite-au}"
dest="$HOME/Library/Application Support/VAE Synth/models"
mkdir -p "$dest/same" "$dest/generator"
cp "$src/native-same/"{native-bank.json,same-encoder.onnx} "$dest/same/"
cp "$src/native-generator/"{generator.json,text.onnx,dit.onnx,decoder.onnx,tokenizer.json} "$dest/generator/"
echo "Installed to $dest"
