"""Stage self-contained inference bundles for both native installers."""
import json
from pathlib import Path
import shutil

LAB_FILES = ("bundle.json", "decoder.onnx", "sound_encoder.onnx", "sound_decoder.onnx",
             "schema.json", "pruning.json", "sound_pca.npz", "xypad.json")
PROMPT_FILES = ("prompt.json", "text_encoder.onnx", "prior.onnx", "tokenizer.json",
                "bank.json", "bank.bin")


def require_files(directory, names):
    for name in names:
        path = directory / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty inference asset: {path}")


def stage_models(destination, model_root, surge_data, native_root=None):
    lab = model_root / "joint-vae-j48w_b05"
    prompt = model_root / "prompt-p1"
    require_files(lab, LAB_FILES)
    require_files(prompt, PROMPT_FILES)
    if json.loads((prompt / "prompt.json").read_text())["lab_bundle_id"] != json.loads(
            (lab / "bundle.json").read_text())["id"]:
        raise ValueError("Prompt model and joint VAE bundle IDs do not match")
    template = Path("patches_factory/Templates/Init Saw.fxp")
    require_files(surge_data, [template])
    schema = json.loads((lab / "schema.json").read_text())
    names = {o for f in schema["features"] if f["name"].endswith("/wt")
             for o in f["options"]} - {"other"} | {"Sine Octaves"}
    index = {}
    for folder in ("wavetables", "wavetables_3rdparty"):
        for path in sorted((surge_data / folder).rglob("*")):
            if path.suffix.lower() in (".wav", ".wt"):
                index.setdefault(path.stem, path)
    missing = names - index.keys()
    if missing:
        raise FileNotFoundError(f"Missing Surge wavetables: {', '.join(sorted(missing))}")
    bundles = [(lab, "joint-vae"), (prompt, "prompt-p1")]
    if native_root is not None:
        same, generator = native_root / "native-same", native_root / "native-generator"
        require_files(same, ["native-bank.json"])
        model = json.loads((same / "native-bank.json").read_text())["model_file"]
        if Path(model).name != model or "\\" in model:
            raise ValueError("SAME model must be a filename beside native-bank.json")
        require_files(same, [model])
        require_files(generator, ["generator.json", "text.onnx", "dit.onnx", "decoder.onnx", "tokenizer.json"])
        bundles += [(same, "same"), (generator, "generator")]
    # Preserve complete bundles, including external ONNX tensor data and metadata.
    for source, name in bundles:
        shutil.copytree(source, destination / name)
    for source in [surge_data / template, *(index[name] for name in sorted(names))]:
        target = destination / "surge-data" / source.relative_to(surge_data)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return destination
