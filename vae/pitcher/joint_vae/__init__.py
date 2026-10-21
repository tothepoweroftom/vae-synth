"""Joint VAE (TEXT2PRESET_PLAN.md §6): one latent z for a preset's parameters AND its sound.

Separate from the parameter-only CVAE (pitcher/step4_vae, runs/vae/h256_d01), which stays as it is.
  sound_code.py  mel -> 48 PCA numbers (k chosen by measurement, Round 36)
  data.py        layout (stepped knobs as categories, NumCat++), training rows, labels
  model.py       two encoders (parameters, sound) -> product of experts -> z -> two decoders
  train.py       modality dropout, beta warm-up from 50%; model kept by human-val score
  evaluate.py    sound-only -> preset (inverter role) on the human val / test presets; z probe
"""
