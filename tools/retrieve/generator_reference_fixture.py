#!/usr/bin/env python3
"""Record upstream FP32 generation and its random draws for native integration parity."""
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from stable_audio_3 import StableAudioModel
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
prompt='A bright metallic bell, single strike, clean decay'
torch.set_num_threads(4)
m=StableAudioModel.from_pretrained('small-sfx',device='cpu',model_half=False)
m.model.conditioner.conditioners['prompt'].model.float().set_attn_implementation('eager')
original=torch.randn_like; draws=[]
def record(x,*args,**kwargs):
    y=original(x,*args,**kwargs);draws.append(y.clone().flatten().numpy());return y
torch.randn_like=record
try:
    with torch.inference_mode():audio=m.generate(prompt=prompt,duration=4,steps=8,seed=42)
finally:torch.randn_like=original
initial=torch.randn((1,256,108),generator=torch.Generator().manual_seed(42)).numpy().flatten()
# The final ping-pong random draw is multiplied by zero; C++ deliberately skips it.
noise=np.concatenate([initial]+draws[:7]+draws[8:])
noise.tofile(out/'noise.raw')
sf.write(out/'python-reference.wav',audio[0].numpy().T,44100,subtype='FLOAT')
print('Draw sizes',list(map(len,draws)),'fixture',len(noise),flush=True)
