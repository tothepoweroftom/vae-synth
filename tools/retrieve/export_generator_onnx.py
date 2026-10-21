#!/usr/bin/env python3
"""Build the fixed 4-second Small-SFX native package (development only)."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort
from stable_audio_3 import StableAudioModel
from stable_audio_3.inference.sampling import build_schedule

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('output',type=Path)
p.add_argument('--stage',choices=['text','dit','decoder'],required=True)
a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4)
# PyTorch 2.7's legacy exporter has no RMSNorm symbolic. Equivalent primitive ops.
def export_rms_norm(x, normalized_shape, weight=None, eps=None):
    dims=tuple(range(x.ndim-len(normalized_shape),x.ndim))
    y=x*torch.rsqrt((x*x).mean(dim=dims,keepdim=True)+(eps if eps is not None else torch.finfo(x.dtype).eps))
    return y if weight is None else y*weight
torch.nn.functional.rms_norm=export_rms_norm
m=StableAudioModel.from_pretrained('small-sfx',device='cpu',model_half=False)
c=m.model.conditioner.conditioners['prompt']
c.model.float()
c.model.set_attn_implementation('eager')
tokens=c.tokenizer(['A bright metallic bell, single strike, clean decay'],padding='max_length',truncation=True,max_length=256,return_tensors='pt')
(a.output/'tokenizer.json').write_text(c.tokenizer.backend_tokenizer.to_str())
L=m._adapt_sample_size([{'seconds_total':4}],5292032,6)//4096
with torch.inference_mode():
    conditions=m.model.conditioner([{'prompt':'A bright metallic bell, single strike, clean decay','seconds_total':4}], 'cpu')
    conditions['inpaint_mask']=[torch.zeros(1,1,L)]
    conditions['inpaint_masked_input']=[torch.zeros(1,256,L)]
    inputs=m.model.get_conditioning_inputs(conditions)
    schedule=build_schedule(8,dist_shift=m.model.sampling_dist_shift,effective_seq_len=torch.tensor([44]),fallback_seq_len=L)

class Text(torch.nn.Module):
    def __init__(self):
        super().__init__();self.encoder=c.model;self.conditioner=c
    def forward(self,ids,mask):
        hidden=self.encoder(input_ids=ids.long(),attention_mask=mask.bool()).last_hidden_state
        return self.conditioner.apply_padding(hidden,mask.bool())
class Dit(torch.nn.Module):
    def __init__(self):
        super().__init__();self.dit=m.dit
        for key,value in inputs.items():
            if value is not None and key not in ['cross_attn_cond','cross_attn_mask']:
                self.register_buffer(key,value)
        self.register_buffer('seconds',inputs['cross_attn_cond'][:,-1:,:])
        # Upstream ceil(4s /4096) + floor(6s /4096) =108 valid tokens.
        self.register_buffer('padding',torch.arange(L)[None,:]<108)
    def forward(self,x,t,hidden):
        return self.dit(x,t,cross_attn_cond=torch.cat([hidden,self.seconds],dim=1),
            global_cond=self.global_cond,local_add_cond=self.local_add_cond,padding_mask=self.padding,cfg_scale=1.)
class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__();self.ae=m.same
    def forward(self,z,noise):
        # Make every upstream decoder random draw explicit, so C++ owns the RNG
        # and PyTorch/ORT can be compared with identical stochastic inputs.
        original=torch.randn_like
        offset=0
        def supplied(x,*args,**kwargs):
            nonlocal offset
            count=x.numel(); result=noise[offset:offset+count].reshape_as(x);offset+=count
            return result
        torch.randn_like=supplied
        try: return self.ae.decode(z)[:,:,:176400].float().clamp(-1,1)
        finally: torch.randn_like=original

torch.manual_seed(42)
if a.stage=='text':
    wrapper=Text().eval(); args=(tokens['input_ids'].float(),tokens['attention_mask'].float()); names=['ids','mask']
elif a.stage=='dit':
    wrapper=Dit().eval();args=(torch.randn(1,256,L),torch.tensor([.6]),conditions['prompt'][0]);names=['x','t','hidden']
else:
    z=torch.randn(1,256,L)
    shapes=[]; original=torch.randn_like
    def record(x,*args,**kwargs):
        shapes.append(list(x.shape));return original(x,*args,**kwargs)
    torch.randn_like=record
    try:
        with torch.inference_mode(): m.same.decode(z)
    finally: torch.randn_like=original
    noise_count=sum(int(np.prod(shape)) for shape in shapes)
    print('DECODER NOISE',shapes,noise_count,flush=True)
    wrapper=Decoder().eval();args=(z,torch.randn(noise_count));names=['latents','noise']
path=a.output/(a.stage+'.onnx')
with torch.inference_mode():
    expected=wrapper(*args).numpy()
    print('EXPORT',a.stage,[list(x.shape) for x in args],list(expected.shape),flush=True)
    torch.onnx.export(wrapper,args,str(path),input_names=names,output_names=['output'],opset_version=17,dynamo=False)
    print('VERIFY',flush=True)
    opts=ort.SessionOptions();opts.intra_op_num_threads=4
    session=ort.InferenceSession(str(path),opts,providers=['CPUExecutionProvider'])
    actual=session.run(None,{n:x.numpy() for n,x in zip(names,args)})[0]
    error=float(np.linalg.norm(actual-expected)/max(np.linalg.norm(expected),1e-10))
    print('RELATIVE RMS',error,flush=True)
    if not np.isfinite(actual).all() or error>0.002:raise RuntimeError('Parity failed')
    np.savez(a.output/(a.stage+'-fixture.npz'),expected=expected,actual=actual,**{n:x.numpy() for n,x in zip(names,args)})
    for n,x in zip(names,args): x.numpy().tofile(a.output/(a.stage+'-'+n+'.raw'))
    expected.tofile(a.output/(a.stage+'-expected.raw'))
    report=dict(stage=a.stage,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),relative_rms=error,shapes=[list(x.shape) for x in args],output_shape=list(actual.shape),schedule=schedule.flatten().tolist(),latent_length=L)
    (a.output/(a.stage+'.parity.json')).write_text(json.dumps(report,indent=2))
