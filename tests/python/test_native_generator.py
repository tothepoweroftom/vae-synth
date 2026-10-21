"""Real tokenizer parity against transformers, including Unicode and truncation."""
import subprocess,sys
from pathlib import Path
from transformers import AutoTokenizer
exe,package=map(Path,sys.argv[1:3])
tokenizer=AutoTokenizer.from_pretrained('stabilityai/stable-audio-3-small-sfx',subfolder='t5gemma-b-b-ul2')
for prompt in ['bell','A bright metallic bell, single strike, clean decay','  spaced\twords\nline',
               'écho 日本語 🔔 <bos> <eos>','long prompt '*500,'<unused99> [toxicity=0]']:
    expected=tokenizer(prompt,truncation=True,max_length=256,padding='max_length')
    lines=subprocess.check_output([str(exe),'--tokenize',str(package/'tokenizer.json'),prompt],text=True).splitlines()
    assert list(map(int,lines[0].split()))==expected['input_ids'],repr(prompt[:80])
    assert list(map(int,lines[1].split()))==expected['attention_mask']
print('Native token IDs and masks match transformers: 6 prompts')
