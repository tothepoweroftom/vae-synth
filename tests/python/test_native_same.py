"""Optional real-model parity: native audio/ONNX/search against the Python bank."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tools/retrieve')]
from same_search import Encoder, read_manifest, rank


def main():
    exe,bank,manifest,index,*refs=sys.argv[1:]
    entries=read_manifest(Path(manifest))
    with np.load(index,allow_pickle=False) as cache:
        embeddings=cache['embeddings']
    encoder=Encoder('cpu')
    with tempfile.TemporaryDirectory() as tmp:
        for i,ref in enumerate(refs):
            output=Path(tmp)/str(i)
            subprocess.run([exe,bank,ref,str(output)],check=True)
            expected=encoder.audio(Path(ref)).numpy()
            actual=np.fromfile(output/'input.raw',np.float32).reshape(2,-1)
            error=float(np.max(np.abs(expected-actual)))
            assert error<1e-4,(ref,error)
            query=encoder.encode([Path(ref)])[0]
            native=np.fromfile(output/'embedding.raw',np.float32)
            cosine=float(native@query)
            assert cosine>.99999,(ref,cosine)
            expected_matches=rank(query,embeddings,entries)
            cutoff=expected_matches[-1][1]
            rows=json.loads((output/'results.json').read_text())
            assert len(rows)==5 and len({tuple(r['values']) for r in rows})==5
            distances=1-embeddings@query
            for r in rows:
                candidates=[j for j,e in enumerate(entries) if e['values']==r['values'] and e['note']==r['note']]
                assert candidates, r['name']
                distance=float(min(distances[candidates]))
                assert abs(distance-r['score'])<3e-5,(r['name'],distance,r['score'])
                assert distance<=cutoff+3e-5,(ref,r['name'],distance,cutoff)
            print(f'{ref}: preprocessing max error {error:.2g}, embedding cosine {cosine:.9f}, ranking parity passed')
        # A corrupted model fingerprint must be rejected before inference.
        bad=Path(tmp)/'bad.json'
        data=json.loads(Path(bank).read_text());data['model_sha256']='0'*64
        bad.write_text(json.dumps(data))
        r=subprocess.run([exe,str(bad),refs[0],str(Path(tmp)/'bad-output')],capture_output=True)
        assert r.returncode!=0
    print('Native SAME integration passed')


if __name__=='__main__':main()
