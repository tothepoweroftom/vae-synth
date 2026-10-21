#!/usr/bin/env python3
"""Local Stable Audio Small -> reference -> SAME -> editable presets."""
from __future__ import annotations
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import secrets
import sys
import threading
import time
from types import SimpleNamespace
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import same_search


class Pipeline:
    def __init__(self, manifest, index, device='cpu'):
        self.manifest, self.index, self.device = manifest, index, device
        self.generator = None
        self.generator_name = None
        self.encoder = None

    def run(self, prompt, output, model='small-sfx', seed=42, progress=print):
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 2000:
            raise ValueError('Enter a prompt of 1–2000 characters')
        if model not in ('small-sfx', 'small-music'):
            raise ValueError('Choose Small-SFX or Small-Music')
        if type(seed) is not int or not 0 <= seed < 2**31:
            raise ValueError('Seed must be an integer from 0 to 2147483647')
        if output.exists():
            raise ValueError('Choose a new output directory')
        if not self.manifest.is_file() or not self.index.is_file():
            raise ValueError('Build the SAME audio bank and index first')
        import torch
        import soundfile as sf
        from stable_audio_3 import StableAudioModel
        torch.set_num_threads(4)
        progress(f'Loading {model}; first use downloads model weights…')
        if self.generator_name != model:
            self.generator = None
            self.generator_name = None
            self.generator = StableAudioModel.from_pretrained(model, device=self.device, model_half=False)
            self.generator_name = model
        output.mkdir(parents=True)
        # Preserve precisely what was sent to the model; no hidden prompt rewriting.
        metadata = dict(prompt=prompt.strip(), model=model, seed=seed, duration=4, steps=8,
                        device=self.device, status='generating')
        request = output / 'request.json'
        request.write_text(json.dumps(metadata, indent=2))
        started = time.monotonic()
        try:
            progress('Generating a four-second reference…')
            audio = self.generator.generate(prompt=prompt.strip(), duration=4, steps=8, seed=seed)
            if audio.ndim != 3 or audio.shape[0] != 1 or audio.shape[1] not in (1, 2) or not torch.isfinite(audio).all():
                raise ValueError('Generator returned invalid audio')
            if audio.abs().max() < 1e-5:
                raise ValueError('Generator returned silence; try another prompt or seed')
            # Floating WAV preserves model output without clipping PCM conversion.
            sf.write(output / 'reference.wav', audio[0].detach().cpu().numpy().T,
                     self.generator.model.sample_rate, subtype='FLOAT')
            progress('Encoding the reference with SAME-Small and finding presets…')
            if self.encoder is None:
                self.encoder = same_search.Encoder(self.device)
            same_search.search(SimpleNamespace(manifest=self.manifest, index=self.index,
                               reference=output/'reference.wav', output=output/'matches', device=self.device),
                               encoder=self.encoder)
            metadata.update(status='complete', elapsed_seconds=time.monotonic()-started)
            request.write_text(json.dumps(metadata, indent=2))
            return json.loads((output/'matches/results.json').read_text())
        except Exception as error:
            metadata.update(status='failed', error=str(error))
            request.write_text(json.dumps(metadata, indent=2))
            raise


def public_error(error):
    if 'GatedRepo' in type(error).__name__ or 'gated repo' in str(error).lower():
        return ('Model access is required. Accept the selected model’s terms on Hugging Face, '
                'then run .cache/same-venv/bin/hf auth login locally and retry. Do not paste tokens here.')
    return str(error)


def serve(args):
    pipeline = Pipeline(args.manifest, args.index, args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    root = args.output.resolve()
    token = secrets.token_urlsafe(32)
    lock = threading.Lock()
    state = {'busy': False, 'message': 'Ready', 'results': []}

    def update(message):
        with lock:
            state['message'] = message

    def work(data, job):
        try:
            results = pipeline.run(data.get('prompt'), root/job, data.get('model', 'small-sfx'),
                                   data.get('seed', 42), update)
            with lock:
                state.update(message='Ready to audition. Load any match JSON with Preset… in VAE Synth.',
                             results=results, job=job)
        except Exception as error:
            with lock:
                state.update(message=public_error(error), results=[], job=None)
        finally:
            with lock:
                state['busy'] = False

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, body, mime='application/json'):
            if not isinstance(body, bytes):
                body = json.dumps(body).encode()
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == '/':
                page = Path(__file__).with_name('prompt_ui.html').read_text().replace('__TOKEN__', token)
                self.reply(200, page.encode(), 'text/html; charset=utf-8')
            elif self.path == '/status':
                with lock:
                    self.reply(200, dict(state))
            elif self.path.startswith('/files/'):
                path = (root / unquote(self.path[len('/files/'):])).resolve()
                if not path.is_relative_to(root) or not path.is_file() or path.suffix not in ('.json', '.wav'):
                    self.reply(404, {'error': 'File not found'})
                else:
                    self.reply(200, path.read_bytes(), mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
            else:
                self.reply(404, {'error': 'Not found'})

        def do_POST(self):
            if self.path != '/generate' or self.headers.get('X-Request-Token') != token:
                self.reply(403, {'error': 'Reload this local page before generating'})
                return
            try:
                size = int(self.headers.get('Content-Length', 0))
                if not 0 < size <= 10000:
                    raise ValueError('Invalid request size')
                data = json.loads(self.rfile.read(size))
                if not isinstance(data, dict):
                    raise ValueError('Expected an object')
            except (ValueError, json.JSONDecodeError) as error:
                self.reply(400, {'error': str(error)})
                return
            with lock:
                if state['busy']:
                    self.reply(409, {'error': 'A generation is already running'})
                    return
                state.update(busy=True, message='Starting…', results=[], job=None)
            job = time.strftime('%Y%m%d-%H%M%S') + '-' + secrets.token_hex(3)
            threading.Thread(target=work, args=(data, job), daemon=True).start()
            self.reply(202, {'accepted': True})

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'Prompt UI: http://127.0.0.1:{server.server_port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=ROOT/'build-lite-au/same-audio/manifest.json')
    p.add_argument('--index', type=Path, default=ROOT/'build-lite-au/same-index.npz')
    p.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('serve')
    s.add_argument('--port', type=int, default=8765)
    s.add_argument('--output', type=Path, default=ROOT/'build-lite-au/prompt-runs')
    r = sub.add_parser('run')
    r.add_argument('--prompt', required=True)
    r.add_argument('--model', choices=['small-sfx', 'small-music'], default='small-sfx')
    r.add_argument('--seed', type=int, default=42)
    r.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    try:
        if args.command == 'serve':
            serve(args)
        else:
            Pipeline(args.manifest, args.index, args.device).run(args.prompt, args.output, args.model, args.seed)
    except Exception as error:
        p.exit(1, public_error(error)+'\n')


if __name__ == '__main__':
    main()
