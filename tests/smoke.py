#!/usr/bin/env python3
"""Real-model HTTP contract test. Inputs are public, authored examples only."""
import argparse
import base64
import json
import math
import os
import socket
import struct
import subprocess
import tempfile
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument('--binary')
    target.add_argument('--image')
    parser.add_argument('--model', default='model')
    args = parser.parse_args()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    base = f'http://127.0.0.1:{port}'
    key = 'smoke-test-only'
    container = None
    process = None
    with tempfile.TemporaryFile() as log:
        try:
            if args.image:
                container = subprocess.check_output([
                    'docker', 'run', '-d', '--rm', '--cpus=2', '--memory=3g',
                    '-p', f'127.0.0.1:{port}:8080', '-e', f'API_KEY={key}', args.image,
                ], text=True).strip()
            else:
                env = {**os.environ, 'MODEL_DIR': os.path.abspath(args.model),
                       'LISTEN_ADDR': f'127.0.0.1:{port}', 'API_KEY': key, 'THREADS': '2'}
                process = subprocess.Popen([args.binary], env=env, stdout=log, stderr=log)

            def request(path, payload=None, auth=True):
                headers = {'Content-Type': 'application/json'}
                if auth:
                    headers['Authorization'] = f'Bearer {key}'
                data = json.dumps(payload).encode() if payload is not None else None
                req = urllib.request.Request(base+path, data=data, headers=headers)
                try:
                    with urllib.request.urlopen(req, timeout=90) as response:
                        return response.status, json.load(response)
                except urllib.error.HTTPError as error:
                    return error.code, json.load(error)

            for _ in range(120):
                try:
                    if request('/ready')[0] == 200:
                        break
                except (OSError, ValueError):
                    pass
                if process and process.poll() is not None:
                    raise AssertionError('server exited during startup')
                time.sleep(1)
            else:
                raise AssertionError('server did not become ready')
            assert request('/health')[0] == 200
            assert request('/v1/models', auth=False)[0] == 401
            assert request('/v1/models')[1]['data'][0]['id'] == 'google/embeddinggemma-2'
            payload = {'model': 'google/embeddinggemma-2', 'input': [
                'task: search result | query: What causes the northern lights?',
                'title: Northern lights | text: Charged particles from the sun cause auroras.',
            ]}
            status, batch = request('/v1/embeddings', payload)
            assert status == 200, batch
            assert [row['index'] for row in batch['data']] == [0, 1]
            vectors = [row['embedding'] for row in batch['data']]
            for vector in vectors:
                assert len(vector) == 768 and all(math.isfinite(x) for x in vector)
                assert abs(sum(x*x for x in vector)-1) < 1e-4
            assert batch['usage']['prompt_tokens'] > 0
            assert batch['usage']['prompt_tokens'] == batch['usage']['total_tokens']
            single = {**payload, 'input': payload['input'][0]}
            status, encoded = request('/v1/embeddings', {**single, 'encoding_format': 'base64'})
            assert status == 200
            decoded = struct.unpack('<768f', base64.b64decode(encoded['data'][0]['embedding']))
            assert max(abs(a-b) for a,b in zip(decoded, vectors[0])) < 1e-6
            status, short = request('/v1/embeddings', {**single, 'dimensions': 128})
            assert status == 200
            prefix = vectors[0][:128]
            norm = math.sqrt(sum(x*x for x in prefix))
            assert max(abs(a/norm-b) for a,b in zip(prefix, short['data'][0]['embedding'])) < 1e-5
            for invalid in [
                {**single, 'dimensions': 129}, {**single, 'input': ''},
                {**single, 'input': [262144]}, {**single, 'input': [1]*1025},
                {**single, 'input': 'word '*2000},
            ]:
                status, error = request('/v1/embeddings', invalid)
                assert status == 400 and 'error' in error, (status, error)
            assert request('/health')[0] == 200
            print('PASS: real model, health/auth, ordered batch, normalized vectors, base64, dimensions, invalid/overlong inputs')
        except BaseException:
            if container:
                subprocess.run(['docker', 'logs', container], check=False)
            else:
                log.seek(0)
                print(log.read().decode(errors='replace'))
            raise
        finally:
            if container:
                subprocess.run(['docker', 'stop', '-t', '15', container], stdout=subprocess.DEVNULL, check=False)
            if process:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

if __name__ == '__main__':
    main()
