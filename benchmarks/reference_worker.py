#!/usr/bin/env python3
"""Official Transformers/PyTorch text model behind the same JSONL protocol."""
import argparse
import json
import os
import re
from pathlib import Path
import sys
import time

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('--attention', choices=['eager', 'sdpa'], default='sdpa')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    started = time.perf_counter()
    import torch
    import transformers
    from tokenizers import Tokenizer
    from transformers import EmbeddingGemma2Config, EmbeddingGemma2Model

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    raw = json.loads((args.model/'config.json').read_text())
    raw['vision_config'] = None
    raw['audio_config'] = None
    config = EmbeddingGemma2Config(**raw)
    model, info = EmbeddingGemma2Model.from_pretrained(
        args.model, config=config, dtype=torch.float32, local_files_only=True,
        attn_implementation=args.attention, output_loading_info=True,
    )
    if info.get('missing_keys') or info.get('unexpected_keys') or info.get('mismatched_keys') or info.get('error_msgs'):
        raise RuntimeError('Reference model did not load the exact text state dictionary')
    model.eval()
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise RuntimeError('Reference parameters must all use float32')
    if model.language_model.config._attn_implementation != args.attention:
        raise RuntimeError('Attention backend did not match request')
    if next(model.parameters()).device.type != 'cpu':
        raise RuntimeError('Reference must run on CPU')
    blas = re.search(r'BLAS_INFO=([^,]+)', torch.__config__.show())
    tokenizer = Tokenizer.from_file(str(args.model/'tokenizer.json'))
    print(json.dumps({'ready': True, 'load_ms': 1000*(time.perf_counter()-started),
                      'torch': torch.__version__, 'transformers': transformers.__version__,
                      'python': sys.version.split()[0], 'parameter_dtype': str(next(model.parameters()).dtype),
                      'device': str(next(model.parameters()).device), 'blas': blas.group(1) if blas else 'unknown',
                      'veclib_maximum_threads': os.environ.get('VECLIB_MAXIMUM_THREADS'),
                      'attention': model.language_model.config._attn_implementation, 'dtype': 'float32',
                      'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads()}),
          file=sys.stderr, flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            ids = request.get('ids')
            if ids is None:
                ids = tokenizer.encode(request['text'], add_special_tokens=True).ids
            dims = request.get('dims', 768)
            if not 1 <= len(ids) <= 1024 or dims not in (128, 256, 512, 768):
                raise ValueError('unsupported input shape')
            start = time.perf_counter()
            with torch.inference_mode():
                inputs = torch.tensor([ids], dtype=torch.long)
                # Official text model projects each token; mean pooling and
                # normalization match the documented SentenceTransformers recipe.
                hidden = model(input_ids=inputs).last_hidden_state
                if hidden.shape[-1] != 768:
                    raise RuntimeError('Official model must return projected 768-dimensional token states')
                output = hidden.mean(dim=1)[0, :dims]
                output = torch.nn.functional.normalize(output, dim=0).tolist()
            response = {'embedding': output, 'tokens': len(ids), 'elapsed_ms': 1000*(time.perf_counter()-start)}
        except Exception as error:
            response = {'error': type(error).__name__ + ': ' + str(error)}
        print(json.dumps(response, allow_nan=False), flush=True)

if __name__ == '__main__':
    main()
