# CPU comparison

The comparison uses the actual pinned `google/embeddinggemma-2` text weights
with native CPU execution. Reference backends use the official Hugging Face
Transformers model implementation on PyTorch, with eager and SDPA attention.
Rust uses Candle 0.11.0 with fused RMS normalization and rotary embedding
operations, plus parallel fused GELU/multiply on Linux. Native macOS uses
Apple Accelerate; optimized Linux x86_64 builds use AOCL-DLP 5.3.2 with
persistent F32 weight reordering. Source builds without `--features aocl` use
portable GEMM. All compute uses F32; the unchanged vocabulary table stays
stored in BF16 in Rust and is expanded to F32 by PyTorch.

## Reproduce

Download the model once, build the Rust binaries, then install the optional
comparison dependencies in a separate Python environment:

```sh
python3 scripts/download_model.py
cargo build --release --locked --bins
python3.12 -m venv .venv-benchmark
# macOS:
.venv-benchmark/bin/pip install -r benchmarks/requirements.txt
# Linux CPU (use this instead of the preceding pip command):
.venv-benchmark/bin/pip install --extra-index-url https://download.pytorch.org/whl/cpu \
  -r benchmarks/requirements.txt
.venv-benchmark/bin/python benchmarks/compare.py \
  --python .venv-benchmark/bin/python --rust target/release/embedding-jsonl \
  --output benchmarks/results/my-machine.json
```

No model download, network request or external service is used during timing.
The model revision and all three asset checksums are checked before inference.
The fixture checksum in `freeze.json` was fixed before the measured runs.

For the v0.3 AMD comparison, first build the optimized backend as described in
the root README, then run all six backend orders. Choose two online CPU IDs
that identify distinct physical cores in one last-level cache domain; the
driver validates this and all worker thread affinity masks. Place the driver on
a disjoint CPU pair. IDs 0,1 and 8,9 below are the measured laptop's topology;
inspect your machine and choose suitable IDs before running:

```sh
OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0 \
  taskset -c 8,9 .venv-benchmark/bin/python benchmarks/amd_matrix.py \
  --python .venv-benchmark/bin/python --rust target/release/embedding-jsonl \
  --cpus 0,1 \
  --build-provenance 'cargo build --release --locked --features aocl --bins; record AOCL_ROOT, AOCL_STATIC_RUNTIME, compiler versions and flags used' \
  --output-dir benchmarks/results/my-amd-matrix
```

The build provenance string is a caller declaration. Supply the commands and
flags actually used; runtime environment variables cannot prove compiler flags.
Each output directory must be new. Raw runs, matrix schedule, source/binary
hashes, quality gates and an aggregate report are retained together.
The driver records the four whitelisted OpenMP wait/binding settings and checks
them against each Linux worker's environment; all six orders must match.
The [AMD report](AMD_REPORT.md) links the completed six-order comparison.

`scaling.py` separately screens equal 1/2/4/8/16-thread configurations, using
distinct physical cores and one rotated backend order per setting. It records
cache domains, numerical gates, short/post-long/peak RSS and all raw timings.
This exploratory sweep is not a substitute for the six-order acceptance test:

```sh
.venv-benchmark/bin/python benchmarks/scaling.py \
  --python .venv-benchmark/bin/python --rust target/release/embedding-jsonl \
  --cpus 0-15 --build-provenance 'record the exact build used' \
  --output-dir benchmarks/results/my-scaling
```

`wait_policy.py` separately tests Rust in default–passive–passive–default order
at 2/4/8/16 threads, with actual process environment checks, fixed quality gates,
CPU-time deltas and both memory phases. It uses the same arguments as
`scaling.py` except `--python` is unnecessary. Run these CPU probes sequentially.

## Method

- Batch size one, two Rayon/PyTorch intra-op threads, one PyTorch inter-op thread.
  BLAS/OpenMP thread environment variables are also set to two, including
  Apple Accelerate’s `VECLIB_MAXIMUM_THREADS`. Native laptop execution; these are thread limits, not reserved cores or a CPU quota.
- Same model revision, input prefixes, tokenizer asset and dimensions (768).
  Text tokenization is checked against common explicit token IDs for every query.
- Twenty authored multilingual retrieval queries, ten authored documents,
  five warmups, five query repeats (100 measured requests per backend).
  Document indexing is excluded from query latency. The p95 describes this
  mixed prompt set, including its different token lengths; it is not isolated
  scheduling jitter.
- Separate 128/512/1,024-token inputs, with one warmup and three measured repeats
  per length. These length cases use identical explicit token IDs across backends.
- Model time excludes tokenization and JSONL IPC. Request wall time includes
  tokenization, IPC and vector serialization. Neither includes HTTP or ranking.
- Memory is measured after the long-input tests. Linux reports process RSS and
  peak RSS; macOS reports the kernel physical-footprint ledger (including
  compressed-memory accounting) as well as RSS. macOS RSS alone can undercount
  memory under compression, so the comparison uses physical footprint there. Framework
  overhead and vocabulary storage precision differ. Keeping the vocabulary
  table in BF16 saves 256 MiB relative to its F32 representation, without
  changing the source values or F32 encoder compute. Startup wall time includes
  framework imports and Rust's asset verification; it is not isolated weight-load time.
- Numerical agreement compares normalized output vectors with PyTorch eager,
  including documents, queries and each long input. Retrieval hit@1/5 uses only
  these small authored labels; it is not a general relevance benchmark.

Compare backends within each machine. The Mac and laptop have different CPUs,
operating systems and libraries, so the results cannot isolate an ISA effect.
The historical v0.2 comparison uses two sequential runs per backend on shared
laptops, reversing backend order for the second run. It has no CPU isolation
or fixed frequency. Repeat locally before drawing
capacity or production tail-latency conclusions.

The official reference follows the model's documented text-only loading,
mean-pooling and normalization recipe:
[model card](https://huggingface.co/google/embeddinggemma-2/blob/914f7f89142e33e77833254d9c9b90c3cef7303b/README.md).
See [the historical report](REPORT.md), [v0.2 aggregate JSON](results/summary.json),
and individual raw runs under `results/`. The historical `summarize.py` requires
the unchanged v0.2 source; run it from a `v0.2.2` checkout. The v0.3 matrix
validates and summarizes its own six runs. To repeat the old second run,
pass `--order rust_f32,transformers_sdpa,transformers_eager` and a new output path.

The README charts are static PNGs generated from historical Apple data in
`results/summary.json` and the new AMD six-order summary; they
do not run inference. To regenerate them in a Python environment:

```sh
python3 -m pip install -r benchmarks/plot-requirements.txt
python3 benchmarks/plot.py
```
