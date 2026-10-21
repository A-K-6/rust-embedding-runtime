# CPU results: EmbeddingGemma 2 text inference

Measured 2026-10-22 with actual pinned model weights. The goal is **lower
uncached latency and lower memory**, while preserving output agreement.

- **Apple M2:** Rust short-query median/p95 is 57.3/75.7 ms versus the faster
  PyTorch baseline (SDPA) at 62.6/98.4 ms: 8.5% lower median, 23.0% lower p95,
  and 20.9% lower measured physical footprint. Query latency wins in both orders.
- **AMD Ryzen 9 9955HX:** Rust uses 36.4% less measured RSS. Pooled query
  median/p95 is 48.4/58.1 ms versus SDPA at 49.0/58.4 ms, but the winner changes
  with run order. Treat this as **latency parity, not an established speedup**.
  Rust is slower on the longer AMD inputs. The speed-and-memory goal remains
  open there.
- All three backends get hit@1 and hit@5 of **20/20** on this tiny authored
  pilot. Rust agrees with eager PyTorch's ordered top five on all 20 queries.
  This does not establish general relevance quality.

Tables pool two opposite-order runs: **200 measured short-query requests per
backend per machine**, reusing 20 authored multilingual queries; long-input
medians use six measurements per length. These are repeated timings, not 200
independent relevance labels. Short queries contain 15–28 tokens including
prefix and special tokens. Memory is the **larger of two post-long-input
samples**, not a short-query-only measurement. Linux uses RSS; macOS uses the
kernel physical-footprint ledger, which accounts for compressed memory.
Neither latency includes HTTP, networking, document indexing or ranking.

## Apple M2

| Backend | Query median / p95 (ms) | Memory (MiB) | 128 / 512 / 1,024 tokens, median (ms) |
| --- | ---: | ---: | ---: |
| PyTorch eager | 73.5 / 146.1 | 1707 | 149.6 / 518.4 / 1058.4 |
| PyTorch SDPA | 62.6 / 98.4 | 1669 | 175.4 / 563.6 / 1116.4 |
| Rust F32 | 57.3 / 75.7 | 1321 | 171.7 / 421.6 / 1054.0 |

| Order | PyTorch eager median / p95 | PyTorch SDPA median / p95 | Rust median / p95 |
| --- | ---: | ---: | ---: |
| Forward | 80.2 / 170.3 | 63.5 / 102.3 | 56.6 / 71.5 |
| Reverse | 66.8 / 123.9 | 61.9 / 91.8 | 59.8 / 78.1 |

Raw runs: [arm64-apple-m2-forward.json](results/arm64-apple-m2-forward.json), [arm64-apple-m2-reverse.json](results/arm64-apple-m2-reverse.json).

## AMD Ryzen 9 9955HX 16-Core Processor

| Backend | Query median / p95 (ms) | Memory (MiB) | 128 / 512 / 1,024 tokens, median (ms) |
| --- | ---: | ---: | ---: |
| PyTorch eager | 50.0 / 58.9 | 1669 | 174.1 / 710.6 / 1725.3 |
| PyTorch SDPA | 49.0 / 58.4 | 1669 | 173.8 / 708.1 / 1652.7 |
| Rust F32 | 48.4 / 58.1 | 1062 | 179.7 / 745.2 / 1795.0 |

| Order | PyTorch eager median / p95 | PyTorch SDPA median / p95 | Rust median / p95 |
| --- | ---: | ---: | ---: |
| Forward | 49.0 / 58.7 | 48.7 / 57.3 | 50.9 / 60.7 |
| Reverse | 50.1 / 59.4 | 49.9 / 58.6 | 47.7 / 56.0 |

Raw runs: [amd64-ryzen-9955hx-forward.json](results/amd64-ryzen-9955hx-forward.json), [amd64-ryzen-9955hx-reverse.json](results/amd64-ryzen-9955hx-reverse.json).

## Numerical checks and provenance

- Same `google/embeddinggemma-2` revision
  `914f7f89142e33e77833254d9c9b90c3cef7303b`; unchanged tokenizer, explicit
  retrieval prefixes, normalized 768-dimensional outputs, batch one.
- Candle **0.11.0**, Rust release builds; native Mac uses Apple Accelerate,
  Linux uses portable GEMM. Fused RMS normalization and rotary kernels;
  fused parallel GELU/multiply on Linux. No result cache or GPU.
- PyTorch **2.14.1** (Linux `+cpu`), Transformers **5.20.2**, tokenizer package
  **0.23.3**; Rust tokenizer **0.22.2**, checked against identical explicit IDs.
  Mac PyTorch reports Accelerate BLAS; Linux reports MKL. Both references use
  the official model class, with eager and SDPA attention respectively.
- Rust compiler **1.98.0 on macOS**, **1.91.0 on Linux**. Driver Python
  **3.12.14 on macOS**, **3.12.13 on Linux**. Complete readiness/version,
  hardware, binary and source hashes are retained in each raw result.
- Rust's maximum absolute output error against eager PyTorch is below
  **3.24e-7**, minimum cosine agreement above **0.999999999996**, over 33 vectors
  per run: 10 documents, 20 queries and three long inputs. The acceptance gates
  are max error <=1e-4 and cosine >=0.99999. All four runs pass.
- Model/activation code was measured at
  [`ce1b663`](https://github.com/A-K-6/rust-embedding-runtime/tree/ce1b6632149722abf2f7a17d22b0cfe1a42aa662).
  The v0.2.2 packaging patches change the build compiler to Rust 1.98, package
  metadata and Linux ARM64 build configuration;
  native benchmark model/activation source is identical. The aggregation script
  verifies those files and the measured commit's lockfile before summarizing.
- Frozen fixture SHA-256:
  `8c61c2f94cc2996cff4546d9ade325da2e26da09d47e4a667bf5b87eee37c525`.
  [Aggregate JSON](results/summary.json), [fixtures](fixtures.json),
  [driver](compare.py), [official reference worker](reference_worker.py),
  [method and reproduction commands](README.md).

## Limits and interpretation

Two kernel threads per backend; PyTorch inter-op one. BLAS/OpenMP environment
variables are capped at two. These shared native machines have **no reserved
cores, CPU quota, fixed frequency or thermal isolation**. Backend order is
reversed for a second run, but this does not eliminate interference. The AMD
order reversal is direct evidence against claiming a reliable short-query
speedup from its tiny pooled advantage. Mixed-query p95 includes token-length
variation; long-input samples are too few for a tail-latency claim.

Rust keeps the original BF16 vocabulary table and converts gathered rows to
F32; PyTorch expands it to F32. All encoder math and linear weights are F32.
This saves 256 MiB of table storage without changing the original weight
values. Remaining differences include framework overhead and allocator state;
this is a practical runtime comparison, not an equal-storage-precision
microbenchmark. Mac physical footprint and Linux RSS are different memory
metrics and should only be compared within the same machine. Linux peak RSS
is also available in the raw results.

**Native macOS benchmarks do not measure the Linux ARM64 Docker image.**
That image uses Linux CPU kernels rather than Apple Accelerate. Docker images
are separately validated for real model/API correctness on native ARM64 and
AMD64; no container speed claim is made here. Linux ARM64 requires NEON/FP16
CPU instructions because the GEMM dependency's ARM assembly needs that build
feature; inference remains F32. This is a build constraint also discussed in
[the upstream Candle issue](https://github.com/huggingface/candle/issues/494).

Only this text checkpoint is implemented. Context is capped at 1,024 tokens;
the upstream multimodal and longer-context capabilities are outside this
runtime. These tests do not compare BF16 inference, torch.compile, ONNX,
OpenVINO, other runtimes, concurrency, throughput or production HTTP p95.

## Rejected CPU experiment

The earlier [`c47224b`](https://github.com/A-K-6/rust-embedding-runtime/tree/c47224b92bbb79ef83be2ccb0b366ecc93de1879)
build enabled runtime-dispatched AVX-512 GEMM. On the tested AMD CPU it regressed
query median to about 95 ms versus 48–51 ms for the references. That option was
removed; it is **not** in the released image. The
[raw experiment runs](experiments/c47224b/) are retained separately, including
noisy Mac runs, and are excluded from the current aggregate. No speedup is
inferred simply from enabling newer instructions or fused kernels.

The next performance acceptance target is a repeatable AMD latency win in both
orders, including longer inputs, alongside the retained memory advantage and
numerical gates. The present results do not close that target.

## Published image verification

`ghcr.io/a-k-6/rust-embedding-runtime:0.2.2` is a public multi-platform image
with the actual pinned text model bundled. Both native GitHub runner jobs
passed 18 Rust tests, Clippy and real-model API smoke tests before publication:
[successful release workflow](https://github.com/A-K-6/rust-embedding-runtime/actions/runs/39672712134).

The combined tag was then pulled using **empty Docker authentication
configuration** and tested on the Apple M2's native Linux ARM64 Docker daemon
and the AMD laptop's native Linux AMD64 Docker daemon. Both passed real-model
health/auth, ordered-batch, normalized output, float/base64 agreement,
dimension and invalid/overlong-input checks at two CPUs and 3 GiB limits.
The tests use the bundled model without a model volume. These are functional
checks; they are not container latency benchmarks.

[Platform manifest and immutable digest](image-manifest-0.2.2.json),
[ARM64 verification](image-verification-arm64.json),
[AMD64 verification](image-verification-amd64.json).
