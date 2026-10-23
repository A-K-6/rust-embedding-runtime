# Comparing startup cleanup and resident memory

Model and tokenizer loading can leave freed buffers resident in the process
allocator. Releasing those pages once before readiness may reduce steady RSS.
This is an allocator hypothesis, independent of the F32 weights, model revision,
tokenizer, inference kernels and embedding computation. A cleanup does not make
the encoder smaller and does not establish a latency improvement.

`startup_memory.py` compares **two actual `embedding-jsonl` executables**. It uses
no preload shim, no cached answers and no additional runtime dependency beyond
Python's standard library, Linux `/proc`, cgroup v2 and `taskset`.

Build the previous and candidate versions with the same compiler, features and
backend. Supply local pinned model assets. Start the caller in a dedicated
cgroup with **CPU quota 200% and memory maximum 3 GiB**, then choose two distinct
physical worker cores sharing their last-level cache:

```sh
python3 benchmarks/startup_memory.py \
  --previous ./previous/embedding-jsonl \
  --candidate ./candidate/embedding-jsonl \
  --model ./model --cpus 0,1 --output ./startup-memory.json
```

CPU IDs are an example; inspect the machine's topology first. The probe validates
physical cores, each worker thread's affinity, the actual worker cgroup limits,
two-thread F32 CPU readiness and its declared environment. It fixes Rayon and
OpenMP to two threads, `OMP_WAIT_POLICY=PASSIVE`, `GOMP_SPINCOUNT=0`, and removes
inherited preload/audit libraries. It does not reserve cores or control competing
processes, clocks or thermal state. The Python driver shares the quota; record
and consider its affinity when interpreting measurements.

For exploratory runs without these exact resource limits,
`--allow-uncontrolled` records an explicit `uncontrolled` label. Such runs do not
demonstrate behavior under the controlled allocation. The JSON retains actual
limits and resource counters so missing controls are visible.

The fixed fresh-process order is previous/candidate/candidate/previous. Each
trial performs six cycles of the three shipped official F32 reference cases and
one synthetic 128-token ID input: **24 calls per trial, at most 96 overall**.
There is a 300-second overall deadline, a 60-second startup deadline and a
30-second response deadline. Failed trials are retained, never retried; an
overall timeout leaves later trials explicitly unstarted. Every first-use call
is timed. The probe does not download assets or use external services.

For every request the output includes wall and encoder latency, RSS, kernel peak
RSS, `smaps_rollup`, CPU ticks, minor/major faults, cgroup CPU/throttling and memory
counters. Startup wall time, readiness, post-readiness RSS, first-call RSS,
first-cycle RSS and final RSS are separate. **Compare post-readiness and final
RSS for retained-memory savings; retain peak RSS as a separate startup metric.**
A one-time release after loading may leave startup peak unchanged and may add
startup cost or subsequent page faults.

All three model asset hashes and the shipped reference checksum must match.
Every official vector must stay within maximum absolute error `1e-6` and cosine
at least `0.99999999`. Synthetic vectors are compared with the first in-memory
synthetic result, which tests agreement rather than official correctness.
Expected token counts are checked; this is not an exact token-ID or labelled
retrieval-quality test. No texts, token IDs, vectors, full executable paths,
hostnames, worker logs or exception messages are saved. Executable identity is
recorded as basename plus SHA-256. Use distinct artifact names to identify builds.

The JSON reports every trial and scalar check. It makes no speed or memory
acceptance claim automatically. Assess every trial, cold calls, failed requests,
page faults, startup cost and backend/build differences before drawing a
conclusion. These short inputs do not establish maximum-length memory behavior
or broad tail-latency performance.

For the retained October 22 run, build runtime revisions `8683d55` and `297524b`
with Rust 1.98.0, the same pinned AOCL build and `--features aocl,profile`. Use a
separate Cargo target directory for each checkout, for example:

```sh
CARGO_TARGET_DIR=./target-before AOCL_ROOT=/path/to/aocl AOCL_STATIC_RUNTIME=1 \
  RUSTFLAGS='-C target-cpu=x86-64-v3' \
  cargo +1.98.0 build --release --locked --features aocl,profile --bins
```

Use `target-after` in the other checkout. Verify distinct binary hashes before
running. The candidate's JSONL readiness reports `startup_allocator_trimmed`;
the older executable does not have that field. The raw report retains both
hashes and actual readiness rather than relying on a filename as build identity.
