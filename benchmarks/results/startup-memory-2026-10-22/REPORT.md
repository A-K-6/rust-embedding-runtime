# Startup cleanup on Ryzen 9 9955HX

Four fresh processes, previous/candidate/candidate/previous. Same pinned F32 model, two physical worker cores, two threads, a two-core quota and 3 GiB memory limit. The driver shares the quota. All 96 uncached calls and cold calls are retained. Inputs are only the three shipped official reference cases and one synthetic 128-token input, each repeated six times.

| Trial | Build | After readiness MiB | After 24 inputs MiB | Process peak MiB | Official p95 ms |
|---:|---|---:|---:|---:|---:|
| 1 | previous | 1119.94 | 1122.27 | 1150.22 | 31.94 |
| 2 | candidate | 891.96 | 929.81 | 1149.93 | 32.25 |
| 3 | candidate | 892.05 | 924.80 | 1149.89 | 32.28 |
| 4 | previous | 1119.73 | 1121.93 | 1149.64 | 31.91 |

Post-inference RSS fell about 17%, with no demonstrated latency improvement. Startup peak stayed about 1,150 MiB. Cleanup can increase subsequent minor page faults. All numerical checks passed against the official F32 cases (1e-6 maximum-error and 0.99999999 minimum-cosine gates). Repeats are not additional independent reference cases.

Baseline runtime source: `8683d5517ec4758d93e53f17095cde1a1155ac94`. Candidate runtime source: `297524b` (startup cleanup plus phase profiling). Both were built with Rust 1.98.0, `aocl,profile`, the same locked dependencies, x86-64-v3 and pinned AOCL-DLP 5.3.2. Separate clean Cargo target directories were used. Binaries, model, reference, driver and helper hashes are recorded in [result.json](result.json).

This small probe measures retained memory and first-use behavior. It does not establish maximum-length memory, HTTP service performance, retrieval quality, or a new PyTorch comparison. Only Linux/glibc AOCL builds perform this cleanup. Other targets retain their existing behavior.

See [the reproducible protocol](../../STARTUP_MEMORY.md) and [the plot source](../../plot_startup_memory.py).
