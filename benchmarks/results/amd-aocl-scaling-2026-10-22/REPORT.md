# Exploratory CPU thread scaling

**Secondary screening: one rotated backend order per thread setting. This does not replace the six-order primary benchmark.**

Status: **complete**. CPU: AMD Ryzen 9 9955HX 16-Core Processor.

Each completed setting uses 20 authored multilingual queries, 100 measured requests per backend, and identical explicit 128/512/1,024-token inputs with three measurements per length. Thread count and physical-core allocation change together. No fixed frequency or exclusive core reservation; cross-cache-domain settings are reported, not hidden.

| Threads | Backend | Engine median / p95 (ms) | Wall median / p95 (ms) | Short / post-long / peak RSS (MiB) | 128 / 512 / 1,024 token wall medians (ms) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | PyTorch eager | 71.03 / 89.04 | 72.42 / 90.34 | 1621.0 / 1621.1 / 1887.9 | 316.05 / 1346.03 / 3244.98 |
| 1 | PyTorch SDPA | 70.75 / 88.38 | 72.03 / 89.75 | 1621.1 / 1645.3 / 1887.9 | 316.78 / 1370.41 / 3190.64 |
| 1 | Rust F32 | 33.58 / 42.48 | 34.26 / 43.23 | 1122.1 / 1164.1 / 1212.1 | 189.52 / 935.61 / 2419.86 |
| 2 | PyTorch SDPA | 48.70 / 59.11 | 50.00 / 60.45 | 1630.9 / 1645.7 / 1888.3 | 178.36 / 723.56 / 1682.02 |
| 2 | Rust F32 | 25.02 / 31.49 | 25.70 / 32.16 | 1138.7 / 1156.0 / 1203.5 | 133.52 / 653.59 / 1492.92 |
| 2 | PyTorch eager | 48.99 / 58.70 | 50.25 / 59.63 | 1630.7 / 1645.4 / 1887.8 | 177.60 / 728.13 / 1769.89 |
| 4 | Rust F32 | 28.51 / 51.02 | 29.20 / 51.67 | 1122.2 / 1123.3 / 1150.4 | 284.17 / 616.07 / 1158.12 |
| 4 | PyTorch eager | 33.19 / 39.69 | 34.57 / 41.01 | 1634.0 / 1672.5 / 1888.2 | 124.41 / 498.45 / 1212.08 |
| 4 | PyTorch SDPA | 32.82 / 39.34 | 34.13 / 40.86 | 1634.1 / 1648.3 / 1887.3 | 125.99 / 501.35 / 1161.73 |
| 8 | PyTorch eager | 25.49 / 28.54 | 26.44 / 29.46 | 1647.4 / 1698.5 / 1887.9 | 65.71 / 243.29 / 615.68 |
| 8 | PyTorch SDPA | 25.01 / 28.28 | 25.93 / 29.19 | 1647.4 / 1698.6 / 1887.6 | 66.09 / 243.87 / 593.52 |
| 8 | Rust F32 | 29.78 / 109.73 | 30.24 / 110.23 | 1121.8 / 1148.2 / 1195.1 | 290.70 / 590.53 / 1015.61 |
| 16 | PyTorch SDPA | 35.85 / 39.83 | 36.48 / 40.47 | 1649.3 / 1732.9 / 1888.1 | 73.94 / 197.57 / 409.81 |
| 16 | Rust F32 | 254.51 / 969.50 | 255.45 / 970.72 | 1122.6 / 1127.0 / 1150.6 | 1130.95 / 1609.15 / 1705.22 |
| 16 | PyTorch eager | 35.33 / 38.54 | 35.96 / 39.19 | 1649.9 / 1732.9 / 1888.4 | 78.95 / 210.93 / 543.56 |

1 threads: quality status **complete**, worker CPUs [0]; LLC domains [[0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23]]; order PyTorch eager, PyTorch SDPA, Rust F32.


2 threads: quality status **complete**, worker CPUs [0, 1]; LLC domains [[0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23]]; order PyTorch SDPA, Rust F32, PyTorch eager.


4 threads: quality status **complete**, worker CPUs [0, 1, 2, 3]; LLC domains [[0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23]]; order Rust F32, PyTorch eager, PyTorch SDPA.


8 threads: quality status **complete**, worker CPUs [0, 1, 2, 3, 4, 5, 6, 7]; LLC domains [[0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23]]; order PyTorch eager, PyTorch SDPA, Rust F32.


16 threads: quality status **complete**, worker CPUs [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]; LLC domains [[0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23], [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31]]; order PyTorch SDPA, Rust F32, PyTorch eager.

Numerical agreement requires 33 vectors/backend, maximum absolute error at most 1e-4 and cosine at least .99999. Every completed setting requires 20/20 authored hit@1/5 plus ordered top-five agreement across all 100 requests. The small authored pilot and repeated measurements do not establish general relevance or performance across all CPUs. Memory is process RSS after short queries and long inputs, plus kernel peak RSS after long inputs.
