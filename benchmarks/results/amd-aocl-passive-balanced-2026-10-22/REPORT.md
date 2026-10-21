# AMD64 CPU matrix

Native AMD64 CPU; same frozen public fixture; batch one; two kernel threads; six sequential orders. No core reservation, fixed frequency, HTTP or database in these timings.

CPU: AMD Ryzen 9 9955HX 16-Core Processor. Each backend has 600 measured short-query requests; each long-input length has 18.

Runtime wait environment: {"GOMP_SPINCOUNT": "0", "OMP_DYNAMIC": null, "OMP_PROC_BIND": null, "OMP_WAIT_POLICY": "PASSIVE"}.

| Backend | Engine median / p95 (ms) | Wall median / p95 (ms) | Short RSS / post-long RSS / peak RSS (MiB) |
| --- | ---: | ---: | ---: |
| PyTorch eager | 50.86 / 61.09 | 52.13 / 62.50 | 1630.9 / 1693.7 / 1888.2 |
| PyTorch SDPA | 50.21 / 60.24 | 51.46 / 61.74 | 1631.0 / 1669.8 / 1888.1 |
| Rust F32 | 23.32 / 28.84 | 23.93 / 29.51 | 1138.7 / 1164.0 / 1211.4 |

| Backend | 128 tokens engine / wall median (ms) | 512 tokens engine / wall median (ms) | 1,024 tokens engine / wall median (ms) |
| --- | ---: | ---: | ---: |
| PyTorch eager | 178.96 / 180.46 | 726.86 / 728.38 | 1758.49 / 1760.00 |
| PyTorch SDPA | 177.73 / 179.05 | 724.59 / 725.91 | 1683.36 / 1684.91 |
| Rust F32 | 104.18 / 104.87 | 512.52 / 513.31 | 1345.54 / 1346.62 |

| Order | Sequence | Eager engine / wall median / p95 (ms) | SDPA engine / wall median / p95 (ms) | Rust engine / wall median / p95 (ms) |
| --- | --- | ---: | ---: | ---: |
| 1 | PyTorch eager, PyTorch SDPA, Rust F32 | 50.22 / 60.92; 51.70 / 62.29 | 50.39 / 60.59; 51.85 / 61.89 | 23.54 / 28.91; 24.25 / 29.58 |
| 2 | PyTorch eager, Rust F32, PyTorch SDPA | 50.90 / 60.05; 52.14 / 61.60 | 50.50 / 60.88; 51.71 / 62.23 | 23.26 / 28.46; 23.50 / 28.72 |
| 3 | PyTorch SDPA, PyTorch eager, Rust F32 | 50.76 / 61.01; 52.06 / 62.33 | 48.94 / 59.86; 50.26 / 61.16 | 23.63 / 29.16; 24.30 / 29.76 |
| 4 | PyTorch SDPA, Rust F32, PyTorch eager | 50.79 / 61.29; 52.07 / 62.82 | 50.79 / 59.66; 52.01 / 61.74 | 23.39 / 28.60; 24.07 / 29.29 |
| 5 | Rust F32, PyTorch eager, PyTorch SDPA | 51.61 / 61.35; 52.83 / 62.66 | 48.91 / 58.00; 50.24 / 59.34 | 23.22 / 28.63; 23.90 / 29.15 |
| 6 | Rust F32, PyTorch SDPA, PyTorch eager | 50.34 / 66.00; 51.71 / 66.58 | 50.21 / 60.60; 51.59 / 61.89 | 23.22 / 28.82; 23.90 / 29.55 |

Short-query speed and memory goal versus both baselines, including every order: **True**.
Faster at every reported long-input metric versus both: **True**.

Quality gates every run. Speed and memory goal requires lower engine AND wall median/p95 and all three memory metrics versus BOTH baselines, aggregated and in every order. Long-input comparisons are reported separately; no statistical significance claim.

Process RSS after short queries and after long inputs; kernel peak RSS after long inputs. Aggregates use the largest of six samples, not sums or averages.

The 20 authored retrieval labels are a small correctness pilot. Repeating them does not create additional relevance labels. Results describe this CPU and software build; they do not establish performance on every consumer or server CPU.

Raw runs: [run-01.json](run-01.json), [run-02.json](run-02.json), [run-03.json](run-03.json), [run-04.json](run-04.json), [run-05.json](run-05.json), [run-06.json](run-06.json).
