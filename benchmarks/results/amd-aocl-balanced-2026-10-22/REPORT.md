# AMD64 CPU matrix

Native AMD64 CPU; same frozen public fixture; batch one; two kernel threads; six sequential orders. No core reservation, fixed frequency, HTTP or database in these timings.

CPU: AMD Ryzen 9 9955HX 16-Core Processor. Each backend has 600 measured short-query requests; each long-input length has 18.

| Backend | Engine median / p95 (ms) | Wall median / p95 (ms) | Short RSS / post-long RSS / peak RSS (MiB) |
| --- | ---: | ---: | ---: |
| PyTorch eager | 49.01 / 58.57 | 50.29 / 59.87 | 1630.9 / 1693.7 / 1888.0 |
| PyTorch SDPA | 48.58 / 58.45 | 49.90 / 59.74 | 1631.0 / 1693.7 / 1888.0 |
| Rust F32 | 25.00 / 31.96 | 25.72 / 32.56 | 1122.2 / 1163.5 / 1211.3 |

| Backend | 128 tokens engine / wall median (ms) | 512 tokens engine / wall median (ms) | 1,024 tokens engine / wall median (ms) |
| --- | ---: | ---: | ---: |
| PyTorch eager | 176.48 / 177.69 | 722.86 / 724.24 | 1755.68 / 1757.02 |
| PyTorch SDPA | 175.87 / 177.31 | 719.13 / 720.34 | 1670.92 / 1672.33 |
| Rust F32 | 132.57 / 133.20 | 647.49 / 648.28 | 1472.67 / 1473.54 |

| Order | Sequence | Eager engine / wall median / p95 (ms) | SDPA engine / wall median / p95 (ms) | Rust engine / wall median / p95 (ms) |
| --- | --- | ---: | ---: | ---: |
| 1 | PyTorch eager, PyTorch SDPA, Rust F32 | 48.33 / 58.19; 49.66 / 59.27 | 48.52 / 58.39; 49.81 / 59.67 | 25.01 / 31.75; 25.74 / 32.53 |
| 2 | PyTorch eager, Rust F32, PyTorch SDPA | 48.79 / 57.92; 50.12 / 59.02 | 48.62 / 57.09; 49.94 / 58.49 | 24.93 / 31.58; 25.63 / 31.93 |
| 3 | PyTorch SDPA, PyTorch eager, Rust F32 | 49.12 / 57.73; 50.42 / 59.20 | 48.09 / 57.43; 49.48 / 58.29 | 25.03 / 33.82; 25.73 / 34.72 |
| 4 | PyTorch SDPA, Rust F32, PyTorch eager | 49.11 / 58.01; 50.25 / 59.33 | 48.59 / 57.48; 49.89 / 58.86 | 24.97 / 33.78; 25.70 / 34.76 |
| 5 | Rust F32, PyTorch eager, PyTorch SDPA | 49.00 / 58.57; 50.29 / 59.87 | 48.89 / 59.10; 50.23 / 60.47 | 25.08 / 31.66; 25.76 / 31.98 |
| 6 | Rust F32, PyTorch SDPA, PyTorch eager | 49.12 / 58.18; 50.51 / 59.84 | 48.72 / 57.91; 49.95 / 59.25 | 25.00 / 32.10; 25.74 / 32.66 |

Short-query speed and memory goal versus both baselines, including every order: **True**.
Faster at every reported long-input metric versus both: **True**.

Quality gates every run. Speed and memory goal requires lower engine AND wall median/p95 and all three memory metrics versus BOTH baselines, aggregated and in every order. Long-input comparisons are reported separately; no statistical significance claim.

Process RSS after short queries and after long inputs; kernel peak RSS after long inputs. Aggregates use the largest of six samples, not sums or averages.

The 20 authored retrieval labels are a small correctness pilot. Repeating them does not create additional relevance labels. Results describe this CPU and software build; they do not establish performance on every consumer or server CPU.

Raw runs: [run-01.json](run-01.json), [run-02.json](run-02.json), [run-03.json](run-03.json), [run-04.json](run-04.json), [run-05.json](run-05.json), [run-06.json](run-06.json).
