# Exploratory Rust OpenMP wait-policy probe

**Rust scheduling screening only; this is not a new PyTorch comparison or a headline performance claim.**

Status: **complete**. CPU: AMD Ryzen 9 9955HX 16-Core Processor.

Default explicitly unsets OMP_WAIT_POLICY and GOMP_SPINCOUNT. Passive sets PASSIVE and 0. Each thread count runs both fresh-process orders, default → passive and passive → default. Idle OpenMP workers competing with Rayon is a hypothesis; no speedup is assumed.

| Run | Threads | Order | Policy | Engine median / p95 (ms) | Wall median / p95 (ms) | Short / post-long / peak RSS (MiB) | 128 / 512 / 1,024 token wall medians (ms) |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 2 | 1 | default | 24.98 / 31.67 | 25.24 / 31.92 | 1121.9 / 1131.2 / 1150.2 | 131.88 / 605.07 / 1451.26 |
| 2 | 2 | 1 | passive | 22.64 / 27.97 | 22.94 / 28.21 | 1122.2 / 1123.6 / 1150.9 | 102.74 / 505.24 / 1326.19 |
| 3 | 2 | 2 | passive | 23.36 / 28.69 | 24.04 / 29.38 | 1122.3 / 1147.6 / 1195.1 | 104.69 / 514.43 / 1348.67 |
| 4 | 2 | 2 | default | 25.07 / 31.73 | 25.76 / 32.54 | 1121.6 / 1139.3 / 1186.7 | 132.82 / 639.09 / 1511.51 |
| 5 | 4 | 1 | default | 27.83 / 52.07 | 28.53 / 52.74 | 1122.0 / 1123.0 / 1150.4 | 223.11 / 531.50 / 1168.48 |
| 6 | 4 | 1 | passive | 19.59 / 22.40 | 20.30 / 23.15 | 1122.4 / 1169.4 / 1216.8 | 69.14 / 301.29 / 811.52 |
| 7 | 4 | 2 | passive | 19.70 / 22.42 | 20.38 / 23.15 | 1138.5 / 1163.6 / 1211.0 | 69.95 / 309.58 / 805.26 |
| 8 | 4 | 2 | default | 26.23 / 49.49 | 26.87 / 50.21 | 1122.0 / 1147.0 / 1194.3 | 312.43 / 551.91 / 1187.25 |
| 9 | 8 | 1 | default | 31.00 / 174.10 | 31.54 / 174.68 | 1121.9 / 1132.0 / 1149.5 | 635.89 / 611.30 / 938.92 |
| 10 | 8 | 1 | passive | 18.71 / 20.75 | 19.11 / 21.25 | 1122.3 / 1124.4 / 1150.0 | 52.19 / 209.65 / 536.55 |
| 11 | 8 | 2 | passive | 18.73 / 20.83 | 19.23 / 21.18 | 1122.3 / 1132.4 / 1150.3 | 52.40 / 209.18 / 541.33 |
| 12 | 8 | 2 | default | 31.85 / 155.43 | 32.35 / 155.95 | 1122.3 / 1132.4 / 1149.6 | 706.14 / 535.43 / 960.24 |
| 13 | 16 | 1 | default | 294.68 / 1180.85 | 295.59 / 1181.94 | 1122.4 / 1142.9 / 1149.0 | 1245.05 / 1618.40 / 1696.78 |
| 14 | 16 | 1 | passive | 25.48 / 29.08 | 25.69 / 29.36 | 1122.5 / 1158.7 / 1204.1 | 72.46 / 262.85 / 535.31 |
| 15 | 16 | 2 | passive | 25.60 / 30.48 | 25.76 / 30.72 | 1122.7 / 1143.1 / 1149.1 | 73.50 / 256.64 / 623.06 |
| 16 | 16 | 2 | default | 180.20 / 1159.11 | 180.77 / 1159.94 | 1122.3 / 1151.0 / 1196.6 | 1337.95 / 1461.00 / 1793.07 |

Every completed worker has 100 raw short-query timings, five warmups and ten document encodes excluded from query latency. Long inputs use one warmup and three timed samples per length. All 33-vector and 100-ranking agreement uses the first default Rust worker; the small 20-label retrieval pilot does not establish general relevance. Engine time excludes tokenization and IPC; wall time includes tokenization, JSONL IPC and serialization. Neither includes ranking or HTTP. Actual worker masks, physical-core/cache topology and whitelisted process environment are saved in wait-policy.json. Cross-cache allocation, shared-host interference and frequency changes remain uncontrolled.
