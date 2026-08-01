# Topology transport migration validation, 2026-07-29

This report validates the first performance-first migration step: exposing the
legacy CCDL topology-aware tree/p2p algorithms through the refactored
`ccdl_comm` API as `strategy="topology"`.

## Environment and workload

- Host: `user@192.168.8.156 -p 360`
- GPU: NVIDIA RTX A6000
- Docker image: `ccdl-comm-a6000:cu126-torch25`
- PyTorch: `2.5.0a0+872d972e41.nv24.08`
- CUDA: `12.6`
- Model: FP16 synthetic MLP
- Parameters: `62,914,560`
- `input_dim=2048`
- `width=4096`
- `depth=4`
- `output_dim=1024`
- `batch_size_per_rank=16`
- `steps=20`
- `warmup_steps=5`
- Compression: INT8, group size 64, top-k 0

## Results

| Version / path | GPUs | Bucket cap MB | Avg step ms | Samples/s | Train loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Legacy CCDL tree | 2 | flat manual sync | 13.247 | 2415.72 | 0.9986604303 |
| Refactor topology | 2 | 25 | 13.998 | 2285.97 | 0.9986624777 |
| Refactor topology | 2 | 512 | 13.622 | 2349.14 | 0.9986624777 |
| Legacy CCDL p2p | 4 | flat manual sync | 20.808 | 3075.72 | 0.9972144723 |
| Refactor topology | 4 | 25 | 22.180 | 2885.52 | 0.9972114623 |
| Refactor topology | 4 | 512 | 20.926 | 3058.39 | 0.9972114623 |

## Interpretation

- With the default 25 MiB DDP bucket, the refactored topology path already
  recovers most of the legacy performance:
  - 2 GPUs: `2285.97 / 2415.72 = 94.6%` of legacy throughput.
  - 4 GPUs: `2885.52 / 3075.72 = 93.8%` of legacy throughput.
- With a larger 512 MiB bucket, the口径 becomes closer to the legacy flat manual
  gradient sync:
  - 2 GPUs: `2349.14 / 2415.72 = 97.2%` of legacy throughput.
  - 4 GPUs: `3058.39 / 3075.72 = 99.4%` of legacy throughput.

This confirms the migration direction: keep the refactored API/contract and
bring the legacy topology-aware communication algorithms underneath it. The next
step should internalize these algorithms into native `ccdl_comm` transport code
instead of importing the legacy `ccdl` package, then make `auto` select topology
for replicated DDP when benchmarks indicate it is fastest.
