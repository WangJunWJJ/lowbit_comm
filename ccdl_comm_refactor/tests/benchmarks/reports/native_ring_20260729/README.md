# Native ring topology validation, 2026-07-29

This report records the migration of legacy CCDL `qall_reduce(method="ring")`
semantics into the native `ccdl_comm` topology transport. The refactored API can
now force `topology_method="ring"` for regression/performance tests, and the
default topology selector uses:

- 2 GPUs: `tree`
- 4+ GPUs: `ring`

The explicit `topology_method="p2p"` path remains available for old behavior
checks.

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
- Bucket cap: `512 MiB`
- Compression: INT8, group size 64, top-k 0

## Results

| Version / path | GPUs | Topology method | Avg step ms | Samples/s | Train loss |
| --- | ---: | --- | ---: | ---: | ---: |
| Legacy CCDL baseline | 2 | tree | 13.247 | 2415.72 | 0.9986604303 |
| Native topology auto | 2 | auto -> tree | 13.241 | 2416.72 | 0.9986624777 |
| Native topology forced ring | 2 | ring | 14.351 | 2229.82 | 0.9986624777 |
| Legacy CCDL baseline | 4 | p2p | 20.808 | 3075.72 | 0.9972144723 |
| Native topology forced p2p | 4 | p2p | 21.017 | 3045.11 | 0.9972114623 |
| Native topology forced ring | 4 | ring | 19.146 | 3342.71 | 0.9972114623 |
| Native topology auto | 4 | auto -> ring | 19.138 | 3344.14 | 0.9972114623 |

## Interpretation

- On 2 GPUs, tree remains the best tested topology:
  - Native auto reaches `2416.72 / 2415.72 = 100.0%` of the legacy tree
    baseline.
  - Forced ring is slower on 2 GPUs, so it is not selected by default.
- On 4 GPUs, ring is faster than the previous p2p default on the tested A6000
  topology:
  - Native auto reaches `3344.14 / 3075.72 = 108.7%` of the legacy p2p
    baseline.
  - Forced ring is also `3342.71 / 3045.11 = 109.8%` of the native forced p2p
    path measured in this run family.
- Training loss remains aligned across tree, p2p, and ring at this short
  synthetic scale, indicating the topology change preserves replicated gradient
  mean semantics for this workload.

This completes one more old-functionality migration step while improving the
default topology choice for 4-GPU A6000 DDP benchmarks.
