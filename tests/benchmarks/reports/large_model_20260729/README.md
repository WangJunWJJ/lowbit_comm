# Large-model and sharded reduce-scatter validation, 2026-07-29

This run checks whether the previous weak 4-GPU scaling was caused by the very
small smoke model, then separately measures the CCDL true sharded consumer path
that avoids restoring a full DDP bucket.

## Environment

- Host: `user@192.168.8.156 -p 360`
- GPU: NVIDIA RTX A6000
- Docker image: `ccdl-comm-a6000:cu126-torch25`
- PyTorch: `2.5.0a0+872d972e41.nv24.08`
- CUDA: `12.6`

## Larger DDP training test

Script: `tests/distributed/synthetic_ddp_compare.py`

Model/config:

- FP16 synthetic MLP
- `input_dim=2048`
- `width=4096`
- `depth=4`
- `output_dim=1024`
- parameters: `62,914,560`
- `batch_size_per_rank=16`
- `steps=20`, `warmup_steps=5`
- CCDL mode: INT8, group size 64, all-gather strategy, async gather, async EF

| GPUs | Mode | Global batch | Avg step ms | Samples/s | Train loss |
| ---: | --- | -----------: | ----------: | --------: | ---------: |
| 2 | PyTorch DDP | 32 | 20.207 | 1583.63 | 0.9986624777 |
| 2 | CCDL all-gather + async EF | 32 | 16.646 | 1922.35 | 0.9986624777 |
| 4 | PyTorch DDP | 64 | 28.440 | 2250.35 | 0.9972114623 |
| 4 | CCDL all-gather + async EF | 64 | 34.671 | 1845.90 | 0.9972114623 |

Observations:

- Larger-model PyTorch DDP scales from 2 to 4 GPUs by about `1.42x`
  (`2250.35 / 1583.63`), better than the tiny smoke model but still below ideal
  `2x`.
- CCDL all-gather + async EF is faster than PyTorch DDP on 2 GPUs in this run,
  but does not scale to 4 GPUs. The final full-bucket semantics and all-gather
  fan-out dominate as world size increases.
- Loss stays finite and matches the baseline for this synthetic deterministic
  setup.

## True sharded communication benchmark

Script: `tests/distributed/sharded_reduce_scatter_perf.py`

This benchmark measures the shard consumer contract directly. CCDL returns only
the local `ReducedShard`; it does not all-gather restored shards back into a
full bucket.

Config:

- tensor numel: `16,777,216`
- dtype: FP16
- CCDL: INT8, group size 64
- `warmup=10`, `repeat=30`

The PyTorch reference in this script performs a full `all_reduce` and slices the
local shard. That represents the replicated-gradient baseline a sharded consumer
wants to avoid.

| GPUs | Torch full reduce + slice ms | CCDL shard ms | CCDL / Torch | Relative L2 |
| ---: | ---------------------------: | ------------: | -----------: | ----------: |
| 2 | 4.945 | 1.542 | 0.312 | 0.0059378 |
| 4 | 7.161 | 2.542 | 0.355 | 0.0059427 |

Observations:

- The true sharded path is materially faster in this benchmark:
  - 2 GPUs: about `3.21x` faster than full reduce + slice.
  - 4 GPUs: about `2.82x` faster than full reduce + slice.
- This validates the architecture direction: 4+ GPU scaling should avoid DDP
  full-bucket restoration and let a sharded consumer consume `ReducedShard`
  directly.
- This is a communication benchmark, not a full optimizer integration. The next
  product step is to connect `ReducedShard` to a real sharded training consumer
  while preserving optimizer state ownership and gradient layout metadata.
