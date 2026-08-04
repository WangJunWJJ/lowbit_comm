# Task 12 compressed reduce-scatter gate

Environment: 4 × NVIDIA RTX A6000, PyTorch 2.5 nightly, CUDA 12.6,
NCCL single-node process group. Both candidates use INT8, group size 64, and
the same compile-once production Executor, immutable ChunkPlan, workspace pool,
and compressed all-to-all shard exchange. The baseline only adds the final
full-precision all-gather. Measurements use the balanced
`full-shard-shard-full` order with 20 warm-up and 100 measured iterations per
position. Send/receive workspaces were reused; reduced output pooling and the
fused dequant-reduce path were not active in this Task 12 measurement.

| FP16 bucket | Compressed full restore | ReducedShard output | Speedup | Peak memory reduction | Relative L2 |
|---:|---:|---:|---:|---:|---:|
| 16 MiB | 2.9880 ms | 1.2506 ms | 2.389× | 33.14% | 0.005944 |
| 64 MiB | 12.2305 ms | 5.0324 ms | 2.430× | 33.14% | 0.005942 |

The ReducedShard path skips the final full-precision all-gather. Every rank
received one compressed target-shard payload from each source rank, and no
rank allocated a `world_size × full_payload` receive list. Rank metadata was
gathered after correctness validation: all invariant fields matched, shard
indices covered ranks 0–3, and both accuracy runs stayed below the approved
INT8 relative-L2 threshold of 0.02.

An additional 4-GPU uneven-shape smoke used 8,388,611 elements. It produced
8,388,612 padded elements, a 2,097,153-element shard on every rank, and exactly
one padding element on rank 3 while preserving the same 0.005944 relative L2.

Raw results:

- `raw/4gpu_16mib.json`
- `raw/4gpu_64mib.json`
- `raw/4gpu_uneven_smoke.json`
