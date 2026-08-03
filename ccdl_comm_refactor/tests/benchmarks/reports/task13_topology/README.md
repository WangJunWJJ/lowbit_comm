# Task 13 A6000 Topology Executor Benchmark

## Scope

This report validates the Task 13 pipelined ring and tree executors against
the existing compressed all-gather path and native PyTorch/NCCL. It also
records compressed reduce-scatter latency as a sharded-consumer reference.
ReducedShard timings are not a drop-in full-output DDP comparison because the
final full-gradient all-gather is intentionally absent.

## Environment and method

- Host: five-GPU NVIDIA RTX A6000 server; 2 and 4 local ranks were selected.
- Container: `ccdl-comm-a6000:cu126-torch25`.
- PyTorch: `2.5.0a0+872d972e41.nv24.08`.
- Data type: FP16; compression: INT8, group size 64.
- Buckets: 1, 16, and 64 MiB per rank.
- Timing: 10 warmups, then five rounds of 30 operations. Every sample is the
  maximum elapsed time across all ranks; the table reports the sample median.
- Native reference: FP16 `all_reduce(SUM)` followed by in-place division by
  world size. Full-output CCDL paths produce the same mean-reduced layout.
- Accuracy gate: relative L2 error against native FP16 must be below 0.1.

Reproduction command for each world-size/bucket pair:

```bash
torchrun --standalone --nproc-per-node=${WORLD_SIZE} \
  tests/distributed/topology_executor_perf.py \
  --numel=${NUMEL} --warmup=10 --repeat=30 --rounds=5 \
  --output-json=tests/benchmarks/reports/task13_topology/raw/${WORLD_SIZE}gpu_${NUMEL}.json
```

## Full-output latency

Lower latency is better. Speedup is native FP16 latency divided by ring
latency.

| GPUs | Bucket | Native FP16 ms | Compressed all-gather ms | Pipelined ring ms | Tree ms | Ring speedup vs native | Ring speedup vs all-gather |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1 MiB | 0.182 | 0.361 | 1.075 | 0.678 | 0.17x | 0.34x |
| 2 | 16 MiB | 2.492 | 1.811 | 1.533 | 2.050 | 1.63x | 1.18x |
| 2 | 64 MiB | 9.836 | 7.001 | 5.459 | 7.692 | 1.80x | 1.28x |
| 4 | 1 MiB | 0.252 | 0.490 | 2.701 | 0.979 | 0.09x | 0.18x |
| 4 | 16 MiB | 3.596 | 4.422 | 2.702 | 5.419 | 1.33x | 1.64x |
| 4 | 64 MiB | 14.291 | 17.179 | 9.500 | 20.710 | 1.50x | 1.81x |

## Sharded-output reference

These results apply only when the consumer can directly consume a
`ReducedShard` and therefore avoids reconstructing the full gradient.

| GPUs | Bucket | Compressed ReducedShard ms | Speedup vs native full-output reference |
|---:|---:|---:|---:|
| 2 | 1 MiB | 0.602 | 0.30x |
| 2 | 16 MiB | 0.873 | 2.86x |
| 2 | 64 MiB | 3.241 | 3.03x |
| 4 | 1 MiB | 0.751 | 0.34x |
| 4 | 16 MiB | 1.381 | 2.60x |
| 4 | 64 MiB | 5.234 | 2.73x |

Across all cases and compressed strategies, the maximum relative L2 error was
`0.002487`, below the `0.1` gate.

## Decision for Task 14

- Do not select a compressed topology for the 1 MiB class. Launch,
  quantization, and P2P scheduling overhead dominate communication savings.
- Admit pipelined ring as an auto-strategy candidate for validated FP16 A6000
  buckets at or above 16 MiB for both 2 and 4 ranks.
- Do not admit tree based on this data; it lost to native NCCL and ring at every
  measured point.
- Admit compressed reduce-scatter only for `output_layout="shard"` consumers
  at or above the validated 16 MiB class. Never select it for a full-output
  request without separately accounting for reconstruction cost.
- Treat these as A6000-specific evidence. Task 14 must retain a safe fallback
  for unknown devices, dtypes, rank counts, and unmeasured bucket classes.

The raw JSON files in [`raw`](raw) contain all per-round samples, environment
metadata, speedups, and accuracy values.
