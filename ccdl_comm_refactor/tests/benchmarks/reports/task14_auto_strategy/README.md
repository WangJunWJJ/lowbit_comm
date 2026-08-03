# Task 14 A6000 Compile-Time Auto Strategy Gate

## Scope

Task 14 converts the checked-in Task 13 A6000 evidence into an immutable,
compile-time strategy policy. The policy never changes collective or output
layout semantics:

- `all_reduce/full` may select native NCCL or full-output topology.
- `reduce_scatter/shard` may select the compressed ReducedShard transport.
- Explicit strategies bypass the selector.
- Unknown dimensions select a semantically safe path with an explainable
  reason. No runtime strategy lookup occurs in `CompiledCommunicationPlan.run`.

## Policy boundary

Policy ID: `cuda-task13-a6000-v1`.

The benchmark-backed fast-path region is restricted to NVIDIA RTX A6000,
FP16, exact world size 2 or 4, the complete default linear INT8/group-64
compression profile, ring-aligned tensors, and 16–64 MiB buckets. Values
outside that region do not inherit the measured speedup.

The 16–64 MiB expected values use the conservative minimum of the two Task 13
endpoints. ReducedShard observations retain an explicit
`native_fp16_full_output_reference` baseline and are not represented as a
same-semantics native reduce-scatter speedup.

## A6000 Gate G6 method

- Container: `ccdl-comm-a6000:cu126-torch25`.
- PyTorch: `2.5.0a0+872d972e41.nv24.08`.
- World sizes: 2 and 4 local NCCL ranks.
- Small full bucket: 524,288 FP16 elements (1 MiB), expected `native_nccl`.
- Large full bucket: 8,388,608 FP16 elements (16 MiB), expected `topology`
  backed by the pipelined ring executor.
- Large shard bucket: 8,388,608 FP16 elements, expected `compressed`.
- Timing: 10 warmups; three samples of 30 operations; each sample uses the
  maximum latency across ranks and the table reports the median.
- Gates: expected concrete strategy, `fallback_used=false`, and relative L2
  below 0.1.

Example reproduction:

```bash
torchrun --standalone --nproc-per-node=${WORLD_SIZE} \
  tests/distributed/auto_strategy_smoke.py \
  --numel=${NUMEL} --output-layout=${LAYOUT} \
  --expect-strategy=${EXPECTED} --warmup=10 --repeat=30 \
  --output-json=tests/benchmarks/reports/task14_auto_strategy/raw/${CASE}.json
```

## Results

| GPUs | Case | Selected strategy | Fallback | Native full ms | Auto ms | Ratio vs native full | Relative L2 |
|---:|---|---|---|---:|---:|---:|---:|
| 2 | 1 MiB full | `native_nccl` | no | 0.179 | 0.179 | 1.000x | 0.000000 |
| 2 | 16 MiB full | `topology` | no | 2.421 | 1.531 | 1.582x | 0.001912 |
| 2 | 16 MiB shard | `compressed` | no | 2.448 | 0.873 | 2.804x | 0.002100 |
| 4 | 1 MiB full | `native_nccl` | no | 0.252 | 0.251 | 1.005x | 0.000000 |
| 4 | 16 MiB full | `topology` | no | 3.532 | 2.704 | 1.306x | 0.002459 |
| 4 | 16 MiB shard | `compressed` | no | 3.545 | 1.377 | 2.575x | 0.001824 |

All six cases selected the expected concrete backend without fallback. The
maximum relative L2 was `0.002459`. Small-bucket compiled native NCCL stayed
within roughly 0.5% of the direct native measurement in this run.

The shard ratios compare a ReducedShard result with a full-output native NCCL
all-reduce reference and therefore describe the benefit available only to a
consumer that directly accepts the shard.

Raw per-run evidence is stored in [`raw`](raw).
