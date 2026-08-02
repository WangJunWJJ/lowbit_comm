# Task 12.1 Fused ReducedShard A6000 Gate

This directory is intentionally a template until the authoritative A6000 run
is complete. Do not add estimated, copied, or synthetic benchmark results.

## Scope

- Device matrix: 2 and 4 RTX A6000 GPUs only.
- Configuration: FP16, INT8 linear quantization, group size 64, top-k 0.
- Buckets: 1 MiB, 16 MiB, and 64 MiB.
- Output ownership: `caller` and `lease`.
- Repetitions: five independent `torchrun` invocations for each of the twelve
  `(world_size, bucket_mib, output_mode)` cases.
- Per invocation: warmup 20, repeat 100, ABBA order
  `task12-fused-fused-task12`.

## Reproduction

Build the CUDA extension from the source revision being measured, then run one
command for every case/run index inside `ccdl-comm-a6000:cu126-torch25` with
`--gpus all --shm-size=8g`:

```bash
torchrun --standalone --nproc-per-node=${WORLD_SIZE} \
  tests/distributed/fused_reduced_shard_perf.py \
  --bucket-mib=${BUCKET_MIB} --dtype=fp16 --bit=8 --group-size=64 \
  --mode=${MODE} --warmup=20 --repeat=100 \
  --output-json=tests/benchmarks/reports/task12_1_fused_reduced_shard/raw/${WORLD_SIZE}gpu_${BUCKET_MIB}mib_${MODE}_run${RUN}.json

python tests/benchmarks/fused_reduced_shard_gate.py \
  --results-dir tests/benchmarks/reports/task12_1_fused_reduced_shard/raw
```

## Evidence to Record

Each raw JSON is generated only by rank 0 and contains source/host identity,
per-position timing and peak-memory samples, fused callback metadata, one-kernel
profiler evidence, pointer samples, steady allocation evidence, and accuracy
against the separately computed FP16 all-reduce reference.

Gate G5a passes only when every case has exactly five raw files, no candidate
falls back, every candidate observes exactly one production fused
`dequant_reduce_fused_*` launch, steady allocation is zero, caller and lease
accuracy match, and neither 16 MiB nor 64 MiB fused median regresses against
its same-run Task 12 median.

## Authoritative Result

Status: awaiting A6000 matrix execution. Raw result files and a measured
median summary must be added by the validation owner after the gate exits zero.
