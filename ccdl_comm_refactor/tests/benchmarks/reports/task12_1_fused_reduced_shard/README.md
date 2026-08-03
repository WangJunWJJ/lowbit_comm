# Task 12.1 Fused ReducedShard A6000 Gate

This directory contains the authoritative A6000 Gate G5a evidence. All values
below are medians of five independent `torchrun` invocations; the 60 raw JSON
records in `raw/` are the source of truth.

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

Each raw JSON is generated only by rank 0 and contains a rank-zero generated,
broadcast `run_id` and `started_at`, source/host identity, per-position timing
and peak-memory samples, all observed profiler kernel names, fused callback
metadata, per-rank pointer/allocation/accuracy evidence, and accuracy against
the separately computed FP16 all-reduce reference. The Task 12 baseline and
Task 12.1 candidate are each precompiled through the same CUDA compiler,
executor, and workspace-cache layer; only the baseline extension proxy hides
the fused-mean callback.

Gate G5a passes only when every case has exactly five unique run identifiers,
no candidate falls back, every rank observes exactly one production fused
`dequant_reduce_fused_*` launch, steady allocation is zero, caller and lease
accuracy match under the fixed FP16/INT8/group64/seed configuration, and
neither 16 MiB nor 64 MiB fused median regresses against its same-run Task 12
median. `lease` is an API ownership mode: transport metadata continues to
report the raw tensor as caller-owned after the executor unwraps the lease.

## Authoritative Result

Status: **PASSED** on 2026-08-03.

- Source revision measured: `d44ddff`.
- Host: 5 × NVIDIA RTX A6000; the matrix used isolated 2- and 4-process jobs.
- Container: `ccdl-comm-a6000:cu126-torch25`.
- PyTorch: `2.5.0a0+872d972e41.nv24.08`; CUDA: 12.6.
- Gate output: `Task 12.1 fused ReducedShard gate passed for 60 result files`.

| GPUs | Bucket | Mode | Task 12 ms | Task 12.1 ms | Speedup | Max relative L2 |
|---:|---:|:---|---:|---:|---:|---:|
| 2 | 1 MiB | caller | 0.484422 | 0.415086 | 1.1670× | 0.005933 |
| 2 | 1 MiB | lease | 0.523156 | 0.566540 | 0.9234× | 0.005933 |
| 2 | 16 MiB | caller | 0.804150 | 0.776013 | 1.0363× | 0.005940 |
| 2 | 16 MiB | lease | 0.803619 | 0.774469 | 1.0376× | 0.005940 |
| 2 | 64 MiB | caller | 3.006604 | 2.899629 | 1.0369× | 0.005939 |
| 2 | 64 MiB | lease | 3.012825 | 2.902324 | 1.0381× | 0.005939 |
| 4 | 1 MiB | caller | 0.678339 | 0.592297 | 1.1453× | 0.005793 |
| 4 | 1 MiB | lease | 0.691128 | 0.713291 | 0.9689× | 0.005793 |
| 4 | 16 MiB | caller | 1.251003 | 1.240758 | 1.0083× | 0.005789 |
| 4 | 16 MiB | lease | 1.252782 | 1.244022 | 1.0070× | 0.005789 |
| 4 | 64 MiB | caller | 5.050000 | 4.980871 | 1.0139× | 0.005787 |
| 4 | 64 MiB | lease | 5.027462 | 4.979845 | 1.0096× | 0.005787 |

Every rank in every run observed exactly one `dequant_reduce_fused_*` kernel,
zero steady-state CUDA allocation, a stable output pointer, no fallback, and no
non-finite values. Median peak-memory savings were 1/16/64 MiB per rank on the
2-GPU cases and 0.5/8/32 MiB per rank on the 4-GPU cases.

The explicit lease state machine adds visible fixed overhead for 1 MiB buckets:
7.66% on 2 GPUs and 3.11% on 4 GPUs versus the Task 12 baseline. Gate G5a only
forbids regressions for 16/64 MiB, where communication and allocation savings
dominate; all eight large-bucket caller/lease cases passed that requirement.
