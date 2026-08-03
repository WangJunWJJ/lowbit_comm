# Task 11: Fused dequant-reduce-mean-EF executor benchmark

## Environment

- Host: five NVIDIA RTX A6000 GPUs (48 GiB each)
- Validation sizes: 2 GPUs and 4 GPUs
- Container: `ccdl-comm-a6000:cu126-torch25`
- PyTorch: `2.5.0a0+872d972e41.nv24.08`
- CUDA runtime: 12.6
- Payload: FP16 source, INT8 linear quantization, group size 64
- Measurement: five independent processes per case; each process uses 30 warmup and 150 measured iterations per segment
- Order control: each process measures baseline/fused/fused/baseline and averages each pair
- Reported value: median of the five per-process balanced mean latencies

## Like-for-like Task 11 result

Both paths reuse preallocated `prepared`, quantized send, gathered receive,
output, and residual buffers. They execute the same fused quantization and NCCL
all-gather. The baseline restores with dequant-reduce, in-place mean, and EF
update kernels; the candidate replaces that restore chain with the single
Executor-bound dequant-reduce-mean-EF kernel.

| GPUs | Bucket | Baseline ms | Fused ms | Speedup | Latency reduction |
|---:|---:|---:|---:|---:|---:|
| 2 | 1 MiB | 0.176920 | 0.138936 | 1.2734x | 21.47% |
| 2 | 16 MiB | 1.807544 | 1.717013 | 1.0527x | 5.01% |
| 2 | 64 MiB | 7.051547 | 6.702629 | 1.0521x | 4.95% |
| 4 | 1 MiB | 0.288130 | 0.284133 | 1.0141x | 1.39% |
| 4 | 16 MiB | 4.213381 | 4.122720 | 1.0220x | 2.15% |
| 4 | 64 MiB | 16.521800 | 16.174408 | 1.0215x | 2.10% |

All 30 measured runs used `cuda_fused_dequant_reduce_mean_ef`, reported no
fallback, and had zero steady-state CUDA allocation growth. The maximum FP16
relative L2 error was `0.00594828`; fused and baseline errors were identical in
every run.

The smaller four-GPU percentage is expected for this all-gather transport: as
world size grows, NCCL payload exchange accounts for more of the end-to-end
latency while Task 11 only removes restore-side launches. The 16 MiB and 64 MiB
performance gates remain positive for both world sizes.

## Comparison with the frozen Task 0 CCDL baseline

The table below compares the fused candidate with the frozen
`gpu_first_baseline` CCDL all-gather-reduce results. This comparison includes
the cumulative effect of preallocated workspaces and later codec improvements,
so it must not be attributed to Task 11 alone; the like-for-like table above is
the isolated Task 11 measurement.

| GPUs | Bucket | Task 0 CCDL ms | Task 11 fused ms | Cumulative speedup | Latency reduction |
|---:|---:|---:|---:|---:|---:|
| 2 | 1 MiB | 0.383788 | 0.138936 | 2.7623x | 63.80% |
| 2 | 16 MiB | 1.808861 | 1.717013 | 1.0535x | 5.08% |
| 2 | 64 MiB | 6.993455 | 6.702629 | 1.0434x | 4.16% |
| 4 | 1 MiB | 0.447938 | 0.284133 | 1.5765x | 36.57% |
| 4 | 16 MiB | 4.436925 | 4.122720 | 1.0762x | 7.08% |
| 4 | 64 MiB | 17.239680 | 16.174408 | 1.0659x | 6.18% |

Raw results are stored in [`raw`](raw). Reproduce one case with:

```bash
torchrun --standalone --nproc-per-node=4 \
  tests/distributed/fused_dequant_executor_perf.py \
  --bucket-mib=64 --dtype=fp16 --warmup=30 --repeat=150 \
  --output-json=tests/benchmarks/reports/task11_fused_dequant/raw/result.json
```
