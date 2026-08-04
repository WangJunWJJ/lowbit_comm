# Fused EF Synthetic Benchmark

Date: 2026-07-25  
Host: `user@192.168.8.156:360`  
GPU: NVIDIA RTX A6000  
Image: `ccdl-comm-a6000:cu126-torch25`  
Commit: `4151e7a`

This is a synthetic DDP communication-pressure benchmark, not a real model/data training result.

## Configuration

- Script: `tests/distributed/synthetic_ddp_compare.py`
- Model: synthetic MLP
- Parameters: 46,137,344
- Steps: 40
- Warmup: 10
- Batch size per rank: 8
- Precision: fp32
- Bucket cap: 25 MB
- CCDL strategy: `all_gather`
- Compression: int8, group size 64

## Results

| GPUs | Mode | samples/s | avg step ms | loss | peak memory MB |
|---:|---|---:|---:|---:|---:|
| 2 | async no EF | 1371.26 | 11.668 | 0.9989066854 | 1130.7 |
| 2 | async EF safe | 1038.20 | 15.411 | 0.9989066862 | 1251.7 |
| 2 | async EF fused | 1093.10 | 14.637 | 0.9989066862 | 1306.7 |
| 4 | async no EF | 1269.71 | 25.203 | 0.9995085482 | 1318.2 |
| 4 | async EF safe | 1102.70 | 29.020 | 0.9995085601 | 1306.2 |
| 4 | async EF fused | 1142.18 | 28.017 | 0.9995085601 | 1494.2 |

## Interpretation

The fused EF path improves throughput relative to the previous safe EF path:

- 2 GPUs: `1093.10 / 1038.20 - 1 = +5.29%`
- 4 GPUs: `1142.18 / 1102.70 - 1 = +3.58%`

Loss matches the safe EF path for this short synthetic run. Peak memory is higher because the current DDP path still allocates restored workspace per bucket instead of reusing a persistent bucket workspace.

The next optimization target is persistent workspace reuse and CUDA event/Future completion semantics, not additional Python-level wrapping.
