# Workspace Ownership Lifecycle A6000 Benchmark

Date: 2026-07-27  
Host: `user@192.168.8.156:360`  
GPU: 5 x NVIDIA RTX A6000, benchmark used 2 and 4 local ranks  
Image: `ccdl-comm-a6000:cu126-torch25`  
Commit: `bb60a5b`

This is a synthetic DDP communication-pressure benchmark, not a real model/data training result.

## Validation

CUDA extension rebuild:

- `CCDL_COMM_BUILD_CUDA=1`
- `TORCH_CUDA_ARCH_LIST=8.6`
- `MAX_JOBS=1`

Extension status:

- `extension_available=True`
- `has_inplace_fused=True`

Remote pytest:

- `181 passed, 1 skipped`
- Skip reason: `torch_npu` is not installed in the CUDA image.

## Benchmark Configuration

- Script: `tests/distributed/synthetic_ddp_compare.py`
- Model: synthetic MLP
- Parameters: 46,137,344
- Steps: 50
- Warmup: 10
- Batch size per rank: 8
- Precision: fp32
- Bucket cap: 25 MB
- CCDL strategy: `all_gather`
- Compression: int8, group size 64
- Workspace policy: default bounded ownership, `workspace_cache_max_entries=1`

## Results

| GPUs | Mode | samples/s | avg step ms | loss | peak memory MB |
|---:|---|---:|---:|---:|---:|
| 2 | async no EF | 1368.34 | 11.693 | 0.9980579811 | 1130.7 |
| 2 | async EF safe | 1035.10 | 15.458 | 0.9980579770 | 1251.7 |
| 2 | async EF bounded workspace | 1080.30 | 14.811 | 0.9980579770 | 1306.7 |
| 4 | async no EF | 1263.69 | 25.323 | 0.9986362621 | 1318.2 |
| 4 | async EF safe | 1107.32 | 28.899 | 0.9986362767 | 1306.2 |
| 4 | async EF bounded workspace | 1130.03 | 28.318 | 0.9986362767 | 1494.2 |

## Interpretation

Compared with the safe EF path, bounded workspace/fused EF improved synthetic throughput:

- 2 GPUs: `1080.30 / 1035.10 - 1 = +4.37%`
- 4 GPUs: `1130.03 / 1107.32 - 1 = +2.05%`

Loss matches the safe EF path in this short synthetic run.

Peak memory did not drop relative to the previous workspace run. The bounded cache reduces CCDL's long-lived Python references, but PyTorch CUDA peak memory still records the highest allocation reached during the process. In this benchmark, the fused path still needs an additional restored workspace allocation, so the measured peak remains higher.

## Next Optimization

The next meaningful memory optimization should avoid allocating a separate restored workspace at all, or make the fused kernel write directly into the DDP bucket result buffer with a carefully owned output contract. Python-side ownership policy alone is not enough to reduce the CUDA peak metric.
