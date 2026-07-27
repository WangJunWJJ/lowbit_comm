# Workspace Cache A6000 Retest

Date: 2026-07-27  
Host: `user@192.168.8.156:360`  
GPU: 5 x NVIDIA RTX A6000, benchmark used 2 and 4 local ranks  
Image: `ccdl-comm-a6000:cu126-torch25`  
Commit: `11afcf8`

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

- `177 passed, 1 skipped`
- Skip reason: `torch_npu` is not installed in the CUDA image.

## Stability Finding

The first 4-GPU retest against commit `9c7538d` exposed non-finite loss at step 5 when async error-feedback skipped CPU completion synchronization by default. The fix in `11afcf8` restores safe synchronized completion as the default and leaves non-blocking completion behind an explicit experimental switch.

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

## Results

| GPUs | Mode | samples/s | avg step ms | loss | peak memory MB |
|---:|---|---:|---:|---:|---:|
| 2 | async no EF | 1382.50 | 11.573 | 0.9980579835 | 1130.7 |
| 2 | async EF safe | 1025.83 | 15.597 | 0.9980579770 | 1251.7 |
| 2 | async EF workspace | 1097.88 | 14.574 | 0.9980579770 | 1306.7 |
| 4 | async no EF | 1251.90 | 25.561 | 0.9986362621 | 1318.2 |
| 4 | async EF safe | 1112.10 | 28.774 | 0.9986362767 | 1306.2 |
| 4 | async EF workspace | 1142.86 | 28.000 | 0.9986362767 | 1494.2 |

## Interpretation

Compared with the safe EF path, workspace/fused EF improved synthetic throughput:

- 2 GPUs: `1097.88 / 1025.83 - 1 = +7.02%`
- 4 GPUs: `1142.86 / 1112.10 - 1 = +2.77%`

Loss matches the safe EF path in this short synthetic run. Peak memory is still higher for the workspace/fused path, which means the current cache reduces repeated allocation churn but does not reduce the reserved peak footprint. The next optimization should make workspace ownership more precise and avoid holding padded workspaces longer than necessary.
