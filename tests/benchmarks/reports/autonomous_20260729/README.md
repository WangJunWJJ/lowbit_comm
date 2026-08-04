# Autonomous A6000 validation, 2026-07-29

This report records the validation run after the CCDL communication refactor
added:

- shard communication workspace cache,
- fused shard dequant-reduce fast-path scheduling,
- bucket-level fusion planning contract,
- async shard completion pipeline,
- async compressed shard transport fallback.

## Environment

- Host: `user@192.168.8.156 -p 360`
- GPU: NVIDIA RTX A6000
- Docker image: `ccdl-comm-a6000:cu126-torch25`
- PyTorch: `2.5.0a0+872d972e41.nv24.08`
- CUDA: `12.6`
- Work path: `/home/user/wangjun/ccdl-master`

Before the full test run, the remote CUDA extension was rebuilt because the
previous `ccdl_cuda_ops*.so` was compiled from stale csrc files and missed the
`inplace_dequantize_reduce_mean_update_error_feedback` pybind symbol.

Build command:

```bash
CCDL_COMM_BUILD_CUDA=1 TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=2 python setup.py build_ext --inplace
```

## Verification

Unit/integration tests inside Docker:

```text
232 passed, 1 skipped in 4.83s
```

Short distributed training smoke used `tests/distributed/synthetic_ddp_compare.py`
with CCDL INT8 all-gather, async gather, async error feedback, FP16 model, 8
steps and 2 warmup steps.

| GPUs | Samples/s | Avg step ms | Train loss | Peak memory MB |
| ---: | --------: | ----------: | ---------: | -------------: |
| 2 | 7933.23 | 2.017 | 0.9873286895 | 47.17 |
| 4 | 10073.05 | 3.177 | 0.9959281869 | 54.39 |

These are smoke results, not full convergence/performance claims. The run
confirms that the refactored communication layers and rebuilt CUDA extension can
execute real multi-GPU DDP training on A6000 without non-finite loss in this
short validation window.
