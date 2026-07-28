# A6000 full comparison supplement, 2026-07-28

This report supplements the A6000 benchmark coverage with same-shape synthetic
training comparisons against native PyTorch DDP and PyTorch FSDP.

## Environment

- Host: `user@192.168.8.156 -p 360`
- GPU: NVIDIA RTX A6000
- GPUs used: 2 and 4, selected from GPU `0,1,2,3`
- Docker image: `ccdl-comm-a6000:cu126-torch25`
- PyTorch: `2.5.0a0+872d972e41.nv24.08`
- CUDA: `12.6`
- Script: `ccdl_comm_refactor/tests/distributed/synthetic_ddp_compare.py`

## Workload

- Model: synthetic dense MLP
- Parameters: 46,137,344
- Steps: 50
- Warmup steps excluded from timing: 10
- Per-rank batch size: 8
- Input dim: 2048
- Width: 4096
- Depth: 3
- Output dim: 1024
- Dtype: FP32
- Optimizer: SGD, lr `1e-4`
- DDP bucket cap: 25 MiB
- CCDL mode: INT8, group size 64, `strategy=all_gather`, `async_gather=true`

## End-to-end synthetic training result

| GPUs | Method | Samples/s | Avg step ms | Speedup vs DDP | Train loss | Loss delta vs DDP | Peak memory MB |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2 | PyTorch DDP | 541.98 | 29.52 | 1.00x | 0.9980579740 | 0 | 544.53 |
| 2 | PyTorch FSDP | 482.22 | 33.18 | 0.89x | 0.9979621559 | -9.58e-05 | 632.41 |
| 2 | CCDL async, no EF | 1367.70 | 11.70 | 2.52x | 0.9980579787 | +4.77e-09 | 1130.66 |
| 2 | CCDL async + EF workspace | 1023.72 | 15.63 | 1.89x | 0.9980579770 | +2.98e-09 | 1306.66 |
| 4 | PyTorch DDP | 793.71 | 40.32 | 1.00x | 0.9986362827 | 0 | 544.53 |
| 4 | PyTorch FSDP | 749.14 | 42.72 | 0.94x | 0.9986214426 | -1.48e-05 | 588.41 |
| 4 | CCDL async, no EF | 1260.22 | 25.39 | 1.59x | 0.9986362621 | -2.06e-08 | 1318.16 |
| 4 | CCDL async + EF workspace | 1098.45 | 29.13 | 1.38x | 0.9986362767 | -5.96e-09 | 1494.16 |

## Interpretation

- For this communication-heavy synthetic MLP, the refactored CCDL DDP hook is
  faster than native DDP on both 2-GPU and 4-GPU runs.
- The fastest measured configuration is CCDL async without error feedback:
  2.52x on 2 GPUs and 1.59x on 4 GPUs versus PyTorch DDP.
- The safer training-oriented configuration, CCDL async with error feedback and
  bounded workspace ownership, remains faster than PyTorch DDP:
  1.89x on 2 GPUs and 1.38x on 4 GPUs.
- FSDP is slower than DDP for this small synthetic model because the benchmark is
  not memory-capacity bound and FSDP introduces shard/all-gather overhead.
- Synthetic train-loss deltas are effectively zero for CCDL over 50 steps. This
  is a short numerical smoke rather than a convergence proof on a real dataset.

## Legacy CCDL comparison status

The pre-refactor CCDL code does not expose a DDP communication hook, so it cannot
be compared under the same DDP-hook training API without adding an adapter.

I attempted to run its original collective benchmark:

```bash
torchrun --standalone --nproc_per_node=2 -m benchmarks.cifar10.comm_bench \
  --output /results/full_compare_20260728/legacy_collective/2gpu_comm.jsonl \
  --warmup 5 --repeat 20
```

The first attempt failed because `ccdl_cuda_ops` was not installed. Building the
legacy extension with `python setup.py build_ext --inplace` compiled the objects
but failed to import the extension under this PyTorch 2.5 / CUDA 12.6 container:

```text
ImportError: /workspace/ccdl_cuda_ops.cpython-310-x86_64-linux-gnu.so:
undefined symbol: _Z8quantizeN2at6TensorEllbl9QuantTypeb
```

Therefore, under the current A6000 test environment, the pre-refactor CCDL is not
directly runnable as an apples-to-apples benchmark target. This is also an
engineering-quality data point: the refactored `ccdl_comm` path supports safe
import and extension fallback, while the original code fails at import time when
the extension is missing or ABI-incompatible.

## Raw data

Raw JSON outputs are stored in `raw/`:

- `2gpu_pytorch_ddp.json`
- `2gpu_pytorch_fsdp.json`
- `2gpu_ccdl_async_no_ef.json`
- `2gpu_ccdl_async_ef_workspace.json`
- `4gpu_pytorch_ddp.json`
- `4gpu_pytorch_fsdp.json`
- `4gpu_ccdl_async_no_ef.json`
- `4gpu_ccdl_async_ef_workspace.json`
