# CCDL Comm independent DDP usage

This guide is the minimal path for using `ccdl_comm` as an independent
compressed communication library in a PyTorch DDP training job.

## 1. Build CUDA extension

From the package root:

```bash
cd ccdl_comm_refactor
CCDL_COMM_BUILD_CUDA=1 TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=2 python3 setup.py build_ext --inplace
```

For RTX 4090 / 4090D, `TORCH_CUDA_ARCH_LIST=8.9` avoids compiling unnecessary
architectures.

## 2. Run smoke checks

```bash
PYTHONPATH=. python3 -m pytest tests/test_cuda_extension_smoke.py -q
```

Expected result:

```text
1 passed
```

## 3. Register CCDL DDP communication hook

Use the conservative `all_gather` strategy for full training validation. It
gathers compressed buffers, dequantizes every rank payload locally, and then
reduces in tensor space. This avoids directly summing arbitrary quantized integer
buffers.

```python
from ccdl_comm.communication import create_ddp_comm_hook
from ccdl_comm.config import CompressionConfig

config = CompressionConfig(
    bit=8,
    group_size=64,
    quant_type="linear",
    error_feedback=True,
)

hook = create_ddp_comm_hook(
    config,
    dtype="fp16",          # use "bf16" or "fp32" when the bucket dtype differs
    strategy="all_gather", # safest initial strategy
    reduce="mean",         # DDP-compatible gradient averaging
)

ddp_model.register_comm_hook(state=None, hook=hook)
```

## 4. Recommended first full-training settings

- Start with `bit=8`.
- Keep `error_feedback=True`.
- Use `strategy="all_gather"` for correctness validation.
- Compare against a normal DDP baseline with the same seed, batch size, precision,
  dataset order, and number of steps.
- Only test lower bits after the 8-bit curve matches the baseline.

## 5. Important caveats

- `all_reduce` transport is available as a low-level adapter, but do not use it
  as the first full-training path unless the quantized format/reduce semantics
  are explicitly validated for the workload.
- The hook currently runs synchronously inside the DDP hook and returns a
  completed Future. This is suitable for correctness/full-training validation;
  overlap optimization should be a later step.
