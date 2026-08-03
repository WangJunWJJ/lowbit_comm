# CCDL Comm Refactor

This repository contains CCDL as an independent low-bit communication
library. Training frameworks may schedule it through the public protocols,
but the library does not depend on ParaScale.

The initial scope is intentionally narrow:

- CUDA/NCCL native-DDP gradient-bucket compression.
- Safe default: `linear`, `8-bit`, `group_size=64`, `topk=0`.
- ParaScale owns backend selection, training orchestration, benchmark evidence,
  checkpoint policy, and fallback.
- CCDL owns compression kernels, compressed collectives, DDP hook adaptation,
  buffer lifetime, error-feedback state, and capability reporting.

## Current status

The backend-neutral runtime, CUDA production path, Ascend adapter, collective
protocols, P2P APIs, topology strategies, workspace management and reduced
shard interface are available as independently buildable packages.

Implemented now:

- `CompressionConfig`: stable user-facing compression policy.
- `CapabilityReport`: ParaScale-friendly runtime capability report.
- `CCDLCommunicationPlugin`: initial planning adapter for ParaScale.
- CUDA extension build wiring and safe import fallback.
- CUDA quantize/dequantize facade.
- Error-feedback residual state.
- Compressed DDP bucket processor.
- Conservative `all_gather` DDP comm-hook factory.
- Low-level compressed all-reduce transport adapter.
- No-Torch tests for the public control-plane contract plus CUDA and CANN
  wheel smoke tests.
- Split distributions with one Python source owner: `ccdl-core`, `ccdl-cuda`
  and `ccdl-ascend`.

Release-level long-running training acceptance remains in progress.

For standalone DDP usage, see
[`docs/INDEPENDENT_DDP_USAGE.md`](docs/INDEPENDENT_DDP_USAGE.md).

## Intended ParaScale usage

ParaScale should select CCDL as a communication plugin, not as a training
backend:

```yaml
training_backend: native_ddp
communication:
  plugin: ccdl
  bit: 8
  group_size: 64
  topk: 0
  quant_type: linear
  error_feedback: true
  fallback: bf16_compress
```

## Local validation

```bash
python -m pytest tests -q
```

## Build and install

Build in an environment that already contains the intended PyTorch backend so
the native wheel cannot silently select a different Torch/CUDA/CANN stack:

```bash
python -m build --wheel --no-isolation packages/ccdl-core
CCDL_COMM_BUILD_CUDA=1 TORCH_CUDA_ARCH_LIST=8.6 \
  python -m build --wheel --no-isolation packages/ccdl-cuda
CCDL_COMM_BUILD_CANN=1 \
  python -m build --wheel --no-isolation packages/ccdl-ascend
```

Install exactly one native backend together with Core:

```bash
python -m pip install dist/core/ccdl_core-*.whl dist/cuda/ccdl_cuda-*.whl
# or
python -m pip install dist/core/ccdl_core-*.whl dist/ascend/ccdl_ascend-*.whl
```

`ccdl-comm` is a compatibility meta-package and owns no Python source. Core is
the sole owner of `ccdl_comm`; backend wheels contain only their native
extension. See
[`tests/benchmarks/reports/task18_packaging/README.md`](tests/benchmarks/reports/task18_packaging/README.md)
for the validated build and install matrix.
