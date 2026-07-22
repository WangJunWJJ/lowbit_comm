# CCDL Comm Refactor

This folder starts the CCDL refactor as an independent communication
compression library for ParaScale.

The initial scope is intentionally narrow:

- CUDA/NCCL native-DDP gradient-bucket compression.
- Safe default: `linear`, `8-bit`, `group_size=64`, `topk=0`.
- ParaScale owns backend selection, training orchestration, benchmark evidence,
  checkpoint policy, and fallback.
- CCDL owns compression kernels, compressed collectives, DDP hook adaptation,
  buffer lifetime, error-feedback state, and capability reporting.

## Current status

This is now usable as an independent DDP communication-hook validation package.
It is still not a full replacement for the original `ccdl/` package collectives.

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
- No-Torch tests for the public control-plane contract plus CUDA smoke tests.

Not implemented yet:

- Async overlap optimization.
- Native reduce-scatter/tree collectives parity with the original package.
- Long-running training benchmark suite.

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
python -m pytest ccdl_comm_refactor/tests -q
```
