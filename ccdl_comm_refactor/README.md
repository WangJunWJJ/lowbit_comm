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

This is a scaffolding milestone, not a full replacement for the original
`ccdl/` package yet.

Implemented now:

- `CompressionConfig`: stable user-facing compression policy.
- `CapabilityReport`: ParaScale-friendly runtime capability report.
- `CCDLCommunicationPlugin`: initial planning adapter for ParaScale.
- No-Torch tests for the public control-plane contract.

Not implemented yet:

- CUDA extension migration.
- DDP `register_comm_hook` Future API.
- Error-feedback residual buffers.
- Compressed all-reduce/all-gather/reduce-scatter refactor.
- Distributed correctness and benchmark suite.

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
