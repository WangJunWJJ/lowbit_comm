# Native Error Feedback Update Design

## Goal

Reduce async error-feedback overhead by moving the error-feedback tensor update from Python tensor expressions into the CCDL native CUDA extension while preserving the current training semantics.

## Current Behavior

For each compressed DDP bucket, Python computes:

```python
prepared = original + residual
restored = dequantize_reduce(...)
residual = detach_clone(prepared - restored)
```

This is mathematically correct but expensive in the async path because `prepared - restored`, `detach().clone()`, Python callback scheduling, and the final CPU-side completion guard add latency. The previous event-gated async EF path was stable on 2/4 GPU A6000 only after adding a conservative `synchronize()` before Future completion, which erased much of the async benefit.

## Selected Approach

Implement native CUDA error-feedback update first, then add a combined C++ extension entry point that runs dequant-reduce and residual update in one native call.

The selected route has three stages:

1. Add `inplace_error_feedback_update(prepared, restored, residual)` to CUDA.
2. Add Python codec wrapper `update_error_feedback_residual(prepared, restored, residual, ...)`.
3. Add `dequantize_reduce_update_error_feedback(...)` as a combined native entry point used by async EF.

This keeps the public training semantics unchanged and gives a clean fallback path when native symbols are unavailable.

## Non-Goals

- Do not change the error-feedback formula.
- Do not rewrite NCCL or PyTorch DDP Future internals.
- Do not make async error feedback default in this phase.
- Do not change Ascend/CANN behavior in this phase.

## CUDA Kernel Semantics

The first CUDA kernel computes:

```text
residual[i] = prepared[i] - restored[i]
```

Requirements:

- `prepared`, `restored`, and `residual` must be contiguous CUDA tensors.
- All tensors must have the same number of elements.
- Supported dtypes in phase one: `float32`, `float16`, `bfloat16`.
- The kernel launches on CCDL's current CUDA stream helper.
- The kernel writes into caller-provided `residual` storage and returns nothing.

## Python Integration Semantics

The codec wrapper should:

- load the CUDA extension safely through existing `CudaExtensionStatus`;
- require the native symbol when the caller explicitly requests native update;
- return the residual buffer after in-place update;
- preserve fallback compatibility by allowing DDP hook code to continue using Python `feedback.update` when the native symbol is missing.

## Combined Entry Point Semantics

The combined entry point should:

```text
dequantize_reduce(inputs) -> restored
inplace_error_feedback_update(prepared, restored, residual)
return restored
```

It must not divide by world size internally. The Python wrapper will keep the existing `reduce="mean"` behavior so semantics match `dequantize_reduce_tensors`.

## DDP Hook Usage

The async EF hook should prefer the combined native update only when:

- `async_gather=True`
- `async_error_feedback=True`
- the EF policy applies to the bucket
- the extension exports the native symbols
- a residual buffer exists or can be allocated for the bucket

Otherwise, it must fall back to the stable path from commit `4616fef`.

## Testing Strategy

Local tests:

- source-level tests verify pybind exports and CUDA kernel names;
- wrapper tests verify native calls and missing-symbol errors;
- DDP hook tests verify native EF update path is selected only when available.

Remote tests:

- rebuild CUDA extension on A6000 if Docker daemon permits;
- run extension smoke test for native residual update;
- run 2/4 GPU synthetic benchmark for sync EF, async EF safe fallback, and async EF native update.

## Success Criteria

- Local unit/source tests pass.
- CUDA extension builds on A6000 or a build failure is clearly identified as infrastructure rather than source.
- Native residual update smoke test matches Python residual update numerically.
- 2/4 GPU async EF native benchmark completes with finite loss.
- Performance is no worse than the previous event-gated async EF path.
