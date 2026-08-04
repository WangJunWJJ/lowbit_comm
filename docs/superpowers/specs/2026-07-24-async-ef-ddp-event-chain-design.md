# Async Error Feedback DDP Event Chain Design

## Goal

Allow CCDL's DDP all-gather communication hook to use asynchronous gather with error feedback without exposing DDP or the optimizer to partially completed CUDA work. The feature must keep the current synchronous error-feedback path as the safe fallback.

## Problem

The current async all-gather path is safe for buckets that do not use error feedback. Error feedback adds stateful `compensate()` and `update()` operations around quantized communication. If the DDP Future is completed before fused dequant-reduce and feedback update are safely ordered on CUDA streams, later training steps can consume stale or partially written tensors, which can cause non-finite loss on multi-GPU runs.

## Approaches Considered

### Recommended: Python-level event-gated async pipeline

Introduce a small runtime abstraction that records and waits on CUDA events when tensors are CUDA-backed. The async DDP hook will run gather completion, fused dequant-reduce, feedback update, event record, and Future completion through a single helper. Unsupported runtimes fall back to synchronization-free no-op semantics or the existing synchronous EF path.

Pros:

- Minimal C++/CUDA risk.
- Easy to unit test with fake event/stream objects.
- Keeps full-training stability as the first requirement.
- Allows later replacement with a native C++ scheduler.

Cons:

- Some orchestration remains in Python.
- Performance is bounded by PyTorch Future callback overhead.

### Alternative: Native C++/CUDA scheduler

Move dequant-reduce, error-feedback update, event recording, and Future completion into a native extension.

Pros:

- Best long-term performance ceiling.
- Fewer Python callbacks and less interpreter overhead.

Cons:

- Higher implementation and debugging cost.
- Harder to keep safe fallback behavior across CUDA, CPU-only, and Ascend environments.
- Riskier for near-term full training.

### Alternative: Keep EF synchronous

Keep the current guard permanently and only allow async all-gather for no-EF buckets.

Pros:

- Highest stability.
- Already verified on 2/4 GPU A6000 tests.

Cons:

- Leaves async+EF performance on the table.
- Does not validate the future direction needed for product-grade compressed training.

## Selected Design

Use the Python-level event-gated async pipeline first. The DDP hook receives a new `async_error_feedback` flag. When `async_gather=True`, `async_error_feedback=True`, and the bucket's error-feedback policy applies, the hook uses an async pipeline instead of falling back to sync gather. If the pipeline cannot establish safe completion semantics, it falls back to the existing sync path.

## Components

### `ccdl_comm.communication.cuda_completion`

Provides a minimal completion abstraction:

- `CudaCompletion`: wraps optional event-like state and exposes `wait()` / `synchronize()`.
- `CudaCompletionManager`: creates completions for tensor results.
- `NoopCompletion`: used for non-CUDA tensors or runtimes without CUDA.

The module must import torch lazily and safely. Missing torch or missing CUDA support must not break package import.

### `ccdl_comm.communication.async_pipeline`

Provides `AsyncBucketPipeline`, which:

1. waits for async all-gather work;
2. runs dequant-reduce;
3. optionally runs error-feedback update;
4. records completion for the restored tensor;
5. completes the DDP Future only after the completion has been made safe for the current runtime.

The pipeline accepts injected functions for gather work, dequant-reduce, feedback update, policy advance, and future creation so it can be unit-tested without torch.

### DDP hook integration

The all-gather fast path will choose among three paths:

1. `async_gather=True` and no EF apply/update: existing async no-EF path.
2. `async_gather=True`, EF apply/update, and `async_error_feedback=True`: new event-gated async EF path.
3. otherwise: existing synchronous path.

The default remains `async_error_feedback=False` until multi-GPU verification demonstrates stable behavior.

## Error Handling and Fallback

- If the async gather work has no usable Future, the pipeline completes inline after `wait()`.
- If CUDA events are unavailable, completion manager uses no-op semantics for CPU/fake tensors.
- If pipeline construction or runtime validation fails before communication starts, the DDP hook uses the existing synchronous EF path.
- Exceptions raised inside callbacks are set on the outer Future when the Future supports `set_exception`; otherwise they are re-raised.

## Testing

Unit tests must cover:

- CUDA completion manager is import-safe without torch.
- Async pipeline orders operations as gather wait → dequant-reduce → feedback update → completion wait → Future set_result.
- DDP hook uses async EF only when both `async_gather` and `async_error_feedback` are true.
- DDP hook keeps the synchronous EF fallback when `async_error_feedback=False`.

Remote validation must cover:

- Container pytest on A6000 with CUDA extension import.
- 2 GPU and 4 GPU synthetic benchmark for baseline, sync EF, async no-EF, and async EF.
- Check that async EF produces finite loss and train loss is consistent with sync EF for the short synthetic run.

## Non-Goals

- Do not rewrite NCCL.
- Do not move EF update into CUDA in this phase.
- Do not make async EF the default in this phase.
- Do not change the public behavior of CPU fallback paths.

## Success Criteria

- Local unit tests pass.
- Remote CUDA test suite passes or reuses a compatible previously built extension if Docker daemon compilation is unstable.
- 2/4 GPU synthetic async EF runs complete with finite loss.
- The final report clearly states whether async EF is safe to recommend for tonight's full training or should remain opt-in.
