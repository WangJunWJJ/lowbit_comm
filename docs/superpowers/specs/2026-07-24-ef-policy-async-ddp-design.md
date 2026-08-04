# CCDL Comm EF Policy and Async DDP Hook Design

Date: 2026-07-24

## Purpose

The next CCDL Comm phase should stop adding broad collective variants and focus on making the existing native-DDP compressed communication path easier for ParaScale to schedule, explain, and validate.

Two problems drive this phase:

1. Error feedback improves low-bit training robustness, but the current boolean switch is too coarse and can slow small-scale runs, especially the observed 2-GPU case.
2. The current DDP hook returns a Future, but the internal communication path is effectively synchronous: quantize, all-gather, dequantize/reduce, then complete the Future. This limits future communication/computation overlap.

This design keeps the validated CUDA fused dequant-reduce path intact and adds scheduling control around it.

## Scope

In scope:

- Native PyTorch DDP gradient bucket compression on CUDA/NCCL.
- Error feedback policy selection and per-bucket policy decisions.
- Async all-gather transport for compressed payload buffers.
- DDP hook integration that completes the returned Future after async communication and fused dequant-reduce.
- Unit tests and 2-GPU/4-GPU benchmark validation.

Out of scope for this phase:

- CANN/NPU async path.
- FSDP integration.
- New ring/tree/p2p algorithm migration.
- Multi-node transport optimization.
- Changing the existing quantization format.

## Error Feedback Policy

### Configuration

Extend `CompressionConfig` with an explicit error-feedback policy while keeping backwards compatibility:

- `error_feedback: bool` remains accepted.
- `error_feedback_policy: str = "always"` is introduced.
- `error_feedback_min_numel: int = 0` controls bucket-size gating.
- `error_feedback_warmup_steps: int = 0` controls delayed activation.
- `error_feedback_period: int = 1` controls periodic residual updates.

When `error_feedback=False`, the effective policy is `none`.
When `error_feedback=True` and no explicit policy is supplied, the effective policy is `always`.

Supported policies:

- `none`: do not compensate or update residuals.
- `always`: current behavior.
- `large_bucket_only`: apply EF only when bucket numel is at least `error_feedback_min_numel`.
- `warmup_then_enable`: skip EF until bucket-local step count reaches `error_feedback_warmup_steps`.
- `periodic`: compensate every step when a residual exists, but update residual only every `error_feedback_period` steps.

### Runtime component

Add a small policy object, for example `ErrorFeedbackPolicy`, responsible for:

- deriving whether a bucket should apply compensation;
- deriving whether a bucket should update residual;
- tracking bucket-local step counts;
- producing a compact reason string for diagnostics.

The existing `ErrorFeedbackState` remains responsible only for storing and clearing residual tensors.

This separation keeps numerical state and scheduling policy independent.

## Async DDP Hook

### Current behavior

The current hook creates a Future only after work is already complete. This is simple and correct, but it does not expose real asynchronous communication.

### Target behavior

For the compressed all-gather strategy, add an async path:

```text
bucket tensor
  -> optional EF compensate
  -> quantize
  -> async all_gather compressed payload buffers
  -> callback/wait continuation
  -> fused dequant_reduce
  -> optional EF update
  -> complete returned Future
```

The synchronous path remains available as a fallback and for tests.

### Transport interface

Add a transport object that can return an async work handle:

```python
work = async_all_gather(buffer)
future = work.get_future()
```

The work object should expose:

- gathered payload buffers;
- world size;
- completion Future when PyTorch provides one;
- a safe fallback that uses `.wait()` and completes a Future manually when `get_future()` is unavailable.

The initial version can limit async support to same-size compressed buffers, which matches the current default compact-disabled CUDA path. Metadata-rich fused payload packing can remain synchronous until its async behavior is tested separately.

### DDP hook behavior

Add an opt-in parameter to `create_ddp_comm_hook`, such as:

- `async_gather: bool = False`

Default stays synchronous initially to avoid silently changing runtime semantics.
ParaScale can enable it after benchmark validation.

When `async_gather=True` and the strategy/path supports it, the returned DDP Future should complete only after dequant-reduce and EF update are complete.
When unsupported, the hook should fall back to the current synchronous path.

## Interfaces for ParaScale

ParaScale should be able to reason about CCDL with structured decisions:

- whether EF is active for a bucket;
- why EF was skipped or applied;
- whether async gather was used or fell back;
- whether fused CUDA dequant-reduce fastpath was eligible.

This phase does not need a full metrics subsystem, but policy objects should expose deterministic reason strings so ParaScale can surface explanations in benchmark reports.

## Testing Plan

Unit tests:

- `CompressionConfig` validates EF policy fields.
- `ErrorFeedbackPolicy` decides apply/update correctly for each policy.
- DDP hook skips compensation/update when policy says no.
- DDP hook uses async all-gather when enabled and available.
- DDP hook falls back safely when async work lacks a Future.
- Returned Future is completed with the dequant-reduced tensor.

Remote validation:

- Re-run local full tests.
- Rebuild CUDA extension on A6000.
- Run remote full tests.
- Run same-scope 2-GPU and 4-GPU benchmark:
  - DDP baseline
  - legacy CCDL gather
  - refactor no EF
  - refactor always EF
  - refactor large-bucket EF
  - refactor async gather no EF
  - refactor async gather with selected EF policy

Success criteria:

- No regression in local and remote test suites.
- `no EF` path remains faster than legacy CCDL on 2-GPU and 4-GPU.
- `large_bucket_only` or another selected EF policy reduces the 2-GPU EF penalty versus current `always`.
- Async gather path is functionally correct and does not regress throughput against synchronous gather by more than 3% before further tuning.

## Risks

- PyTorch DDP communication hooks have strict Future semantics; incorrect completion timing can cause silent gradient races.
- Async NCCL work completion and CUDA stream ordering must be handled conservatively.
- EF policy can change convergence behavior; performance wins must be paired with real training validation.
- Small synthetic benchmarks may overstate communication benefits and understate dataloader/model overhead.

## Implementation Order

1. Add EF policy config and tests.
2. Implement `ErrorFeedbackPolicy` and integrate it into DDP hook.
3. Benchmark policy variants on 2-GPU and 4-GPU.
4. Add async all-gather transport and tests.
5. Integrate async path into DDP hook behind `async_gather=False` default.
6. Benchmark async path and decide whether ParaScale should enable it by default.

## Decision

Use a conservative, opt-in implementation. Keep existing synchronous fused CUDA path as the stable default while adding EF policy control and async gather as separately testable capabilities.
