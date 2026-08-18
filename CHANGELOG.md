# Changelog

## [0.4.0.dev0] - 2026-08-15

### Phase 1 architecture foundation

- Add immutable communication Intent, Policy, Strategy, Result, Context,
  ExecutionPlan, Work, and error-feedback contracts.
- Add deterministic capability Registry, evidence promotion gates, conservative
  Auto selection, strict Explicit compilation, and stable plan caching.
- Add the narrow `compile_communicator()` / `CompiledCommunicator.execute()`
  facade with an exact stable semantic export surface and import-safe behavior.
- Publish `LowbitCommError`, `CompileError`, `CapabilityError`, and
  `ExecutionError` from both stable facades without exposing other internals.
- Clarify that Phase 1 error feedback is a caller-driven visibility and state
  transaction; Work/event/token binding, stale completion rejection, and CUDA
  stream ordering remain Phase 2 work.
- Add deterministic FullTensor and ReducedShard Reference oracles while keeping
  them outside the production Backend Registry and Compiler.
- Establish the root `lowbit_comm/` and `csrc/` layout and remove active v0.3
  source, test, architecture, migration, and process-document paths.
- Phase 1 does not include a CUDA/quantized production Backend and makes no
  training acceleration claim.

### Phase 2 CUDA foundation

- Add safe extension discovery, exact FullTensor CUDA capability lowering, and
  deterministic compile-time workspace layouts.
- Replace Python callback completion with native CUDA-event `CudaWork`, unique
  launch tokens, and capacity-bounded device workspace leases.
- Add rank-local `create_fulltensor_plan` execution over an explicit c10d
  `ProcessGroup`: Native uses NCCL all-reduce; INT8 uses compact quantize-pack,
  quantized all-gather, and fused dequant-reduce.
- Validate FP16/BF16 SUM/MEAN on 2 and 4 A6000 ranks, including zero, tail, and
  large tensors, group sizes 16/32/64, repeated execution, and sanitizer runs.
- Publish same-scope FP16 communication evidence. INT8 improves 2-rank 16 MiB
  and 64 MiB buckets by 28.91% and 42.00% versus PyTorch native, but regresses
  every measured 4-rank bucket; those cases remain Native fallback candidates.
- Keep token-bound error feedback, DDP/FSDP adapters, multi-node validation,
  and end-to-end training acceleration outside the delivered boundary.

## [0.3.0] - Unreleased

### BREAKING - Major Architecture Refactor

- Introduce typed Semantic and Lowered communication IR.
- Separate operation, output contract, wire format, and algorithm.
- Keep compressed FullTensor wire quantized through both collectives.
- Introduce compile-once/run-many and unified Work/Event/WorkspaceLease semantics.
- Move DDP, ReducedShard, and qWD training state into explicit adapters.
- Add typed CUDA P2P, fixed device metadata, dynamic quantized all-gather,
  native collective facade, and general ring/tree/hierarchical lowering.
- Add rank-local FP32 master-shard and qWD adapters with checkpoint-forced
  full-precision refresh.
- Remove the legacy public API, `restore_mode`, ParaScale-specific plugin,
  compatibility wheels, and obsolete old-API tests/examples.
- Package only `lowbit_comm` and the reviewed package-local CUDA source assets.
