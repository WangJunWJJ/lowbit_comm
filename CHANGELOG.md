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
- Keep real compressed collectives, token-bound error feedback, and end-to-end
  training acceleration outside the delivered boundary.

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
