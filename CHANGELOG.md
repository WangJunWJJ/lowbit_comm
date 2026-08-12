# Changelog

## [0.3.0] - Unreleased

### BREAKING — Major Architecture Refactor

- Introduce typed Semantic and Lowered communication IR.
- Separate operation, output contract, wire format, and algorithm.
- Keep the normal compressed FullTensor wire quantized through both collectives.
- Introduce compile-once/run-many and unified Work/Event/WorkspaceLease semantics.
- Move DDP, ReducedShard, and qWD training state into explicit adapters.
- Remove the legacy public API, `restore_mode`, and the ParaScale-specific core plugin.

See `docs/MIGRATION_0.3.0_ZH.md`.
