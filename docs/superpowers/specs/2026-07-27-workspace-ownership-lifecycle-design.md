# Workspace Ownership Lifecycle Design

## Goal

Reduce peak active GPU memory in the fused async error-feedback path by making restored workspace ownership bounded and explicit.

## Problem

The current `DequantizedWorkspaceCache` stores one restored workspace per bucket key for the lifetime of a DDP hook instance. This reduces repeated allocation churn, but it can also keep multiple padded bucket workspaces alive at once. On the A6000 synthetic benchmark this kept the fused workspace path faster than safe EF, but peak memory stayed higher.

## Design

Turn the cache into a bounded LRU owner:

- each cache record owns exactly one restored workspace tensor;
- records are keyed by bucket key;
- matching metadata reuses the existing tensor;
- misses allocate a new tensor;
- after insertion, the cache evicts least-recently-used records until both limits are satisfied:
  - `max_entries`
  - `max_cached_bytes`

Eviction only removes the cache reference. It does not mutate or reuse a tensor that may still be held by the DDP Future result. This keeps correctness safe while allowing Python/PyTorch to release active allocation ownership sooner.

## DDP Hook Policy

Add DDP hook knobs:

```text
workspace_cache_max_entries: int | None = 1
workspace_cache_max_bytes: int | None = None
```

The default is intentionally memory-conservative: keep only one restored workspace. This favors lower peak memory over aggressive multi-bucket reuse. Advanced users can increase the entry limit when throughput matters more than peak memory.

## Byte Accounting

The cache estimates bytes from:

- padded numel from `CompressionConfig.group_size`
- tensor dtype size inferred from tensor dtype string

This does not try to model PyTorch caching allocator reserved memory. It only bounds the tensors the CCDL cache itself strongly owns.

## Safety

This change does not alter CUDA kernel behavior or collective semantics. It only changes Python-side ownership duration for restored workspace references. The safe completion synchronization default from `11afcf8` remains unchanged.

## Tests

- cache evicts least-recently-used records by entry count;
- cache evicts by byte budget;
- DDP hook can configure workspace cache size;
- existing same-bucket reuse remains intact.
