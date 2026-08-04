# Workspace Ownership Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bound DDP fused EF restored workspace ownership to reduce peak active memory.

**Architecture:** Extend `DequantizedWorkspaceCache` into a small LRU owner with entry and byte limits. Wire DDP hook defaults to a memory-conservative cache size while keeping explicit knobs for larger reuse windows.

**Tech Stack:** Python, pytest, A6000 CUDA remote validation.

## Global Constraints

- Do not change CUDA kernel semantics.
- Keep safe async feedback completion synchronization as the default.
- Preserve extension-missing fallback behavior.
- Benchmark claims must be synthetic unless run with real model/data.

## Files

- `ccdl_comm/communication/workspace.py`: add LRU ownership and byte budget.
- `ccdl_comm/communication/ddp_hook.py`: expose and pass cache limits.
- `tests/test_workspace_cache.py`: cache lifecycle behavior.
- `tests/test_ddp_comm_hook.py`: DDP hook configuration behavior.
- `tests/benchmarks/reports/workspace_lifecycle_<commit>/`: A6000 result files.

## Task 1: Add failing tests

- [ ] Test LRU entry eviction.
- [ ] Test byte-budget eviction.
- [ ] Test DDP hook configured cache capacity affects allocation count.
- [ ] Run focused tests and confirm failure.

## Task 2: Implement bounded cache

- [ ] Use ordered records to track recency.
- [ ] Estimate workspace bytes from padded numel and dtype.
- [ ] Enforce `max_entries`.
- [ ] Enforce `max_cached_bytes`.
- [ ] Keep `clear`.
- [ ] Run focused tests.
- [ ] Commit.

## Task 3: Wire DDP hook knobs

- [ ] Add `workspace_cache_max_entries`.
- [ ] Add `workspace_cache_max_bytes`.
- [ ] Default entries to `1`.
- [ ] Pass limits to cache construction.
- [ ] Run focused and full tests.
- [ ] Commit.

## Task 4: A6000 validation

- [ ] Rebuild CUDA extension remotely.
- [ ] Run remote full pytest.
- [ ] Run 2/4 GPU synthetic benchmark.
- [ ] Pull result JSON.
- [ ] Record report and commit.
- [ ] Push `wj_dev`.
