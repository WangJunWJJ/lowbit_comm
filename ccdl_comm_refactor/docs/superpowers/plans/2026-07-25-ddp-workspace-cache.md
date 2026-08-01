# DDP Workspace Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reuse restored dequantization workspace in the async fused error-feedback DDP path.

**Architecture:** Add a small cache class under the communication package and have each DDP hook instance own one cache. The cache delegates allocation to the existing codec allocator and invalidates on incompatible shape/dtype/device/padded-size metadata.

**Tech Stack:** Python, pytest, existing CCDL CUDA codec wrapper.

## Global Constraints

- Preserve extension-missing fallback behavior.
- Do not change CUDA kernel semantics.
- Keep cache scoped to a DDP hook instance; do not introduce global tensor state.
- Do not claim training performance without same-contract benchmark evidence.

## Files

- `ccdl_comm/communication/workspace.py`: new cache class.
- `ccdl_comm/communication/ddp_hook.py`: use cache in fused async EF path.
- `tests/test_workspace_cache.py`: cache behavior tests.
- `tests/test_ddp_comm_hook.py`: repeated-bucket allocator reuse test.

## Task 1: Add failing tests

- [ ] Add cache hit test for same key/shape/dtype/device.
- [ ] Add cache miss test for changed shape.
- [ ] Add DDP hook test proving allocator is called once across two same-bucket fused async EF invocations.
- [ ] Run focused tests and confirm failure.

## Task 2: Implement cache

- [ ] Create `DequantizedWorkspaceCache`.
- [ ] Store workspace records keyed by bucket key.
- [ ] Compare metadata before reuse.
- [ ] Add `clear`.
- [ ] Run cache tests.
- [ ] Commit.

## Task 3: Integrate DDP hook

- [ ] Instantiate cache once in `create_ddp_comm_hook`.
- [ ] Replace direct `allocate_dequantized_buffer` call with cache lookup.
- [ ] Preserve injected allocator support.
- [ ] Run focused DDP hook tests.
- [ ] Commit.

## Task 4: Verify and benchmark

- [ ] Run local full pytest.
- [ ] Push `wj_dev`.
- [ ] Rebuild CUDA extension on A6000.
- [ ] Run remote full pytest.
- [ ] Run short synthetic 2/4 GPU benchmark and record results if meaningful.
