# Fused Dequant Reduce Mean Error Feedback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fuse dequantize, reduce, mean scaling, and error-feedback residual update into one CUDA kernel for the hot compressed DDP path.

**Architecture:** Add an inplace fused CUDA entry point that writes into caller-owned restored/residual workspace and returns a capability boolean. Python codec exposes this as an optional fast path, and DDP hook uses it only when residual workspace already exists.

**Tech Stack:** PyTorch C++ extension, CUDA, pybind11, pytest.

## Global Constraints

- Preserve safe import and extension-missing fallback behavior.
- Do not change public pre-refactor CCDL API compatibility assumptions; `ccdl_comm` is the new API surface.
- Keep unsupported quantization modes on existing fallback paths.
- Do not claim benchmark wins without same-machine, same-workload measurement.

## Files

- `ccdl_comm/csrc/quantization/dequant_api.cuh`: declare the fused inplace native entry point.
- `ccdl_comm/csrc/quantization/dequant_reduce_kernel.cu`: implement fused kernel and capability checks.
- `ccdl_comm/csrc/pybind.cpp`: export the fused inplace entry point.
- `ccdl_comm/quantization/codec.py`: add Python wrapper with optional workspace.
- `ccdl_comm/communication/ddp_hook.py`: route async EF through the workspace-aware fused wrapper.
- `tests/test_pybind_exports.py`: assert pybind symbol exists.
- `tests/test_quantization_codec.py`: validate wrapper behavior without CUDA.
- `tests/test_cuda_extension_smoke.py`: validate CUDA numerical equivalence when extension is available.
- `tests/test_ddp_comm_hook.py`: validate hook dependency injection and fallback.

## Task 1: Add API tests

- [ ] Add pybind export expectation for `inplace_dequantize_reduce_mean_update_error_feedback`.
- [ ] Add codec wrapper test that passes `output` workspace and asserts native symbol receives it.
- [ ] Add invalid reduce-mode test.
- [ ] Run focused tests and verify expected failure before implementation.

## Task 2: Implement native fused inplace entry

- [ ] Declare native function in `dequant_api.cuh`.
- [ ] Implement CUDA kernel that writes restored mean and residual in one launch.
- [ ] Return `false` for unsupported fast-path predicates.
- [ ] Export symbol in `pybind.cpp`.
- [ ] Run focused tests.
- [ ] Commit.

## Task 3: Integrate Python workspace path

- [ ] Add codec wrapper `inplace_dequantize_reduce_mean_update_error_feedback`.
- [ ] Keep existing allocation-returning wrapper for fallback.
- [ ] Update DDP hook injection points to prefer workspace-aware fused entry when residual exists.
- [ ] Run focused tests.
- [ ] Commit.

## Task 4: Verify

- [ ] Run local full pytest suite.
- [ ] Build and run CUDA tests remotely if environment is reachable.
- [ ] Record benchmark status honestly.
