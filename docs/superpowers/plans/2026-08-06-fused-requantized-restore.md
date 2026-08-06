# Fused Requantized ReducedShard Restore Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the intermediate FP ReducedShard and per-rank restore launches by fusing dequant-reduce-mean with INT8 requantization, gathering into reusable storage, and restoring every gathered rank payload in one CUDA kernel.

**Architecture:** Keep the Python transport and native ABI capability driven. The production fast path specializes linear INT8, group size 64, topk 0, compact false, FP16/BF16/FP32, and at most eight payload inputs; every unsupported policy uses the existing compressed restore chain. A stream-safe workspace session owns the requantized send buffer, aligned gather buffer, and full output until the final CUDA completion.

**Tech Stack:** Python 3.10+, PyTorch distributed/NCCL, PyBind11, CUDA C++, pytest, Ruff.

## Global Constraints

- Preserve the public `ReducedShard` contract for shard consumers.
- Preserve current numerical semantics: reduce in FP32, round the mean to the configured tensor dtype, then compute and store the requantization scale.
- Never select the native fast path after a collective has started; capability rejection must happen before communication or use the established fallback.
- Keep 16-byte rank payload stride alignment and initialize transmitted padding bytes in the fused requantize kernel.
- Support arbitrary global world sizes through topology grouping; one fused invocation accepts one through eight current-group payloads.
- Do not add host synchronization to the transport hot path.

---

### Task 1: Fused restore contracts and dispatch

**Files:**
- Modify: `ccdl_comm/quantization/codec.py`
- Modify: `ccdl_comm/communication/reduce_scatter_transport.py`
- Test: `tests/test_quantization_codec.py`
- Test: `tests/test_reduce_scatter_transport.py`

**Interfaces:**
- Produces: `inplace_dequantize_reduce_mean_requantize(buffers, output, config, *, dtype, extension_status, divisor) -> bool`.
- Produces: `inplace_dequantize_gathered(buffer, output, config, *, dtype, extension_status, world_size, payload_numel, payload_stride, shard_numel) -> bool`.
- The transport accepts optional fused callbacks and falls back to the existing quantize/dequantize chain when either returns `False`.

- [ ] Write codec tests proving missing native symbols return `False`, supported symbols receive complete layout metadata, and invalid Python arguments fail before native dispatch.
- [ ] Run the focused tests and confirm failure because the APIs do not exist.
- [ ] Add minimal capability wrappers without allocation or synchronization.
- [ ] Write transport tests proving one fused requantize call and one fused gathered-dequantize call replace the old per-rank loop, while callback rejection preserves the old path.
- [ ] Run the focused tests and confirm failure because transport dispatch is absent.
- [ ] Add the minimal transport dispatch and metadata reporting.
- [ ] Run `python -m pytest tests/test_quantization_codec.py tests/test_reduce_scatter_transport.py -q`.
- [ ] Commit as `perf(transport): dispatch fused requantized restore`.

### Task 2: Stream-safe restore workspace reuse

**Files:**
- Modify: `ccdl_comm/cuda/workspace.py`
- Modify: `ccdl_comm/communication/reduce_scatter_transport.py`
- Test: `tests/cuda/test_cuda_workspace_pool.py`
- Test: `tests/test_reduce_scatter_transport.py`

**Interfaces:**
- Produces session methods `get_requantized_shard(...)`, `get_gathered_restore(...)`, and reuses existing `get_full_output(...)`.
- Each key includes shape, dtype, world size, bit, group size, aligned payload stride, and workspace kind.

- [ ] Write pool tests that acquire, release after an event, and reuse all three restore buffers without aliasing concurrent leases.
- [ ] Run the tests and confirm the session methods are missing.
- [ ] Implement the three typed workspace keys and allocations.
- [ ] Connect the all-gather restore transport to one workspace session so leases survive the final restore kernel.
- [ ] Run `python -m pytest tests/cuda/test_cuda_workspace_pool.py tests/test_reduce_scatter_transport.py -q`.
- [ ] Commit as `perf(workspace): pool compressed restore buffers`.

### Task 3: Fused dequant-reduce-mean-requantize CUDA kernel

**Files:**
- Modify: `ccdl_comm/csrc/quantization/dequant_api.cuh`
- Modify: `ccdl_comm/csrc/quantization/dequant_reduce_kernel.cu`
- Modify: `ccdl_comm/csrc/pybind.cpp`
- Test: `tests/cuda/test_fused_requantized_restore.py`

**Interfaces:**
- Produces native `inplace_dequantize_reduce_mean_requantize(inputs, output, group_size, topk, bit, quant_type, compact, dtype, divisor, payload_stride) -> bool`.
- Capability: linear INT8/group64/topk0/noncompact, one through eight equal CUDA uint8 inputs, aligned uint8 output, FP16/BF16/FP32 scale dtype.

- [ ] Write CUDA tests comparing the native output after dequantization against the established `dequantize_reduce(mean) -> quantize` reference for 1/2/4/8 inputs, all supported dtypes, zeros, extrema, and non-finite rejection behavior.
- [ ] Run locally and confirm skip without CUDA or failure from the missing symbol on a built extension.
- [ ] Implement one block-per-group reduction with FP32 accumulation, dtype rounding, shared group values, max-abs reduction, packed INT8 output, stored dtype scale, and zeroed alignment tail.
- [ ] Bind the capability-returning function through PyBind11.
- [ ] Build and run the CUDA tests on A6000.
- [ ] Commit as `perf(cuda): fuse reduced shard requantization`.

### Task 4: Single-launch gathered payload dequantization

**Files:**
- Modify: `ccdl_comm/csrc/quantization/dequant_api.cuh`
- Modify: `ccdl_comm/csrc/quantization/dequant_reduce_kernel.cu`
- Modify: `ccdl_comm/csrc/pybind.cpp`
- Test: `tests/cuda/test_fused_requantized_restore.py`

**Interfaces:**
- Produces native `inplace_dequantize_gathered(input, output, group_size, topk, bit, quant_type, compact, dtype, world_size, payload_numel, payload_stride, shard_numel) -> bool`.
- One launch maps output groups to rank-strided payloads and writes the full padded FP bucket.

- [ ] Add a failing CUDA test comparing one native call with a Python loop over rank payloads for 2/4/8 ranks and validating stride, capacity, dtype, device, and group alignment contracts.
- [ ] Run it and confirm failure from the missing symbol.
- [ ] Implement the rank-strided dequantization kernel and strict host-side contract validation.
- [ ] Bind the API and rerun the focused CUDA test on A6000.
- [ ] Commit as `perf(cuda): restore gathered shards in one launch`.

### Task 5: End-to-end validation and performance gate

**Files:**
- Modify: `tests/distributed/int8_restore_compare.py`
- Create: `tests/benchmarks/reports/psi_policy_fused_restore_20260806/README.md`
- Create: `tests/benchmarks/reports/psi_policy_fused_restore_20260806/summary.json`

**Interfaces:**
- Compares FP16 restore, current compressed restore, and fused compressed restore with identical model, data, batch, warmup, measurement window, devices, and optimizer semantics.

- [ ] Add benchmark counters for fused path selection, fallback reason, workspace reuse, restore latency, end-to-end samples/s, and maximum numerical delta.
- [ ] Run correctness tests and Ruff locally.
- [ ] Build the extension and run CUDA unit tests on A6000.
- [ ] Run 2-GPU and 4-GPU microbenchmarks and end-to-end PSI Policy training comparisons on A6000.
- [ ] Reject default enablement if fused restore regresses the current compressed path outside benchmark noise; retain explicit opt-in and document evidence.
- [ ] Run the full local test suite and inspect the complete result.
- [ ] Commit as `test(benchmark): validate fused compressed restoration`.
