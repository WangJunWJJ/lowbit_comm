# INT8 ReducedShard Restore Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Gather ReducedShard restoration payloads as INT8 bytes and dequantize only into the final DDP gradient bucket.

**Architecture:** Preserve the existing full-precision restore as the default and add an explicit compressed restore strategy inside the reduce-scatter transport. The new strategy reuses the existing CCDL packed codec, gathers fixed-size byte payloads, and decodes each rank payload directly into a slice of the caller-visible FP output.

**Tech Stack:** Python 3.10, PyTorch distributed/NCCL, CCDL CUDA codec, pytest, A6000 Docker benchmark environment.

## Global Constraints

- Existing `restore_mode="fp16"` behavior and API calls remain compatible.
- Compressed restore must return the original tensor shape and dtype.
- All ranks must reconstruct identical gradients.
- Production code is written only after a failing test demonstrates the missing behavior.
- The 2/4-GPU benchmark uses the same PSI Policy model, dataset, batch size, and epoch count as the 2026-08-06 baseline.

---

### Task 1: Compressed ReducedShard restoration

**Files:**
- Modify: `ccdl_comm/communication/reduce_scatter_transport.py`
- Test: `tests/test_reduce_scatter_transport.py`

**Interfaces:**
- Consumes: `quantize_tensor(tensor, config, extension_status=...)` and `dequantize_tensor(buffer, shape, config, dtype=..., output=...)`.
- Produces: `make_torch_compressed_reduce_scatter_all_gather(..., restore_mode="compressed")`.

- [ ] Write a unit test whose fake distributed backend records that the final all-gather input is a quantized payload and whose fake decoder writes each gathered payload into the correct full-output slice.
- [ ] Run `python -m pytest tests/test_reduce_scatter_transport.py -k compressed_restore -vv` and verify that construction fails because `restore_mode` is not accepted.
- [ ] Add restore-mode validation, injectable restore codec callbacks, compressed workspace allocation, byte all-gather, and direct-to-slice decoding.
- [ ] Re-run the focused test and the complete reduce-scatter transport suite.
- [ ] Commit with `perf(transport): compress reduced shard restoration`.

### Task 2: CUDA codec and distributed correctness validation

**Files:**
- Modify: `tests/test_cuda_extension_smoke.py`
- Create: `tests/distributed/int8_restore_compare.py`

**Interfaces:**
- Consumes: the compressed restore transport from Task 1.
- Produces: rank-consistency, relative-L2, and communication-byte evidence.

- [ ] Write CUDA and distributed tests covering FP16 output, non-group-aligned padding, and equal results on every rank.
- [ ] Run the CUDA smoke test in the A6000 image and verify the new test fails before the remote tree receives Task 1.
- [ ] Synchronize Task 1 and run the focused CUDA/distributed tests on 2 and 4 A6000 GPUs.
- [ ] Record payload bytes, relative L2 error, and rank-to-rank maximum difference.
- [ ] Commit with `test(cuda): validate compressed shard restoration`.

### Task 3: PSI Policy end-to-end performance comparison

**Files:**
- Modify the existing remote PSI Policy CCDL adapter to select compressed restore for the benchmark only.
- Create: `tests/benchmarks/reports/psi_policy_int8_restore_20260806/README.md`
- Create: `tests/benchmarks/reports/psi_policy_int8_restore_20260806/summary.json`

**Interfaces:**
- Consumes: the validated compressed restore mode and the 2026-08-06 native/FP16-restore baselines.
- Produces: comparable 2/4-GPU throughput, latency, loss, validation loss, and effective-strategy evidence.

- [ ] Run a short warm-up/smoke training job and reject fallback, non-finite loss, or rank divergence.
- [ ] Run full 2-GPU and 4-GPU epochs with the existing model, per-rank batch size 16, and requested 4-GPU devices 1,2,3,4.
- [ ] Compare against native DDP and FP16-restore CCDL using samples/s, mean/median/P95 step time, convergence metrics, and data-wait share.
- [ ] Write the report with limitations and the next bottleneck supported by measurements.
- [ ] Commit with `test(benchmark): compare int8 shard restoration`.

