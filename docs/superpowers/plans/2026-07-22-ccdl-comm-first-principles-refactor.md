# CCDL Comm First-Principles Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `ccdl_comm` as a new high-performance, robust, highly available communication library for ParaScale while reusing proven C/CUDA assets where they preserve performance.

**Architecture:** The Python API is redesigned around typed configs, clear exceptions, safe CUDA loading, modern collectives, and DDP/ParaScale adapters. The C/CUDA quantization kernels remain the performance foundation and are reused through the existing extension build path. Legacy Python APIs are not compatibility targets.

**Tech Stack:** Python 3.10, PyTorch distributed/NCCL, CUDA extension `ccdl_cuda_ops`, pytest, torchrun remote validation on dual RTX 4090D.

## Global Constraints

- Do not preserve old `ccdl.comm` API compatibility as a design goal.
- Reuse original C/CUDA kernels unless benchmark evidence requires changes.
- Python code must follow PEP8-style naming, type hints, dataclass config objects, explicit exceptions, and safe import behavior.
- Performance must not be lower than the pre-refactor implementation for migrated primitives; benchmark gates compare old vs new under equal tensor size, dtype, world size, strategy, and method.
- Each independently useful feature must have tests and its own commit.
- ParaScale integration continues through plugin/DDP hook paths, not by making CCDL own training orchestration.

---

### Task 1: Public exception hierarchy

**Files:**
- Create: `ccdl_comm_refactor/ccdl_comm/exceptions.py`
- Modify: `ccdl_comm_refactor/ccdl_comm/__init__.py`
- Modify: `ccdl_comm_refactor/ccdl_comm/quantization/codec.py`
- Modify: `ccdl_comm_refactor/ccdl_comm/communication/torch_transport.py`
- Test: `ccdl_comm_refactor/tests/test_exceptions.py`

**Interfaces:**
- Produces: `CCDLError`, `CCDLUnavailableError`, `UnsupportedCollective`, `TorchDistributedUnavailableError`
- Consumes: existing `CCDLUnavailableError` and `TorchDistributedUnavailableError` call sites.

- [ ] Write tests importing all public exception types from `ccdl_comm`.
- [ ] Verify tests fail because `ccdl_comm.exceptions` does not exist.
- [ ] Add the exception module and re-export names.
- [ ] Update existing codec and torch transport modules to import canonical exceptions.
- [ ] Run focused and full local tests.
- [ ] Commit with `feat(ccdl_comm): add public exception hierarchy`.

### Task 2: Modern collectives namespace scaffold

**Files:**
- Create: `ccdl_comm_refactor/ccdl_comm/collectives/__init__.py`
- Create: `ccdl_comm_refactor/ccdl_comm/collectives/work.py`
- Create: `ccdl_comm_refactor/ccdl_comm/collectives/all_reduce.py`
- Test: `ccdl_comm_refactor/tests/test_collectives_api.py`

**Interfaces:**
- Produces: `CollectiveWork`, `ImmediateWork`, `compressed_all_reduce(tensor, config, op="mean", strategy="all_gather", async_op=False)`
- Consumes: `CompressionConfig`, existing quantization facade, and existing torch transports.

- [ ] Write tests for unsupported strategy raising `UnsupportedCollective`.
- [ ] Write tests for `async_op=True` returning a work object with `wait()`.
- [ ] Verify tests fail because collectives namespace does not exist.
- [ ] Implement the minimal all-reduce wrapper over the existing compressed all-gather/all-reduce adapter.
- [ ] Run focused and full local tests.
- [ ] Commit with `feat(ccdl_comm): add compressed all reduce api`.

### Task 3: Performance benchmark gate scaffold

**Files:**
- Create: `ccdl_comm_refactor/tests/distributed/collective_perf_compare.py`
- Test: local compile plus remote execution.

**Interfaces:**
- Produces: a torchrun script that measures baseline `torch.distributed.all_reduce`, old CCDL if importable, and new `ccdl_comm.collectives.compressed_all_reduce`.

- [ ] Add a benchmark script that records latency, tensor size, dtype, method, world size, relative L2, and output JSON.
- [ ] Run local compile.
- [ ] Run remote dual-GPU validation with a small tensor matrix.
- [ ] Commit with `test(ccdl_comm): add collective performance comparison`.

### Task 4: Remote CUDA and real training gate

**Files:**
- Reuse: `ccdl_comm_refactor/tests/distributed/ddp_hook_smoke.py`
- Reuse: `ccdl_comm_refactor/tests/distributed/cifar10_ddp_compare.py`

**Interfaces:**
- Produces: remote evidence that CUDA extension, DDP hook, and real ImageFolder training still pass after refactor.

- [ ] Package and copy `ccdl_comm_refactor` to `wangjun@192.168.1.100`.
- [ ] Build CUDA extension in `parascale-ci:cu121-torch24`.
- [ ] Run local package tests inside the container.
- [ ] Run `ddp_hook_smoke.py` on 2 GPUs.
- [ ] Run ImageFolder baseline vs CCDL training on `/home/wangjun/work/dataset/real102_small`.
- [ ] Report exact metrics and any regression.
