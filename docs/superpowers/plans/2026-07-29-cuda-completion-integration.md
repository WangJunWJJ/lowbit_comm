# CUDA Completion Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give CCDL point-to-point and collective transports one result-bearing, CUDA-stream-ordered asynchronous completion protocol and validate its overlap behavior on two and four A6000 GPUs.

**Architecture:** Extend `CollectiveWork` with side-effect-free readiness queries and implement a callback-capable work object through `CudaCompletionManager`. Integrate it at transport boundaries so each operation owns its in-flight buffers and post-processing, while synchronous and unsupported-runtime paths retain safe fallback behavior.

**Tech Stack:** Python 3.10+, PyTorch 2.5, `torch.distributed`, NCCL, CUDA streams/events, pytest, Docker.

## Global Constraints

- Preserve current public function signatures and result values.
- Do not add unconditional `torch.cuda.synchronize()` to library code.
- `query()` must not run deferred work or synchronize the CPU.
- A work object must retain every buffer used by an in-flight operation.
- Non-CUDA and unsupported backends must retain a correctness-preserving fallback.
- Each implementation task follows RED, GREEN, full regression, and an independent conventional commit.

---

### Task 1: Unified result-bearing completion work

**Files:**
- Modify: `ccdl_comm_refactor/ccdl_comm/collectives/work.py`
- Modify: `ccdl_comm_refactor/ccdl_comm/communication/cuda_completion.py`
- Modify: `ccdl_comm_refactor/tests/test_cuda_completion.py`

**Interfaces:**
- Produces: `CompletionWork(result, handle=None, complete=None, completion=None, resources=())`
- Produces: `CudaCompletionManager.create_work(*, result, handle=None, complete=None, resources=())`
- Produces: `CollectiveWork.query() -> bool` and `CollectiveWork.get_future() -> Any | None`

- [ ] **Step 1: Write failing tests**

Add tests proving that `query()` does not invoke the completion callback,
`wait()` waits for the backend before running the callback, repeated `wait()`
runs the callback once, resources remain owned, and callback exceptions are
cached and re-raised.

- [ ] **Step 2: Verify RED**

Run:

```powershell
python -m pytest ccdl_comm_refactor\tests\test_cuda_completion.py -q
```

Expected: failure because `create_work` and the common readiness protocol do
not exist.

- [ ] **Step 3: Implement the minimal work protocol**

Implement a generic work object with this state transition:

```python
def wait(self) -> Any:
    if not self._finished:
        self._wait_handle()
        self._result = self._run_complete_once()
        self._wait_completion()
        self._finished = True
    if self._error is not None:
        raise self._error
    return self._result
```

`query()` checks the handle and completion event only. It must return `False`
when readiness cannot be observed safely.

- [ ] **Step 4: Verify GREEN and regressions**

Run:

```powershell
python -m pytest ccdl_comm_refactor\tests\test_cuda_completion.py ccdl_comm_refactor\tests\test_async_bucket_pipeline.py ccdl_comm_refactor\tests\test_async_shard_pipeline.py -q
python -m pytest ccdl_comm_refactor\tests -q
```

Expected: all selected and full-suite tests pass.

- [ ] **Step 5: Commit**

```powershell
git add ccdl_comm_refactor/ccdl_comm/collectives/work.py ccdl_comm_refactor/ccdl_comm/communication/cuda_completion.py ccdl_comm_refactor/tests/test_cuda_completion.py
git commit -m "refactor(ccdl_comm): unify async completion work"
```

### Task 2: Point-to-point completion integration

**Files:**
- Modify: `ccdl_comm_refactor/ccdl_comm/communication/point_to_point.py`
- Modify: `ccdl_comm_refactor/tests/test_point_to_point.py`
- Modify: `ccdl_comm_refactor/tests/distributed/point_to_point_smoke.py`

**Interfaces:**
- Consumes: `CudaCompletionManager.create_work`
- Produces: `iqsend(..., completion_manager=None) -> CollectiveWork[Any]`
- Produces: `iqrecv(..., completion_manager=None) -> CollectiveWork[Any]`

- [ ] **Step 1: Write failing tests**

Add tests proving that asynchronous send retains the compressed buffer,
asynchronous receive dequantizes only after the receive handle completes,
`query()` has no receive-side effect, and blocking send/receive reuse the
asynchronous implementation through `wait()`.

- [ ] **Step 2: Verify RED**

Run:

```powershell
python -m pytest ccdl_comm_refactor\tests\test_point_to_point.py -q
```

Expected: failure because P2P does not accept or use a completion manager.

- [ ] **Step 3: Integrate the manager**

Replace `PointToPointWork` internals with the common protocol. Pass the
quantized send buffer or receive buffer through `resources` so it remains live
until `wait()`. Preserve `PointToPointWork` as a public alias if needed for
import compatibility.

- [ ] **Step 4: Verify GREEN and regressions**

Run:

```powershell
python -m pytest ccdl_comm_refactor\tests\test_point_to_point.py -q
python -m pytest ccdl_comm_refactor\tests -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add ccdl_comm_refactor/ccdl_comm/communication/point_to_point.py ccdl_comm_refactor/tests/test_point_to_point.py ccdl_comm_refactor/tests/distributed/point_to_point_smoke.py
git commit -m "refactor(ccdl_comm): order point-to-point completion"
```

### Task 3: Topology work integration

**Files:**
- Modify: `ccdl_comm_refactor/ccdl_comm/communication/topology_transport.py`
- Modify: `ccdl_comm_refactor/tests/test_topology_transport.py`

**Interfaces:**
- Consumes: `CudaCompletionManager.create_work`
- Extends: `make_native_topology_all_reduce(..., completion_manager=None)`
- Extends: `make_native_topology_reduce_scatter_shard(..., completion_manager=None)`

- [ ] **Step 1: Write failing tests**

Add tests proving overlap-gather and overlap-p2p wait for their backend handle
before dequantization, completion callbacks execute once, synchronous calls
return tensors directly, and topology work exposes side-effect-free `query()`.

- [ ] **Step 2: Verify RED**

Run:

```powershell
python -m pytest ccdl_comm_refactor\tests\test_topology_transport.py -q
```

Expected: failure because topology transports return private work wrappers.

- [ ] **Step 3: Replace private wrappers**

Route `_TopologyWork` and `_CallbackTopologyWork` construction through the
manager. Keep the topology algorithms and method selection unchanged. Retain
compressed receive buffers and output tensors through work resources.

- [ ] **Step 4: Verify GREEN and regressions**

Run:

```powershell
python -m pytest ccdl_comm_refactor\tests\test_topology_transport.py -q
python -m pytest ccdl_comm_refactor\tests -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add ccdl_comm_refactor/ccdl_comm/communication/topology_transport.py ccdl_comm_refactor/tests/test_topology_transport.py
git commit -m "refactor(ccdl_comm): unify topology async work"
```

### Task 4: Compressed collective asynchronous transports

**Files:**
- Modify: `ccdl_comm_refactor/ccdl_comm/communication/torch_transport.py`
- Modify: `ccdl_comm_refactor/ccdl_comm/collectives/all_gather.py`
- Modify: `ccdl_comm_refactor/ccdl_comm/collectives/all_reduce.py`
- Modify: `ccdl_comm_refactor/ccdl_comm/collectives/work.py`
- Modify: `ccdl_comm_refactor/tests/test_compressed_all_reduce.py`
- Create: `ccdl_comm_refactor/tests/test_compressed_all_gather.py`

**Interfaces:**
- Produces: async payload all-gather and all-reduce transports retaining output
  buffers and backend handles.
- Extends: `compressed_all_gather(..., completion_manager=None)`
- Extends: `compressed_all_reduce(..., completion_manager=None)`

- [ ] **Step 1: Write failing tests**

Add tests proving `async_op=True` does not dequantize before `wait()`, injected
synchronous transports use a completed-work fallback, mean reduction occurs
exactly once, and `wait()` returns the same value as the synchronous API.

- [ ] **Step 2: Verify RED**

Run:

```powershell
python -m pytest ccdl_comm_refactor\tests\test_compressed_all_gather.py ccdl_comm_refactor\tests\test_compressed_all_reduce.py -q
```

Expected: failure because the public APIs currently perform all work before
returning `ImmediateWork`.

- [ ] **Step 3: Implement async transport paths**

When `async_op=True` and the default torch transport is used, launch
`torch.distributed` with `async_op=True`, retain gathered/reduced buffers, and
defer reconstruction to the common completion work. Preserve immediate
fallback for injected synchronous callables.

- [ ] **Step 4: Verify GREEN and regressions**

Run:

```powershell
python -m pytest ccdl_comm_refactor\tests\test_compressed_all_gather.py ccdl_comm_refactor\tests\test_compressed_all_reduce.py ccdl_comm_refactor\tests\test_compressed_all_gather_reduce.py -q
python -m pytest ccdl_comm_refactor\tests -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add ccdl_comm_refactor/ccdl_comm/communication/torch_transport.py ccdl_comm_refactor/ccdl_comm/collectives/all_gather.py ccdl_comm_refactor/ccdl_comm/collectives/all_reduce.py ccdl_comm_refactor/ccdl_comm/collectives/work.py ccdl_comm_refactor/tests/test_compressed_all_gather.py ccdl_comm_refactor/tests/test_compressed_all_reduce.py
git commit -m "perf(ccdl_comm): add true async compressed collectives"
```

### Task 5: Reduce-scatter protocol alignment

**Files:**
- Modify: `ccdl_comm_refactor/ccdl_comm/communication/reduce_scatter_transport.py`
- Modify: `ccdl_comm_refactor/ccdl_comm/communication/async_shard_pipeline.py`
- Modify: `ccdl_comm_refactor/tests/test_reduce_scatter_transport.py`
- Modify: `ccdl_comm_refactor/tests/test_async_shard_pipeline.py`

**Interfaces:**
- Consumes: common `CollectiveWork` readiness protocol and completion manager.
- Preserves: `ReducedShard` metadata and workspace ownership.

- [ ] **Step 1: Write failing tests**

Add tests proving reduce-scatter async work exposes `query()`, owns its
send/receive/reduced workspaces until completion, and returns the same
`ReducedShard` metadata as the synchronous path.

- [ ] **Step 2: Verify RED**

Run:

```powershell
python -m pytest ccdl_comm_refactor\tests\test_reduce_scatter_transport.py ccdl_comm_refactor\tests\test_async_shard_pipeline.py -q
```

Expected: failure on the missing common readiness/ownership behavior.

- [ ] **Step 3: Align the pipeline**

Return common work objects from the asynchronous transport while retaining
workspace leases in `resources`. Keep existing fused kernel and error-feedback
ordering unchanged.

- [ ] **Step 4: Verify GREEN and regressions**

Run:

```powershell
python -m pytest ccdl_comm_refactor\tests\test_reduce_scatter_transport.py ccdl_comm_refactor\tests\test_async_shard_pipeline.py -q
python -m pytest ccdl_comm_refactor\tests -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add ccdl_comm_refactor/ccdl_comm/communication/reduce_scatter_transport.py ccdl_comm_refactor/ccdl_comm/communication/async_shard_pipeline.py ccdl_comm_refactor/tests/test_reduce_scatter_transport.py ccdl_comm_refactor/tests/test_async_shard_pipeline.py
git commit -m "refactor(ccdl_comm): align reduce-scatter completion"
```

### Task 6: A6000 synchronous and asynchronous benchmark

**Files:**
- Create: `ccdl_comm_refactor/tests/distributed/async_completion_perf.py`
- Create: `ccdl_comm_refactor/tests/test_async_completion_perf_script.py`
- Create: `ccdl_comm_refactor/tests/benchmarks/reports/async_completion_20260729/README.md`
- Create: result JSON files under `ccdl_comm_refactor/tests/benchmarks/reports/async_completion_20260729/raw/`

**Interfaces:**
- Produces CLI arguments for strategy, topology method, tensor elements,
  compression bit, group size, warm-up iterations, measured iterations, and
  overlap compute duration.
- Produces JSON containing launch latency, launch-plus-wait latency, overlapped
  latency, throughput, relative L2 error, maximum absolute error, and runtime
  metadata.

- [ ] **Step 1: Write a failing script contract test**

Test parser defaults and required JSON keys without requiring CUDA.

- [ ] **Step 2: Verify RED**

Run:

```powershell
python -m pytest ccdl_comm_refactor\tests\test_async_completion_perf_script.py -q
```

Expected: failure because the benchmark module does not exist.

- [ ] **Step 3: Implement the benchmark**

Use CUDA events for GPU timing and `time.perf_counter()` for end-to-end host
timing. Measure synchronous, immediate async `wait()`, and async with a
controlled independent CUDA workload between launch and wait.

- [ ] **Step 4: Verify locally**

Run:

```powershell
python -m pytest ccdl_comm_refactor\tests\test_async_completion_perf_script.py -q
python -m pytest ccdl_comm_refactor\tests -q
```

Expected: all tests pass.

- [ ] **Step 5: Run A6000 two-rank and four-rank cases**

Run inside `ccdl-comm-a6000:cu126-torch25` with the repository mounted at
`/workspace`, using `torchrun --nproc_per_node=2` and
`torchrun --nproc_per_node=4`. Test P2P, topology overlap, all-gather,
all-reduce, and reduce-scatter with identical inputs and settings.

- [ ] **Step 6: Record and commit evidence**

```powershell
git add ccdl_comm_refactor/tests/distributed/async_completion_perf.py ccdl_comm_refactor/tests/test_async_completion_perf_script.py ccdl_comm_refactor/tests/benchmarks/reports/async_completion_20260729
git commit -m "test(ccdl_comm): benchmark async completion"
```

### Task 7: Final verification and push

**Files:**
- Verify all changed files and commit history.

- [ ] **Step 1: Run the full local suite**

```powershell
python -m pytest ccdl_comm_refactor\tests -q
git diff --check
git status --short --branch
```

Expected: zero failures, no whitespace errors, and no uncommitted changes.

- [ ] **Step 2: Confirm remote artifacts and push**

```powershell
git push origin wj_dev
```

Expected: `wj_dev` is updated successfully.
