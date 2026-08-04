# Task 12.1 Fused ReducedShard Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fuse compressed ReducedShard dequant-reduce-mean into one CUDA launch and provide zero-allocation caller-owned and explicitly leased output buffers without weakening the safe default ownership contract.

**Architecture:** Extend the existing fused dequant-reduce CUDA kernels with a divisor and expose a non-EF in-place ABI. Bind that ABI at CUDA compile time, propagate an optional output target through `CompressedReduceScatterExecutor`, and build an opt-in output lease on the existing stream-safe `CudaWorkspacePool`; default outputs remain unpooled. Task 12.1 is a prerequisite for Task 13 topology pipelines.

**Tech Stack:** Python 3.10+, PyTorch distributed/NCCL, C++17, CUDA, pybind11, pytest, torch.profiler, RTX A6000 Docker environment.

## Global Constraints

- GPU performance is the first priority, but no fast path may weaken correctness, fallback transparency, or buffer ownership.
- The default `run(tensor)` result remains caller-owned and is never returned automatically to a pool.
- The hot path performs no policy selection and creates no per-rank restored tensors.
- Fused production support is initially linear INT8, group size 64, top-k 0, contiguous FP16/BF16/FP32, and at most 8 input ranks.
- Unsupported inputs use an explicit, preallocated fallback and record the reason; they must not be reported as fused.
- Local verification is required before A6000 testing. A6000 gates use 2 and 4 GPUs only.
- Every independently reviewable task ends in a Conventional Commit compliant with `CONTRIBUTING.md`.

---

### Task 1: Add the non-EF fused dequant-reduce-mean CUDA ABI

**Files:**
- Modify: `ccdl_comm/csrc/quantization/dequant_reduce_kernel.cu`
- Modify: `ccdl_comm/csrc/quantization/dequant_api.cuh`
- Modify: `ccdl_comm/csrc/pybind.cpp`
- Modify: `ccdl_comm/quantization/codec.py`
- Create: `tests/cuda/test_fused_reduced_shard.py`
- Modify: `tests/test_quantization_codec.py`

**Interfaces:**
- Consumes: existing packed INT8 payload layout and `can_use_fused_dequant_reduce(...)` capability checks.
- Produces: `inplace_dequantize_reduce_mean(inputs, output, config, *, extension_status, reduce) -> bool`.
- Produces native ABI: `inplace_dequantize_reduce_mean(inputs, output, group_size, topk, bit, quant_type, compact, divisor) -> bool`.

- [ ] **Step 1: Write failing codec ABI tests**

Add tests that require the codec to call the new symbol with the exact divisor and preserve output identity:

```python
def test_inplace_dequantize_reduce_mean_calls_native_symbol() -> None:
    output = FakeTensor((0.0, 0.0))
    extension = FakeExtension(fused_result=True)
    result = inplace_dequantize_reduce_mean(
        ["p0", "p1", "p2", "p3"],
        output,
        CompressionConfig(bit=8, group_size=64),
        extension_status=CudaExtensionStatus(True, extension),
        reduce="mean",
    )
    assert result is True
    assert extension.calls == [("fused_mean", output, 4)]
```

Also require `reduce="sum"` to pass divisor 1, reject unknown reductions, and return `False` when the native symbol declines.

- [ ] **Step 2: Run the RED tests**

Run:

```bash
python -m pytest tests/test_quantization_codec.py tests/cuda/test_fused_reduced_shard.py -q
```

Expected: FAIL because `inplace_dequantize_reduce_mean` and its pybind symbol do not exist.

- [ ] **Step 3: Add native capability and launch tests**

Parameterize CUDA tests over FP16/BF16/FP32, input counts 1/2/3/4/5/8, compact true/false, and non-group-aligned logical numel represented by a padded output. Require:

```python
assert extension.inplace_dequantize_reduce_mean(
    payloads, output, 64, 0, 8, extension.Linear, compact, len(payloads)
)
assert output.data_ptr() == original_ptr
torch.testing.assert_close(output, reference_mean, rtol=2e-2, atol=2e-2)
```

Add negative cases for INT4, group size 32, top-k, non-linear quantization, more than 8 inputs, non-contiguous output, wrong payload size, CPU output, and zero divisor. Negative capability cases return `False`; invalid divisor raises.

- [ ] **Step 4: Implement the CUDA ABI**

Change `dequant_reduce_fused_16bit_kernel` and `dequant_reduce_fused_fp32_kernel` to accept `float inv_divisor` and write `sum * inv_divisor`. Existing sum callers pass `1.0f`. Add:

```cpp
bool try_inplace_dequantize_reduce_mean_fused(
    std::vector<torch::Tensor> inputs,
    torch::Tensor output,
    int64_t group_size,
    int64_t topk,
    int64_t bit,
    QuantType quant_type,
    bool compact,
    int64_t divisor
);
```

The function checks `divisor > 0`, reuses `can_use_fused_dequant_reduce`, launches exactly one existing fused kernel with `1.0f / divisor`, and returns whether the launch was selected. Bind it in pybind as `inplace_dequantize_reduce_mean`.

- [ ] **Step 5: Implement the Python codec and allocation-free fallback**

Add:

```python
def inplace_dequantize_reduce_mean(
    buffers: list[object],
    output: object,
    config: CompressionConfig,
    *,
    extension_status: CudaExtensionStatus | None = None,
    reduce: str = "mean",
) -> bool:
    divisor = len(buffers) if reduce == "mean" else 1
    return bool(module.inplace_dequantize_reduce_mean(..., divisor))
```

When `dequantize_reduce_tensors(..., output=output, reduce="mean")` uses the fallback, call `output.div_(len(buffers))`; do not assign `output / len(buffers)` to a new tensor. Tests must assert output identity in both fused and fallback paths.

- [ ] **Step 6: Build and verify on A6000**

In `/home/user/wangjun/lowbit_comm_task12_1_validation/ccdl_comm_refactor`:

```bash
CCDL_COMM_BUILD_CUDA=1 TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=2 \
python setup.py build_ext --inplace
python -m pytest tests/test_quantization_codec.py tests/cuda/test_fused_reduced_shard.py -q
```

Expected: all tests pass; profiler test observes one `dequant_reduce_fused_*` launch and no separate divide kernel.

- [ ] **Step 7: Commit Task 1**

```bash
git add ccdl_comm/csrc/quantization/dequant_reduce_kernel.cu \
  ccdl_comm/csrc/quantization/dequant_api.cuh \
  ccdl_comm/csrc/pybind.cpp \
  ccdl_comm/quantization/codec.py \
  tests/cuda/test_fused_reduced_shard.py \
  tests/test_quantization_codec.py
git commit -m "perf(ccdl_comm): fuse reduced shard dequant mean"
```

---

### Task 2: Propagate caller-owned output through transport and Executor

**Files:**
- Modify: `ccdl_comm/communication/reduce_scatter_transport.py`
- Modify: `ccdl_comm/cuda/compiler.py`
- Modify: `ccdl_comm/cuda/executors.py`
- Modify: `ccdl_comm/collectives/reduce_scatter.py`
- Modify: `ccdl_comm/build/setuptools.py`
- Modify: `tests/test_reduce_scatter_transport.py`
- Modify: `tests/cuda/test_cuda_backend_compile.py`
- Modify: `tests/test_reduce_scatter_api.py`
- Modify: `tests/test_setuptools_build.py`
- Modify: `tests/distributed/sharded_reduce_scatter_perf.py`

**Interfaces:**
- Consumes: Task 1 `inplace_dequantize_reduce_mean(...) -> bool`.
- Produces: `CompressedReduceScatterExecutor.run(tensor, *, out=None) -> CollectiveWork[ReducedShard]`.
- Produces: transport keyword `out`, validated before quantization or communication begins.

- [ ] **Step 1: Write failing caller-owned output tests**

Require the transport and compiled Executor to write directly into the provided buffer:

```python
work = executor.run(bucket, out=output)
reduced = work.wait()
assert reduced.shard is output
assert reduced.metadata["output_ownership"] == "caller"
assert reduced.metadata["fused_dequant_reduce"] is True
```

Add pre-communication failures for incorrect numel, shape, dtype, device, non-contiguous output, and input/output alias. Fake transport tests must assert neither quantize nor `all_to_all` ran after validation failure.

- [ ] **Step 2: Run the RED tests**

Run:

```bash
python -m pytest tests/test_reduce_scatter_transport.py \
  tests/cuda/test_cuda_backend_compile.py \
  tests/test_reduce_scatter_api.py -q
```

Expected: FAIL because `run(..., out=...)` and transport `out` are unsupported.

- [ ] **Step 3: Add the transport output contract**

Extend the factory callable with `out: Any | None = None`. Before workspace acquisition or communication, validate the flattened output against `ChunkPlan.shard_numel`, source dtype/device, contiguity, and storage pointer. Pass caller output ahead of allocator/cache output:

```python
output_workspace = out or _allocate_reduced_workspace(...)
```

Do not use boolean truth testing on tensors. The actual selection is:

```python
output_workspace = out
if output_workspace is None:
    output_workspace = _allocate_reduced_workspace(...)
```

Bind Task 1 codec through `fused_dequantize_reduce`. Record `output_ownership` as `caller`, `pool`, or `allocated`, plus a concrete `fused_dequant_reduce_reason` when fallback runs.

- [ ] **Step 4: Extend the compiled Executor without per-run strategy selection**

Change only the ReducedShard executor operation signature:

```python
def run(self, tensor: object, *, out: object | None = None) -> CollectiveWork[ReducedShard]:
    result = self._operation(tensor, out=out)
    return bind_execution_work(result, self.execution_info, self.execution_counters)
```

The compiler pre-binds `inplace_dequantize_reduce_mean`, ChunkPlan, dtype, divisor, workspace provider, and completion manager. Capability is recorded as `cuda_fused_reduced_shard`; unsupported static configuration records the exact fallback reason at compile time.

- [ ] **Step 5: Keep source and built-package behavior identical**

Add `"ccdl_comm.cuda.transports"` to the explicit package list in `ccdl_comm/build/setuptools.py`. Extend the build test so a CUDA-enabled editable build contains the Task 12 transport package; this prevents source-tree tests from passing while an installed wheel cannot import `ChunkPlan`. Extend the existing shard benchmark with `--output-mode=default|caller`; caller mode preallocates one shard output and passes it to every sequential iteration.

- [ ] **Step 6: Verify local and A6000 behavior**

Run locally:

```bash
python -m pytest tests/test_reduce_scatter_transport.py \
  tests/cuda/test_cuda_backend_compile.py \
  tests/test_reduce_scatter_api.py \
  tests/test_setuptools_build.py -q
```

Run on A6000 after rebuilding:

```bash
torchrun --standalone --nproc-per-node=4 \
  tests/distributed/sharded_reduce_scatter_perf.py \
  --numel=8388608 --dtype=fp16 --bit=8 --group-size=64 \
  --transport=compressed --output-mode=caller --warmup=5 --repeat=20 \
  --output-json=/tmp/task12_1_caller_out_smoke.json
```

Expected: caller output pointer is stable, metadata reports fused/caller mode, relative L2 is at most 0.02, and no full-gradient all-gather occurs in the candidate path.

- [ ] **Step 7: Commit Task 2**

```bash
git add ccdl_comm/communication/reduce_scatter_transport.py \
  ccdl_comm/cuda/compiler.py ccdl_comm/cuda/executors.py \
  ccdl_comm/collectives/reduce_scatter.py ccdl_comm/build/setuptools.py \
  tests/test_reduce_scatter_transport.py tests/cuda/test_cuda_backend_compile.py \
  tests/test_reduce_scatter_api.py tests/test_setuptools_build.py \
  tests/distributed/sharded_reduce_scatter_perf.py
git commit -m "perf(ccdl_comm): support caller-owned shard output"
```

---

### Task 3: Add explicit stream-safe ReducedShard output leases

**Files:**
- Modify: `ccdl_comm/cuda/workspace.py`
- Modify: `ccdl_comm/cuda/executors.py`
- Modify: `ccdl_comm/cuda/compiler.py`
- Modify: `ccdl_comm/cuda/__init__.py`
- Modify: `tests/cuda/test_cuda_workspace_pool.py`
- Modify: `tests/cuda/test_cuda_backend_compile.py`
- Modify: `tests/test_async_shard_pipeline.py`

**Interfaces:**
- Consumes: existing `CudaWorkspacePool.acquire(...) -> WorkspaceLease` and `CudaCompletionManager.record_for(...)`.
- Produces: `CudaOutputLease.buffer`, `CudaOutputLease.release_after(value_or_completion)`, `CudaOutputLease.release_unused()`, and `CompressedReduceScatterExecutor.acquire_output()`.
- Produces: `executor.run(tensor, out=lease)` with executor ownership validation.

- [ ] **Step 1: Write failing lease ownership tests**

Cover acquire, pending state, stream handoff, release, and misuse:

```python
lease = executor.acquire_output()
work = executor.run(bucket, out=lease)
shard = work.wait()
assert shard.shard is lease.buffer
lease.release_after(shard.shard)
assert executor.workspace_pool.stats.in_flight_bytes > 0
completion.ready = True
second = executor.acquire_output()
assert second.buffer is lease.buffer
```

Require double release, `release_unused()` after a run, lease from a different executor, and reuse while active to raise deterministic errors. A never-used lease must support `release_unused()` without leaking pool capacity. Require an async Work to retain the lease buffer until completion, while never auto-releasing the output lease.

- [ ] **Step 2: Run the RED tests**

Run:

```bash
python -m pytest tests/cuda/test_cuda_workspace_pool.py \
  tests/cuda/test_cuda_backend_compile.py \
  tests/test_async_shard_pipeline.py -q
```

Expected: FAIL because `CudaOutputLease` and `acquire_output()` do not exist.

- [ ] **Step 3: Implement `CudaOutputLease`**

Wrap one `WorkspaceLease` with an executor identity token and completion manager:

```python
class CudaOutputLease:
    @property
    def buffer(self) -> object: ...

    def mark_used(self, owner_token: object) -> object: ...

    def release_after(self, value_or_completion: object) -> None:
        completion = _as_completion(value_or_completion)
        self._lease.release(completion=completion)

    def release_unused(self) -> None: ...
```

`mark_used` rejects foreign owner tokens and multiple concurrent runs. `release_after` records a CUDA event for tensors, accepts an existing object implementing `query()`, and resets active state only after handing the workspace to the pool. The pool remains responsible for event-ready reuse.
`release_unused()` is valid only before `mark_used`; it records the acquisition stream and returns the untouched buffer safely.

- [ ] **Step 4: Bind output acquisition at compile time**

Construct a dedicated reduced-output `WorkspaceKey` from compile context and ChunkPlan. Attach a zero-argument acquisition callable and an opaque owner token to `CompressedReduceScatterExecutor`. `acquire_output()` raises a clear error when workspace caching is disabled or the budget cannot represent one shard output.

`run(..., out=lease)` unwraps only a lease owned by that executor. Plain tensors remain caller-owned outputs. The transport receives the unwrapped tensor and never sees CUDA lease policy.

- [ ] **Step 5: Verify pool budgets and stream ownership**

Add tests for max entries/bytes, pending event handoff with `wait_stream`, LRU eviction after release, and pool stats. Run:

```bash
python -m pytest tests/cuda/test_cuda_workspace_pool.py \
  tests/cuda/test_cuda_backend_compile.py \
  tests/test_async_shard_pipeline.py -q
```

Expected: all pass; an output buffer is never reused before explicit release plus completion ordering.

- [ ] **Step 6: Commit Task 3**

```bash
git add ccdl_comm/cuda/workspace.py ccdl_comm/cuda/executors.py \
  ccdl_comm/cuda/compiler.py ccdl_comm/cuda/__init__.py \
  tests/cuda/test_cuda_workspace_pool.py tests/cuda/test_cuda_backend_compile.py \
  tests/test_async_shard_pipeline.py
git commit -m "perf(ccdl_comm): pool leased reduced shard outputs"
```

---

### Task 4: Gate correctness, launches, allocations, and A6000 performance

**Files:**
- Create: `tests/distributed/fused_reduced_shard_perf.py`
- Create: `tests/benchmarks/fused_reduced_shard_gate.py`
- Create: `tests/test_fused_reduced_shard_perf_script.py`
- Create: `tests/benchmarks/reports/task12_1_fused_reduced_shard/README.md`
- Create: `tests/benchmarks/reports/task12_1_fused_reduced_shard/raw/*.json`
- Modify: `docs/superpowers/plans/2026-07-31-gpu-first-ccdl-development.md`

**Interfaces:**
- Consumes: Task 2 caller-owned output and Task 3 `CudaOutputLease`.
- Produces: reproducible ABBA benchmark JSON and a gate evaluator for all required 2/4-GPU cases.

- [ ] **Step 1: Write failing benchmark contract tests**

Require the script to expose `--bucket-mib`, `--world-size` through torchrun, `--warmup`, `--repeat`, `--output-json`, and `--mode=caller|lease`. JSON must contain:

```python
required = {
    "world_size", "bucket_mib", "measurement_order",
    "task12_ms", "fused_ms", "speedup",
    "task12_peak_memory_bytes", "fused_peak_memory_bytes",
    "steady_allocation_bytes", "relative_l2", "max_abs_error",
    "fused_kernel_launches", "fallback_used", "output_mode",
}
```

The gate evaluator requires five runs for every `(world_size, bucket_mib, output_mode)` combination where world size is 2/4, bucket is 1/16/64 MiB, and output mode is caller/lease.

- [ ] **Step 2: Run the RED tests**

Run:

```bash
python -m pytest tests/test_fused_reduced_shard_perf_script.py -q
```

Expected: FAIL because the benchmark and gate evaluator do not exist.

- [ ] **Step 3: Implement the ABBA benchmark**

Compile two stable paths once per process:

- Task 12 baseline: same compressed transport with fused callback disabled and default unpooled result.
- Task 12.1 candidate: fused callback enabled with one pre-acquired caller or leased output.

Measure in `task12-fused-fused-task12` order. Reset CUDA peak memory statistics per position, synchronize only outside timed operations, gather rank metadata once after measurement, and validate output against an FP16 all-reduce reference.

- [ ] **Step 4: Implement the gate evaluator**

For each required case, use the median of five independent runs. Fail when:

```python
if result["fallback_used"]:
    failures.append("production fused path used fallback")
if result["fused_kernel_launches"] != 1:
    failures.append("expected one fused dequant-reduce-mean launch")
if result["steady_allocation_bytes"] != 0:
    failures.append("steady-state allocation is non-zero")
if bucket_mib in {16, 64} and fused_median_ms > task12_median_ms:
    failures.append("large-bucket latency regressed")
if result["relative_l2"] > 0.02 or result["non_finite"] != 0:
    failures.append("accuracy gate failed")
```

Also require lease and caller modes to produce equal accuracy and stable output pointers.

- [ ] **Step 5: Run A6000 2/4-GPU matrix**

Use Docker image `ccdl-comm-a6000:cu126-torch25`, `--gpus all`, and `--shm-size=8g`. For each world size, bucket, mode, and run index 1–5:

```bash
torchrun --standalone --nproc-per-node=${WORLD_SIZE} \
  tests/distributed/fused_reduced_shard_perf.py \
  --bucket-mib=${BUCKET_MIB} --dtype=fp16 --bit=8 --group-size=64 \
  --mode=${MODE} --warmup=20 --repeat=100 \
  --output-json=tests/benchmarks/reports/task12_1_fused_reduced_shard/raw/${WORLD_SIZE}gpu_${BUCKET_MIB}mib_${MODE}_run${RUN}.json
```

Then run:

```bash
python tests/benchmarks/fused_reduced_shard_gate.py \
  --results-dir tests/benchmarks/reports/task12_1_fused_reduced_shard/raw
```

Expected: exit 0 with all 12 case groups represented by five runs each; 16/64 MiB candidate median does not regress on 2 or 4 GPUs.

- [ ] **Step 6: Run full regression and independent review**

Run locally:

```bash
python -m pytest -q
python -m ruff check ccdl_comm tests
git diff --check
```

Run in the rebuilt A6000 container:

```bash
python -m pytest -q
```

Request independent review focused on CUDA bounds, output aliasing, Work/lease lifetime, truthful fallback metadata, benchmark fairness, and accidental full-gradient restoration. Resolve all Critical and Important findings before commit.

- [ ] **Step 7: Write the report and close Task 12.1**

The report records hardware/software identity, exact commit base, all medians, accuracy, profiler launch count, allocation evidence, and any configurations that remain fallback. Update the GPU-first master plan so Task 12.1 precedes Task 13 and record Gate G5a:

```text
G5a ReducedShard fusion: one fused dequant-reduce-mean launch, explicit output
ownership, zero steady allocation, no 16/64 MiB regression on 2/4 A6000.
```

- [ ] **Step 8: Commit Task 4**

```bash
git add tests/distributed/fused_reduced_shard_perf.py \
  tests/benchmarks/fused_reduced_shard_gate.py \
  tests/test_fused_reduced_shard_perf_script.py \
  tests/benchmarks/reports/task12_1_fused_reduced_shard \
  docs/superpowers/plans/2026-07-31-gpu-first-ccdl-development.md
git commit -m "test(ccdl_comm): gate fused reduced shard performance"
```

---

## Completion Gate

Task 12.1 is complete only when all four commits exist, local and A6000 full suites pass, profiler evidence shows one production fused kernel, output lease stress tests pass, and the 2/4-GPU 16/64 MiB median latency gate passes. If any large-bucket case regresses, keep the capability-gated fallback and do not start Task 13.
