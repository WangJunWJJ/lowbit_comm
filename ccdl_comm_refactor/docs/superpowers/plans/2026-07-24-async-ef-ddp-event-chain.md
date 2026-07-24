# Async Error Feedback DDP Event Chain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an opt-in event-gated async error-feedback path for CCDL's DDP all-gather hook.

**Architecture:** Add a small import-safe completion abstraction, then add an async bucket pipeline that sequences gather wait, fused dequant-reduce, error-feedback update, completion wait, and DDP Future completion. Integrate it behind `async_error_feedback=False` by default so the current synchronous EF path remains the stable fallback.

**Tech Stack:** Python 3.10+, pytest, PyTorch DDP Future API, optional CUDA events through lazy torch imports, existing CCDL CUDA extension for dequant-reduce.

## Global Constraints

- Do not rewrite NCCL.
- Do not move EF update into CUDA in this phase.
- Do not make async EF the default in this phase.
- Do not break safe import when torch or CUDA is unavailable.
- Use TDD for every behavior change.
- Commit after each independently testable task.

---

## File Structure

- Create `ccdl_comm_refactor/ccdl_comm/communication/cuda_completion.py`: import-safe completion/event abstraction.
- Create `ccdl_comm_refactor/tests/test_cuda_completion.py`: unit tests for no-op and fake CUDA completion behavior.
- Create `ccdl_comm_refactor/ccdl_comm/communication/async_pipeline.py`: async bucket pipeline that owns callback ordering.
- Create `ccdl_comm_refactor/tests/test_async_bucket_pipeline.py`: unit tests for operation ordering and exception propagation.
- Modify `ccdl_comm_refactor/ccdl_comm/communication/ddp_hook.py`: add `async_error_feedback` flag and route EF buckets into async pipeline when explicitly enabled.
- Modify `ccdl_comm_refactor/tests/test_ddp_comm_hook.py`: add hook-level tests for async EF opt-in and existing sync fallback.
- Modify `ccdl_comm_refactor/tests/distributed/synthetic_ddp_compare.py`: expose `--async-error-feedback`.
- Modify `ccdl_comm_refactor/tests/test_synthetic_ddp_script.py`: verify benchmark flag and JSON metadata.

---

### Task 1: Completion Abstraction

**Files:**
- Create: `ccdl_comm_refactor/ccdl_comm/communication/cuda_completion.py`
- Test: `ccdl_comm_refactor/tests/test_cuda_completion.py`

**Interfaces:**
- Produces: `NoopCompletion.wait() -> None`
- Produces: `NoopCompletion.synchronize() -> None`
- Produces: `CudaCompletion(event: Any | None).wait() -> None`
- Produces: `CudaCompletion.synchronize() -> None`
- Produces: `CudaCompletionManager.record_for(tensor: Any) -> CudaCompletion | NoopCompletion`

- [ ] **Step 1: Write failing tests**

```python
def test_noop_completion_is_safe_without_torch():
    completion = NoopCompletion()
    completion.wait()
    completion.synchronize()


def test_manager_records_event_for_cuda_tensor_with_injected_torch():
    calls = []

    class FakeEvent:
        def record(self):
            calls.append("record")

        def wait(self):
            calls.append("wait")

        def synchronize(self):
            calls.append("synchronize")

    class FakeCuda:
        Event = FakeEvent

        @staticmethod
        def is_available():
            return True

    class FakeTorch:
        cuda = FakeCuda

    class FakeTensor:
        is_cuda = True

    manager = CudaCompletionManager(torch_provider=lambda: FakeTorch)
    completion = manager.record_for(FakeTensor())
    completion.wait()
    completion.synchronize()
    assert calls == ["record", "wait", "synchronize"]
```

- [ ] **Step 2: Run test to verify RED**

Run: `python -m pytest ccdl_comm_refactor/tests/test_cuda_completion.py -q`

Expected: FAIL because `ccdl_comm.communication.cuda_completion` does not exist.

- [ ] **Step 3: Implement minimal completion abstraction**

Create `cuda_completion.py` with lazy torch handling and injected `torch_provider`.

- [ ] **Step 4: Run tests to verify GREEN**

Run: `python -m pytest ccdl_comm_refactor/tests/test_cuda_completion.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ccdl_comm_refactor/ccdl_comm/communication/cuda_completion.py ccdl_comm_refactor/tests/test_cuda_completion.py
git commit -m "feat(ccdl_comm): add cuda completion abstraction"
```

---

### Task 2: Async Bucket Pipeline

**Files:**
- Create: `ccdl_comm_refactor/ccdl_comm/communication/async_pipeline.py`
- Test: `ccdl_comm_refactor/tests/test_async_bucket_pipeline.py`

**Interfaces:**
- Consumes: `CudaCompletionManager.record_for(tensor)`
- Produces: `AsyncBucketPipeline(...).run() -> Any`

- [ ] **Step 1: Write failing tests**

```python
def test_async_pipeline_orders_gather_reduce_feedback_completion_and_future():
    calls = []
    outer = FakeFuture()
    work = FakeWork(calls)
    manager = FakeCompletionManager(calls)

    pipeline = AsyncBucketPipeline(
        gather_work=work,
        future=outer,
        dequantize_reduce=lambda gathered: calls.append(("reduce", gathered)) or "restored",
        update_feedback=lambda restored: calls.append(("feedback", restored)),
        advance_policy=lambda: calls.append("advance"),
        completion_manager=manager,
    )

    returned = pipeline.run()

    assert returned is outer
    assert outer.result == "restored"
    assert calls == [
        "get_future",
        "then",
        "wait",
        ("reduce", ["rank0", "rank1"]),
        ("feedback", "restored"),
        "advance",
        ("record", "restored"),
        "completion_wait",
        ("set_result", "restored"),
    ]
```

- [ ] **Step 2: Run test to verify RED**

Run: `python -m pytest ccdl_comm_refactor/tests/test_async_bucket_pipeline.py -q`

Expected: FAIL because `async_pipeline` does not exist.

- [ ] **Step 3: Implement minimal pipeline**

Create `AsyncBucketPipeline` that attaches to `gather_work.get_future().then(callback)` when available, otherwise runs callback inline. The callback must call `gather_work.wait()` before reduce, call feedback update and policy advance before completion record, call `completion.wait()` before `future.set_result(restored)`, and return `restored`.

- [ ] **Step 4: Run tests to verify GREEN**

Run: `python -m pytest ccdl_comm_refactor/tests/test_async_bucket_pipeline.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ccdl_comm_refactor/ccdl_comm/communication/async_pipeline.py ccdl_comm_refactor/tests/test_async_bucket_pipeline.py
git commit -m "feat(ccdl_comm): add async bucket pipeline"
```

---

### Task 3: DDP Hook Async EF Opt-In

**Files:**
- Modify: `ccdl_comm_refactor/ccdl_comm/communication/ddp_hook.py`
- Modify: `ccdl_comm_refactor/tests/test_ddp_comm_hook.py`

**Interfaces:**
- Consumes: `AsyncBucketPipeline`
- Adds parameter: `create_ddp_comm_hook(..., async_error_feedback: bool = False, ...)`

- [ ] **Step 1: Write failing tests**

Add one test proving `async_error_feedback=True` allows EF buckets to use async gather and still update feedback. Keep the existing test proving `async_error_feedback=False` uses sync fallback.

- [ ] **Step 2: Run test to verify RED**

Run: `python -m pytest ccdl_comm_refactor/tests/test_ddp_comm_hook.py -q`

Expected: FAIL because `async_error_feedback` is not accepted and EF still falls back to sync.

- [ ] **Step 3: Implement minimal DDP hook integration**

Import `AsyncBucketPipeline` and `CudaCompletionManager`. Add the flag. Use async no-EF path for no-EF buckets. Use async EF pipeline only when `async_gather and async_error_feedback and (feedback_decision.apply or feedback_decision.update)`.

- [ ] **Step 4: Run tests to verify GREEN**

Run: `python -m pytest ccdl_comm_refactor/tests/test_ddp_comm_hook.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ccdl_comm_refactor/ccdl_comm/communication/ddp_hook.py ccdl_comm_refactor/tests/test_ddp_comm_hook.py
git commit -m "feat(ccdl_comm): gate async error feedback ddp hook"
```

---

### Task 4: Benchmark Flag

**Files:**
- Modify: `ccdl_comm_refactor/tests/distributed/synthetic_ddp_compare.py`
- Modify: `ccdl_comm_refactor/tests/test_synthetic_ddp_script.py`

**Interfaces:**
- Adds CLI flag: `--async-error-feedback {true,false}`
- Adds JSON metadata key: `"async_error_feedback"`

- [ ] **Step 1: Write failing test**

Add assertions that `--async-error-feedback` exists, is passed to `create_ddp_comm_hook`, and is included in JSON output.

- [ ] **Step 2: Run test to verify RED**

Run: `python -m pytest ccdl_comm_refactor/tests/test_synthetic_ddp_script.py -q`

Expected: FAIL because the script does not expose the new flag.

- [ ] **Step 3: Implement minimal benchmark flag**

Add parser argument, pass `async_error_feedback=(args.async_error_feedback == "true")`, and include JSON metadata.

- [ ] **Step 4: Run tests to verify GREEN**

Run: `python -m pytest ccdl_comm_refactor/tests/test_synthetic_ddp_script.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ccdl_comm_refactor/tests/distributed/synthetic_ddp_compare.py ccdl_comm_refactor/tests/test_synthetic_ddp_script.py
git commit -m "test(ccdl_comm): benchmark async error feedback"
```

---

### Task 5: Verification

**Files:**
- No source changes expected.

**Interfaces:**
- Consumes all previous tasks.

- [ ] **Step 1: Run full local test suite**

Run: `python -m pytest ccdl_comm_refactor/tests -q -rs --tb=short`

Expected: PASS with only documented local skips for missing torch/setuptools if the Windows environment lacks them.

- [ ] **Step 2: Run remote A6000 container tests**

Run inside `/home/user/wangjun/ccdl-master/ccdl_comm_refactor` with `PYTHONPATH` set and compatible CUDA extension available:

```bash
python -m pytest tests -q -rs --tb=short
```

Expected: PASS with only `torch_npu` skip.

- [ ] **Step 3: Run 2/4 GPU synthetic benchmark**

Run baseline, sync EF, async no-EF, and async EF with `--async-error-feedback true`.

Expected:

- all runs complete with finite loss;
- async EF train loss is close to sync EF on the short synthetic workload;
- final report states whether async EF should remain opt-in.

---

## Self-Review

- Spec coverage: completion abstraction, pipeline ordering, hook integration, benchmark flag, local/remote verification are all mapped to tasks.
- Placeholder scan: no TBD/TODO/later placeholders remain.
- Type consistency: `async_error_feedback`, `AsyncBucketPipeline`, and `CudaCompletionManager` names are consistent across tasks.
