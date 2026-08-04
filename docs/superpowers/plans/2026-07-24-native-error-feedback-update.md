# Native Error Feedback Update Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move CCDL error-feedback residual update into native CUDA/C++ code and expose it through safe Python wrappers.

**Architecture:** Add a CUDA kernel for `residual = prepared - restored`, export it through pybind, wrap it in `ccdl_comm.quantization.codec`, then use it from the DDP async EF path when available. A later task adds a combined dequant-reduce + residual-update entry point.

**Tech Stack:** Python 3.10+, pytest, PyTorch C++ extension, CUDA C++, pybind11, existing CCDL quantization extension.

## Global Constraints

- Preserve the current error-feedback formula.
- Keep native update opt-in through symbol availability and safe fallback.
- Do not make async error feedback the default.
- Do not change Ascend/CANN behavior.
- Use TDD for every source behavior change.
- Commit after each independently testable task.

---

## File Structure

- Modify `ccdl_comm_refactor/ccdl_comm/csrc/quantization/dequant_api.cuh`: declare native EF update functions.
- Modify `ccdl_comm_refactor/ccdl_comm/csrc/quantization/dequant_reduce_kernel.cu`: implement CUDA EF update kernel.
- Modify `ccdl_comm_refactor/ccdl_comm/csrc/pybind.cpp`: export native EF update.
- Modify `ccdl_comm_refactor/tests/test_pybind_exports.py`: source-level export and kernel tests.
- Modify `ccdl_comm_refactor/ccdl_comm/quantization/codec.py`: add Python wrapper.
- Modify `ccdl_comm_refactor/tests/test_quantization_codec.py`: wrapper unit tests.
- Modify `ccdl_comm_refactor/ccdl_comm/communication/ddp_hook.py`: select native update in async EF path when provided.
- Modify `ccdl_comm_refactor/tests/test_ddp_comm_hook.py`: hook selection tests.
- Add or modify CUDA smoke tests after native symbol is available.

---

### Task 1: Native EF Update Source and Export

**Files:**
- Modify: `ccdl_comm_refactor/tests/test_pybind_exports.py`
- Modify: `ccdl_comm_refactor/ccdl_comm/csrc/quantization/dequant_api.cuh`
- Modify: `ccdl_comm_refactor/ccdl_comm/csrc/quantization/dequant_reduce_kernel.cu`
- Modify: `ccdl_comm_refactor/ccdl_comm/csrc/pybind.cpp`

**Interfaces:**
- Produces: `void inplace_error_feedback_update(torch::Tensor prepared, torch::Tensor restored, torch::Tensor residual)`
- Produces pybind symbol: `inplace_error_feedback_update`

- [ ] **Step 1: Write failing source tests**

Add assertions that:

```python
assert "inplace_error_feedback_update" in header_source
assert "error_feedback_update_kernel" in kernel_source
assert 'm.def("inplace_error_feedback_update", &inplace_error_feedback_update);' in pybind_source
```

- [ ] **Step 2: Run source test to verify RED**

Run: `python -m pytest ccdl_comm_refactor/tests/test_pybind_exports.py -q`

Expected: FAIL because the symbol and kernel do not exist.

- [ ] **Step 3: Implement minimal CUDA source**

Declare the function in `dequant_api.cuh`. Implement dtype-dispatched CUDA kernel in `dequant_reduce_kernel.cu`. Export through pybind.

- [ ] **Step 4: Run source test to verify GREEN**

Run: `python -m pytest ccdl_comm_refactor/tests/test_pybind_exports.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ccdl_comm_refactor/tests/test_pybind_exports.py ccdl_comm_refactor/ccdl_comm/csrc/quantization/dequant_api.cuh ccdl_comm_refactor/ccdl_comm/csrc/quantization/dequant_reduce_kernel.cu ccdl_comm_refactor/ccdl_comm/csrc/pybind.cpp
git commit -m "feat(ccdl_comm): add native error feedback update kernel"
```

---

### Task 2: Python Codec Wrapper

**Files:**
- Modify: `ccdl_comm_refactor/ccdl_comm/quantization/codec.py`
- Modify: `ccdl_comm_refactor/tests/test_quantization_codec.py`

**Interfaces:**
- Produces: `update_error_feedback_residual(prepared, restored, residual, *, extension_status=None) -> object`

- [ ] **Step 1: Write failing wrapper tests**

Add a fake extension test:

```python
def test_update_error_feedback_residual_calls_native_inplace_symbol():
    extension = FakeExtension()
    status = CudaExtensionStatus(available=True, module=extension)
    result = update_error_feedback_residual("prepared", "restored", "residual", extension_status=status)
    assert result == "residual"
    assert extension.calls == [("prepared", "restored", "residual")]
```

Add a missing-symbol test:

```python
with pytest.raises(CCDLUnavailableError, match="inplace_error_feedback_update"):
    update_error_feedback_residual(object(), object(), object(), extension_status=status)
```

- [ ] **Step 2: Run wrapper tests to verify RED**

Run: `python -m pytest ccdl_comm_refactor/tests/test_quantization_codec.py -q`

Expected: FAIL because wrapper does not exist.

- [ ] **Step 3: Implement wrapper**

Use `_require_available_extension` and `_get_required_attr(module, "inplace_error_feedback_update")`, call it, return `residual`.

- [ ] **Step 4: Run wrapper tests to verify GREEN**

Run: `python -m pytest ccdl_comm_refactor/tests/test_quantization_codec.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ccdl_comm_refactor/ccdl_comm/quantization/codec.py ccdl_comm_refactor/tests/test_quantization_codec.py
git commit -m "feat(ccdl_comm): wrap native error feedback update"
```

---

### Task 3: DDP Hook Native Update Selection

**Files:**
- Modify: `ccdl_comm_refactor/ccdl_comm/communication/ddp_hook.py`
- Modify: `ccdl_comm_refactor/tests/test_ddp_comm_hook.py`

**Interfaces:**
- Consumes: `update_error_feedback_residual(prepared, restored, residual)`
- Adds optional injected callable: `native_error_feedback_update`

- [ ] **Step 1: Write failing hook test**

Add a test that passes `native_error_feedback_update` and verifies async EF pipeline calls it instead of `feedback.update`.

- [ ] **Step 2: Run hook tests to verify RED**

Run: `python -m pytest ccdl_comm_refactor/tests/test_ddp_comm_hook.py -q`

Expected: FAIL because hook does not accept native update callable.

- [ ] **Step 3: Implement hook integration**

Add optional parameter `native_error_feedback_update`. In async EF pipeline, when `feedback_decision.update` is true and native callable is available, update the existing residual storage if present; otherwise allocate/store through fallback for the first bucket iteration. Keep Python fallback unchanged.

- [ ] **Step 4: Run hook tests to verify GREEN**

Run: `python -m pytest ccdl_comm_refactor/tests/test_ddp_comm_hook.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ccdl_comm_refactor/ccdl_comm/communication/ddp_hook.py ccdl_comm_refactor/tests/test_ddp_comm_hook.py
git commit -m "perf(ccdl_comm): use native error feedback update in ddp hook"
```

---

### Task 4: CUDA Smoke and Remote Build

**Files:**
- Modify: `ccdl_comm_refactor/tests/test_cuda_extension_smoke.py`

**Interfaces:**
- Consumes pybind symbol: `inplace_error_feedback_update`

- [ ] **Step 1: Add smoke test**

Add a CUDA-only test that creates `prepared`, `restored`, and `residual`, calls native update, synchronizes, and compares residual to `prepared - restored`.

- [ ] **Step 2: Run local smoke test**

Run: `python -m pytest ccdl_comm_refactor/tests/test_cuda_extension_smoke.py -q`

Expected on Windows without torch: SKIP.

- [ ] **Step 3: Run remote build and smoke**

Run A6000 Docker build:

```bash
export CCDL_COMM_BUILD_CUDA=1
export TORCH_CUDA_ARCH_LIST=8.6
export MAX_JOBS=2
python setup.py build_ext --inplace
python -m pytest tests/test_cuda_extension_smoke.py tests/test_pybind_exports.py tests/test_quantization_codec.py -q
```

Expected: PASS.

- [ ] **Step 4: Commit smoke test**

```bash
git add ccdl_comm_refactor/tests/test_cuda_extension_smoke.py
git commit -m "test(ccdl_comm): verify native error feedback update"
```

---

### Task 5: Combined Native Entry Point

**Files:**
- Modify: `ccdl_comm_refactor/tests/test_pybind_exports.py`
- Modify: `ccdl_comm_refactor/ccdl_comm/csrc/quantization/dequant_api.cuh`
- Modify: `ccdl_comm_refactor/ccdl_comm/csrc/pybind.cpp`
- Modify: `ccdl_comm_refactor/ccdl_comm/quantization/codec.py`
- Modify: `ccdl_comm_refactor/tests/test_quantization_codec.py`

**Interfaces:**
- Produces pybind symbol: `dequantize_reduce_update_error_feedback`
- Produces Python wrapper: `dequantize_reduce_update_error_feedback(...)`

- [ ] **Step 1: Add failing source and wrapper tests**

Assert export exists and fake extension receives buffers, prepared, residual, config enums, dtype enum, and compact flag.

- [ ] **Step 2: Run tests to verify RED**

Run: `python -m pytest ccdl_comm_refactor/tests/test_pybind_exports.py ccdl_comm_refactor/tests/test_quantization_codec.py -q`

Expected: FAIL.

- [ ] **Step 3: Implement combined entry point**

In C++, allocate `restored`, call `inplace_dequantize_reduce`, then `inplace_error_feedback_update(prepared, restored, residual)`, and return `restored`.

- [ ] **Step 4: Run tests to verify GREEN**

Run: `python -m pytest ccdl_comm_refactor/tests/test_pybind_exports.py ccdl_comm_refactor/tests/test_quantization_codec.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ccdl_comm_refactor/tests/test_pybind_exports.py ccdl_comm_refactor/ccdl_comm/csrc/quantization/dequant_api.cuh ccdl_comm_refactor/ccdl_comm/csrc/pybind.cpp ccdl_comm_refactor/ccdl_comm/quantization/codec.py ccdl_comm_refactor/tests/test_quantization_codec.py
git commit -m "feat(ccdl_comm): combine dequant reduce and feedback update"
```

---

### Task 6: Full Verification and Benchmark

**Files:**
- No source changes expected.

- [ ] **Step 1: Run full local tests**

Run: `python -m pytest ccdl_comm_refactor/tests -q -rs --tb=short`

Expected: PASS with documented local skips.

- [ ] **Step 2: Run remote CUDA test suite**

Run A6000 Docker tests after rebuild.

Expected: PASS with only non-CUDA-platform skips.

- [ ] **Step 3: Run 2/4 GPU benchmark**

Compare:

- sync EF
- async EF safe fallback
- async EF native update
- async no-EF

Expected:

- finite loss for all runs;
- async EF native update no slower than prior event-gated async EF;
- report whether native update is safe for larger training.

---

## Self-Review

- Spec coverage: kernel, wrapper, DDP hook selection, smoke, combined entry point, and benchmark are covered.
- Placeholder scan: no TBD/TODO/later placeholders remain.
- Type consistency: native symbol and wrapper names are consistent across tasks.
