# CCDL CANN Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional Ascend CANN backend for CCDL compressed communication, starting with verifiable group-wise linear INT8 quant/dequant and ending with HCCL benchmark and DDP hook validation.

**Architecture:** Keep CUDA and Ascend backends side by side. CANN imports remain safe and optional; ParaScale injects the selected codec into the existing `compressed_all_reduce` and DDP hook APIs. Communication optimization first preserves the existing payload contract, then adds packed payload support after kernel correctness is proven.

**Tech Stack:** Python 3.10+, pytest, torch, torch-npu, CANN 9.0.0, AscendC, HCCL, pybind/torch-npu extension loading.

## Global Constraints

- Do not make CANN required for importing `ccdl_comm`.
- Do not replace HCCL or rewrite distributed collectives.
- Do not change the CUDA extension module name `ccdl_cuda_ops`.
- Initial CANN kernel supports `bit=8`, `quant_type="linear"`, `topk=0`, `group_size in {16, 32, 64}`.
- Every implementation task must start with a failing test and end with a commit.
- Remote performance claims require execution on `user1@47.107.62.29 -p 30303`.

---

### Task 1: Safe CANN Loader and Public Status

**Files:**
- Create: `ccdl_comm_refactor/ccdl_comm/ascend/__init__.py`
- Create: `ccdl_comm_refactor/ccdl_comm/ascend/loader.py`
- Modify: `ccdl_comm_refactor/ccdl_comm/build/setuptools.py`
- Test: `ccdl_comm_refactor/tests/test_cann_loader.py`
- Test: `ccdl_comm_refactor/tests/test_setuptools_build.py`

**Interfaces:**
- Produces: `CannExtensionStatus(available: bool, module: object | None, reason: str | None = None)`
- Produces: `load_cann_extension(module_name: str = "ccdl_cann_ops", import_module: Callable = importlib.import_module) -> CannExtensionStatus`

- [ ] **Step 1: Write failing loader tests**

```python
from ccdl_comm.ascend.loader import CannExtensionStatus, load_cann_extension

def test_load_cann_extension_reports_missing_module():
    status = load_cann_extension(import_module=lambda name: (_ for _ in ()).throw(ModuleNotFoundError(name)))
    assert status.available is False
    assert status.module is None
    assert "ccdl_cann_ops" in status.reason

def test_load_cann_extension_returns_module_when_available():
    module = object()
    status = load_cann_extension(import_module=lambda name: module)
    assert status == CannExtensionStatus(available=True, module=module)
```

- [ ] **Step 2: Verify red**

Run: `python -m pytest ccdl_comm_refactor/tests/test_cann_loader.py -q`

Expected: FAIL because `ccdl_comm.ascend.loader` does not exist.

- [ ] **Step 3: Implement loader**

Create `ccdl_comm/ascend/loader.py` mirroring CUDA loader semantics.

- [ ] **Step 4: Add build flag test**

Extend `tests/test_setuptools_build.py`:

```python
def test_setup_kwargs_enable_cann_extension_when_requested():
    kwargs = build_setup_kwargs(env={"CCDL_COMM_BUILD_CANN": "1"}, create_cann_extension=lambda: "cann")
    assert "cann" in kwargs["ext_modules"]
```

- [ ] **Step 5: Implement minimal build hook**

Modify `build_setup_kwargs()` so `CCDL_COMM_BUILD_CANN=1` can append a CANN extension spec without changing CUDA defaults.

- [ ] **Step 6: Verify and commit**

Run:

```bash
python -m pytest ccdl_comm_refactor/tests/test_cann_loader.py ccdl_comm_refactor/tests/test_setuptools_build.py -q
python -m pytest ccdl_comm_refactor/tests -q
git add ccdl_comm_refactor/ccdl_comm/ascend ccdl_comm_refactor/ccdl_comm/build/setuptools.py ccdl_comm_refactor/tests/test_cann_loader.py ccdl_comm_refactor/tests/test_setuptools_build.py
git commit -m "feat(ccdl_comm): add safe cann extension loader"
```

---

### Task 2: CANN Codec Contract

**Files:**
- Create: `ccdl_comm_refactor/ccdl_comm/ascend/codec.py`
- Modify: `ccdl_comm_refactor/ccdl_comm/ascend/__init__.py`
- Test: `ccdl_comm_refactor/tests/test_cann_codec.py`

**Interfaces:**
- Consumes: `load_cann_extension()`
- Produces: `quantize_tensor_cann(tensor, config, *, extension_status=None) -> CompressedPayload`
- Produces: `dequantize_tensor_cann(payload, shape, config, dtype, *, extension_status=None) -> Any`

- [ ] **Step 1: Write failing codec tests**

```python
from types import SimpleNamespace
import pytest
from ccdl_comm.ascend.codec import quantize_tensor_cann, dequantize_tensor_cann
from ccdl_comm.ascend.loader import CannExtensionStatus
from ccdl_comm.communication.collectives import CompressedPayload
from ccdl_comm.config import CompressionConfig
from ccdl_comm.exceptions import CCDLUnavailableError

def test_cann_codec_requires_available_extension():
    status = CannExtensionStatus(False, None, "ccdl_cann_ops is not installed")
    with pytest.raises(CCDLUnavailableError, match="ccdl_cann_ops is not installed"):
        quantize_tensor_cann(object(), CompressionConfig(), extension_status=status)

def test_cann_quantize_wraps_extension_payload():
    class FakeCann:
        def quantize_linear_int8(self, tensor, group_size):
            return SimpleNamespace(buffer="q", scales="s", original_numel=4)
    payload = quantize_tensor_cann(FakeTensor(dtype="torch.float16", shape=(4,)), CompressionConfig(), extension_status=CannExtensionStatus(True, FakeCann()))
    assert isinstance(payload, CompressedPayload)
    assert payload.buffer == "q"
    assert payload.metadata == {"scales": "s", "original_numel": 4}
```

- [ ] **Step 2: Verify red**

Run: `python -m pytest ccdl_comm_refactor/tests/test_cann_codec.py -q`

Expected: FAIL because `ccdl_comm.ascend.codec` does not exist.

- [ ] **Step 3: Implement codec wrapper**

Call extension methods:

```python
module.quantize_linear_int8(tensor, config.group_size)
module.dequantize_linear_int8(payload.buffer, payload.metadata["scales"], payload.metadata["original_numel"], shape, dtype, config.group_size)
```

Reject unsupported `bit`, `quant_type`, and `topk` with clear `ValueError`.

- [ ] **Step 4: Verify and commit**

Run:

```bash
python -m pytest ccdl_comm_refactor/tests/test_cann_codec.py -q
python -m pytest ccdl_comm_refactor/tests -q
git add ccdl_comm_refactor/ccdl_comm/ascend ccdl_comm_refactor/tests/test_cann_codec.py
git commit -m "feat(ccdl_comm): add cann codec contract"
```

---

### Task 3: CANN Build Probe on Ascend

**Files:**
- Create: `ccdl_comm_refactor/ccdl_comm/build/cann.py`
- Create: `ccdl_comm_refactor/tests/test_cann_build.py`
- Remote-only scratch: `/home/user1/work/ccdl_comm_ascend_test_latest`

**Interfaces:**
- Produces: `create_cann_extension(name: str = "ccdl_cann_ops") -> object`
- Produces: deterministic source discovery for `ccdl_comm/csrc_ascend`

- [ ] **Step 1: Write failing build tests**

```python
from ccdl_comm.build.cann import collect_cann_sources

def test_collect_cann_sources_is_deterministic(tmp_path):
    root = tmp_path / "csrc_ascend"
    root.mkdir()
    (root / "pybind.cpp").write_text("")
    (root / "b.cpp").write_text("")
    (root / "a.cpp").write_text("")
    assert [p.name for p in collect_cann_sources(root)] == ["pybind.cpp", "a.cpp", "b.cpp"]
```

- [ ] **Step 2: Verify red**

Run: `python -m pytest ccdl_comm_refactor/tests/test_cann_build.py -q`

- [ ] **Step 3: Implement build discovery and extension factory**

Use `torch_npu.utils.cpp_extension` only inside the default factory so local CPU imports stay safe.

- [ ] **Step 4: Remote probe**

Upload HEAD to Ascend and run a minimal build command inside `parascale-test:ascend-llamafactory-pytest` with `CCDL_COMM_BUILD_CANN=1`. If `torch_npu.utils.cpp_extension` cannot compile AscendC kernels directly, record the exact error and switch Task 4 to `opbuild` project layout.

- [ ] **Step 5: Commit**

Commit once local tests pass and remote probe result is captured in commentary/final.

---

### Task 4: AscendC Linear INT8 Kernels

**Files:**
- Create: `ccdl_comm_refactor/ccdl_comm/csrc_ascend/pybind.cpp`
- Create: `ccdl_comm_refactor/ccdl_comm/csrc_ascend/quant_linear_int8.cpp`
- Create: `ccdl_comm_refactor/ccdl_comm/csrc_ascend/kernels/quant_linear_int8.cpp`
- Create: `ccdl_comm_refactor/ccdl_comm/csrc_ascend/kernels/dequant_linear_int8.cpp`
- Test: `ccdl_comm_refactor/tests/test_cann_kernel_sources.py`
- Remote validation: `ccdl_comm_refactor/tests/test_cann_extension_smoke.py`

**Interfaces:**
- Extension symbol: `quantize_linear_int8(tensor, group_size) -> object with buffer/scales/original_numel`
- Extension symbol: `dequantize_linear_int8(buffer, scales, original_numel, shape, dtype, group_size) -> tensor`

- [ ] **Step 1: Add source export tests**

Assert `pybind.cpp` exposes both symbols and source files contain kernel entry names.

- [ ] **Step 2: Verify red**

Run: `python -m pytest ccdl_comm_refactor/tests/test_cann_kernel_sources.py -q`

- [ ] **Step 3: Implement host binding and kernels**

Implement group-wise linear INT8 quant/dequant. If CANN custom op build requires generated operator project files, add only the minimum required descriptors.

- [ ] **Step 4: Remote build and smoke**

Run on Ascend:

```bash
CCDL_COMM_BUILD_CANN=1 python3 setup.py build_ext --inplace
python3 -m pytest tests/test_cann_extension_smoke.py -q
```

Expected: relative L2 `< 0.02` for fp16 1M tensor.

- [ ] **Step 5: Commit**

Commit CANN source and smoke tests.

---

### Task 5: HCCL Benchmark and DDP Hook with CANN Codec

**Files:**
- Modify: `ccdl_comm_refactor/tests/distributed/torch_fallback_collective_perf.py`
- Create: `ccdl_comm_refactor/tests/distributed/cann_collective_perf.py`
- Create: `ccdl_comm_refactor/tests/distributed/npu_cann_ddp_smoke.py`
- Test: `ccdl_comm_refactor/tests/test_cann_perf_scripts.py`

**Interfaces:**
- Consumes: `quantize_tensor_cann`, `dequantize_tensor_cann`
- Produces JSON report with native HCCL, torch fallback, and CANN compressed timings.

- [ ] **Step 1: Write script contract tests**

Assert scripts import `ccdl_comm.ascend.codec` and report `"ccdl_cann_ms"` and `"relative_l2"`.

- [ ] **Step 2: Verify red**

Run: `python -m pytest ccdl_comm_refactor/tests/test_cann_perf_scripts.py -q`

- [ ] **Step 3: Implement scripts**

Follow the existing fallback benchmark structure, replacing fallback codec with CANN codec.

- [ ] **Step 4: Remote run**

Run two-card HCCL benchmark on Ascend cards 4/5 first. If available later, run 8-card only after confirming the server is idle.

- [ ] **Step 5: Commit**

Commit benchmark and smoke scripts with verified output paths.

---

### Task 6: Packed Payload Format

**Files:**
- Create: `ccdl_comm_refactor/ccdl_comm/communication/packed_payload.py`
- Modify: `ccdl_comm_refactor/ccdl_comm/collectives/all_reduce.py`
- Modify: `ccdl_comm_refactor/ccdl_comm/collectives/all_gather.py`
- Test: `ccdl_comm_refactor/tests/test_packed_payload.py`

**Interfaces:**
- Produces: `PackedPayload(buffer: Any, original_numel: int, scale_count: int, dtype: str, shape: tuple[int, ...])`
- Produces: `pack_payload(payload: CompressedPayload) -> PackedPayload`
- Produces: `unpack_payload(packed: PackedPayload) -> CompressedPayload`

- [ ] **Step 1: Write failing pack/unpack tests**

Use fake tensor objects for local tests and torch tensors on Ascend smoke.

- [ ] **Step 2: Verify red**

Run: `python -m pytest ccdl_comm_refactor/tests/test_packed_payload.py -q`

- [ ] **Step 3: Implement pack/unpack**

For first version, pack metadata into a sidecar header object locally and keep tensor buffer single for HCCL. Only enable fully flat tensor packing after dtype-safe serialization is verified on NPU.

- [ ] **Step 4: Benchmark packed all-gather**

Add benchmark mode `--payload-format packed` and compare HCCL call count/latency.

- [ ] **Step 5: Commit**

Commit packed payload implementation and benchmark.

---

## Self-Review

- Spec coverage: loader, codec, CANN kernels, benchmark, DDP, and payload optimization are each represented by tasks.
- Placeholder scan: No task uses TBD/TODO language; Task 4 intentionally contains a CANN build branch because the exact remote compiler behavior must be determined by Task 3.
- Type consistency: Codec signatures match existing `compressed_all_reduce` and DDP hook injection contracts.
