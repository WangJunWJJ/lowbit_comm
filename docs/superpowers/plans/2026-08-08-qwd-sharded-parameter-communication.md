# qWD Sharded Parameter Communication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace direct quantized parameter restoration with an FP32-master qWD loop that preserves sharded AdamW semantics, restores replicated model parameters safely, and passes the A6000 correctness and performance gates.

**Architecture:** `TorchShardedAdamWStep` owns an FP32 master shard and rank-local FP32 AdamW state while the model remains replicated in FP16/BF16. A dedicated delta provider computes `master - model_copy`, a policy selects INT8 qWD or FP refresh, and a dedicated restore component performs either dequantize-add or full overwrite with CUDA event ordering. Correctness lands before fused kernels; the same contract is then accelerated without changing its numerical semantics.

**Tech Stack:** Python 3.10+, PyTorch distributed/NCCL, CUDA C++/PyBind11, pytest, existing CCDL workspace/completion abstractions, single-node A6000 GPUs 1/2/3/4.

## Global Constraints

- The authoritative parameter and AdamW moment shards are FP32.
- Forward/backward parameters remain replicated FP16 or BF16 model copies.
- Default compressed parameter communication is INT8 linear qWD with group size 64.
- The qWD writeback operation adds decoded deltas; FP refresh overwrites the model copy.
- No implementation may hard-code world size 2, 4, or 8; these are validation sizes only.
- INT4, TopK, Hadamard, and LoCo EMA are outside this implementation.
- A collective that has already started may not silently fall back to a second collective.
- Every production behavior starts with a failing test and each independently reviewable task ends in a conventional commit.
- Existing direct `TorchCompressedParameterRestore` behavior remains available and unchanged.
- Parameter rank difference after restore must be exactly zero.
- Three-epoch validation loss may be at most 2% above `sharded_fp` under the same benchmark contract.
- Four-GPU qWD median throughput must be at least 98% of direct INT8 restore and target at least 1.05x Native DDP.

---

## File Structure

- `ccdl_comm/optim/sharded.py`: allow a validated gradient shard to update an FP32 master shard without weakening device/shape checks.
- `ccdl_comm/communication/parameter_delta.py`: immutable qWD metadata, FP32 delta preparation, policy decision, and safe policy implementation.
- `ccdl_comm/communication/parameter_delta_restore.py`: reference qWD all-gather/add and FP refresh/overwrite paths with workspace and completion ownership.
- `ccdl_comm/communication/__init__.py`: public qWD communication exports.
- `examples/training/torch_sharded_adamw.py`: FP32-master training adapter, checkpoint state, policy scheduling, and restore orchestration.
- `ccdl_comm/csrc/quantization/dequant_api.cuh`: declaration of gathered dequantize-add API.
- `ccdl_comm/csrc/quantization/dequant_reduce_kernel.cu`: fused gathered INT8 dequantize-add CUDA kernel.
- `ccdl_comm/csrc/quantization/quant_api.cuh`: declaration of fused master-minus-model quant-pack API.
- `ccdl_comm/csrc/quantization/quant_pack_kernel.cu`: fused qWD delta calculation and INT8 pack kernel.
- `ccdl_comm/csrc/pybind.cpp`: native qWD symbol registration.
- `ccdl_comm/quantization/codec.py`: safe Python capability adapters for the two native qWD kernels.
- `examples/training/compressed_sharded_optimizer.py`: add `sharded_qwd` benchmark mode and stage/strategy reporting.
- `tests/core/test_sharded_optimizer.py`: FP32-master update semantics.
- `tests/test_parameter_delta.py`: qWD delta and policy unit tests.
- `tests/test_parameter_delta_restore.py`: restore/fallback/completion/workspace tests.
- `tests/examples/test_torch_sharded_adamw.py`: model/master/checkpoint orchestration tests.
- `tests/cuda/test_qwd_kernels.py`: CUDA correctness and tail-group tests.
- `tests/distributed/torch_sharded_qwd_smoke.py`: 2/4-rank NCCL consistency smoke.
- `tests/benchmarks/qwd_parameter_pipeline_gate.py`: precision and performance evidence gate.
- `tests/benchmarks/reports/qwd_parameter_pipeline_<date>/`: raw A6000 runs, summary, and report.

---

### Task 1: FP32 Master-Shard Optimizer Semantics

**Files:**
- Modify: `ccdl_comm/optim/sharded.py`
- Modify: `tests/core/test_sharded_optimizer.py`

**Interfaces:**
- Consumes: `ReducedShard`, `FlatShardLayout`, `AdamWShardUpdateRule`.
- Produces: `ShardedOptimizerConsumer(..., gradient_transform: Callable[[Any, Any], Any] | None = None)` and FP32 `UpdatedParameterShard.shard`.

- [ ] **Step 1: Write the failing mixed-dtype master test**

Append a test that constructs an FP32 parameter shard and FP16 gradient shard, passes `gradient_transform=lambda gradient, master: gradient.to(master.dtype)`, and compares two AdamW steps to `torch.optim.AdamW`:

```python
def test_adamw_consumer_updates_fp32_master_from_fp16_gradient() -> None:
    master = torch.tensor([4.0, 5.0, 0.0], dtype=torch.float32)
    state: dict[str, object] = {}
    consumer = ShardedOptimizerConsumer(
        layout=layout(),
        parameter_shard=master,
        update_rule=AdamWShardUpdateRule(0.01),
        state=state,
        gradient_transform=lambda gradient, parameter: gradient.to(parameter.dtype),
    )
    reduced = reduced_shard((1.0, 2.0, 99.0))
    reduced = dataclasses.replace(reduced, shard=reduced.shard.half())

    updated = consumer.consume(reduced, step=1)

    assert updated.shard is master
    assert master.dtype == torch.float32
    assert state["exp_avg"].dtype == torch.float32
    assert state["exp_avg_sq"].dtype == torch.float32
    assert master[2].item() == 0.0
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `python -m pytest tests/core/test_sharded_optimizer.py::test_adamw_consumer_updates_fp32_master_from_fp16_gradient -q`

Expected: FAIL because `ShardedOptimizerConsumer` does not accept `gradient_transform` and currently rejects mixed dtypes.

- [ ] **Step 3: Add the minimal transform boundary**

Add the constructor parameter, validate it once, transform immediately before the update, and keep all existing device/contiguity/numel validation:

```python
self._gradient_transform = gradient_transform

gradient = reduced.shard
if self._gradient_transform is not None:
    gradient = self._gradient_transform(gradient, self._parameter_shard)
_require_contiguous(gradient, "gradient shard")
_require_matching_tensor_property(self._parameter_shard, gradient, "dtype")
_require_matching_tensor_property(self._parameter_shard, gradient, "device")
if _tensor_numel(gradient, "gradient shard") != self._layout.shard_numel:
    raise ValueError("transformed gradient shard must match layout shard_numel")
```

- [ ] **Step 4: Run optimizer tests and verify GREEN**

Run: `python -m pytest tests/core/test_sharded_optimizer.py -q`

Expected: all optimizer tests pass; invalid transformed dtype/device/shape tests also pass.

- [ ] **Step 5: Commit**

```bash
git add ccdl_comm/optim/sharded.py tests/core/test_sharded_optimizer.py
git commit -m "feat(optim): support fp32 master shard updates"
```

---

### Task 2: qWD Delta and Safe Policy Core

**Files:**
- Create: `ccdl_comm/communication/parameter_delta.py`
- Create: `tests/test_parameter_delta.py`
- Modify: `ccdl_comm/communication/__init__.py`

**Interfaces:**
- Consumes: master/model tensor-like objects and backend capability flag.
- Produces: `ParameterDeltaShard`, `ParameterCommunicationDecision`, `TorchParameterDeltaProvider`, and `SafeInt8QWDPolicy`.

- [ ] **Step 1: Write failing delta and policy tests**

Cover FP32 subtraction, zero padding, caller-owned output, warmup, interval refresh, error-triggered refresh, sensitive roles, and missing capability:

```python
def test_delta_provider_computes_fp32_master_minus_model_and_zeros_padding() -> None:
    master = torch.tensor([1.125, 2.25, 99.0], dtype=torch.float32)
    model = torch.tensor([1.0, 2.0, 7.0], dtype=torch.float16)
    output = torch.empty(3, dtype=torch.float32)

    result = TorchParameterDeltaProvider().prepare_delta(
        master, model, out=output, valid_numel=2
    )

    assert result is output
    torch.testing.assert_close(output, torch.tensor([0.125, 0.25, 0.0]))


@pytest.mark.parametrize(
    ("step", "role", "error", "capability", "mode", "reason"),
    (
        (1, "weight", None, True, "fp_refresh", "warmup"),
        (101, "weight", None, True, "qwd", "int8_qwd"),
        (512, "weight", None, True, "fp_refresh", "periodic_refresh"),
        (129, "weight", 0.02, True, "fp_refresh", "error_threshold"),
        (129, "sensitive", None, True, "fp_refresh", "sensitive_tensor"),
        (129, "weight", None, False, "fp_refresh", "capability"),
    ),
)
def test_safe_policy_decisions(step, role, error, capability, mode, reason) -> None:
    decision = SafeInt8QWDPolicy().decide(
        step=step,
        tensor_role=role,
        numel=4096,
        relative_error=error,
        capability=capability,
    )
    assert decision.mode == mode
    assert decision.reason == reason
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_parameter_delta.py -q`

Expected: collection fails because `parameter_delta` does not exist.

- [ ] **Step 3: Implement immutable metadata and policy**

Create the exact public contracts, including runtime-checkable strategy boundaries:

```python
@dataclass(frozen=True, slots=True)
class ParameterDeltaShard:
    shard: Any
    shard_index: int
    shard_numel: int
    valid_numel: int
    original_numel: int
    padded_numel: int
    world_size: int
    layout_version: int = 0


@dataclass(frozen=True, slots=True)
class ParameterCommunicationDecision:
    mode: Literal["fp_refresh", "qwd"]
    bit: Literal[8]
    reason: str


@runtime_checkable
class ParameterDeltaProvider(Protocol):
    def prepare_delta(
        self, master_shard, model_shard, *, out, valid_numel: int
    ) -> Any: ...


@runtime_checkable
class ParameterCommunicationPolicy(Protocol):
    def decide(
        self,
        *,
        step: int,
        tensor_role: str,
        numel: int,
        relative_error: float | None,
        capability: bool,
    ) -> ParameterCommunicationDecision: ...


class TorchParameterDeltaProvider:
    def prepare_delta(self, master_shard, model_shard, *, out, valid_numel):
        out.copy_(master_shard)
        out.sub_(model_shard)
        out[valid_numel:].zero_()
        return out
```

Implement `SafeInt8QWDPolicy` with constructor defaults `warmup_steps=100`, `refresh_interval=512`, `relative_error_threshold=1e-2`, `error_check_interval=128`, and deterministic decision precedence: invalid input -> warmup -> sensitive -> missing capability -> error threshold -> periodic refresh -> qWD.

Expose a stable `configuration_packet()` tuple containing integer-scaled policy fields. Do not use
Python `hash()`; encode the tuple directly into the fixed metadata packet so separate processes
produce identical values:

```python
def configuration_packet(self) -> tuple[int, int, int, int]:
    return (
        self.warmup_steps,
        self.refresh_interval,
        int(round(self.relative_error_threshold * 1_000_000_000)),
        self.error_check_interval,
    )
```

- [ ] **Step 4: Run policy/core tests and existing import tests**

Run: `python -m pytest tests/test_parameter_delta.py tests/conformance/test_cuda_backend.py::test_cuda_package_import_does_not_import_torch -q`

Expected: all tests pass without importing CUDA at module import time.

- [ ] **Step 5: Commit**

```bash
git add ccdl_comm/communication/parameter_delta.py ccdl_comm/communication/__init__.py tests/test_parameter_delta.py
git commit -m "feat(communication): add qwd delta policy core"
```

---

### Task 3: Reference qWD Restore and FP Refresh

**Files:**
- Create: `ccdl_comm/communication/parameter_delta_restore.py`
- Create: `tests/test_parameter_delta_restore.py`
- Modify: `ccdl_comm/communication/__init__.py`

**Interfaces:**
- Consumes: `ParameterDeltaShard`, `UpdatedParameterShard`, `ParameterCommunicationDecision`, caller-owned replicated output.
- Produces: `TorchQuantizedParameterDeltaRestore.supports_qwd(...)`, `.restore_delta(...)`, and `.refresh(...)`; restore methods return `CollectiveWork[Any]`.

- [ ] **Step 1: Write failing reference restore tests**

Use fake distributed and codec callables to prove add versus overwrite, collective ordering, hard failure after collective, and workspace reuse:

```python
def test_qwd_restore_adds_decoded_delta_to_model_copy() -> None:
    runtime = FakeQWDRuntime(decoded=(0.25, -0.5, 0.0, 0.0))
    restore = restore_for(runtime)
    model = FakeTensor((1.0, 2.0, 3.0, 4.0))

    result = restore.restore_delta(delta_shard(), out=model).wait()

    assert result is model
    assert model.values == pytest.approx((1.25, 1.5, 3.0, 4.0))
    assert runtime.calls == ["quantize", "all_gather", "dequantize_add"]


def test_fp_refresh_overwrites_model_copy() -> None:
    runtime = FakeQWDRuntime(gathered_fp=(7.0, 8.0, 9.0, 10.0))
    model = FakeTensor((1.0, 2.0, 3.0, 4.0))

    restore_for(runtime).refresh(updated_master(), out=model).wait()

    assert model.values == pytest.approx((7.0, 8.0, 9.0, 10.0))
    assert runtime.calls == ["fp_all_gather", "overwrite"]
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_parameter_delta_restore.py -q`

Expected: collection fails because the qWD restore module does not exist.

- [ ] **Step 3: Implement the reference restore**

Implement separate methods so callers cannot accidentally select overwrite/add with a boolean:

```python
class TorchQuantizedParameterDeltaRestore:
    def supports_qwd(self, updated: UpdatedParameterShard, out: Any) -> bool:
        return self._supports_qwd(updated, out)

    def restore_delta(self, delta: ParameterDeltaShard, *, out, async_op=True):
        workspace = self._qwd_workspace_for(delta, out)
        packed = self._quantize(delta.shard, self.config, output=workspace.send)
        handle = self._dist.all_gather_into_tensor(
            workspace.gathered, packed, async_op=async_op
        )
        return self._completion_manager.create_work(
            result=out,
            handle=handle,
            complete=lambda: self._finish_add(delta, workspace, out),
            resources=(delta.shard, workspace.send, workspace.gathered, out),
        )

    def refresh(self, master: UpdatedParameterShard, *, out, async_op=True):
        workspace = self._refresh_workspace_for(master, out)
        workspace.send.copy_(master.shard)
        handle = self._dist.all_gather_into_tensor(
            workspace.gathered, workspace.send, async_op=async_op
        )
        return self._completion_manager.create_work(
            result=out,
            handle=handle,
            complete=lambda: self._finish_overwrite(workspace, out),
            resources=(master.shard, workspace.send, workspace.gathered, out),
        )
```

The Python reference `_finish_add` decodes each rank payload and adds into its logical output slice. Capability rejection happens before `all_gather_into_tensor`; a post-collective decode rejection raises without retry.

- [ ] **Step 4: Run restore, completion, and old-restore regression tests**

Run: `python -m pytest tests/test_parameter_delta_restore.py tests/test_parameter_restore.py tests/test_cuda_completion.py -q`

Expected: all tests pass and the old restore tests remain unchanged.

- [ ] **Step 5: Commit**

```bash
git add ccdl_comm/communication/parameter_delta_restore.py ccdl_comm/communication/__init__.py tests/test_parameter_delta_restore.py
git commit -m "feat(communication): restore quantized parameter deltas"
```

---

### Task 4: FP32-Master AdamW Adapter and Checkpoint

**Files:**
- Modify: `examples/training/torch_sharded_adamw.py`
- Modify: `tests/examples/test_torch_sharded_adamw.py`

**Interfaces:**
- Consumes: `TorchParameterDeltaProvider`, `TorchQuantizedParameterDeltaRestore`, `SafeInt8QWDPolicy`.
- Produces: `ShardedAdamWState.master_shard`, qWD-aware `TorchShardedAdamWStep.step()`, and policy metrics.

- [ ] **Step 1: Write failing master/model/error-loop tests**

Add tests proving model parameters remain low precision, master/moments are FP32, a deliberately lossy first delta is recovered in the second delta, and load forces refresh:

```python
def test_qwd_error_is_carried_by_next_master_minus_model_delta() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float16))
    restore = RecordingLossyQWDRestore(first_scale=0.5)
    adapter = build_qwd_adapter(parameter, restore=restore)

    parameter.grad = torch.tensor([0.25, -0.5], dtype=torch.float16)
    adapter.step(step=1)
    first_unrestored = adapter.master_shard - parameter.detach().float()

    parameter.grad = torch.zeros_like(parameter)
    adapter.step(step=2)

    torch.testing.assert_close(restore.deltas[1], first_unrestored)
    assert adapter.master_shard.dtype == torch.float32
    assert parameter.dtype == torch.float16
```

Update checkpoint tests to assert saved master/moments are FP32 clones and the restored adapter performs `refresh` before its next forward-visible completion.

- [ ] **Step 2: Run adapter tests and verify RED**

Run: `python -m pytest tests/examples/test_torch_sharded_adamw.py -q`

Expected: FAIL because the adapter currently updates `storage.local_shard` directly and checkpoint omits master state.

- [ ] **Step 3: Rewire the adapter around FP32 master**

Allocate persistent state and construct the consumer with the mixed-dtype transform:

```python
self._master_shard = storage.local_shard.detach().float().clone()
self._delta_workspace = self._master_shard.new_empty(self._master_shard.shape)
self._consumer = ShardedOptimizerConsumer(
    layout=storage.layout,
    parameter_shard=self._master_shard,
    update_rule=update_rule,
    state=state,
    gradient_transform=lambda gradient, master: gradient.to(master.dtype),
)
```

After update, call policy once. For `qwd`, prepare the delta from master and `storage.local_shard`, bind `ParameterDeltaShard`, and invoke `restore_delta`; for `fp_refresh`, invoke `refresh`. Add `mode`, `reason`, and sampled `relative_error` to `ShardedAdamWStepMetrics`.

Remove `ShardedStepPipeline` from this adapter only: its existing restore contract is
intentionally overwrite-based and cannot represent qWD add semantics. Preserve its public API
and tests for SGD/direct restore users. Sequence the new path explicitly and wait on the returned
CCDL work before returning from `step()`:

```python
updated_master = self._consumer.consume(reduced, step=step)
decision = self._policy.decide(
    step=step,
    tensor_role="weight",
    numel=self._storage.layout.original_numel,
    relative_error=self._last_relative_error,
    capability=self._restore.supports_qwd(updated_master, self._storage.padded_flat),
)
if self._refresh_required or decision.mode == "fp_refresh":
    work = self._restore.refresh(
        updated_master, out=self._storage.padded_flat, async_op=True
    )
    self._refresh_required = False
else:
    delta = self._prepare_parameter_delta(updated_master)
    work = self._restore.restore_delta(
        delta, out=self._storage.padded_flat, async_op=True
    )
work.wait()
```

After a sampled qWD step completes, compute the next implicit error from
`master_shard - model_copy_local` and divide its FP32 L2 norm by the precommunication
delta L2 norm. Persist this scalar as `_last_relative_error`; pass it to the policy on
the next step. Only sample when `step % error_check_interval == 0`, and synchronize
only the two reduced scalar norms at the existing completion boundary.

Extend the checkpoint dataclass exactly as follows:

```python
@dataclass(frozen=True, slots=True)
class ShardedAdamWState:
    step: int
    layout_version: int
    world_size: int
    shard_index: int
    shard_numel: int
    master_shard: Any
    exp_avg: Any
    exp_avg_sq: Any
    policy_state: Mapping[str, Any]
```

Loading validates FP32 dtype/device/layout, clones all state, and sets `_refresh_required=True`; the next `step` must select refresh regardless of normal policy.

- [ ] **Step 4: Run adapter and optimizer regression tests**

Run: `python -m pytest tests/examples/test_torch_sharded_adamw.py tests/core/test_sharded_optimizer.py tests/test_sharded_step_pipeline.py -q`

Expected: all tests pass, including exact rank-one AdamW parity under FP32 model parameters.

- [ ] **Step 5: Commit**

```bash
git add examples/training/torch_sharded_adamw.py tests/examples/test_torch_sharded_adamw.py
git commit -m "feat(examples): train with qwd fp32 master shards"
```

---

### Task 5: Distributed qWD Correctness Smoke

**Files:**
- Create: `tests/distributed/torch_sharded_qwd_smoke.py`
- Modify: `tests/test_sharded_training_gate.py`

**Interfaces:**
- Consumes: public qWD adapter and existing compiled compressed reduce-scatter.
- Produces: JSON evidence containing `world_size`, `losses`, `max_rank_parameter_difference`, `max_master_reference_difference`, `decision_counts`, and `workspace_stable`.

- [ ] **Step 1: Write the failing smoke launcher assertions**

Add an import-safe payload validation test that exercises the same gate called by the
distributed script without requiring CUDA locally:

```python
def test_qwd_smoke_reports_master_and_model_consistency() -> None:
    payload = {
        "world_size": 2,
        "losses": [2.0, 1.5],
        "max_rank_parameter_difference": 0.0,
        "max_master_reference_difference": 1.0e-7,
        "decision_counts": {"qwd": 1, "fp_refresh": 1},
        "workspace_stable": True,
    }

    validate_qwd_smoke_payload(payload)
```

- [ ] **Step 2: Run and verify RED**

Run: `python -m pytest tests/test_sharded_training_gate.py -q`

Expected: FAIL because the qWD smoke helper/script is missing.

- [ ] **Step 3: Implement the NCCL smoke**

Build the same MLP, input generation, compressed reduce-scatter, and clipping contract as `torch_sharded_adamw_smoke.py`, but run FP16 autocast with FP32 master qWD. At completion gather the FP32 master shards, compare against a full FP32 AdamW oracle for a deterministic one-rank-equivalent gradient fixture, broadcast rank-zero model copy, and emit the required JSON fields. Define `validate_qwd_smoke_payload(payload)` in the script without initializing torch.distributed; call it from both the unit test and `main()` before printing evidence.

At initialization, all-gather one fixed-length device metadata packet encoding layout version,
world size, qWD mode, bit, group size, and policy fingerprint; reject any mismatch before the
first parameter collective. For sampled relative error, all-reduce the local residual and delta
squared norms with SUM before calculating the ratio, so every rank passes the same scalar to the
deterministic policy. Tests inject one mismatched packet and require failure before qWD payload
communication.

Use exact failure gates:

```python
if max_rank_parameter_difference != 0.0:
    raise RuntimeError("qWD model copies differ across ranks")
if max_master_reference_difference > 1.0e-6:
    raise RuntimeError("FP32 master differs from AdamW reference")
if not workspace_stable:
    raise RuntimeError("qWD steady-state workspace pointers changed")
```

- [ ] **Step 4: Run local tests, then A6000 2/4-rank smoke**

Local: `python -m pytest tests/test_sharded_training_gate.py -q`

Remote 2 GPU: `CUDA_VISIBLE_DEVICES=1,2 torchrun --standalone --nproc_per_node=2 tests/distributed/torch_sharded_qwd_smoke.py`

Remote 4 GPU: `CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --standalone --nproc_per_node=4 tests/distributed/torch_sharded_qwd_smoke.py`

Expected: both emit finite losses, zero rank difference, master reference difference at most `1e-6`, and stable workspace pointers.

- [ ] **Step 5: Commit**

```bash
git add tests/distributed/torch_sharded_qwd_smoke.py tests/test_sharded_training_gate.py
git commit -m "test(distributed): validate qwd shard consistency"
```

---

### Task 6: Fused qWD CUDA Kernels

**Files:**
- Modify: `ccdl_comm/csrc/quantization/quant_api.cuh`
- Modify: `ccdl_comm/csrc/quantization/quant_pack_kernel.cu`
- Modify: `ccdl_comm/csrc/quantization/dequant_api.cuh`
- Modify: `ccdl_comm/csrc/quantization/dequant_reduce_kernel.cu`
- Modify: `ccdl_comm/csrc/pybind.cpp`
- Modify: `ccdl_comm/quantization/codec.py`
- Modify: `ccdl_comm/communication/parameter_delta_restore.py`
- Create: `tests/cuda/test_qwd_kernels.py`

**Interfaces:**
- Consumes: FP32 master shard, FP16/BF16/FP32 model shard, rank-strided INT8 payload.
- Produces: `quantize_parameter_delta(..., output) -> bool` and `inplace_dequantize_gathered_add(...) -> bool`.

- [ ] **Step 1: Write CUDA oracle tests before native symbols**

Use `pytest.importorskip("torch")`, skip when CUDA extension is unavailable, and compare native kernels with pure PyTorch for FP16/BF16/FP32, world sizes 1/2/4/8, and shard lengths 1/63/64/65/4097:

```python
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
@pytest.mark.parametrize("shard_numel", (1, 63, 64, 65, 4097))
def test_fused_qwd_matches_torch_oracle(dtype, shard_numel) -> None:
    master = torch.randn(shard_numel, device="cuda", dtype=torch.float32)
    model = torch.randn(shard_numel, device="cuda", dtype=dtype)
    expected_delta = master - model.float()
    packed = allocate_qwd_payload(master, config)

    assert quantize_parameter_delta(master, model, config, output=packed)
    actual = model.clone()
    assert inplace_dequantize_gathered_add(
        packed, actual, config, world_size=1, shard_numel=shard_numel
    )

    expected = model + dequantize_payload(packed, config, dtype=dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.isfinite(expected_delta).all()
```

- [ ] **Step 2: Run CUDA tests and verify RED**

Run: `python -m pytest tests/cuda/test_qwd_kernels.py -q`

Expected: FAIL because both codec/native qWD symbols are absent.

- [ ] **Step 3: Implement fused master-minus-model quant-pack**

Add a runtime-dispatched kernel whose per-group reduction reads `master[index]` as FP32 and `model[index]` as target dtype, computes the FP32 delta, derives linear INT8 scale, and writes the existing compact payload layout. Padding lanes contribute zero and write zero. Export:

```cpp
bool quantize_parameter_delta(
    torch::Tensor master,
    torch::Tensor model,
    torch::Tensor output,
    int64_t group_size,
    int64_t bit,
    QuantType quant_type,
    bool compact,
    DType dtype);
```

Reject unsupported bit/group/topk/quant-type/dtype before launching the kernel and use `CUDAGuard` plus the current PyTorch CUDA stream.

- [ ] **Step 4: Implement gathered-dequant-add**

Clone the proven rank-strided index/scale decoding structure of `inplace_dequantize_gathered`, but replace overwrite with FP32-accumulated add cast to output dtype:

```cpp
const float delta = dequantize_value(payload, scale, local_index);
output[global_index] = static_cast<scalar_t>(
    static_cast<float>(output[global_index]) + delta);
```

Never write beyond `original_numel`; support partial final groups and runtime world size.

- [ ] **Step 5: Add safe codec wrappers and capability selection**

Add Python wrappers returning `False` when the symbol/config is unsupported and raising native execution failures. Update qWD restore to choose fused kernels only before the collective and retain the reference path for CPU/unit tests.

- [ ] **Step 6: Build and run CUDA/regression tests**

Run: `python -m pytest tests/cuda/test_qwd_kernels.py tests/cuda/test_fused_quant_pack.py tests/cuda/test_fused_requantized_restore.py tests/test_parameter_delta_restore.py -q`

Expected: all cases pass with no CUDA warning, illegal access, or dtype mismatch.

- [ ] **Step 7: Commit**

```bash
git add ccdl_comm/csrc/quantization/quant_api.cuh ccdl_comm/csrc/quantization/quant_pack_kernel.cu ccdl_comm/csrc/quantization/dequant_api.cuh ccdl_comm/csrc/quantization/dequant_reduce_kernel.cu ccdl_comm/csrc/pybind.cpp ccdl_comm/quantization/codec.py ccdl_comm/communication/parameter_delta_restore.py tests/cuda/test_qwd_kernels.py
git commit -m "perf(cuda): fuse qwd quantize and add restore"
```

---

### Task 7: End-to-End qWD Benchmark Mode and Gate

**Files:**
- Modify: `examples/training/compressed_sharded_optimizer.py`
- Modify: `tests/examples/test_compressed_sharded_optimizer.py`
- Create: `tests/benchmarks/qwd_parameter_pipeline_gate.py`
- Create: `tests/test_qwd_parameter_pipeline_perf.py`
- Modify: `tests/benchmarks/run_sharded_training_gate.py`

**Interfaces:**
- Consumes: existing common workload/configuration and qWD adapter.
- Produces: mode `sharded_qwd`, comparable JSON timing/error/decision data, and a deterministic gate evaluator.

- [ ] **Step 1: Write failing CLI/schema/gate tests**

Require the new mode and its additional evidence fields:

```python
def test_qwd_gate_requires_precision_and_direct_int8_performance(tmp_path) -> None:
    write_three_trials(tmp_path, mode="sharded_fp", throughput=350.0, val_loss=2.94)
    write_three_trials(tmp_path, mode="sharded_compressed", throughput=380.0, val_loss=4.70)
    write_three_trials(tmp_path, mode="sharded_qwd", throughput=375.0, val_loss=2.97)
    write_three_trials(tmp_path, mode="native_ddp", throughput=353.0, val_loss=2.98)

    result = evaluate(tmp_path, world_size=4)

    assert result["qwd_speedup_vs_native"] >= 1.05
    assert result["qwd_ratio_vs_direct_int8"] >= 0.98
    assert result["qwd_loss_ratio_vs_sharded_fp"] <= 1.02
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/examples/test_compressed_sharded_optimizer.py tests/test_qwd_parameter_pipeline_perf.py -q`

Expected: FAIL because `sharded_qwd` and its gate do not exist.

- [ ] **Step 3: Add the qWD benchmark path**

Extend `MODES` with `sharded_qwd`, instantiate `TorchShardedAdamWStep` with FP32 master and `SafeInt8QWDPolicy`, and preserve the same model/data/batch/seed/autocast contract used by the other modes. Report:

```python
result["parameter_communication"] = {
    "algorithm": "qwd",
    "bit": 8,
    "warmup_steps": policy.warmup_steps,
    "refresh_interval": policy.refresh_interval,
    "relative_error_threshold": policy.relative_error_threshold,
    "decision_counts": dict(decision_counts),
    "sampled_relative_errors": relative_errors,
}
```

Add stage timings `parameter_delta_quantize`, `parameter_all_gather`, `parameter_add_writeback`, and `fp_refresh`, without changing common throughput accounting.

- [ ] **Step 4: Implement strict benchmark evaluator**

Require three trials per mode, identical environment signatures, finite loss, zero rank difference, stable workspace, no fallback on measured qWD steps, qWD/direct throughput ratio at least 0.98, qWD/native ratio at least 1.05, and qWD/sharded-fp validation-loss ratio at most 1.02.

- [ ] **Step 5: Run local benchmark contract tests**

Run: `python -m pytest tests/examples/test_compressed_sharded_optimizer.py tests/test_qwd_parameter_pipeline_perf.py tests/test_sharded_parameter_pipeline_perf.py -q`

Expected: all tests pass and old result schemas remain accepted for old modes.

- [ ] **Step 6: Commit**

```bash
git add examples/training/compressed_sharded_optimizer.py tests/examples/test_compressed_sharded_optimizer.py tests/benchmarks/qwd_parameter_pipeline_gate.py tests/benchmarks/run_sharded_training_gate.py tests/test_qwd_parameter_pipeline_perf.py
git commit -m "feat(examples): benchmark qwd parameter training"
```

---

### Task 8: Full Verification and A6000 Acceptance

**Files:**
- Create: `tests/benchmarks/reports/qwd_parameter_pipeline_20260808/README.md`
- Create: `tests/benchmarks/reports/qwd_parameter_pipeline_20260808/summary.json`
- Create: `tests/benchmarks/reports/qwd_parameter_pipeline_20260808/raw/*.json`

**Interfaces:**
- Consumes: committed qWD implementation, A6000 Docker environment, 21 GB dataset, 44.956M model workload.
- Produces: reproducible raw evidence and a pass/fail report; no production API changes.

- [ ] **Step 1: Run the complete local suite**

Run: `python -m pytest -q`

Expected: all applicable tests pass; CUDA-only tests skip locally for an explicit extension/device reason.

- [ ] **Step 2: Synchronize the exact source to the A6000 test workspace and build once**

Record `git rev-parse HEAD`, Docker image ID, CUDA extension SHA-256, PyTorch/CUDA/NCCL versions, GPU UUIDs, driver version, dataset path/hash manifest, and model/config hash in every raw JSON file.

- [ ] **Step 3: Run 2/4-GPU correctness smokes**

Run the Task 5 commands with GPUs `1,2` and `1,2,3,4`.

Expected: finite loss, zero rank difference, master reference difference at most `1e-6`, no fallback, and stable workspace.

- [ ] **Step 4: Run alternating four-GPU full training trials**

Use the exact order below to reduce thermal/order bias:

```text
round 1: native_ddp -> sharded_fp -> sharded_compressed -> sharded_qwd
round 2: sharded_qwd -> sharded_compressed -> sharded_fp -> native_ddp
round 3: sharded_fp -> native_ddp -> sharded_qwd -> sharded_compressed
```

Each run uses `CUDA_VISIBLE_DEVICES=1,2,3,4`, the same container, 21 GB dataset, 44.956M model, FP16 mixed precision, per-rank batch 16, identical seed/data order, and three full epochs.

- [ ] **Step 5: Evaluate gates and diagnose failures before changing policy**

Run:

```bash
python tests/benchmarks/qwd_parameter_pipeline_gate.py \
  --results-dir tests/benchmarks/reports/qwd_parameter_pipeline_20260808/raw \
  --world-size 4 \
  --output tests/benchmarks/reports/qwd_parameter_pipeline_20260808/summary.json
```

Expected: qWD/direct INT8 throughput ratio `>= 0.98`, qWD/native speedup `>= 1.05`, qWD/sharded-fp validation-loss ratio `<= 1.02`, and all correctness gates pass. If a gate fails, preserve raw evidence, add a failing regression test for the diagnosed cause, and return to the relevant task instead of weakening the gate.

- [ ] **Step 6: Write the evidence report**

The report includes all raw trials, medians, P50/P95 step latency, stage timing, peak memory, allocation stability, loss curves, decision counts, refresh-step cost, observed relative errors, environment signature, and an explicit default-enable/opt-in conclusion.

- [ ] **Step 7: Commit**

```bash
git add tests/benchmarks/reports/qwd_parameter_pipeline_20260808
git commit -m "test(benchmark): validate qwd parameter pipeline"
```

---

## Final Verification Checklist

- [ ] Every new public class/function has a focused test that was observed failing first.
- [ ] Old direct parameter restore and existing sharded training modes remain green.
- [ ] FP32 master and moments never alias the low-precision model copy.
- [ ] qWD uses add writeback; FP refresh uses overwrite.
- [ ] The implicit error is visible in the next `master - model` delta.
- [ ] Checkpoint reload forces refresh and never reconstructs master from model copy.
- [ ] Unsupported capability falls back before collective launch only.
- [ ] 2/4-rank A6000 smokes have exact rank equality.
- [ ] Full local and remote CUDA suites pass without warnings introduced by this work.
- [ ] Three alternating full-training trials per mode meet both precision and performance gates.
- [ ] Each task has its own conventional commit and the worktree is clean.
