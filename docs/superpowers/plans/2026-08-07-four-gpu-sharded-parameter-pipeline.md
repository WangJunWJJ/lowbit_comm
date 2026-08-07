# Four-GPU Sharded Parameter Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a generic ReducedShard consumer that updates only the local parameter shard, restores replicated parameters through INT8 all-gather and one gathered-dequant/writeback CUDA launch, and demonstrates a stable four-GPU end-to-end throughput improvement.

**Architecture:** Add backend-neutral updated-shard and update-rule contracts under `ccdl_comm.optim`, a Torch-distributed compressed parameter restore transport with caller-owned direct output, and a bounded asynchronous bucket pipeline. Reuse the existing quantization codec and extend the gathered-dequant kernel to decode compact payloads emitted by the fused quant-pack path. Keep unsupported layouts on the existing FP gather fallback.

**Tech Stack:** Python 3.10+, PyTorch 2.4+, `torch.distributed`/NCCL, CUDA C++, pytest, Ruff, A6000 sm_86, Docker.

## Global Constraints

- CCDL remains an independent repository; do not add ParaScale imports or modify ParaScale.
- Performance is the first priority, but rank parameter equality, finite loss, and explicit completion ordering are hard gates.
- Do not hard-code four ranks; production code accepts arbitrary positive `world_size`, with CUDA validation on 2, 4, and 8 ranks.
- The fused fast path is linear INT8, group size 64, top-k 0, compact payload; unsupported policies use an explicit FP gather fallback.
- Parameter compression does not use error feedback in this phase.
- No next forward may observe a parameter bucket before its restore completion event.
- All new behavior is developed test-first and committed as an independently reviewable feature.
- A6000 four-GPU tests use physical GPUs 1, 2, 3, and 4 in one fixed Docker image.

---

## File Structure

- `ccdl_comm/optim/sharded.py`: backend-neutral updated parameter shard, update-rule protocol, and local consumer.
- `ccdl_comm/optim/__init__.py`: public optimizer contract exports.
- `ccdl_comm/communication/parameter_restore.py`: compressed parameter all-gather, fallback, workspace ownership, and work completion.
- `ccdl_comm/communication/sharded_step.py`: bounded bucket pipeline and step completion.
- `ccdl_comm/csrc/quantization/dequant_reduce_kernel.cu`: compact gathered payload decoding support.
- `ccdl_comm/quantization/codec.py`: existing Python facade reused without a second CUDA API.
- `examples/training/compressed_sharded_optimizer.py`: Torch flat-storage integration and executable three-mode example.
- `tests/core/test_sharded_optimizer.py`: core contracts and mutation safety.
- `tests/test_parameter_restore.py`: injected transport, fallback, and resource ownership.
- `tests/test_sharded_step_pipeline.py`: bounded ordering and failure propagation.
- `tests/cuda/test_fused_requantized_restore.py`: compact direct-writeback CUDA correctness.
- `tests/examples/test_compressed_sharded_optimizer.py`: flat storage and CLI tests.
- `tests/distributed/compressed_sharded_optimizer_perf.py`: 2/4-rank correctness and performance runner.
- `tests/benchmarks/sharded_parameter_pipeline_gate.py`: result aggregation and performance gates.
- `tests/benchmarks/reports/sharded_parameter_pipeline_20260807/`: A6000 raw results and report.
- `packages/ccdl-core/pyproject.toml` and `ccdl_comm/build/distributions.py`: include the new `ccdl_comm.optim` package in the Core wheel.

---

### Task 1: Updated parameter shard and local optimizer consumer

**Files:**
- Create: `ccdl_comm/optim/sharded.py`
- Create: `ccdl_comm/optim/__init__.py`
- Modify: `ccdl_comm/__init__.py`
- Modify: `ccdl_comm/build/distributions.py`
- Test: `tests/core/test_sharded_optimizer.py`
- Test: `tests/test_package_build.py`

**Interfaces:**
- Consumes: `ReducedShard`, `FlatShardLayout`.
- Produces: `UpdatedParameterShard`, `ShardUpdateRule`, `SgdShardUpdateRule`, `ShardedOptimizerConsumer.consume(reduced, *, step)`.

- [ ] **Step 1: Write failing core tests**

```python
def test_consumer_updates_only_valid_values_and_returns_layout_version():
    parameter = FakeTensor([10.0, 20.0, 0.0])
    gradient = FakeTensor([1.0, 2.0, 99.0])
    consumer = ShardedOptimizerConsumer(
        layout=layout(original_numel=5, padded_numel=6, rank=1),
        parameter_shard=parameter,
        update_rule=SgdShardUpdateRule(learning_rate=0.1),
        layout_version=7,
    )
    updated = consumer.consume(reduced_shard(gradient, rank=1), step=1)
    assert updated.layout_version == 7
    assert updated.valid_numel == 2
    assert parameter.values == [9.9, 19.8, 0.0]


def test_layout_mismatch_does_not_mutate_parameter_shard():
    consumer = consumer_for_rank(0)
    before = consumer.parameter_shard.clone()
    with pytest.raises(ValueError, match="does not match optimizer layout"):
        consumer.consume(reduced_shard(rank=1), step=1)
    assert consumer.parameter_shard == before
```

- [ ] **Step 2: Run the RED tests**

Run: `python -m pytest tests/core/test_sharded_optimizer.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'ccdl_comm.optim'`.

- [ ] **Step 3: Implement the minimal backend-neutral contracts**

```python
@dataclass(frozen=True)
class UpdatedParameterShard:
    shard: Any
    shard_index: int
    shard_numel: int
    valid_numel: int
    original_numel: int
    padded_numel: int
    world_size: int
    dtype: str
    layout_version: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class ShardUpdateRule(Protocol):
    def update(self, parameter_shard, gradient_shard, state, *, valid_numel: int, step: int): ...


class SgdShardUpdateRule:
    def update(self, parameter_shard, gradient_shard, state, *, valid_numel, step):
        del state, step
        parameter_shard[:valid_numel].add_(gradient_shard[:valid_numel], alpha=-self.learning_rate)
        return parameter_shard
```

Validate the complete `ReducedShard` before calling the rule; zero padding after update; make `metadata` immutable; reject boolean, non-finite, or non-positive learning rates and steps.

- [ ] **Step 4: Export and package the new module**

Add `ccdl_comm.optim` to `CORE_PACKAGES`, export the four public types from `ccdl_comm.optim`, and export `UpdatedParameterShard` plus `ShardedOptimizerConsumer` from `ccdl_comm`.

- [ ] **Step 5: Run GREEN and packaging tests**

Run: `python -m pytest tests/core/test_sharded_optimizer.py tests/test_package_build.py tests/packaging/test_backend_wheels.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add ccdl_comm/optim ccdl_comm/__init__.py ccdl_comm/build/distributions.py tests/core/test_sharded_optimizer.py tests/test_package_build.py
git commit -m "feat(optim): add reduced shard optimizer consumer"
```

---

### Task 2: Shared flat parameter storage with direct model views

**Files:**
- Create: `examples/training/compressed_sharded_optimizer.py`
- Test: `tests/examples/test_compressed_sharded_optimizer.py`

**Interfaces:**
- Consumes: ordered homogeneous Torch parameters and `compile_torch_shard_layout`.
- Produces: `TorchFlatParameterStorage`, whose `padded_flat` is the direct CUDA restore target and whose parameters are views into its logical prefix.

- [ ] **Step 1: Write failing flat-storage tests**

```python
def test_flat_storage_rebinds_parameters_to_one_padded_buffer():
    model = TinyModel()
    storage = TorchFlatParameterStorage.from_parameters(
        model.parameters(), rank=0, world_size=4, group_size=64
    )
    storage.padded_flat[: storage.original_numel].add_(1)
    assert model.weight.data_ptr() == storage.padded_flat.data_ptr()
    torch.testing.assert_close(model.weight.flatten(), storage.padded_flat[: model.weight.numel()])
    assert storage.layout.shard_numel % 512 == 0


def test_incompatible_parameter_storage_is_rejected_before_rebinding():
    model = MixedDtypeModel()
    before = tuple(parameter.data_ptr() for parameter in model.parameters())
    with pytest.raises(ValueError, match="same dtype"):
        TorchFlatParameterStorage.from_parameters(model.parameters(), rank=0, world_size=2)
    assert tuple(parameter.data_ptr() for parameter in model.parameters()) == before
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/examples/test_compressed_sharded_optimizer.py -q`

Expected: import fails because `TorchFlatParameterStorage` is absent.

- [ ] **Step 3: Implement deterministic aligned flat storage**

Allocate `shard_numel = ceil(original_numel / (world_size * 512)) * 512` and
`padded_numel = shard_numel * world_size`. Copy original parameter values into the logical prefix, zero padding, then rebind each `parameter.data` to a shaped view into `padded_flat`. Perform all validation and allocation before the first rebind so failure is atomic.

```python
class TorchFlatParameterStorage:
    @classmethod
    def from_parameters(cls, parameters, *, rank, world_size, group_size=64):
        active = tuple(parameters)
        validated = _validate_homogeneous_parameters(active)
        shard_alignment = math.lcm(group_size, 512)
        shard_numel = _ceil_div(validated.original_numel, world_size * shard_alignment) * shard_alignment
        flat = active[0].new_zeros((shard_numel * world_size,))
        _copy_and_rebind(active, validated.slices, flat)
        return cls(active, validated, flat, rank, world_size, shard_numel)
```

- [ ] **Step 4: Verify storage identity and backward gradients**

Add a test that runs forward/backward after rebinding and confirms every parameter receives a finite gradient with its original shape.

- [ ] **Step 5: Run GREEN**

Run: `python -m pytest tests/examples/test_compressed_sharded_optimizer.py -q`

Expected: all flat-storage tests pass.

- [ ] **Step 6: Commit**

```bash
git add examples/training/compressed_sharded_optimizer.py tests/examples/test_compressed_sharded_optimizer.py
git commit -m "feat(examples): add direct flat parameter storage"
```

---

### Task 3: Compact gathered-dequant CUDA direct writeback

**Files:**
- Modify: `ccdl_comm/csrc/quantization/dequant_reduce_kernel.cu`
- Test: `tests/cuda/test_fused_requantized_restore.py`

**Interfaces:**
- Consumes: compact rank-strided INT8 payloads produced by `inplace_quantize_pack`.
- Produces: existing `inplace_dequantize_gathered(...) -> bool` supporting both `compact=True` and `compact=False` without changing its Python signature.

- [ ] **Step 1: Write failing compact-layout CUDA tests**

```python
@pytest.mark.parametrize("dtype_name", ("fp16", "bf16", "fp32"))
@pytest.mark.parametrize("world_size", (2, 4, 8))
def test_gathered_dequantize_writes_compact_payloads_directly(extension_status, dtype_name, world_size):
    config = CompressionConfig(bit=8, group_size=64, compact=True)
    sources = [torch.randn(126, device="cuda", dtype=DTYPES[dtype_name]) for _ in range(world_size)]
    payloads = [quantize_tensor(source, config, extension_status=extension_status) for source in sources]
    gathered, payload_numel, payload_stride = rank_strided(payloads)
    output = torch.empty(world_size * 126, device="cuda", dtype=DTYPES[dtype_name])
    assert inplace_dequantize_gathered(
        gathered, output, config, dtype=dtype_name, extension_status=extension_status,
        world_size=world_size, payload_numel=payload_numel,
        payload_stride=payload_stride, shard_numel=126,
    )
    torch.testing.assert_close(output, reference_dequantize(payloads, 126, config, dtype_name), rtol=0, atol=0)
```

- [ ] **Step 2: Build the current extension and verify RED on A6000**

Run in the fixed A6000 container:

```bash
CCDL_COMM_BUILD_CUDA=1 TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=2 \
python packages/ccdl-cuda/setup.py build_ext --inplace
pytest -q tests/cuda/test_fused_requantized_restore.py -k compact
```

Expected: the new compact cases fail because native capability returns `False`.

- [ ] **Step 3: Pass compact layout into the kernel**

Change `dequantize_gathered_kernel` to accept `bool compact` and pass it to
`dequant_one_fp32_scale` / `dequant_one_16bit_scale`. Remove only the
`compact` rejection from `can_use_fused_gathered_dequantize`; retain linear INT8,
group-size 64, stride, dtype, device, and output-size checks.

```cpp
template <typename scalar_t>
__global__ void dequantize_gathered_kernel(
    const uint8_t* input, scalar_t* output, int64_t world_size,
    int64_t payload_stride, int64_t shard_numel, bool compact) {
    // derive rank/group/element from output index
    value = dequant_one_16bit_scale<scalar_t>(payload, group, element, compact, num_groups);
    output[index] = float2half<scalar_t>(value);
}
```

- [ ] **Step 4: Rebuild and run GREEN CUDA tests**

Run: `pytest -q tests/cuda/test_fused_quant_pack.py tests/cuda/test_fused_requantized_restore.py`

Expected: all existing non-compact and new compact tests pass for FP16/BF16/FP32.

- [ ] **Step 5: Commit**

```bash
git add ccdl_comm/csrc/quantization/dequant_reduce_kernel.cu tests/cuda/test_fused_requantized_restore.py
git commit -m "perf(cuda): decode compact gathered parameter shards"
```

---

### Task 4: Compressed parameter restore transport

**Files:**
- Create: `ccdl_comm/communication/parameter_restore.py`
- Modify: `ccdl_comm/communication/__init__.py`
- Modify: `ccdl_comm/__init__.py`
- Test: `tests/test_parameter_restore.py`

**Interfaces:**
- Consumes: `UpdatedParameterShard`, `CompressionConfig`, caller-owned padded output.
- Produces: `TorchCompressedParameterRestore.restore(updated, *, out, async_op=True) -> CollectiveWork[Any]`.

- [ ] **Step 1: Write failing transport tests with injected Torch/distributed facades**

```python
def test_restore_quantizes_gathers_and_dequantizes_directly_into_out():
    runtime = FakeRuntime(world_size=4)
    restore = TorchCompressedParameterRestore(
        config=CompressionConfig(compact=True), dtype="fp32",
        import_module=runtime.import_module,
        quantize=runtime.quantize, dequantize_gathered=runtime.dequantize_gathered,
    )
    out = FakeTensor(numel=updated.padded_numel)
    work = restore.restore(updated, out=out, async_op=False)
    assert work.wait() is out
    assert runtime.calls == ["quantize", "all_gather_into_tensor", "dequantize_gathered"]


def test_capability_rejection_uses_fp_gather_without_partial_int8_writeback():
    runtime = FakeRuntime(dequantize_supported=False)
    out = FakeTensor(numel=updated.padded_numel)
    result = restore_for(runtime).restore(updated, out=out, async_op=False).wait()
    assert result is out
    assert runtime.calls[-1] == "fp_all_gather_into_tensor"
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/test_parameter_restore.py -q`

Expected: import fails because the transport class is absent.

- [ ] **Step 3: Implement validation and stable workspaces**

The constructor preallocates exact `uint8` payload and gathered workspaces lazily by key
`(device, dtype, shard_numel, world_size, bit, group_size, compact)`. `restore` validates
the updated shard and output before quantization, calls `quantize_tensor(..., output=send)`,
launches `all_gather_into_tensor`, and completes with `inplace_dequantize_gathered`.

```python
def restore(self, updated, *, out, async_op=True):
    self._validate(updated, out)
    workspace = self._workspace_for(updated)
    self._quantize(updated.shard, self.config, output=workspace.send)
    handle = self._dist.all_gather_into_tensor(
        workspace.gathered, workspace.send, async_op=async_op
    )
    return self._completion_manager.create_work(
        result=out, handle=handle,
        complete=lambda: self._finish_int8_restore(updated, workspace, out),
        resources=(workspace.send, workspace.gathered, out),
    )
```

If capability is known to be unsupported before collective launch, use FP gather. A native
kernel returning `False` after INT8 collective launch is a hard invariant error; do not start a
second mismatched collective.

- [ ] **Step 4: Add steady-state pointer and failure tests**

Verify 100 sequential restores keep send/gather pointers stable, concurrent restore before
the previous work finishes raises `RuntimeError("parameter restore workspace is in flight")`,
and `wait()` propagates collective/kernel exceptions without returning `out` as complete.

- [ ] **Step 5: Run GREEN and related regressions**

Run: `python -m pytest tests/test_parameter_restore.py tests/test_quantization_codec.py tests/test_cuda_completion.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add ccdl_comm/communication/parameter_restore.py ccdl_comm/communication/__init__.py ccdl_comm/__init__.py tests/test_parameter_restore.py
git commit -m "feat(transport): add compressed parameter restoration"
```

---

### Task 5: Bounded asynchronous sharded-step pipeline

**Files:**
- Create: `ccdl_comm/communication/sharded_step.py`
- Modify: `ccdl_comm/communication/__init__.py`
- Test: `tests/test_sharded_step_pipeline.py`

**Interfaces:**
- Consumes: `ShardedOptimizerConsumer`, `TorchCompressedParameterRestore`, bucket-specific output views.
- Produces: `ShardedStepPipeline.consume_bucket(bucket_id, reduced, *, parameter_view, step) -> CollectiveWork[Any]`, `finish_step() -> tuple[Any, ...]`.

- [ ] **Step 1: Write failing ordering tests**

```python
def test_pipeline_waits_oldest_work_at_inflight_limit():
    pipeline = ShardedStepPipeline(
        consumer_for_bucket=consumers.__getitem__,
        restore_for_bucket=restores.__getitem__,
        max_inflight=2,
    )
    first = pipeline.consume_bucket("bucket0", reduced(0), parameter_view=view(0), step=1)
    second = pipeline.consume_bucket("bucket1", reduced(1), parameter_view=view(1), step=1)
    third = pipeline.consume_bucket("bucket2", reduced(2), parameter_view=view(2), step=1)
    assert first.wait_count == 1
    assert second.wait_count == 0
    assert third.wait_count == 0


def test_finish_step_waits_every_bucket_in_submission_order():
    pipeline = populated_pipeline(3)
    result = pipeline.finish_step()
    assert waits == ["bucket0", "bucket1", "bucket2"]
    assert result == (view0, view1, view2)
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/test_sharded_step_pipeline.py -q`

Expected: import fails because `ShardedStepPipeline` is absent.

- [ ] **Step 3: Implement bounded submission and completion**

Use a `deque[CollectiveWork]`. `consume_bucket` resolves the bucket-specific consumer and
restore through constructor-injected callables, validates a monotonically increasing step,
calls consumer then restore, appends work, and waits/pops the oldest when `len > max_inflight`.
`finish_step` waits all remaining work in order and clears step state only after success.

- [ ] **Step 4: Add exception and reentrancy tests**

Confirm a failed work is re-raised by `finish_step`, remaining resources stay referenced until
their works complete, a second step cannot start before `finish_step`, and `max_inflight < 1`
is rejected.

- [ ] **Step 5: Run GREEN**

Run: `python -m pytest tests/test_sharded_step_pipeline.py tests/test_parameter_restore.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add ccdl_comm/communication/sharded_step.py ccdl_comm/communication/__init__.py tests/test_sharded_step_pipeline.py
git commit -m "feat(pipeline): overlap sharded parameter restoration"
```

---

### Task 6: Executable three-mode training example

**Files:**
- Modify: `examples/training/compressed_sharded_optimizer.py`
- Test: `tests/examples/test_compressed_sharded_optimizer.py`
- Test: `tests/test_example_smoke.py`

**Interfaces:**
- Consumes: compiled CCDL ReducedShard operation, `TorchFlatParameterStorage`, optimizer consumer, restore transport, pipeline.
- Produces: CLI modes `native_ddp`, `full_fused`, `sharded_fp`, `sharded_compressed` and one JSON metrics record per rank-0 run.

- [ ] **Step 1: Write failing CLI and metric tests**

```python
def test_parser_exposes_four_comparable_modes():
    parser = build_parser()
    assert parser.parse_args(["--mode", "native_ddp"]).mode == "native_ddp"
    assert parser.parse_args(["--mode", "full_fused"]).mode == "full_fused"
    assert parser.parse_args(["--mode", "sharded_fp"]).mode == "sharded_fp"
    assert parser.parse_args(["--mode", "sharded_compressed"]).mode == "sharded_compressed"


def test_metrics_report_every_pipeline_stage():
    metrics = run_fake_step(mode="sharded_compressed")
    assert set(metrics["stage_ms"]) == {
        "backward_flatten", "compressed_reduce_scatter", "local_update",
        "parameter_quantize_gather", "parameter_restore_writeback",
    }
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/examples/test_compressed_sharded_optimizer.py tests/test_example_smoke.py -q`

Expected: parser/runner imports fail.

- [ ] **Step 3: Implement the common training loop**

Use one model constructor, deterministic synthetic dataset, optimizer hyperparameters, batch
size, warmup, measured steps, and loss function for all modes. Native uses DDP/SGD;
`full_fused` uses the current fused full-gradient CCDL path; `sharded_fp` uses
`TorchShardedSgdConsumer`; `sharded_compressed` uses the new consumer and pipeline.

- [ ] **Step 4: Add rank consistency and checkpoint output**

At the final step, all-reduce the maximum parameter difference against rank 0 and store
`max_rank_parameter_difference`, final loss, throughput, P50/P95, stage timings, peak memory,
selected fast path, fallback reason, and workspace pointers in JSON. Reject non-finite loss or
nonzero rank difference.

- [ ] **Step 5: Run local smoke tests**

Run: `python -m pytest tests/examples/test_compressed_sharded_optimizer.py tests/test_example_smoke.py -q`

Expected: all tests pass without requiring CUDA.

- [ ] **Step 6: Commit**

```bash
git add examples/training/compressed_sharded_optimizer.py tests/examples/test_compressed_sharded_optimizer.py tests/test_example_smoke.py
git commit -m "feat(examples): train with compressed parameter shards"
```

---

### Task 7: A6000 distributed correctness and microbenchmark gate

**Files:**
- Create: `tests/distributed/compressed_sharded_optimizer_perf.py`
- Create: `tests/benchmarks/sharded_parameter_pipeline_gate.py`
- Test: `tests/test_sharded_parameter_pipeline_perf.py`

**Interfaces:**
- Consumes: example modes and JSON records.
- Produces: alternating 2/4-GPU runs, median aggregation, correctness gate, and regression gate.

- [ ] **Step 1: Write failing gate tests using fixed result fixtures**

```python
def test_gate_uses_trial_median_and_requires_exact_rank_equality(tmp_path):
    write_trials(tmp_path, native=[100, 101, 99], compressed=[106, 107, 105], rank_diff=0.0)
    result = evaluate(tmp_path, min_speedup_vs_native=1.05)
    assert result["native_median"] == 100
    assert result["compressed_median"] == 106


def test_gate_rejects_any_nonzero_rank_difference(tmp_path):
    write_trials(tmp_path, native=[100] * 3, compressed=[110] * 3, rank_diff=1e-7)
    with pytest.raises(GateFailure, match="rank parameter difference"):
        evaluate(tmp_path, min_speedup_vs_native=1.0)
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/test_sharded_parameter_pipeline_perf.py -q`

Expected: import fails because gate module is absent.

- [ ] **Step 3: Implement runner and gate**

Runner accepts `--mode`, `--world-size`, `--model-numel`, `--batch-size`, `--warmup`,
`--steps`, and `--output-json`. Gate requires three successful trials per mode, identical GPU
set/image/source hash, zero rank difference, finite loss, no fallback for compressed mode, and
uses the median throughput rather than best run.

- [ ] **Step 4: Run local gate tests**

Run: `python -m pytest tests/test_sharded_parameter_pipeline_perf.py tests/test_synthetic_ddp_script.py -q`

Expected: all tests pass.

- [ ] **Step 5: Run A6000 2/4-GPU alternating matrix**

For two GPUs use physical GPUs 1,2; for four GPUs use 1,2,3,4. Use one exact image and rotate
the four modes across trials:

```text
trial1: native_ddp -> full_fused -> sharded_fp -> sharded_compressed
trial2: full_fused -> sharded_fp -> sharded_compressed -> native_ddp
trial3: sharded_fp -> sharded_compressed -> native_ddp -> full_fused
```

Run at least 50 warmup and 200 measured steps for approximately 44.96M parameters and batch
size 16/rank. Record raw JSON, `nvidia-smi topo -m`, image ID, extension SHA-256, and GPU clocks.

- [ ] **Step 6: Enforce initial performance gate**

Run:

```bash
python tests/benchmarks/sharded_parameter_pipeline_gate.py \
  --results-dir /results/sharded_parameter_pipeline \
  --world-size 4 --min-speedup-vs-native 1.05
```

Expected: pass if compressed median/native median is at least 1.05; otherwise retain the path as
explicit opt-in and use stage timings to select the next optimization before real-data claims.

- [ ] **Step 7: Commit test infrastructure**

```bash
git add tests/distributed/compressed_sharded_optimizer_perf.py tests/benchmarks/sharded_parameter_pipeline_gate.py tests/test_sharded_parameter_pipeline_perf.py
git commit -m "test(benchmark): gate sharded parameter pipeline"
```

---

### Task 8: 21 GB real-data end-to-end validation and report

**Files:**
- Create: `tests/benchmarks/reports/sharded_parameter_pipeline_20260807/README.md`
- Create: `tests/benchmarks/reports/sharded_parameter_pipeline_20260807/summary.json`
- Create: `tests/benchmarks/reports/sharded_parameter_pipeline_20260807/raw/.gitkeep` initially; replace with compact raw result JSON after execution.

**Interfaces:**
- Consumes: the fixed A6000 environment, 21 GB PSI dataset, four production modes.
- Produces: strict FP16 4-GPU median comparison, loss/validation evidence, and a default/opt-in recommendation.

- [ ] **Step 1: Prepare one isolated real-data adapter**

Copy the verified PSI training source to a new remote directory. The adapter may import CCDL and
the example integration, but no PSI code is copied into the CCDL package. Assert at startup:

```python
assert accelerator.mixed_precision == "fp16"
assert accelerator.num_processes == 4
assert tuple(physical_gpu_ids) == (1, 2, 3, 4)
```

- [ ] **Step 2: Run three alternating full epochs per mode**

Use the same dataset view, model initialization, batch 16/rank, dataloader workers, validation,
checkpoint policy, Docker image, source commit, CUDA extension SHA, and GPUs for all twelve runs.
Rotate the four modes in the same trial order used by Task 7.

- [ ] **Step 3: Validate every run before aggregation**

Require one completion marker, structured log, and checkpoint per run; grep for
`Mixed Precision: fp16`; reject traceback, CUDA error, NCCL warning, NaN, Inf, nonzero rank
parameter difference, or any compressed-path fallback.

- [ ] **Step 4: Aggregate the report**

Discard the same first 20 training steps from every run. Report all three throughput values and
their median for Native DDP, existing fused full-gradient CCDL, old sharded FP restore, and the
new compressed sharded pipeline. Also report P50/P95, five stage timings, peak memory, epoch loss,
validation loss, checkpoint size, 4/2 scaling where available, and raw environment hashes.

- [ ] **Step 5: Set the recommendation from evidence**

- Mark the new path eligible for a four-GPU recommended profile only if its median throughput is
  at least 1.05 times Native DDP, it is not slower than existing fused CCDL, and all correctness
  gates pass.
- Otherwise document the measured bottleneck and retain explicit opt-in without changing the
  default strategy.

- [ ] **Step 6: Run final repository verification**

Run locally:

```bash
python -m ruff check ccdl_comm examples tests
python -m pytest -q
python -m json.tool tests/benchmarks/reports/sharded_parameter_pipeline_20260807/summary.json
git diff --check
```

Run remotely with the final extension:

```bash
pytest -q tests/cuda/test_fused_quant_pack.py \
  tests/cuda/test_fused_requantized_restore.py \
  tests/cuda/test_fused_reduced_shard.py
```

Expected: zero failures, zero lint errors, valid JSON, and no diff whitespace errors.

- [ ] **Step 7: Commit the validated report**

```bash
git add tests/benchmarks/reports/sharded_parameter_pipeline_20260807
git commit -m "test(benchmark): validate sharded parameter pipeline"
```
