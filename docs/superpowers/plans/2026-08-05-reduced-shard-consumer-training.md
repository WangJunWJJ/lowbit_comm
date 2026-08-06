# ReducedShard Consumer End-to-End Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a backend-neutral ReducedShard consumer contract and an `examples/` ZeRO-2-style sharded SGD path that converts compressed reduce-scatter savings into measurable 2/4-GPU end-to-end training throughput.

**Architecture:** Core defines immutable flat-shard metadata and a consumer protocol. Torch example code owns reusable flat-gradient, reduced-gradient, local-parameter-shard, and gathered-parameter buffers; it updates only the local parameter shard, gathers updated FP16 shards into one contiguous buffer, and writes the valid prefix back to the replicated model. A correctness-first gate compares native DDP, current CCDL full-gradient compression, and the sharded consumer before accepting speedup.

**Tech Stack:** Python 3.10+, PyTorch 2.5 distributed/NCCL and Gloo, CCDL CUDA extension, pytest, Ruff, RTX A6000 Docker image `ccdl-comm-a6000:cu126-torch25`.

## Global Constraints

- CCDL remains independent; do not import ParaScale or PyTorch FSDP private APIs.
- Training code lives under `examples/`; Torch optimizer helpers do not enter backend-neutral Core.
- First optimizer: SGD without momentum, weight decay, master weights, or parameter groups.
- ReducedShard and parameter/gradient workspaces are caller-owned and reused in steady state.
- Hot path uses contiguous buffers and `all_gather_into_tensor`, never per-rank tensor lists or `torch.cat`.
- Padding never enters model parameters or loss computation.
- Correctness, CUDA extension execution, and rank equality gate every performance claim.
- Local tests precede A6000 tests. A6000 acceptance covers 2 and 4 GPUs, three independent runs per mode, and median aggregation.
- Each independently reviewable task ends with a Conventional Commit matching `CONTRIBUTING.md`.

## File Structure

- `ccdl_comm/consumer.py`: backend-neutral consumer protocol.
- `ccdl_comm/shard_layout.py`: immutable parameter and rank-local shard metadata.
- `examples/training/sharded_sgd.py`: Torch flat buffers and sharded SGD consumer.
- `examples/training/sharded_metrics.py`: stable result schema.
- `examples/sharded_training.py`: three-mode executable comparison.
- `tests/core/test_reduced_shard_consumer.py`: Core tests.
- `tests/examples/test_sharded_sgd.py`: Torch helper tests.
- `tests/examples/test_sharded_training.py`: example/config/schema tests.
- `tests/distributed/sharded_training_smoke.py`: distributed correctness oracle.
- `tests/benchmarks/run_sharded_training_gate.py`: correctness-first gate.
- `tests/test_sharded_training_gate.py`: gate anti-bypass tests.

---

### Task 1: Core consumer and flat layout contract

**Files:**
- Create: `ccdl_comm/consumer.py`
- Create: `ccdl_comm/shard_layout.py`
- Modify: `ccdl_comm/__init__.py`
- Create: `tests/core/test_reduced_shard_consumer.py`

**Interfaces:**
- Consumes: `ccdl_comm.shard.ReducedShard`.
- Produces: `ReducedShardConsumer.consume(reduced: ReducedShard) -> object`.
- Produces: `FlatParameterSlice(index, offset, numel, shape, dtype, requires_grad)`.
- Produces: `FlatShardLayout.validate_reduced_shard(reduced) -> None`.

- [ ] **Step 1: Write failing protocol and layout tests**

Require a runtime-checkable consumer, contiguous slices, exact padding metadata, immutable tuples, and strict ReducedShard validation:

```python
def test_layout_validates_matching_reduced_shard() -> None:
    layout = FlatShardLayout(
        original_numel=5, padded_numel=6, shard_numel=3,
        world_size=2, shard_index=1,
        parameters=(
            FlatParameterSlice(0, 0, 4, (2, 2), "fp16", True),
            FlatParameterSlice(1, 4, 1, (1,), "fp16", True),
        ),
    )
    reduced = ReducedShard(
        shard=FakeTensor(3), shard_index=1, shard_numel=3,
        original_shape=(5,), original_numel=5, padded_numel=6,
        world_size=2, reduce="mean", dtype="fp16",
    )
    layout.validate_reduced_shard(reduced)
    assert (layout.valid_numel, layout.padding_numel) == (2, 1)
```

Negative tests cover bool-as-int, slice gaps/overlaps, inconsistent padding, invalid rank, dtype/range mismatch, and shard tensor numel mismatch.

- [ ] **Step 2: Verify RED**

Run `python -m pytest tests/core/test_reduced_shard_consumer.py -q`.

Expected: collection fails because the two modules do not exist.

- [ ] **Step 3: Implement minimal Core types**

```python
@runtime_checkable
class ReducedShardConsumer(Protocol):
    def consume(self, reduced: ReducedShard) -> object: ...

@dataclass(frozen=True, slots=True)
class FlatParameterSlice:
    index: int
    offset: int
    numel: int
    shape: tuple[int, ...]
    dtype: str
    requires_grad: bool

@dataclass(frozen=True, slots=True)
class FlatShardLayout:
    original_numel: int
    padded_numel: int
    shard_numel: int
    world_size: int
    shard_index: int
    parameters: tuple[FlatParameterSlice, ...]
    def validate_reduced_shard(self, reduced: ReducedShard) -> None: ...
```

Use strict integer checks, require complete coverage of `[0, original_numel)`, one dtype, and exact ReducedShard ownership fields. Export all public types.

- [ ] **Step 4: Verify GREEN and lint**

Run:

```bash
python -m pytest tests/core/test_reduced_shard_consumer.py tests/test_reduce_scatter_api.py tests/test_communication_exports.py -q
python -m ruff check ccdl_comm/consumer.py ccdl_comm/shard_layout.py tests/core/test_reduced_shard_consumer.py
```

Expected: all pass without warnings.

- [ ] **Step 5: Commit**

```bash
git add ccdl_comm/consumer.py ccdl_comm/shard_layout.py ccdl_comm/__init__.py tests/core/test_reduced_shard_consumer.py
git commit -m "feat(consumer): add reduced shard layout contract"
```

---

### Task 2: Reusable Torch buffers and sharded SGD consumer

**Files:**
- Create: `examples/training/sharded_sgd.py`
- Create: `tests/examples/test_sharded_sgd.py`

**Interfaces:**
- Produces: `compile_torch_shard_layout(parameters, *, rank, world_size) -> FlatShardLayout`.
- Produces: `TorchShardedSgdConsumer(parameters, layout, learning_rate, all_gather_into_tensor, torch)`.
- Produces: `flatten_gradients()`, `reduced_output()`, and `consume(reduced)`.

- [ ] **Step 1: Write failing real-tensor tests**

Use CPU torch tensors. Require stable order, reusable gradient storage, zero filling for `grad is None`, padding preservation, and rejection of mixed dtype/device:

```python
consumer = TorchShardedSgdConsumer(
    model.parameters(), layout=layout, learning_rate=0.1,
    all_gather_into_tensor=fake_gather, torch=torch,
)
first = consumer.flatten_gradients()
second = consumer.flatten_gradients()
assert first.data_ptr() == second.data_ptr()
assert second[missing_gradient_range].count_nonzero() == 0
assert consumer.reduced_output().numel() == layout.shard_numel
```

- [ ] **Step 2: Verify RED**

Run `python -m pytest tests/examples/test_sharded_sgd.py -q`.

Expected: FAIL because `examples.training.sharded_sgd` does not exist.

- [ ] **Step 3: Implement buffers and local update**

Allocate once at construction:

```python
self._flat_gradients = first_parameter.new_zeros(layout.padded_numel)
self._reduced_gradient = first_parameter.new_empty(layout.shard_numel)
self._local_parameters = first_parameter.new_zeros(layout.shard_numel)
self._gathered_parameters = first_parameter.new_empty(layout.padded_numel)
```

Initialize the valid local parameter prefix from the owned flattened range. `flatten_gradients()` copies each gradient to its fixed slice and zeros absent gradients. `consume()` validates before mutation, applies `local_parameters.add_(reduced.shard, alpha=-learning_rate)`, calls injected `all_gather_into_tensor`, writes only the original valid prefix to model tensors, and returns the gathered logical view.

- [ ] **Step 4: Test exact update and reuse**

Add a padded-rank test requiring exact SGD values, zero padding, correct model writeback, one gather call with contiguous buffers, and identical four workspace pointers for ten iterations. A mismatched shard must leave parameters unchanged.

Run:

```bash
python -m pytest tests/examples/test_sharded_sgd.py tests/core/test_reduced_shard_consumer.py -q
python -m ruff check examples/training/sharded_sgd.py tests/examples/test_sharded_sgd.py
```

- [ ] **Step 5: Commit**

```bash
git add examples/training/sharded_sgd.py tests/examples/test_sharded_sgd.py
git commit -m "feat(examples): add reusable sharded SGD consumer"
```

---

### Task 3: Three-mode end-to-end training example

**Files:**
- Create: `examples/training/sharded_metrics.py`
- Create: `examples/sharded_training.py`
- Create: `examples/configs/a6000_sharded_2gpu.json`
- Create: `examples/configs/a6000_sharded_4gpu.json`
- Modify: `examples/README.md`
- Create: `tests/examples/test_sharded_training.py`

**Interfaces:**
- CLI modes: `native_ddp`, `ccdl_full_gradient`, `ccdl_sharded_sgd`.
- Result schema: workload, correctness, execution, timing, phase timing, memory, loss, and buffer pointers.

- [ ] **Step 1: Write failing parser and schema tests**

Require identical mode-independent workload signatures and exactly these phases:

```python
assert payload["phase_timing_ms"].keys() == {
    "backward_and_flatten", "compressed_reduce_scatter",
    "local_shard_update", "parameter_all_gather", "parameter_writeback",
}
assert payload["execution"]["fallback_reason"] is None
assert payload["correctness"]["rank_parameters_consistent"] is True
```

Reject non-finite phases, missing workload fields, and pointer changes when reuse is claimed.

- [ ] **Step 2: Verify RED**

Run `python -m pytest tests/examples/test_sharded_training.py -q`.

Expected: FAIL because the example and schema do not exist.

- [ ] **Step 3: Implement three modes**

Native mode uses DDP+SGD. Full-gradient mode uses the existing synchronous INT8 all-gather DDP hook+SGD. Sharded mode uses the raw replicated model and runs:

```python
loss.backward()
flat_gradients = consumer.flatten_gradients()
reduced = compiled_plan.run(flat_gradients, out=consumer.reduced_output()).wait()
consumer.consume(reduced)
model.zero_grad(set_to_none=True)
```

Compile once before warm-up with `collective="reduce_scatter"`, `strategy="compressed"`, `output_layout="shard"`, INT8/group 64, and caller-owned output. Require CUDA reduced-shard fast path and no fallback before measuring.

Use CUDA events for the five phases and synchronize only at warm-up/measurement boundaries. All phases remain inside end-to-end step latency.

- [ ] **Step 4: Verify CPU smoke, schema, and lint**

Run:

```bash
python -m pytest tests/examples/test_sharded_training.py tests/examples/test_training_config.py tests/examples/test_training_metrics.py -q
python examples/sharded_training.py --mode native_ddp --synthetic --device cpu --steps 2 --warmup-steps 1 --batch-size-per-rank 2 --input-dim 32 --hidden-dim 64 --depth 2 --num-classes 8 --output dist/sharded-training-smoke.json
python -m ruff check examples/sharded_training.py examples/training/sharded_metrics.py tests/examples/test_sharded_training.py
```

Expected: smoke JSON has finite loss, one measured step, and schema version 1.

- [ ] **Step 5: Commit**

```bash
git add examples/sharded_training.py examples/training/sharded_metrics.py examples/configs/a6000_sharded_2gpu.json examples/configs/a6000_sharded_4gpu.json examples/README.md tests/examples/test_sharded_training.py
git commit -m "feat(examples): add end-to-end sharded SGD training"
```

---

### Task 4: Distributed correctness oracle and performance gate

**Files:**
- Create: `tests/distributed/sharded_training_smoke.py`
- Create: `tests/benchmarks/run_sharded_training_gate.py`
- Create: `tests/test_sharded_training_gate.py`

**Interfaces:**
- Produces: `evaluate_sharded_runs(native, full_gradient, sharded, thresholds) -> dict[str, object]`.
- Persists failure stages: `input`, `comparability`, `correctness`, `execution`, `convergence`, `performance`.

- [ ] **Step 1: Write failing Gloo semantic smoke wrapper**

Launch two ranks with a tiny FP32 model and injected exact reduce-scatter. After every step require zero rank parameter difference, matching layout/shard logical ranges, and stable buffer pointers. This isolates consumer correctness from quantization error.

- [ ] **Step 2: Write RED gate tests**

Reject mismatched workload/world size/global batch, NaN thresholds, string booleans, nonzero rank difference, fallback, non-CUDA sharded capability, unstable pointers, loss divergence above `0.02`, a 2-GPU native ratio below `0.95`, and a 4-GPU sharded/full-gradient ratio not greater than `1.0`. Malformed JSON must still persist a failure report.

```python
with pytest.raises(GateFailure, match="sharded consumer benefit"):
    evaluate_sharded_runs(
        candidate("native_ddp", 100.0),
        candidate("ccdl_full_gradient", 90.0),
        candidate("ccdl_sharded_sgd", 89.0),
    )
```

- [ ] **Step 3: Verify RED**

Run `python -m pytest tests/test_sharded_training_gate.py tests/examples/test_sharded_training.py -q`.

Expected: FAIL because the gate module does not exist.

- [ ] **Step 4: Implement strict ordered gate**

Validation order is: full workload identity; finite/decreasing loss and exact rank equality; no fallback plus `cuda_extension` and compressed reduce-scatter; stable caller-owned pointers; final loss relative difference `<=0.02`; two-GPU sharded/native ratio `>=0.95`; four-GPU sharded/full-gradient ratio `>1.0`. Always write raw inputs and failure stage, and disallow overrides below these safety floors.

- [ ] **Step 5: Verify GREEN**

Run:

```bash
torchrun --standalone --nproc-per-node=2 tests/distributed/sharded_training_smoke.py
python -m pytest tests/test_sharded_training_gate.py tests/examples/test_sharded_sgd.py tests/core/test_reduced_shard_consumer.py -q
python -m ruff check tests/distributed/sharded_training_smoke.py tests/benchmarks/run_sharded_training_gate.py tests/test_sharded_training_gate.py
```

Expected: all pass and Gloo reports exact rank equality.

- [ ] **Step 6: Commit**

```bash
git add tests/distributed/sharded_training_smoke.py tests/benchmarks/run_sharded_training_gate.py tests/test_sharded_training_gate.py
git commit -m "test(training): gate sharded consumer correctness and speedup"
```

---

### Task 5: A6000 two/four-GPU end-to-end validation

**Files:**
- Create: `tests/benchmarks/reports/reduced_shard_consumer_20260805/README.md`
- Create: `tests/benchmarks/reports/reduced_shard_consumer_20260805/summary.json`
- Create: `tests/benchmarks/reports/reduced_shard_consumer_20260805/raw/*.json`

**Interfaces:**
- Consumes Tasks 3-4.
- Produces median 2/4-GPU results, phase attribution, raw evidence, and gate decisions.

- [ ] **Step 1: Synchronize exact source and record environment**

Use `user@192.168.8.156:360`, `/home/user/wangjun/lowbit_comm_task9_current`, image `ccdl-comm-a6000:cu126-torch25`, `--ipc=host`, and `--shm-size=8g`. Record commit SHA, image ID, torch/CUDA versions, GPU names, and `nvidia-smi topo -m`.

- [ ] **Step 2: Run focused A6000 correctness tests**

Inside Docker:

```bash
python -m pytest tests/core/test_reduced_shard_consumer.py tests/examples/test_sharded_sgd.py tests/examples/test_sharded_training.py tests/test_sharded_training_gate.py -q
torchrun --standalone --nproc-per-node=2 tests/distributed/sharded_training_smoke.py --backend nccl
```

Expected: CUDA extension selected, no fallback, exact rank equality.

- [ ] **Step 3: Run 18 independent benchmarks**

For 2 and 4 GPUs, run three fresh processes for each mode. Example:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  examples/sharded_training.py --config examples/configs/a6000_sharded_2gpu.json \
  --mode ccdl_sharded_sgd --output /results/2gpu_ccdl_sharded_sgd_run1.json
```

Four-GPU runs use devices `0,1,2,3`. Both configs use the 44,971,744-parameter FP16 MLP, batch 16/rank, 22 steps, 2 warm-up, learning rate `0.001`, INT8/group 64, and seed `20260805`.

- [ ] **Step 4: Aggregate medians and run gates**

Retain every raw run and choose the median-throughput run per mode. Calculate:

```text
sharded_vs_native = sharded_samples_per_second / native_samples_per_second
sharded_vs_full_gradient = sharded_samples_per_second / full_gradient_samples_per_second
communication_fraction = compressed_reduce_scatter_ms / mean_step_latency_ms
parameter_restore_fraction = (parameter_all_gather_ms + parameter_writeback_ms) / mean_step_latency_ms
```

Run separate 2/4-GPU gates. Preserve nonzero exit status instead of weakening thresholds.

- [ ] **Step 5: Write evidence report**

Report throughput, ratios, P50/P95, peak memory, final loss, rank equality, and all five phases. If four GPUs fail, name the largest phase as the next isolated optimization boundary. State explicitly that 22 synthetic steps do not establish final model accuracy.

- [ ] **Step 6: Run final verification**

```bash
python -m pytest -q
python -m ruff check .
python -m build
git diff --check
```

Expected: full suite, lint, build, and whitespace checks pass.

- [ ] **Step 7: Commit evidence**

```bash
git add tests/benchmarks/reports/reduced_shard_consumer_20260805
git commit -m "test(benchmark): validate sharded SGD on A6000"
```

If a gate fails, this commit records the failure honestly. Any hotspot optimization begins only from measured phase evidence and receives a separate design, RED-GREEN cycle, and `perf` commit.

## Plan Self-Review

- Tasks 1-5 cover every approved design requirement.
- Core remains torch-free; Torch training logic remains in `examples/training/`.
- Interface names and shard ownership metadata are consistent across tasks.
- Correctness gates precede performance gates and reject fallback or malformed evidence.
- A6000 workload, repetitions, aggregation, metrics, and thresholds are explicit.
- No step assumes ParaScale, FSDP private APIs, AdamW, permanent parameter sharding, or multi-node execution.
