# Compressed Strategy Planner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a generic topology-aware compressed communication planner and safe fallback path for future multi-GPU and multi-node CCDL strategies.

**Architecture:** Start with a pure strategy planner that has no torch dependency, then wire `strategy="auto"` into the DDP hook without changing the validated all-gather fast path. Add benchmark/report metadata so ParaScale can explain selected strategy and fallback reasons. Leave real reduce-scatter/hierarchical transport and deeper CUDA fusion for later tasks behind explicit capability gates.

**Tech Stack:** Python 3.10+, pytest, torch distributed DDP hook interfaces, existing `ccdl_comm` collectives and benchmark scripts.

## Global Constraints

- CCDL strategy interfaces must not assume single-node execution.
- 2-GPU, 4-GPU, and 8-GPU runs are validation slices, not architectural limits.
- Missing topology, process groups, CUDA extension symbols, or transport features must fall back to the current all-gather strategy.
- Fallback reason and selected strategy must be visible in benchmark output.
- No production code without a failing test first.
- Commit after every independently testable feature.

---

### Task 1: Pure distributed strategy planner

**Files:**
- Create: `ccdl_comm_refactor/ccdl_comm/communication/strategy.py`
- Test: `ccdl_comm_refactor/tests/test_strategy_planner.py`

**Interfaces:**
- Produces: `TopologyInfo`, `CollectiveCapabilities`, `StrategyPlan`
- Produces: `plan_ddp_compression_strategy(*, requested_strategy: str, world_size: int, rank: int = 0, local_world_size: int | None = None, node_count: int | None = None, bucket_numel: int = 0, topology: TopologyInfo | None = None, capabilities: CollectiveCapabilities | None = None) -> StrategyPlan`

- [ ] **Step 1: Write failing planner tests**

```python
from ccdl_comm.communication.strategy import (
    CollectiveCapabilities,
    TopologyInfo,
    plan_ddp_compression_strategy,
)


def test_auto_prefers_all_gather_for_two_ranks() -> None:
    plan = plan_ddp_compression_strategy(
        requested_strategy="auto",
        world_size=2,
        rank=0,
        capabilities=CollectiveCapabilities(reduce_scatter=True),
    )

    assert plan.strategy == "all_gather"
    assert plan.fallback_strategy == "all_gather"
    assert plan.requires_fallback is False
    assert "world_size<=2" in plan.reason


def test_auto_falls_back_without_reduce_scatter_on_single_node_four_ranks() -> None:
    plan = plan_ddp_compression_strategy(
        requested_strategy="auto",
        world_size=4,
        rank=0,
        local_world_size=4,
        node_count=1,
        capabilities=CollectiveCapabilities(reduce_scatter=False),
    )

    assert plan.strategy == "all_gather"
    assert plan.requires_fallback is True
    assert "reduce_scatter unavailable" in plan.reason


def test_auto_selects_reduce_scatter_when_single_node_capable() -> None:
    plan = plan_ddp_compression_strategy(
        requested_strategy="auto",
        world_size=4,
        rank=0,
        local_world_size=4,
        node_count=1,
        capabilities=CollectiveCapabilities(reduce_scatter=True),
    )

    assert plan.strategy == "reduce_scatter"
    assert plan.requires_fallback is False
    assert "single-node capable" in plan.reason


def test_auto_multi_node_requires_hierarchical_capability_and_groups() -> None:
    topology = TopologyInfo(
        rank=3,
        world_size=8,
        local_rank=1,
        local_world_size=4,
        node_id=0,
        node_count=2,
        has_process_groups=True,
    )

    plan = plan_ddp_compression_strategy(
        requested_strategy="auto",
        world_size=8,
        rank=3,
        topology=topology,
        capabilities=CollectiveCapabilities(hierarchical=True),
    )

    assert plan.strategy == "hierarchical"
    assert plan.requires_fallback is False
    assert "multi-node hierarchical" in plan.reason


def test_auto_multi_node_falls_back_without_process_groups() -> None:
    topology = TopologyInfo(
        rank=3,
        world_size=8,
        local_rank=1,
        local_world_size=4,
        node_id=0,
        node_count=2,
        has_process_groups=False,
    )

    plan = plan_ddp_compression_strategy(
        requested_strategy="auto",
        world_size=8,
        rank=3,
        topology=topology,
        capabilities=CollectiveCapabilities(hierarchical=True),
    )

    assert plan.strategy == "all_gather"
    assert plan.requires_fallback is True
    assert "process groups unavailable" in plan.reason


def test_explicit_strategy_is_preserved() -> None:
    plan = plan_ddp_compression_strategy(
        requested_strategy="all_gather",
        world_size=8,
        rank=0,
        capabilities=CollectiveCapabilities(reduce_scatter=True, hierarchical=True),
    )

    assert plan.strategy == "all_gather"
    assert plan.requested_strategy == "all_gather"
    assert plan.requires_fallback is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest ccdl_comm_refactor/tests/test_strategy_planner.py -q`
Expected: FAIL because `ccdl_comm.communication.strategy` does not exist.

- [ ] **Step 3: Implement planner**

Create the dataclasses and pure planner in `ccdl_comm_refactor/ccdl_comm/communication/strategy.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest ccdl_comm_refactor/tests/test_strategy_planner.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ccdl_comm_refactor/ccdl_comm/communication/strategy.py ccdl_comm_refactor/tests/test_strategy_planner.py
git commit -m "feat(ccdl_comm): add distributed strategy planner"
```

### Task 2: Compressed reduce-scatter API skeleton

**Files:**
- Create: `ccdl_comm_refactor/ccdl_comm/collectives/reduce_scatter.py`
- Modify: `ccdl_comm_refactor/ccdl_comm/collectives/__init__.py`
- Modify: `ccdl_comm_refactor/ccdl_comm/__init__.py`
- Test: `ccdl_comm_refactor/tests/test_reduce_scatter_api.py`

**Interfaces:**
- Consumes: `CompressionConfig`
- Produces: `compressed_reduce_scatter(tensor, *, config, op="mean", async_op=False, dtype="auto", reduce_scatter=None, all_gather_fallback=None, extension_status=None)`

- [ ] **Step 1: Write failing API tests**

```python
import pytest

from ccdl_comm import CompressionConfig, compressed_reduce_scatter
from ccdl_comm.exceptions import UnsupportedCollective


class FakeTensor:
    shape = (4,)
    dtype = "float32"


def test_reduce_scatter_falls_back_when_transport_missing() -> None:
    calls = []

    def fallback(tensor, *, config, op, async_op, dtype, extension_status):
        calls.append((tensor, config.bit, op, async_op, dtype, extension_status))
        return "fallback-result"

    result = compressed_reduce_scatter(
        FakeTensor(),
        config=CompressionConfig(bit=8, group_size=64),
        all_gather_fallback=fallback,
    )

    assert result == "fallback-result"
    assert calls[0][1:5] == (8, "mean", False, "auto")


def test_reduce_scatter_rejects_unsupported_op() -> None:
    with pytest.raises(UnsupportedCollective, match="reduce_scatter:max"):
        compressed_reduce_scatter(FakeTensor(), config=CompressionConfig(), op="max")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest ccdl_comm_refactor/tests/test_reduce_scatter_api.py -q`
Expected: FAIL because `compressed_reduce_scatter` is not exported.

- [ ] **Step 3: Implement minimal skeleton**

Implement op validation and fallback-only behavior. Raise `UnsupportedCollective("reduce_scatter:transport")` when neither `reduce_scatter` nor `all_gather_fallback` is provided.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest ccdl_comm_refactor/tests/test_reduce_scatter_api.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ccdl_comm_refactor/ccdl_comm/collectives/reduce_scatter.py ccdl_comm_refactor/ccdl_comm/collectives/__init__.py ccdl_comm_refactor/ccdl_comm/__init__.py ccdl_comm_refactor/tests/test_reduce_scatter_api.py
git commit -m "feat(ccdl_comm): add compressed reduce scatter skeleton"
```

### Task 3: Wire `strategy="auto"` into DDP benchmark metadata

**Files:**
- Modify: `ccdl_comm_refactor/ccdl_comm/communication/ddp_hook.py`
- Modify: `ccdl_comm_refactor/tests/distributed/synthetic_ddp_compare.py`
- Test: `ccdl_comm_refactor/tests/test_ddp_comm_hook.py`
- Test: `ccdl_comm_refactor/tests/test_synthetic_ddp_script.py`

**Interfaces:**
- Consumes: `plan_ddp_compression_strategy`
- Produces: hook attributes `_ccdl_strategy_plan` and benchmark JSON fields `selected_strategy`, `strategy_fallback_reason`

- [ ] **Step 1: Write failing hook/script tests**

Add assertions that `create_ddp_comm_hook(..., strategy="auto")` exposes planner metadata and that the synthetic script writes `selected_strategy` and `strategy_fallback_reason`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest ccdl_comm_refactor/tests/test_ddp_comm_hook.py ccdl_comm_refactor/tests/test_synthetic_ddp_script.py -q`
Expected: FAIL because metadata is missing.

- [ ] **Step 3: Implement minimal metadata wiring**

Resolve `strategy="auto"` at hook creation using available non-distributed defaults for tests. Keep actual processing on the existing all-gather path unless a future transport is implemented. Attach the plan to the hook.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest ccdl_comm_refactor/tests/test_ddp_comm_hook.py ccdl_comm_refactor/tests/test_synthetic_ddp_script.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ccdl_comm_refactor/ccdl_comm/communication/ddp_hook.py ccdl_comm_refactor/tests/distributed/synthetic_ddp_compare.py ccdl_comm_refactor/tests/test_ddp_comm_hook.py ccdl_comm_refactor/tests/test_synthetic_ddp_script.py
git commit -m "feat(ccdl_comm): expose auto strategy fallback metadata"
```

### Task 4: Local and A6000 validation report

**Files:**
- Create: `ccdl_comm_refactor/tests/benchmarks/reports/auto_strategy_<commit>/README.md`
- Create: `ccdl_comm_refactor/tests/benchmarks/reports/auto_strategy_<commit>/raw/*.json`

**Interfaces:**
- Consumes: synthetic benchmark JSON fields from Task 3
- Produces: A6000 validation evidence for 2-GPU and 4-GPU `strategy="auto"`

- [ ] **Step 1: Run focused local tests**

Run: `python -m pytest ccdl_comm_refactor/tests/test_strategy_planner.py ccdl_comm_refactor/tests/test_reduce_scatter_api.py ccdl_comm_refactor/tests/test_ddp_comm_hook.py ccdl_comm_refactor/tests/test_synthetic_ddp_script.py -q`
Expected: PASS.

- [ ] **Step 2: Run remote A6000 benchmark**

Use `user@192.168.8.156 -p 360`, Docker image `ccdl-comm-a6000:cu126-torch25`, and run 2-GPU/4-GPU synthetic DDP with `--strategy auto` plus the current all-gather baseline.

- [ ] **Step 3: Pull raw JSON and write report**

Record selected strategy, fallback reason, samples/s, step ms, loss, and peak memory.

- [ ] **Step 4: Commit report**

```bash
git add ccdl_comm_refactor/tests/benchmarks/reports/auto_strategy_<commit>
git commit -m "test(ccdl_comm): record auto strategy a6000 benchmark"
git push origin wj_dev
```

### Task 5: Capability-gated hierarchical compressed transport prototype

**Files:**
- Create: `ccdl_comm_refactor/ccdl_comm/collectives/hierarchical.py`
- Modify: `ccdl_comm_refactor/ccdl_comm/collectives/__init__.py`
- Modify: `ccdl_comm_refactor/ccdl_comm/__init__.py`
- Modify: `ccdl_comm_refactor/ccdl_comm/communication/ddp_hook.py`
- Modify: `ccdl_comm_refactor/tests/distributed/synthetic_ddp_compare.py`
- Test: `ccdl_comm_refactor/tests/test_hierarchical_api.py`
- Test: `ccdl_comm_refactor/tests/test_ddp_comm_hook.py`
- Test: `ccdl_comm_refactor/tests/test_synthetic_ddp_script.py`

**Interfaces:**
- Produces: `compressed_hierarchical_all_reduce(tensor, *, config, op="mean", async_op=False, dtype="auto", hierarchical_all_reduce=None, all_gather_fallback=None, extension_status=None) -> Any`
- Consumes in DDP hook: optional `hierarchical_all_reduce: Callable[..., Any] | None`
- Produces benchmark metadata field: `strategy_requires_fallback`

- [ ] **Step 1: Write failing hierarchical API tests**

Test that a fake hierarchical transport is called when provided, and that the
function falls back to an injected all-gather fallback when no hierarchical
transport is provided.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest ccdl_comm_refactor/tests/test_hierarchical_api.py -q`
Expected: FAIL because `compressed_hierarchical_all_reduce` is not exported.

- [ ] **Step 3: Implement minimal capability-gated API**

Implement validation for `op in {"sum", "mean"}`. Call the injected
`hierarchical_all_reduce` when provided. Otherwise call `all_gather_fallback`
when provided. Raise `UnsupportedCollective("hierarchical:transport")` when
neither is available.

- [ ] **Step 4: Wire DDP hook without changing default behavior**

Add optional `hierarchical_all_reduce` to `create_ddp_comm_hook`. Planner
capabilities should set `hierarchical=True` only when that callable is provided.
If the planner selects `hierarchical`, run the callable through
`compressed_hierarchical_all_reduce`; otherwise continue the existing effective
all-gather/all-reduce paths. Default behavior remains fallback.

- [ ] **Step 5: Add benchmark metadata**

Record `strategy_requires_fallback` in synthetic benchmark JSON from
`hook._ccdl_strategy_plan.requires_fallback`.

- [ ] **Step 6: Run focused tests**

Run: `python -m pytest ccdl_comm_refactor/tests/test_hierarchical_api.py ccdl_comm_refactor/tests/test_strategy_planner.py ccdl_comm_refactor/tests/test_ddp_comm_hook.py ccdl_comm_refactor/tests/test_synthetic_ddp_script.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add ccdl_comm_refactor/ccdl_comm/collectives/hierarchical.py ccdl_comm_refactor/ccdl_comm/collectives/__init__.py ccdl_comm_refactor/ccdl_comm/__init__.py ccdl_comm_refactor/ccdl_comm/communication/ddp_hook.py ccdl_comm_refactor/tests/distributed/synthetic_ddp_compare.py ccdl_comm_refactor/tests/test_hierarchical_api.py ccdl_comm_refactor/tests/test_ddp_comm_hook.py ccdl_comm_refactor/tests/test_synthetic_ddp_script.py
git commit -m "feat(ccdl_comm): add hierarchical transport prototype"
```

### Task 6: A6000 hierarchical fallback validation report

**Files:**
- Create: `ccdl_comm_refactor/tests/benchmarks/reports/hierarchical_proto_<commit>/README.md`
- Create: `ccdl_comm_refactor/tests/benchmarks/reports/hierarchical_proto_<commit>/raw/*.json`

**Interfaces:**
- Consumes: benchmark metadata fields from Task 5
- Produces: A6000 evidence that `strategy="hierarchical"` and `strategy="auto"`
  remain safe on 4 GPUs when no real hierarchical transport is available.

- [ ] **Step 1: Run remote A6000 4-GPU validation**

Use `user@192.168.8.156 -p 360`, Docker image
`ccdl-comm-a6000:cu126-torch25`, and run 4-GPU synthetic DDP with
`--strategy all_gather`, `--strategy auto`, and `--strategy hierarchical`.

- [ ] **Step 2: Pull raw JSON and write report**

Record selected strategy, fallback reason, `strategy_requires_fallback`,
samples/s, step ms, loss, and peak memory.

- [ ] **Step 3: Commit report**

```bash
git add ccdl_comm_refactor/tests/benchmarks/reports/hierarchical_proto_<commit>
git commit -m "test(ccdl_comm): record hierarchical prototype a6000 benchmark"
git push origin wj_dev
```

### Task 7: True compressed reduce-scatter/all-gather transport prototype

**Files:**
- Create: `ccdl_comm_refactor/ccdl_comm/communication/reduce_scatter_transport.py`
- Modify: `ccdl_comm_refactor/ccdl_comm/communication/ddp_hook.py`
- Modify: `ccdl_comm_refactor/tests/distributed/synthetic_ddp_compare.py`
- Test: `ccdl_comm_refactor/tests/test_reduce_scatter_transport.py`
- Test: `ccdl_comm_refactor/tests/test_ddp_comm_hook.py`
- Test: `ccdl_comm_refactor/tests/test_synthetic_ddp_script.py`

**Interfaces:**
- Produces: `make_torch_compressed_reduce_scatter_all_gather(*, import_module=_import_module, quantize=quantize_tensor, dequantize_reduce=dequantize_reduce_tensors) -> Callable[..., Any]`
- Consumes in DDP hook: optional `reduce_scatter_all_gather: Callable[..., Any] | None`
- Produces benchmark CLI flag: `--enable-reduce-scatter-transport`

- [ ] **Step 1: Write failing reduce-scatter transport tests**

```python
from ccdl_comm.communication.reduce_scatter_transport import (
    make_torch_compressed_reduce_scatter_all_gather,
)
from ccdl_comm.config import CompressionConfig


def test_reduce_scatter_transport_quantizes_per_destination_chunk_and_restores_full_bucket():
    calls = []

    class Tensor:
        def __init__(self, values):
            self.values = tuple(values)
            self.shape = (len(self.values),)
            self.dtype = "torch.float32"

        def reshape(self, shape):
            assert shape == (-1,)
            return self

        def chunk(self, chunks):
            assert chunks == 2
            return (Tensor(self.values[:2]), Tensor(self.values[2:]))

        def new_empty(self, shape):
            return Tensor([0.0] * shape[0])

    class Dist:
        ReduceOp = object()

        def is_available(self): return True
        def is_initialized(self): return True
        def get_world_size(self): return 2
        def get_rank(self): return 0
        def all_to_all(self, output, input):
            calls.append(("all_to_all", tuple(payload.values for payload in input)))
            output[:] = [Tensor([10.0, 20.0]), Tensor([30.0, 40.0])]
        def all_gather(self, output, input):
            calls.append(("all_gather", input.values))
            output[:] = [input, Tensor([50.0, 60.0])]

    def import_module(name):
        assert name == "torch.distributed"
        return Dist()

    def quantize(tensor, config, *, extension_status):
        calls.append(("quantize", tensor.values, config.bit))
        return Tensor([sum(tensor.values)])

    def dequantize_reduce(buffers, shape, config, *, dtype, extension_status, reduce):
        calls.append(("dequantize_reduce", tuple(buffer.values for buffer in buffers), shape, reduce))
        return Tensor([1.0, 2.0])

    transport = make_torch_compressed_reduce_scatter_all_gather(
        import_module=import_module,
        quantize=quantize,
        dequantize_reduce=dequantize_reduce,
    )

    result = transport(
        Tensor([1.0, 2.0, 3.0, 4.0]),
        config=CompressionConfig(bit=8),
        op="mean",
        async_op=False,
        dtype="fp32",
        extension_status=None,
    )

    assert result.values == (1.0, 2.0, 50.0, 60.0)
    assert ("quantize", (1.0, 2.0), 8) in calls
    assert ("quantize", (3.0, 4.0), 8) in calls
    assert ("all_to_all", ((3.0,), (7.0,))) in calls
    assert ("dequantize_reduce", ((10.0, 20.0), (30.0, 40.0)), (2,), "mean") in calls
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest ccdl_comm_refactor/tests/test_reduce_scatter_transport.py -q`
Expected: FAIL because `ccdl_comm.communication.reduce_scatter_transport` does not exist.

- [ ] **Step 3: Implement minimal real transport**

Implement a synchronous transport that splits the flattened bucket into one
equal chunk per rank, quantizes each chunk independently, exchanges compressed
chunks with `dist.all_to_all`, reduces the received compressed chunks with
`dequantize_reduce_tensors`, and restores full DDP bucket semantics with
`dist.all_gather` followed by `torch.cat(...).reshape(original_shape)`.
Reject `async_op=True`, unsupported reductions, non-divisible flattened bucket
sizes, and unequal compressed chunk sizes with `UnsupportedCollective`.

- [ ] **Step 4: Wire DDP hook and benchmark flag behind capability gate**

Add `reduce_scatter_all_gather` to `create_ddp_comm_hook`. Set planner
`reduce_scatter=True` only when this callable is provided. If selected, call
`compressed_reduce_scatter(..., reduce_scatter=reduce_scatter_all_gather)`.
Add `--enable-reduce-scatter-transport` to `synthetic_ddp_compare.py`; when
enabled, inject `make_torch_compressed_reduce_scatter_all_gather()`.

- [ ] **Step 5: Run focused tests**

Run: `python -m pytest ccdl_comm_refactor/tests/test_reduce_scatter_transport.py ccdl_comm_refactor/tests/test_reduce_scatter_api.py ccdl_comm_refactor/tests/test_strategy_planner.py ccdl_comm_refactor/tests/test_ddp_comm_hook.py ccdl_comm_refactor/tests/test_synthetic_ddp_script.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add ccdl_comm_refactor/ccdl_comm/communication/reduce_scatter_transport.py ccdl_comm_refactor/ccdl_comm/communication/ddp_hook.py ccdl_comm_refactor/tests/distributed/synthetic_ddp_compare.py ccdl_comm_refactor/tests/test_reduce_scatter_transport.py ccdl_comm_refactor/tests/test_ddp_comm_hook.py ccdl_comm_refactor/tests/test_synthetic_ddp_script.py
git commit -m "feat(ccdl_comm): add compressed reduce scatter transport"
```

### Task 8: A6000 reduce-scatter transport validation report

**Files:**
- Create: `ccdl_comm_refactor/tests/benchmarks/reports/reduce_scatter_transport_<commit>/README.md`
- Create: `ccdl_comm_refactor/tests/benchmarks/reports/reduce_scatter_transport_<commit>/raw/*.json`

**Interfaces:**
- Consumes: benchmark fields from Task 7
- Produces: A6000 4-GPU comparison of current validated all-gather path and
  capability-gated true compressed reduce-scatter/all-gather prototype.

- [ ] **Step 1: Sync code to A6000**

Upload changed Python files to `/home/user/wangjun/ccdl-master` on
`user@192.168.8.156 -p 360`.

- [ ] **Step 2: Run remote focused tests**

Run inside Docker image `ccdl-comm-a6000:cu126-torch25`:
`PYTHONPATH=/workspace/ccdl_comm_refactor python -m pytest tests/test_reduce_scatter_transport.py tests/test_reduce_scatter_api.py tests/test_strategy_planner.py tests/test_ddp_comm_hook.py tests/test_synthetic_ddp_script.py -q`
Expected: PASS.

- [ ] **Step 3: Run remote 4-GPU benchmark**

Run synthetic DDP for:
- `--strategy all_gather`
- `--strategy reduce_scatter --enable-reduce-scatter-transport true`
- `--strategy auto --enable-reduce-scatter-transport true`

Record avg step ms, samples/s, memory, loss, selected strategy, fallback
reason, and whether the result validates performance.

- [ ] **Step 4: Commit report and push**

```bash
git add ccdl_comm_refactor/tests/benchmarks/reports/reduce_scatter_transport_<commit>
git commit -m "test(ccdl_comm): record reduce scatter a6000 benchmark"
git push origin wj_dev
```

### Task 9: Sharded reduced-gradient consumer contract

**Files:**
- Modify: `ccdl_comm_refactor/ccdl_comm/collectives/reduce_scatter.py`
- Modify: `ccdl_comm_refactor/ccdl_comm/collectives/__init__.py`
- Modify: `ccdl_comm_refactor/ccdl_comm/__init__.py`
- Modify: `ccdl_comm_refactor/ccdl_comm/communication/reduce_scatter_transport.py`
- Test: `ccdl_comm_refactor/tests/test_reduce_scatter_api.py`
- Test: `ccdl_comm_refactor/tests/test_reduce_scatter_transport.py`

**Interfaces:**
- Produces: `ReducedShard(shard, shard_index: int, shard_numel: int, original_shape: tuple[int, ...], original_numel: int, world_size: int, reduce: str)`
- Produces: `compressed_reduce_scatter_shard(tensor, *, config, op="mean", async_op=False, dtype="auto", reduce_scatter_shard=None, extension_status=None) -> ReducedShard`
- Produces: `make_torch_compressed_reduce_scatter_shard(...) -> Callable[..., ReducedShard]`

- [ ] **Step 1: Write failing API and transport tests**

Add tests proving that `compressed_reduce_scatter_shard` calls an injected shard
transport and that the torch transport performs compressed all-to-all plus local
dequant-reduce without calling final `all_gather`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest ccdl_comm_refactor/tests/test_reduce_scatter_api.py ccdl_comm_refactor/tests/test_reduce_scatter_transport.py -q`
Expected: FAIL because the shard API and transport do not exist.

- [ ] **Step 3: Implement minimal public shard API**

Add `ReducedShard` and `compressed_reduce_scatter_shard` to
`collectives/reduce_scatter.py`, export them from `collectives/__init__.py` and
`ccdl_comm/__init__.py`, validate `op in {"sum", "mean"}`, reject async until a
transport supports it, and raise `UnsupportedCollective("reduce_scatter_shard:transport")`
when no transport is injected.

- [ ] **Step 4: Implement torch shard transport**

Add `make_torch_compressed_reduce_scatter_shard` beside the existing full-bucket
transport. Reuse compressed per-destination chunk all-to-all and
`dequantize_reduce_tensors`, but return `ReducedShard` immediately after the
local shard is restored. Do not call `dist.all_gather`.

- [ ] **Step 5: Run focused tests**

Run: `python -m pytest ccdl_comm_refactor/tests/test_reduce_scatter_api.py ccdl_comm_refactor/tests/test_reduce_scatter_transport.py ccdl_comm_refactor/tests/test_strategy_planner.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add ccdl_comm_refactor/ccdl_comm/collectives/reduce_scatter.py ccdl_comm_refactor/ccdl_comm/collectives/__init__.py ccdl_comm_refactor/ccdl_comm/__init__.py ccdl_comm_refactor/ccdl_comm/communication/reduce_scatter_transport.py ccdl_comm_refactor/tests/test_reduce_scatter_api.py ccdl_comm_refactor/tests/test_reduce_scatter_transport.py
git commit -m "feat(ccdl_comm): add reduced shard consumer contract"
```
