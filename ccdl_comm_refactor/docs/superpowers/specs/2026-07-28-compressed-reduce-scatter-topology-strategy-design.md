# Compressed reduce-scatter and topology-aware DDP strategy design

## Goal

Improve refactored CCDL scaling on 4-GPU and 8-GPU single-node training by
reducing the current `all_gather -> local dequant-reduce` expansion cost.

The next phase should keep ParaScale integration safe and explainable:

- add topology-aware strategy selection instead of hard-coding all-gather;
- introduce compressed reduce-scatter and hierarchical collective interfaces;
- keep current all-gather as the correctness-preserving fallback;
- fuse `dequant + reduce + mean + error-feedback update` in CUDA for the hot
  DDP bucket path;
- reduce Python callback and kernel-launch overhead with bucket-level fusion.

This design does not attempt to replace NCCL. CCDL remains a compressed
communication layer over torch/NCCL transports with capability-gated fast paths.

## Current context

The validated DDP hook path is:

```text
bucket tensor
  -> optional error-feedback compensation
  -> quantize local bucket
  -> all_gather compressed payload from every rank
  -> dequantize every rank payload locally
  -> reduce sum/mean
  -> optional error-feedback update
  -> complete DDP Future
```

This works and is faster than PyTorch DDP in the A6000 synthetic benchmark, but
its 4-GPU speedup is lower than its 2-GPU speedup because every rank gathers and
dequantizes every other rank payload. The dequant/reduce workload grows with
world size.

## Non-goals

- Do not rewrite NCCL or implement a full standalone collective backend.
- Do not change the public ParaScale integration contract in this phase.
- Do not remove the existing `all_gather` strategy.
- Do not make compressed reduce-scatter the default until it has same-shape
  2/4-GPU benchmark evidence and numerical validation.
- Do not implement FSDP-native reduce-scatter hooks in this phase. FSDP remains
  a baseline/fallback backend.

## Proposed architecture

### 1. Strategy planner

Add a small planner module, for example:

```text
ccdl_comm/communication/strategy.py
```

It should expose a pure, testable function:

```python
plan_ddp_compression_strategy(
    *,
    requested_strategy: str,
    world_size: int,
    bucket_numel: int,
    topology: TopologyInfo | None,
    capabilities: CollectiveCapabilities,
) -> StrategyPlan
```

`StrategyPlan` should include:

- `strategy`: selected strategy, such as `all_gather`, `reduce_scatter`, or
  `hierarchical`;
- `fallback_strategy`: usually `all_gather`;
- `reason`: human-readable explanation for logs and ParaScale reports;
- `requires_fallback`: whether the requested path is unavailable;
- `capability_flags`: which native extension/transport features were detected.

Initial rules:

- `requested_strategy != "auto"` preserves current explicit behavior.
- `auto`, world size `<= 2`: prefer `all_gather`.
- `auto`, world size `>= 4`, reduce-scatter unsupported: choose `all_gather`
  with an explanation that reduce-scatter is unavailable.
- `auto`, world size `>= 4`, reduce-scatter supported: choose
  `reduce_scatter`.
- `hierarchical` remains capability-gated and off by default.

The first implementation can use `torch.distributed.get_world_size()` and a
minimal topology placeholder. Detailed NVLink/PCIe topology parsing should be a
separate follow-up after `strategy="auto"` reports topology-related fallback or
performance bottleneck evidence in benchmark output.

### 2. Compressed reduce-scatter interface

Add an interface layer before implementing a fast transport:

```text
ccdl_comm/collectives/reduce_scatter.py
ccdl_comm/communication/reduce_scatter.py
```

The first public function should be:

```python
compressed_reduce_scatter(
    tensor,
    *,
    config: CompressionConfig,
    op: str = "mean",
    async_op: bool = False,
    dtype: str = "auto",
    reduce_scatter=None,
    all_gather_fallback=None,
    extension_status=None,
)
```

Fast-path contract:

1. Quantize local bucket.
2. Partition the logical output range by world size.
3. Communicate only compressed segments needed for each rank's shard.
4. Dequantize and reduce only the local output shard.
5. Optionally all-gather restored shards only if the caller requires a full DDP
   bucket tensor.

DDP requires the full reduced bucket tensor as hook output. Therefore the first
DDP integration should be explicit about whether it:

- uses reduce-scatter internally and then all-gathers restored shards; or
- keeps reduce-scatter as a lower-level collective for future optimizer/sharded
  paths only.

For native DDP, the first useful target is a hierarchical strategy that reduces
local dequant work while preserving full bucket output semantics.

### 3. Hierarchical compressed collective

For 4/8 single-node GPUs, a safer intermediate strategy is:

```text
within small groups:
  compressed all-gather/reduce
between group leaders:
  compressed exchange/reduce
broadcast/restored result within group
```

This can reduce per-rank payload fan-in compared with flat all-gather while
keeping a full bucket result for DDP.

Initial grouping policy:

- 2 GPUs: one group, use existing all-gather.
- 4 GPUs: two groups of two when topology information is unavailable.
- 8 GPUs: four groups of two or two groups of four, selected by topology when
  available.

The first version should expose the interface and planner choice but fall back
to all-gather until the transport is implemented and verified.

### 4. Fused dequant-reduce-mean-EF kernel

The CUDA fast path should move from Python/C++ orchestration to a true kernel:

```text
compressed payloads + scales + prepared bucket + residual workspace
  -> one CUDA kernel
  -> restored reduced bucket
  -> updated residual
```

Required behavior:

- support INT8 linear group quantization first;
- support contiguous FP32/FP16/BF16 restored output where existing codec supports
  it;
- support `reduce="sum"` and `reduce="mean"`;
- update residual as `prepared - restored` when EF update is enabled;
- write into caller-provided workspace to avoid restored intermediate
  allocation;
- return `False` for unsupported fast-path predicates and let Python use the
  existing safe fallback;
- throw only for caller contract bugs such as mismatched device, non-contiguous
  buffers, or invalid shapes.

The DDP hook must keep CUDA event/Future ordering:

```text
async all-gather completion
  -> fused kernel launch
  -> event record
  -> Future completion after event-safe result
```

### 5. Bucket-level fusion

Bucket-level fusion should be introduced only after the single-bucket fused path
is stable.

The first design should be conservative:

- fuse only small consecutive buckets whose combined numel is below a configured
  limit;
- preserve DDP bucket index ordering for error-feedback state;
- avoid holding references to DDP bucket tensors beyond their safe lifetime;
- expose metrics: fused bucket count, skipped bucket count, fused numel, and
  fallback reason.

## Error handling and fallback

Every new path must be capability-gated:

- missing CUDA extension: fallback to current all-gather;
- unsupported dtype or quantization mode: fallback to current all-gather;
- unsupported world size or topology: fallback to current all-gather;
- reduce-scatter transport unavailable: fallback to current all-gather;
- fused kernel returns `False`: use current safe dequant-reduce + EF update.

Fallback should be visible in benchmark JSON and ParaScale reports. Silent
fallback is not acceptable for performance claims.

## Testing plan

Local tests:

- strategy planner unit tests for 1/2/4/8 world sizes;
- capability-gated fallback tests;
- unsupported strategy error tests;
- compressed reduce-scatter API contract tests with fake transports;
- DDP hook tests proving `strategy="auto"` resolves to expected plan;
- fused kernel export tests and Python wrapper fallback tests.

Remote A6000 tests:

- 2-GPU and 4-GPU synthetic DDP benchmark with:
  - PyTorch DDP;
  - current CCDL all-gather;
  - `strategy="auto"`;
  - reduce-scatter/hierarchical when implemented;
  - fused kernel enabled/disabled.
- Record step time, samples/s, train loss, peak memory, selected strategy, and
  fallback reason.

Correctness thresholds:

- short synthetic loss delta versus DDP should remain near existing levels;
- reduced tensor relative error should be tracked per bucket-size benchmark;
- no non-finite loss in smoke training;
- no DDP Future completion before CUDA result readiness.

Performance gates:

- `strategy="auto"` must not be slower than explicit current all-gather when it
  falls back.
- On 4 GPUs, the new path should target a higher speedup than the current
  `1.38x` CCDL async+EF result against PyTorch DDP before being recommended as
  default.
- If a new fast path is slower, keep it opt-in and report the result.

## Rollout sequence

1. Add planner data structures and `strategy="auto"` resolution with all-gather
   fallback only.
2. Add compressed reduce-scatter interface with fake-transport tests.
3. Add DDP hook wiring for planned strategies, still falling back safely.
4. Add benchmark JSON fields for selected strategy and fallback reason.
5. Implement fused dequant-reduce-mean-EF CUDA fast path with workspace reuse.
6. Validate on A6000 2/4 GPU synthetic benchmark.
7. Add hierarchical strategy implementation if reduce-scatter does not improve
   native DDP full-bucket output semantics enough.
8. Add bucket-level fusion after single-bucket fused path is stable.

## Open decisions

- Whether native DDP should use reduce-scatter plus restored-shard all-gather, or
  reserve reduce-scatter primarily for future sharded optimizer/FSDP-like paths.
  The initial implementation should keep this explicit and benchmark-driven.
- Whether topology detection should parse `nvidia-smi topo -m` directly or accept
  topology injected by ParaScale. The initial version should support injection
  and use a simple world-size heuristic by default.
