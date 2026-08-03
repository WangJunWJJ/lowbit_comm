# Task 15 Hierarchical Stage Executor Gate

## Scope

Task 15 replaces the legacy monolithic hierarchical prototype in the compiled
CUDA path with an immutable three-stage executor:

1. intra-node compressed reduce-scatter: `full -> shard`;
2. inter-node compressed topology ring: `shard -> shard`;
3. intra-node native NCCL all-gather: `shard -> full`.

For a single node, each inter-node group contains one rank, so stage 2 becomes
an identity operation. Process groups are supplied before compilation or are
created once by an explicit group factory. No process group is created in
`HierarchicalExecutor.run()`.

## Compile-time gates

- Stage declaration order and layout continuity are immutable.
- Compilation accepts only the canonical three-stage CUDA signature; missing,
  reordered, substituted, or extra stages fail before any operation is built.
- Every process-group member list must exactly match the rank's expected
  local or cross-node group.
- When a group factory is used, every rank creates all local groups followed
  by all cross-node groups in one deterministic global order; identical rank
  sets are reused by later stages.
- `world_size == local_world_size * node_count` is required.
- The preceding stage event is inserted as a `wait_stream` dependency on the
  next stage stream; the executor performs no intermediate host `wait()`.
- Ring/tree schedules keep group-local ranks internally and map every P2P peer
  to the bound process group's global rank before calling PyTorch.
- Input and intermediate resources remain owned by the final Work.
- If a later stage fails after prior work was submitted, resources are held in
  a completion-fenced quarantine and released by nonblocking reap only after
  the emergency event is ready.
- Single-node hierarchical remains explicit-only. The CUDA auto table is not
  changed by Task 15.
- Multi-node hierarchical remains non-default until a real multi-node
  correctness and performance gate is available.

The individual stage transports are currently synchronous. Task 15 establishes
device-event ordering and a composable execution model, but does not claim a
fully asynchronous multi-stage NCCL pipeline.

## Eight-rank fake topology validation

All ranks in a two-node, four-ranks-per-node topology are covered. For rank
`r`, compilation verifies:

- local group: the four contiguous ranks on `r`'s node;
- inter group: ranks with the same local-rank index on both nodes;
- restore group: the same local group used by stage 1;
- group-local rank/world sizes: `4 -> 2 -> 4`;
- layouts: `full -> shard -> shard -> full`.

The tests also cover mismatched members, missing/extra/substituted stages,
final-layout mismatch, deterministic one-time group factory creation,
non-contiguous inter-group P2P rank mapping, event-to-stream launch order, and
exception-path resource quarantine.

## A6000 Gate method

- Host: `user@192.168.8.156 -p 360`.
- Container: `ccdl-comm-a6000:cu126-torch25`.
- PyTorch: `2.5.0a0+872d972e41.nv24.08`.
- GPUs: 4 NVIDIA RTX A6000, one node.
- Tensor: 8,388,608 FP16 elements (16 MiB).
- Compression: linear INT8, group size 64.
- Timing: 10 warmups; three samples of 30 operations; maximum rank latency;
  median reported.
- Auto baseline is measured before hierarchy-specific NCCL communicators are
  created, preventing those extra communicators from distorting the baseline.

## Result

| GPUs | Path | Latency | Ratio vs native NCCL | Relative L2 |
|---:|---|---:|---:|---:|
| 4 | native NCCL | 3.543 ms | 1.000x | 0 |
| 4 | auto (`topology`) | 2.711 ms | 1.307x | — |
| 4 | explicit hierarchical stages | 3.078 ms | 1.151x | 0.001825 |

The hierarchical path is `0.881x` as fast as the selected auto topology path,
so it is not performance-recommended for single-node auto selection. It is
still faster than native NCCL in this communication-only case and remains an
explicit, correctly reported strategy with no fallback.

Raw Task 15 evidence is stored in [`raw`](raw).

## Environment limitation

The available A6000 host has five GPUs and cannot provide an 8-rank or
two-node run. Task 15 therefore does not mark multi-node hierarchical as a
production default. Its 8-rank semantics are verified with deterministic fake
groups, while the real GPU gate covers the required four-card single-node
degeneration.
