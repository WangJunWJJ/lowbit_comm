# Native topology reduce-scatter shard validation, 2026-07-29

This report records the migration of legacy CCDL reduce-scatter topology
semantics into an independently callable `ccdl_comm` transport:
`make_native_topology_reduce_scatter_shard`.

The transport returns a `ReducedShard` and is intended for sharded consumers. It
does not restore the full DDP bucket and does not import the legacy `ccdl.comm`
package.

## Environment and workload

- Host: `user@192.168.8.156 -p 360`
- GPU: NVIDIA RTX A6000
- Docker image: `ccdl-comm-a6000:cu126-torch25`
- PyTorch: `2.5.0a0+872d972e41.nv24.08`
- CUDA: `12.6`
- Tensor: FP16 flat tensor
- Elements: `16,777,216`
- Compression: INT8, group size 64
- Warmup: `5`
- Repeat: `15`
- Reference: torch full all-reduce + local shard narrow

## Results

| GPUs | Transport | Topology method | Torch reference ms | CCDL shard ms | CCDL / torch | Relative L2 |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 2 | all_to_all shard | n/a | 4.926 | 1.557 | 0.316 | 0.00594 |
| 2 | topology shard | auto -> p2p | 4.945 | 1.608 | 0.325 | 0.00421 |
| 4 | all_to_all shard | n/a | 7.140 | 2.508 | 0.351 | 0.00594 |
| 4 | topology shard | auto -> ring | 7.149 | 2.527 | 0.353 | 0.00729 |

## Interpretation

- The native topology reduce-scatter shard transport is functional on both
  2-GPU and 4-GPU A6000 runs.
- The topology transport is much faster than the torch full all-reduce reference
  used by this shard benchmark:
  - 2 GPUs: `1.608 / 4.945 = 32.5%` of reference latency.
  - 4 GPUs: `2.527 / 7.149 = 35.3%` of reference latency.
- The existing compressed all-to-all shard transport remains slightly faster in
  this benchmark:
  - 2 GPUs: `1.557 ms` vs topology `1.608 ms`.
  - 4 GPUs: `2.508 ms` vs topology `2.527 ms`.

Because performance is the first priority, this migration keeps topology
reduce-scatter as an explicit optional transport rather than changing the
default shard benchmark transport. It fills the old CCDL functionality gap while
preserving the faster all-to-all path for current sharded benchmarks.
