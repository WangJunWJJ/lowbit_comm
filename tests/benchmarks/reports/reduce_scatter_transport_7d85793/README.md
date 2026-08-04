# Reduce-Scatter Transport A6000 Benchmark

Commit under test: `7d85793 feat(ccdl_comm): add compressed reduce scatter transport`

## Environment

- Host: `user@192.168.8.156 -p 360`
- GPU: 4 x NVIDIA RTX A6000
- Docker image: `ccdl-comm-a6000:cu126-torch25`
- Torch: `2.5.0a0+872d972e41.nv24.08`
- CUDA: `12.6`
- Model: synthetic DDP MLP, 62,914,560 parameters
- Steps: 20, warmup: 5
- Batch size per rank: 16, global batch size: 64
- Compression: INT8, group size 64, error feedback disabled

## Results

| Case | Selected strategy | Avg step ms | Samples/s | Peak memory MB | Train loss |
| --- | --- | ---: | ---: | ---: | ---: |
| `all_gather` | `all_gather` | 36.437 | 1756.44 | 1536.56 | 0.997212 |
| `reduce_scatter` | `reduce_scatter` | 41.689 | 1535.18 | 1404.56 | 0.997212 |
| `auto + reduce_scatter capability` | `reduce_scatter` | 41.931 | 1526.31 | 1404.56 | 0.997212 |

## Interpretation

The new transport validates true compressed reduce-scatter semantics: the main
inter-rank exchange is compressed per-destination shard exchange, and the
reduced shard is restored before the final DDP-compatible full bucket gather.
Training loss matches the all-gather baseline in this smoke benchmark.

Performance is not yet faster for DDP full-bucket semantics:

- Explicit reduce-scatter is `0.874x` the all-gather throughput, or about
  `14.4%` slower by samples/s.
- It reduces peak memory by about `132 MB` per max rank, or `8.6%`.
- The slowdown is expected because this prototype still performs a final
  full-precision all-gather to satisfy DDP's full-gradient-bucket contract.

## Decision

Keep the transport capability-gated. It is useful as the correctness bridge
toward ParaScale/FSDP-style sharded consumers, but it should not replace the
validated all-gather DDP fast path as the default until the final full-precision
gather is removed or overlapped.

Next performance step: implement a sharded-consumer path or a native fused
compressed reduce kernel that writes only the locally needed shard, then expose
that path to ParaScale where a full replicated DDP bucket is not required.
