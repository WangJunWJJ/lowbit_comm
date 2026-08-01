# Legacy vs refactored CCDL communication benchmark, 2026-07-29

This report compares the pre-refactor `ccdl` package with the refactored
`ccdl_comm` package on the same A6000 host and the same tensor communication
workload.

## Environment

- Host: `user@192.168.8.156 -p 360`
- GPU: NVIDIA RTX A6000
- Docker image: `ccdl-comm-a6000:cu126-torch25`
- PyTorch: `2.5.0a0+872d972e41.nv24.08`
- CUDA: `12.6`
- Tensor: FP16, `16,777,216` elements
- Compression: INT8, group size 64, top-k 0
- Warmup/repeat: `10 / 30`

## Important loading note

Both the legacy and refactored packages import the extension as
`ccdl_cuda_ops`. A stale root-level legacy `.so` existed on the remote host and
failed to import because it missed the expected legacy `quantize` symbol. To
avoid comparing against a broken binary, the benchmark was run from
`/workspace/ccdl_comm_refactor` with:

```bash
PYTHONPATH=/workspace/ccdl_comm_refactor:/workspace
```

That makes both Python APIs use the already rebuilt and validated refactor CUDA
extension. The legacy Python path is still `ccdl.comm.qall_reduce`; the refactor
path is `ccdl_comm.compressed_all_reduce(strategy="all_gather")`.

This isolates the Python communication-path refactor and current collective
strategy, but it is not a bit-for-bit test of the old stale `.so` artifact.

## Strict same-algorithm comparison: gather vs all_gather

Legacy uses `qall_reduce(..., method="gather")`; refactor uses
`compressed_all_reduce(..., strategy="all_gather")`.

| GPUs | Legacy CCDL gather ms | Refactor CCDL all_gather ms | Refactor / Legacy | Legacy rel L2 | Refactor rel L2 |
| ---: | --------------------: | --------------------------: | ----------------: | ------------: | --------------: |
| 2 | 3.452 | 3.544 | 1.027x latency | 0.0059406 | 0.0059406 |
| 4 | 8.497 | 8.725 | 1.027x latency | 0.0059512 | 0.0059512 |

Interpretation:

- Under the strict gather/all-gather communication shape, the refactored path is
  roughly performance-neutral but about `2.6%` slower in this run.
- Numerical error is identical under this口径.

## Legacy preferred/topology path vs current refactor default

Legacy default behavior uses topology-specific methods: tree for 2 GPUs and p2p
for larger power-of-two world sizes. The current refactored DDP-facing default
path tested here remains all-gather.

| GPUs | Legacy method | Legacy CCDL ms | Refactor all_gather ms | Refactor / Legacy | Legacy rel L2 | Refactor rel L2 |
| ---: | --- | -------------: | ---------------------: | ----------------: | ------------: | --------------: |
| 2 | tree | 2.892 | 3.537 | 1.223x latency | 0.0059406 | 0.0059406 |
| 4 | p2p | 5.171 | 8.754 | 1.693x latency | 0.0078633 | 0.0059512 |

Interpretation:

- Against legacy's faster topology-aware methods, the refactored all-gather path
  is slower:
  - 2 GPUs: refactor is about `18.2%` slower by throughput-equivalent speed
    (`0.818x` speed of legacy tree).
  - 4 GPUs: refactor is about `40.9%` slower by throughput-equivalent speed
    (`0.591x` speed of legacy p2p).
- The refactored path has lower relative L2 than legacy p2p in the 4-GPU test,
  but speed is worse.

## Conclusion

The refactor improved engineering properties around safe import, explicit
contracts, DDP hook integration, async completion, error-feedback scheduling,
workspace ownership, and sharded consumer metadata. It has not yet matched the
legacy package's fastest topology-specific collective implementations.

The immediate performance priority should be to port or redesign the legacy
tree/p2p-style compressed collective under the new safe transport interface,
then combine it with the newer fused dequant-reduce/workspace paths. For 4+ GPU
training, the stronger direction remains true compressed reduce-scatter /
sharded consumption rather than DDP full-bucket all-gather restoration.
